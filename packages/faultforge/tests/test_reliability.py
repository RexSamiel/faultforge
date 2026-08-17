"""Tests for faultforge._internal.reliability."""

import math

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given, settings
from faultforge._internal.reliability import (
    Accumulator,
    AccuracyDegradationScorer,
    AccuracyScorer,
    MultipleChoiceAccuracyScorer,
    PerplexityScorer,
    ReliabilityScorer,
    SdcScorer,
    Top1SdcScorer,
    find_metric_display_name,
    find_metric_identifier,
    metric_from_identifier,
    requires_golden,
)


def test_accumulator_combine() -> None:
    first = Accumulator(value=3.0, count=10)
    second = Accumulator(value=2.0, count=5)
    assert first.combine(second) == Accumulator(value=5.0, count=15)


def test_accuracy_degradation_scorer_preprocess_golden() -> None:
    golden_logits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.25, 0.424]])
    result = AccuracyDegradationScorer().preprocess_golden(golden_logits)
    assert torch.equal(result, torch.tensor([0, 2]))


def test_top1_sdc_scorer_preprocess_golden() -> None:
    golden_logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 1.0, 1.0]])
    result = Top1SdcScorer().preprocess_golden(golden_logits)
    assert torch.equal(result, torch.tensor([1, 0]))


def test_default_preprocess_golden_is_identity() -> None:
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = AccuracyScorer().preprocess_golden(logits)
    assert torch.equal(result, logits)


def test_accuracy_scorer_batch_metric_all_correct() -> None:
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([1, 0])
    metric = AccuracyScorer().batch_metric(logits, logits, targets)
    assert metric == Accumulator(value=2, count=2)


def test_accuracy_scorer_batch_metric_all_wrong() -> None:
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1])
    metric = AccuracyScorer().batch_metric(logits, logits, targets)
    assert metric == Accumulator(value=0, count=2)


def test_accuracy_scorer_batch_metric_partial() -> None:
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([1, 1, 1])
    metric = AccuracyScorer().batch_metric(logits, logits, targets)
    assert metric == Accumulator(value=2, count=3)


def test_accuracy_scorer_score() -> None:
    assert AccuracyScorer().score(Accumulator(value=3, count=4)) == 75.0


def test_accuracy_degradation_scorer_batch_metric_no_degradation() -> None:
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    golden = torch.tensor([1, 0])
    targets = torch.tensor([1, 0])
    metric = AccuracyDegradationScorer().batch_metric(logits, golden, targets)
    assert metric == Accumulator(value=0, count=2)


def test_accuracy_degradation_scorer_batch_metric_degraded() -> None:
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    golden = torch.tensor([1, 0])
    targets = torch.tensor([1, 0])
    metric = AccuracyDegradationScorer().batch_metric(logits, golden, targets)
    assert metric == Accumulator(value=1, count=2)


def test_accuracy_degradation_scorer_batch_metric_shape_mismatch_raises() -> None:
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    golden = torch.tensor([1, 0, 1])
    targets = torch.tensor([1, 0])
    with pytest.raises(AssertionError):
        AccuracyDegradationScorer().batch_metric(logits, golden, targets)


def test_accuracy_degradation_scorer_score() -> None:
    assert AccuracyDegradationScorer().score(Accumulator(value=1, count=4)) == 25.0


def test_sdc_scorer_batch_metric_nothing_changed() -> None:
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metric = SdcScorer().batch_metric(logits, logits, torch.tensor([0, 0]))
    assert metric == Accumulator(value=4, count=4)


def test_sdc_scorer_batch_metric_everything_changed() -> None:
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    golden = torch.tensor([[9.0, 9.0], [9.0, 9.0]])
    metric = SdcScorer().batch_metric(logits, golden, torch.tensor([0, 0]))
    assert metric == Accumulator(value=0, count=4)


def test_sdc_scorer_batch_metric_shape_mismatch_raises() -> None:
    logits = torch.tensor([[1.0, 2.0]])
    golden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(AssertionError):
        SdcScorer().batch_metric(logits, golden, torch.tensor([0]))


def test_sdc_scorer_score() -> None:
    assert SdcScorer().score(Accumulator(value=3, count=4)) == 25.0


def test_top1_sdc_scorer_batch_metric_prediction_unchanged() -> None:
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    golden = torch.tensor([1, 0])
    metric = Top1SdcScorer().batch_metric(logits, golden, torch.tensor([0, 0]))
    assert metric == Accumulator(value=2, count=2)


def test_top1_sdc_scorer_batch_metric_prediction_changed() -> None:
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    golden = torch.tensor([1, 0])
    metric = Top1SdcScorer().batch_metric(logits, golden, torch.tensor([0, 0]))
    assert metric == Accumulator(value=1, count=2)


def test_top1_sdc_scorer_batch_metric_shape_mismatch_raises() -> None:
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    golden = torch.tensor([1, 0, 1])
    with pytest.raises(AssertionError):
        Top1SdcScorer().batch_metric(logits, golden, torch.tensor([0, 0]))


def test_top1_sdc_scorer_score() -> None:
    assert Top1SdcScorer().score(Accumulator(value=3, count=4)) == 25.0


def _cross_entropy(token_logits: list[float], target: int) -> float:
    """`ln(sum(exp(logits))) - logits[target]`, computed independently of
    `torch.nn.functional.cross_entropy` via plain `math`, so tests checking
    against it catch a real bug rather than just re-deriving the same call.

    Uses the log-sum-exp trick (subtracting off `max_logit` before
    exponentiating) for numerical stability - mathematically identical, but
    avoids overflow for large logits."""
    max_logit = max(token_logits)
    log_sum_exp = max_logit + math.log(
        sum(math.exp(logit - max_logit) for logit in token_logits)
    )
    return log_sum_exp - token_logits[target]


def test_perplexity_scorer_batch_metric() -> None:
    token_logits = [[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]]
    targets = [2, 1]
    expected_log_likelihood_sum = -sum(
        _cross_entropy(logits, target) for logits, target in zip(token_logits, targets)
    )

    logits = torch.tensor([token_logits])
    metric = PerplexityScorer().batch_metric(logits, logits, torch.tensor([targets]))

    assert metric.count == len(targets)
    assert metric.value == pytest.approx(expected_log_likelihood_sum, abs=1e-4)


def test_perplexity_scorer_score() -> None:
    metric = Accumulator(value=-2.450595, count=2)
    expected = math.exp(-metric.value / metric.count)
    assert PerplexityScorer().score(metric) == pytest.approx(expected)


def test_multiple_choice_accuracy_scorer_batch_metric_correct_wins() -> None:
    # 1 question, 2 candidates. Candidate 0 (correct, by convention) has the
    # higher summed log-likelihood, so it should win.
    logits = torch.tensor(
        [
            [[1.0, 2.0, 0.5], [0.3, 0.1, 0.9]],  # candidate 0 (correct)
            [[2.0, 0.0, 1.0], [0.5, 0.5, 0.5]],  # candidate 1 (wrong)
        ]
    )
    targets = torch.tensor(
        [
            [1, 2],
            [0, 1],
        ]
    )
    metric = MultipleChoiceAccuracyScorer(num_choices=2).batch_metric(
        logits, logits, targets
    )
    assert metric == Accumulator(value=1, count=1)


def test_multiple_choice_accuracy_scorer_batch_metric_wrong_wins() -> None:
    # 1 question, 2 candidates. Candidate 1 (wrong) has the higher summed
    # log-likelihood, so the question should be marked incorrect.
    logits = torch.tensor(
        [
            [
                [0.1, 0.2, 3.0],
                [0.1, 0.2, 3.0],
            ],  # candidate 0 (correct, low prob for its own target)
            [
                [3.0, 0.1, 0.2],
                [3.0, 0.1, 0.2],
            ],  # candidate 1 (wrong, high prob for its own target)
        ]
    )
    targets = torch.tensor(
        [
            [0, 0],
            [0, 0],
        ]
    )
    metric = MultipleChoiceAccuracyScorer(num_choices=2).batch_metric(
        logits, logits, targets
    )
    assert metric == Accumulator(value=0, count=1)


def test_multiple_choice_accuracy_scorer_batch_metric_multiple_questions() -> None:
    # 2 questions (4 rows), one where the correct candidate wins and one
    # where it doesn't - correct/total should reflect both.
    logits = torch.tensor(
        [
            [[1.0, 2.0, 0.5], [0.3, 0.1, 0.9]],
            [[2.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
            [[0.1, 0.2, 3.0], [0.1, 0.2, 3.0]],
            [[3.0, 0.1, 0.2], [3.0, 0.1, 0.2]],
        ]
    )
    targets = torch.tensor(
        [
            [1, 2],
            [0, 1],
            [0, 0],
            [0, 0],
        ]
    )
    metric = MultipleChoiceAccuracyScorer(num_choices=2).batch_metric(
        logits, logits, targets
    )
    assert metric == Accumulator(value=1, count=2)


def test_multiple_choice_accuracy_scorer_batch_metric_padding_biases_shorter_candidate() -> (
    None
):
    # Candidate 1 is padded (-100) after 1 real token, so it only accumulates
    # one token's worth of negative log-likelihood - this can make it "win"
    # over a longer, more-likely-per-token correct candidate. This documents
    # the known length-bias of unnormalized log-likelihood summing, it does
    # not mean the implementation is wrong.
    logits = torch.tensor(
        [
            [[1.0, 2.0, 0.5], [0.3, 0.1, 0.9]],  # candidate 0 (correct), 2 real tokens
            [
                [2.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],  # candidate 1 (wrong), 1 real token + 1 padding
        ]
    )
    targets = torch.tensor(
        [
            [1, 2],
            [0, -100],
        ]
    )
    metric = MultipleChoiceAccuracyScorer(num_choices=2).batch_metric(
        logits, logits, targets
    )
    assert metric == Accumulator(value=0, count=1)


def test_multiple_choice_accuracy_scorer_batch_metric_shape_mismatch_raises() -> None:
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([0, 0])
    with pytest.raises(AssertionError):
        MultipleChoiceAccuracyScorer(num_choices=2).batch_metric(
            logits, logits, targets
        )


def test_multiple_choice_accuracy_scorer_score() -> None:
    assert (
        MultipleChoiceAccuracyScorer(num_choices=4).score(Accumulator(value=3, count=4))
        == 75.0
    )


@pytest.mark.parametrize(
    ("identifier", "scorer_class"),
    [
        ("accuracy", AccuracyScorer),
        ("accuracy_degradation", AccuracyDegradationScorer),
        ("sdc", SdcScorer),
        ("top1_sdc", Top1SdcScorer),
        ("perplexity", PerplexityScorer),
        ("multiple_choice_accuracy", MultipleChoiceAccuracyScorer),
    ],
)
def test_metric_from_identifier_resolves_registered_class(
    identifier: str, scorer_class: type[ReliabilityScorer]
) -> None:
    assert metric_from_identifier(identifier) is scorer_class


def test_metric_from_identifier_unknown_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        metric_from_identifier("bogus")


@pytest.mark.parametrize(
    ("scorer", "identifier"),
    [
        (AccuracyScorer(), "accuracy"),
        (AccuracyDegradationScorer(), "accuracy_degradation"),
        (SdcScorer(), "sdc"),
        (Top1SdcScorer(), "top1_sdc"),
        (PerplexityScorer(), "perplexity"),
        (MultipleChoiceAccuracyScorer(num_choices=4), "multiple_choice_accuracy"),
    ],
)
def test_find_metric_identifier_round_trips_with_metric_from_identifier(
    scorer: ReliabilityScorer, identifier: str
) -> None:
    assert find_metric_identifier(scorer) == identifier


@pytest.mark.parametrize(
    ("scorer", "expected"),
    [
        (AccuracyScorer(), False),
        (AccuracyDegradationScorer(), True),
        (SdcScorer(), True),
        (Top1SdcScorer(), True),
        (PerplexityScorer(), False),
        (MultipleChoiceAccuracyScorer(num_choices=4), False),
    ],
)
def test_requires_golden_matches_each_metric(
    scorer: ReliabilityScorer, expected: bool
) -> None:
    assert requires_golden(scorer) is expected


@pytest.mark.parametrize(
    ("scorer", "display_name"),
    [
        (AccuracyScorer(), "Accuracy"),
        (AccuracyDegradationScorer(), "Accuracy Degradation"),
        (SdcScorer(), "SDC"),
        (Top1SdcScorer(), "Top-1 SDC"),
        (PerplexityScorer(), "Perplexity"),
        (MultipleChoiceAccuracyScorer(num_choices=4), "Multiple Choice Accuracy"),
    ],
)
def test_find_metric_display_name_matches_each_metric(
    scorer: ReliabilityScorer, display_name: str
) -> None:
    assert find_metric_display_name(scorer) == display_name


_UNSIGNED_PERCENTAGE_SCORERS: list[type[ReliabilityScorer]] = [
    AccuracyScorer,
    SdcScorer,
    Top1SdcScorer,
]
"""Percentage scorers whose `value` is always in `[0, count]` (unlike
`AccuracyDegradationScorer`, which is signed - see
`test_accuracy_degradation_scorer_score_can_go_negative`)."""


@pytest.mark.parametrize("scorer_class", _UNSIGNED_PERCENTAGE_SCORERS)
@given(
    correct=st.integers(min_value=0, max_value=1000),
    extra=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=50)
def test_unsigned_percentage_scorers_score_is_bounded(
    scorer_class: type[ReliabilityScorer], correct: int, extra: int
) -> None:
    metric = Accumulator(value=correct, count=correct + extra + 1)
    result = scorer_class().score(metric)
    assert 0.0 <= result <= 100.0


@given(
    correct=st.integers(min_value=0, max_value=1000),
    extra=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=50)
def test_multiple_choice_accuracy_scorer_score_is_bounded(
    correct: int, extra: int
) -> None:
    metric = Accumulator(value=correct, count=correct + extra + 1)
    result = MultipleChoiceAccuracyScorer(num_choices=4).score(metric)
    assert 0.0 <= result <= 100.0


@given(
    value=st.integers(min_value=-1000, max_value=1000),
    count=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=50)
def test_accuracy_degradation_scorer_score_can_go_negative(
    value: int, count: int
) -> None:
    # Unlike the other percentage scorers, degradation is signed: a faulty
    # model can outscore golden (`golden_correct - correct < 0`), which is a
    # legitimate negative "improvement", not a bug - so this asserts the
    # formula directly instead of a false 0-100 bound.
    result = AccuracyDegradationScorer().score(Accumulator(value=value, count=count))
    assert result == pytest.approx(value / count * 100)
