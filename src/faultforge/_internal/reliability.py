"""Reliability scoring for neural network models."""

import abc
import math
from dataclasses import dataclass
from typing import Self, override

import torch.nn.functional as F
from torch import Tensor


@dataclass(slots=True, frozen=True)
class Accumulator:
    """Holds and accumulates batch data that has been calculated for a metric."""

    value: float
    count: int

    def combine(self, other: Self) -> Self:
        return type(self)(self.value + other.value, self.count + other.count)


class ReliabilityScorer(abc.ABC):
    """Template for measuring the reliability of models."""

    def preprocess_golden(self, result: Tensor) -> Tensor:
        """Preprocess the golden result values before calculation"""
        return result

    @abc.abstractmethod
    def batch_metric(
        self, batch_result: Tensor, golden_result: Tensor, targets: Tensor
    ) -> Accumulator:
        """Create a metric from a batch results."""

    @abc.abstractmethod
    def score(self, metric: Accumulator) -> float:
        """Format calculated metric and give a numeric score to the metric."""


_METRIC_IDENTIFIERS: dict[str, type[ReliabilityScorer]] = {}
_DISPLAY_NAMES: dict[type[ReliabilityScorer], str] = {}
_REQUIRES_GOLDEN: set[type[ReliabilityScorer]] = set()


def register_metric(
    display_name: str, identifier: str, *, requires_golden: bool = False
):
    """Wrapper for registering a reliability metrics display name, identifier, and golden results requirement."""

    def decorator(scorer_class: type[ReliabilityScorer]) -> type[ReliabilityScorer]:
        _METRIC_IDENTIFIERS[identifier] = scorer_class
        _DISPLAY_NAMES[scorer_class] = display_name
        if requires_golden:
            _REQUIRES_GOLDEN.add(scorer_class)
        return scorer_class

    return decorator


@register_metric("Accuracy", "accuracy")
class AccuracyScorer(ReliabilityScorer):
    """Percentage of predictions matching the dataset's true labels."""

    @override
    def batch_metric(
        self, batch_result: Tensor, golden_result: Tensor, targets: Tensor
    ) -> Accumulator:
        assert batch_result.dim() == 2, "AccuracyScorer expects [batch, classes] logits"
        classifications = batch_result.argmax(dim=1)
        correct = int((classifications == targets).sum().item())
        return Accumulator(value=correct, count=classifications.numel())

    @override
    def score(self, metric: Accumulator) -> float:
        return metric.value / metric.count * 100


@register_metric("Accuracy Degradation", "accuracy_degradation", requires_golden=True)
class AccuracyDegradationScorer(ReliabilityScorer):
    """Golden model accuracy minus faulty model accuracy."""

    @override
    def preprocess_golden(self, result: Tensor) -> Tensor:
        return result.argmax(dim=1)

    @override
    def batch_metric(
        self, batch_result: Tensor, golden_result: Tensor, targets: Tensor
    ) -> Accumulator:
        assert batch_result.dim() == 2, (
            "AccuracyDegradationScorer expects [batch, classes] logits"
        )
        classifications = batch_result.argmax(dim=1)
        assert golden_result.shape == classifications.shape
        correct = int((classifications == targets).sum().item())
        golden_correct = int((golden_result == targets).sum().item())
        return Accumulator(
            value=golden_correct - correct, count=classifications.numel()
        )

    @override
    def score(self, metric: Accumulator) -> float:
        return metric.value / metric.count * 100


@register_metric("SDC", "sdc", requires_golden=True)
class SdcScorer(ReliabilityScorer):
    """Silent Data Corruption is the percentage of output values that changed
    versus the golden (unfaulted) run, regardless of whether the prediction
    itself changed."""

    @override
    def batch_metric(
        self, batch_result: Tensor, golden_result: Tensor, targets: Tensor
    ) -> Accumulator:
        assert batch_result.dim() == 2, "SdcScorer expects [batch, classes] logits"
        assert golden_result.shape == batch_result.shape
        unchanged = int((batch_result == golden_result).sum().item())
        return Accumulator(value=unchanged, count=golden_result.numel())

    @override
    def score(self, metric: Accumulator) -> float:
        return 100 - metric.value / metric.count * 100


@register_metric("Top-1 SDC", "top1_sdc", requires_golden=True)
class Top1SdcScorer(ReliabilityScorer):
    """Critical top-1SDC is the percentage of the highest predictions that changed vs
    golden fault-free predictions, regardless of whether they're correct."""

    @override
    def preprocess_golden(self, result: Tensor) -> Tensor:
        return result.argmax(dim=1)

    @override
    def batch_metric(
        self, batch_result: Tensor, golden_result: Tensor, targets: Tensor
    ) -> Accumulator:
        assert batch_result.dim() == 2, "Top1SdcScorer expects [batch, classes] logits"
        classifications = batch_result.argmax(dim=1)
        assert golden_result.shape == classifications.shape
        unchanged = int((classifications == golden_result).sum().item())
        return Accumulator(value=unchanged, count=classifications.numel())

    @override
    def score(self, metric: Accumulator) -> float:
        return 100 - metric.value / metric.count * 100


@register_metric("Perplexity", "perplexity")
class PerplexityScorer(ReliabilityScorer):
    """How surprised the model is by real text.

    Perplexity calculation pipeline is taken from the HuggingFace implementation.
    https://huggingface.co/docs/transformers/en/perplexity"""

    @override
    def batch_metric(
        self, batch_result: Tensor, golden_result: Tensor, targets: Tensor
    ) -> Accumulator:
        assert batch_result.dim() == 3, (
            "PerplexityScorer expects [batch, sequence_length, vocabulary_size] logits"
        )
        mask = targets != -100
        vocabulary_size = batch_result.shape[-1]
        log_likelihood_sum = -F.cross_entropy(
            batch_result.view(-1, vocabulary_size),
            targets.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return Accumulator(
            value=log_likelihood_sum.item(), count=int(mask.sum().item())
        )

    @override
    def score(self, metric: Accumulator) -> float:
        average_log_prob = metric.value / metric.count
        return math.exp(-average_log_prob)


@register_metric("Multiple Choice Accuracy", "multiple_choice_accuracy")
class MultipleChoiceAccuracyScorer(ReliabilityScorer):
    """Score accuracy for multiple choice tasks

    Multiple choice accuracy is calculated in the same principle as meta describes mmlu
    evaluation and lm-evaluation-harness does it.
    https://github.com/EleutherAI/lm-evaluation-harness
    https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md"""

    def __init__(self, num_choices: int) -> None:
        self.num_choices = num_choices

    @override
    def batch_metric(
        self, batch_result: Tensor, golden_result: Tensor, targets: Tensor
    ) -> Accumulator:
        assert batch_result.dim() == 3, (
            "MultipleChoiceAccuracyScorer expects [batch, sequence_length, vocabulary_size] logits"
        )
        vocabulary_size = batch_result.shape[-1]

        negative_log_likelihood = F.cross_entropy(
            batch_result.view(-1, vocabulary_size),
            targets.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        negative_log_likelihood = negative_log_likelihood.view(targets.shape)
        log_likelihood = -negative_log_likelihood.sum(dim=-1)

        grouped = log_likelihood.view(-1, self.num_choices)
        predicted_choice = grouped.argmax(dim=-1)
        correct = int((predicted_choice == 0).sum().item())
        return Accumulator(value=correct, count=grouped.shape[0])

    @override
    def score(self, metric: Accumulator) -> float:
        return metric.value / metric.count * 100


def registered_identifiers() -> list[str]:
    """Every identifier currently registered via `register_metric`.

    For building CLI choices/help text dynamically from whatever's
    registered, rather than a separately-maintained hardcoded list.
    """
    return list(_METRIC_IDENTIFIERS.keys())


def metric_from_identifier(identifier: str) -> type[ReliabilityScorer]:
    """Look up which scorer class a stored `metric_identifier` refers to."""
    try:
        return _METRIC_IDENTIFIERS[identifier]
    except KeyError:
        raise ValueError(
            f"Unknown reliability metric identifier: {identifier!r}"
        ) from None


def find_metric_identifier(scorer: ReliabilityScorer) -> str:
    """The identifier string for an already-constructed scorer instance."""
    for identifier, scorer_class in _METRIC_IDENTIFIERS.items():
        if scorer_class is type(scorer):
            return identifier
    raise ValueError(f"No identifier registered for {type(scorer).__name__}")


def requires_golden(scorer: ReliabilityScorer) -> bool:
    """Whether this scorer needs a golden (unfaulted) baseline to compare against."""
    return type(scorer) in _REQUIRES_GOLDEN


def find_metric_display_name(
    scorer: ReliabilityScorer | type[ReliabilityScorer],
) -> str:
    """The display name for a scorer instance or class."""
    if isinstance(scorer, ReliabilityScorer):
        scorer_class = type(scorer)
    else:
        scorer_class = scorer
    return _DISPLAY_NAMES[scorer_class]
