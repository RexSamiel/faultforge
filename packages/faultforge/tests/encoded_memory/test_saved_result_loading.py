"""Tests for SavedResult standalone loading."""

import pytest
from faultforge._internal.reliability import AccuracyScorer
from faultforge.experiments.encoded_memory import SavedResult

from .conftest import _make_experiment


def test_saved_result_round_trip_scores_and_bit_error_rate(tmp_path):
    experiment = _make_experiment(compare_bitwise=True, faults=3)
    experiment.run()
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save(path)

    loaded = SavedResult.load(path)
    assert loaded.scores() == list(experiment.scores())
    assert loaded.reliability_metric_class() is AccuracyScorer
    assert loaded.bit_error_rate() == pytest.approx(3 / loaded.total_bits)


def test_saved_result_scores_empty_before_first_run():
    experiment = _make_experiment(compare_bitwise=False)
    saved = SavedResult.model_validate_json(experiment.serialize())
    assert saved.total_items is None
    assert saved.scores() == []
