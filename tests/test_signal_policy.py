"""Tests for confidence-aware signal abstention research."""

from __future__ import annotations

import pandas as pd
import pytest

from src.signal_policy import apply_confidence_abstention, evaluate_abstention_thresholds


@pytest.fixture
def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "close": [100, 101, 100, 102, 101],
            "prediction": [1, -1, 1, -1, 1],
            "confidence": [0.40, 0.60, 0.55, 0.30, 0.80],
        }
    )


def test_apply_confidence_abstention_neutralizes_low_confidence(prediction_frame):
    filtered = apply_confidence_abstention(prediction_frame, 0.50)

    assert filtered["prediction"].tolist() == [0, -1, 1, 0, 1]
    assert filtered["raw_prediction"].tolist() == prediction_frame["prediction"].tolist()
    assert filtered["abstained"].tolist() == [True, False, False, True, False]


def test_evaluate_abstention_thresholds_preserves_fixed_grid(prediction_frame):
    scenarios = evaluate_abstention_thresholds(
        prediction_frame,
        thresholds=[0.6, 0.0, 0.5, 0.5],
    )

    assert scenarios["minimum_confidence"].tolist() == [0.0, 0.5, 0.6]
    assert scenarios.loc[0, "abstention_rate"] == 0
    assert scenarios["confidence_coverage"].is_monotonic_decreasing


def test_apply_confidence_abstention_rejects_invalid_threshold(prediction_frame):
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_confidence_abstention(prediction_frame, 1.1)
