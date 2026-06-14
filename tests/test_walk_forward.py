"""Tests for purged walk-forward validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import train_logistic_regression
from src.walk_forward import run_walk_forward_validation


@pytest.fixture
def walk_forward_dataset() -> pd.DataFrame:
    """Return deterministic classification data for multiple folds."""
    rows = 80
    feature = np.sin(np.arange(rows) / 4)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="D"),
            "close": 100 + np.arange(rows),
            "feature": feature,
            "target": np.select([feature > 0.25, feature < -0.25], [1, -1], default=0),
        }
    )


def test_walk_forward_predictions_are_unique_and_out_of_sample(walk_forward_dataset):
    results = run_walk_forward_validation(
        walk_forward_dataset,
        ["feature"],
        train_logistic_regression,
        train_window=30,
        test_window=10,
        step_size=10,
        purge_rows=3,
    )

    assert len(results) == 47
    assert results["source_index"].is_unique
    assert results["source_index"].min() == 33
    assert results.groupby("fold").size().tolist() == [10, 10, 10, 10, 7]
    assert (
        results["train_end_date"] < results["test_start_date"]
    ).all()
    assert {"probability_-1", "probability_0", "probability_1", "confidence"}.issubset(
        results.columns
    )


def test_walk_forward_refits_model_for_every_fold(walk_forward_dataset):
    train_boundaries: list[tuple[int, int]] = []

    def recording_factory(X_train, y_train):
        train_boundaries.append((int(X_train.index.min()), int(X_train.index.max())))
        return train_logistic_regression(X_train, y_train)

    results = run_walk_forward_validation(
        walk_forward_dataset,
        ["feature"],
        recording_factory,
        train_window=20,
        test_window=10,
        step_size=10,
        purge_rows=2,
    )

    assert len(train_boundaries) == results["fold"].nunique()
    assert len(set(train_boundaries)) == len(train_boundaries)


def test_walk_forward_rejects_overlapping_test_windows(walk_forward_dataset):
    with pytest.raises(ValueError, match="avoid overlap"):
        run_walk_forward_validation(
            walk_forward_dataset,
            ["feature"],
            train_logistic_regression,
            train_window=30,
            test_window=10,
            step_size=5,
        )


def test_walk_forward_rejects_target_leakage(walk_forward_dataset):
    walk_forward_dataset["future_return"] = 0.01

    with pytest.raises(ValueError, match="target leakage"):
        run_walk_forward_validation(
            walk_forward_dataset,
            ["feature", "future_return"],
            train_logistic_regression,
            train_window=30,
            test_window=10,
            step_size=10,
        )


def test_walk_forward_rejects_non_callable_model_factory(walk_forward_dataset):
    with pytest.raises(TypeError, match="must be callable"):
        run_walk_forward_validation(
            walk_forward_dataset,
            ["feature"],
            None,
            train_window=30,
            test_window=10,
            step_size=10,
        )
