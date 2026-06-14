"""Tests for chronological model training and evaluation."""

from __future__ import annotations

from functools import partial

import pandas as pd
import pytest

from src.models import (
    chronological_train_test_split,
    evaluate_classifier,
    get_prediction_proba,
    get_predictions,
    train_logistic_regression,
    train_naive_baseline,
    train_random_forest,
)


def test_chronological_split_never_shuffles(model_dataset):
    shuffled = model_dataset.sample(frac=1, random_state=42)
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        shuffled,
        ["feature_1", "feature_2"],
        test_size=24,
    )

    assert len(X_train) == 96
    assert len(X_test) == 24
    assert y_train.index.equals(X_train.index)
    assert y_test.index.equals(X_test.index)
    assert model_dataset.loc[X_train.index, "date"].max() < model_dataset.loc[
        X_test.index, "date"
    ].min()


def test_chronological_split_purges_overlapping_target_rows(model_dataset):
    X_train, X_test, _, _ = chronological_train_test_split(
        model_dataset,
        ["feature_1", "feature_2"],
        test_size=24,
        purge_rows=5,
    )

    assert len(X_train) == 91
    assert len(X_test) == 24
    assert model_dataset.loc[X_train.index, "date"].max() < model_dataset.loc[
        X_test.index, "date"
    ].min() - pd.Timedelta(days=5)


def test_chronological_split_rejects_target_leakage(model_dataset):
    model_dataset["future_return"] = 0.01

    with pytest.raises(ValueError, match="target leakage"):
        chronological_train_test_split(model_dataset, ["feature_1", "future_return"])


@pytest.mark.parametrize(
    "trainer",
    [
        train_naive_baseline,
        train_logistic_regression,
        partial(train_random_forest, n_estimators=20),
    ],
)
def test_trainers_predict_and_evaluate(model_dataset, trainer):
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        model_dataset,
        ["feature_1", "feature_2"],
        test_size=0.2,
    )
    model = trainer(X_train, y_train)
    predictions = get_predictions(model, X_test)
    metrics = evaluate_classifier(model, X_test, y_test)

    assert len(predictions) == len(y_test)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert metrics["labels"]
    assert get_prediction_proba(model, X_test).shape[0] == len(y_test)
