"""Tests for lightweight model explainability."""

from __future__ import annotations

from src.explainability import explain_latest_prediction, get_feature_importance
from src.models import (
    chronological_train_test_split,
    train_logistic_regression,
    train_random_forest,
)


def test_tree_feature_importance_is_normalized(model_dataset):
    feature_cols = ["feature_1", "feature_2"]
    X_train, _, y_train, _ = chronological_train_test_split(
        model_dataset, feature_cols, test_size=0.2
    )
    model = train_random_forest(X_train, y_train, n_estimators=20)
    importance = get_feature_importance(model, feature_cols)

    assert importance["feature"].tolist()
    assert abs(importance["importance"].sum() - 1) < 1e-9
    assert set(importance["method"]) == {"tree_importance"}


def test_latest_prediction_explanation_contains_probabilities(model_dataset):
    feature_cols = ["feature_1", "feature_2"]
    X_train, X_test, y_train, _ = chronological_train_test_split(
        model_dataset, feature_cols, test_size=0.2
    )
    model = train_logistic_regression(X_train, y_train)
    explanation = explain_latest_prediction(
        model, X_test.tail(1), feature_cols, top_n=2
    )

    assert explanation["prediction"] in {-1, 0, 1}
    assert 0 <= explanation["confidence"] <= 1
    assert len(explanation["top_features"]) == 2
    assert set(explanation["probabilities"]) == {"-1", "0", "1"}
