"""Lightweight global and latest-signal model explainability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models import get_prediction_proba, get_predictions


def get_feature_importance(model: Any, feature_cols: list[str]) -> pd.DataFrame:
    """Return normalized global feature importance for supported classifiers."""
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature.")
    estimator = _unwrap_estimator(model)
    if hasattr(estimator, "feature_importances_"):
        raw_importance = np.asarray(estimator.feature_importances_, dtype=float)
        direction = np.full(len(feature_cols), np.nan)
        method = "tree_importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.atleast_2d(np.asarray(estimator.coef_, dtype=float))
        raw_importance = np.mean(np.abs(coefficients), axis=0)
        direction = np.mean(coefficients, axis=0)
        method = "coefficient"
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")
    if len(raw_importance) != len(feature_cols):
        raise ValueError("Model importance length does not match feature columns.")

    total = raw_importance.sum()
    normalized = raw_importance / total if total > 0 else raw_importance
    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": normalized,
                "direction": direction,
                "method": method,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def explain_latest_prediction(
    model: Any,
    latest_row: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 10,
) -> dict[str, Any]:
    """Explain the latest prediction with probabilities and top global features."""
    if len(latest_row) != 1:
        raise ValueError("latest_row must contain exactly one row.")
    missing = set(feature_cols).difference(latest_row.columns)
    if missing:
        raise ValueError(f"latest_row is missing features: {sorted(missing)}")
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    X_latest = latest_row[feature_cols]
    prediction = get_predictions(model, X_latest)[0]
    probabilities = get_prediction_proba(model, X_latest)
    classes = getattr(model, "classes_", None)
    probability_map: dict[str, float] | None = None
    confidence: float | None = None
    if probabilities is not None:
        if classes is None:
            classes = np.arange(probabilities.shape[1])
        probability_map = {
            str(label): float(probabilities[0, index])
            for index, label in enumerate(classes)
        }
        confidence = float(probabilities[0].max())

    importance = get_feature_importance(model, feature_cols).head(top_n).copy()
    importance["value"] = importance["feature"].map(latest_row.iloc[0])
    return {
        "prediction": _to_python_scalar(prediction),
        "confidence": confidence,
        "probabilities": probability_map,
        "top_features": importance.to_dict(orient="records"),
    }


def _unwrap_estimator(model: Any) -> Any:
    estimator = (
        model.estimator
        if hasattr(model, "label_encoder") and hasattr(model, "estimator")
        else model
    )
    if hasattr(estimator, "named_steps"):
        estimator = list(estimator.named_steps.values())[-1]
    return estimator


def _to_python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
