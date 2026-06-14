"""Leakage-safe current-signal generation from the latest completed candle."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from src.explainability import explain_latest_prediction
from src.features import create_features
from src.live_data import assess_market_data_freshness
from src.models import LEAKAGE_COLUMNS


def build_latest_unlabeled_signal(
    raw_df: pd.DataFrame,
    labeled_dataset: pd.DataFrame,
    feature_cols: list[str],
    model_trainer: Callable[[pd.DataFrame, pd.Series], Any],
    excluded_features: tuple[str, ...] = (),
    as_of: pd.Timestamp | str | None = None,
    asset_type: str = "equity",
) -> dict[str, Any]:
    """Train on labeled history and predict the latest feature-only market row."""
    if raw_df.empty or labeled_dataset.empty:
        raise ValueError("Raw and labeled market data must not be empty.")
    if "target" not in labeled_dataset.columns:
        raise ValueError("labeled_dataset must contain a target column.")
    if not callable(model_trainer):
        raise TypeError("model_trainer must be callable.")

    resolved_features = [
        feature
        for feature in feature_cols
        if feature not in excluded_features and feature not in LEAKAGE_COLUMNS
    ]
    if not resolved_features:
        raise ValueError("No leakage-safe feature columns remain for prediction.")
    missing_labeled = set(resolved_features).difference(labeled_dataset.columns)
    if missing_labeled:
        raise ValueError(
            f"Labeled dataset is missing features: {sorted(missing_labeled)}"
        )

    full_features = create_features(raw_df)
    missing_latest = set(resolved_features).difference(full_features.columns)
    if missing_latest:
        raise ValueError(f"Latest market data is missing features: {sorted(missing_latest)}")
    latest_row = full_features.tail(1).copy()
    model = model_trainer(
        labeled_dataset[resolved_features],
        labeled_dataset["target"],
    )
    explanation = explain_latest_prediction(
        model,
        latest_row,
        resolved_features,
        top_n=10,
    )

    latest_date = pd.to_datetime(latest_row.iloc[0]["date"])
    labeled_date = pd.to_datetime(labeled_dataset["date"]).max()
    resolved_as_of = (
        pd.Timestamp(as_of)
        if as_of is not None
        else pd.Timestamp.now(tz="UTC").tz_localize(None)
    )
    if resolved_as_of.tz is not None:
        resolved_as_of = resolved_as_of.tz_convert("UTC").tz_localize(None)
    freshness = assess_market_data_freshness(
        raw_df,
        interval="1d",
        asset_type=asset_type,
        as_of=resolved_as_of,
    )
    freshness_minutes = float(freshness["data_age_minutes"])
    is_stale = bool(freshness["is_stale"])
    warnings: list[str] = []
    if latest_date > labeled_date:
        warnings.append(
            "Current signal uses a newer unlabeled candle than the latest "
            "backtest-evaluable row because its future outcome is not known yet."
        )
    if is_stale:
        warnings.append(str(freshness["message"]))

    return {
        "latest_date": latest_date,
        "latest_close": float(latest_row.iloc[0]["close"]),
        "prediction": int(explanation["prediction"]),
        "confidence": explanation["confidence"],
        "probabilities": explanation["probabilities"],
        "top_features": explanation["top_features"],
        "feature_row": latest_row,
        "model": model,
        "feature_cols": resolved_features,
        "latest_labeled_date": labeled_date,
        "latest_is_newer_than_labeled": bool(latest_date > labeled_date),
        "data_freshness_minutes": freshness_minutes,
        "freshness_status": "stale" if is_stale else "fresh",
        "is_stale": is_stale,
        "market_status": freshness["market_status"],
        "freshness_message": freshness["message"],
        "signal_source": "latest completed candle",
        "warnings": warnings,
    }
