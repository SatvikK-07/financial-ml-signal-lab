"""Automated plausibility diagnostics for financial research results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def analyze_research_plausibility(research: dict[str, Any]) -> dict[str, Any]:
    """Flag unusually strong or structurally suspicious research results."""
    dataset = research["dataset"]
    backtest = research["backtest"]
    summary = research["backtest_summary"]
    signal_return_correlation = _safe_correlation(
        backtest["executed_position"], backtest["asset_return"]
    )
    return_autocorrelation = float(backtest["asset_return"].autocorr(1))
    zero_volume_fraction = (
        float((dataset["volume"] == 0).mean()) if "volume" in dataset.columns else 0.0
    )
    gross_total_return = float(
        (1 + backtest["strategy_return_before_cost"]).prod() - 1
    )
    maximum_absolute_daily_return = float(backtest["asset_return"].abs().max())
    next_return = dataset["close"].pct_change().shift(-1)
    feature_return_correlations = {
        feature: _safe_correlation(dataset[feature], next_return)
        for feature in research["feature_cols"]
    }
    strongest_feature = max(
        feature_return_correlations,
        key=lambda feature: abs(feature_return_correlations[feature]),
    )
    strongest_feature_correlation = feature_return_correlations[strongest_feature]

    warnings: list[str] = []
    if abs(strongest_feature_correlation) > 0.2:
        warnings.append(
            f"Feature {strongest_feature} has unusually high correlation with "
            "next-period returns."
        )
    if abs(signal_return_correlation) > 0.2:
        warnings.append(
            "Delayed positions have unusually high correlation with next-period returns."
        )
    if abs(summary["sharpe_ratio"]) > 3:
        warnings.append(
            "Absolute Sharpe ratio exceeds 3 and requires independent review."
        )
    if gross_total_return > 10:
        warnings.append(
            "Gross strategy return exceeds 1,000% and requires stress testing."
        )
    if zero_volume_fraction > 0.95:
        warnings.append("Volume is missing or zero for nearly all rows.")
    if maximum_absolute_daily_return > 0.5:
        warnings.append("At least one daily asset return exceeds 50%.")

    return {
        "status": "review_required" if warnings else "no_automatic_flags",
        "warnings": warnings,
        "signal_return_correlation": signal_return_correlation,
        "return_autocorrelation_1": return_autocorrelation,
        "zero_volume_fraction": zero_volume_fraction,
        "gross_total_return": gross_total_return,
        "maximum_absolute_daily_return": maximum_absolute_daily_return,
        "strongest_feature_return_correlation": {
            "feature": strongest_feature,
            "correlation": strongest_feature_correlation,
        },
    }


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return 0.0
    correlation = left.corr(right)
    return float(correlation) if np.isfinite(correlation) else 0.0
