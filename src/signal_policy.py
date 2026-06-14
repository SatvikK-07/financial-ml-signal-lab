"""Confidence-aware signal abstention research utilities."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.backtester import run_backtest, summarize_backtest


def apply_confidence_abstention(
    predictions: pd.DataFrame,
    minimum_confidence: float,
    prediction_col: str = "prediction",
    confidence_col: str = "confidence",
) -> pd.DataFrame:
    """Replace predictions below a fixed confidence threshold with neutral."""
    _validate_threshold(minimum_confidence)
    required = {prediction_col, confidence_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Predictions are missing signal-policy columns: {sorted(missing)}"
        )
    if predictions[list(required)].isna().any().any():
        raise ValueError("Prediction and confidence columns must not contain missing values.")

    result = predictions.copy()
    result["raw_prediction"] = result[prediction_col]
    result["abstained"] = result[confidence_col] < minimum_confidence
    result.loc[result["abstained"], prediction_col] = 0
    return result


def evaluate_abstention_thresholds(
    predictions: pd.DataFrame,
    thresholds: Iterable[float],
    initial_capital: float = 10_000,
    transaction_cost: float = 0.0005,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """Evaluate predefined confidence thresholds without selecting a winner."""
    normalized_thresholds = sorted(set(float(value) for value in thresholds))
    if not normalized_thresholds:
        raise ValueError("At least one confidence threshold is required.")

    rows: list[dict[str, float | int]] = []
    for threshold in normalized_thresholds:
        filtered = apply_confidence_abstention(predictions, threshold)
        backtest = run_backtest(
            filtered,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            position_size=position_size,
        )
        summary = summarize_backtest(backtest)
        rows.append(
            {
                "minimum_confidence": threshold,
                "confidence_coverage": float((~filtered["abstained"]).mean()),
                "abstention_rate": float(filtered["abstained"].mean()),
                "active_signal_rate": float((filtered["prediction"] != 0).mean()),
                "total_return": summary["total_return"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "max_drawdown": summary["max_drawdown"],
                "number_of_trades": summary["number_of_trades"],
                "exposure": summary["exposure"],
                "total_transaction_cost": summary["total_transaction_cost"],
            }
        )
    return pd.DataFrame(rows)


def _validate_threshold(value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError("minimum_confidence must be between 0 and 1.")
