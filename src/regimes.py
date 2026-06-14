"""Leakage-safe market regime classification and conditional evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_win_rate,
)


def add_volatility_regime(
    df: pd.DataFrame,
    vol_col: str = "volatility_20",
    lookback: int = 252,
    low_percentile: float = 0.33,
    high_percentile: float = 0.67,
) -> pd.DataFrame:
    """Classify volatility using each row's trailing percentile rank."""
    _validate_dataframe(df, [vol_col])
    if lookback < 2:
        raise ValueError("lookback must be at least 2.")
    if not 0 < low_percentile < high_percentile < 1:
        raise ValueError("Percentile thresholds must satisfy 0 < low < high < 1.")

    result = df.copy()
    result["volatility_percentile"] = result[vol_col].rolling(
        window=lookback, min_periods=lookback
    ).apply(_last_percentile_rank, raw=True)
    result["volatility_regime"] = np.select(
        [
            result["volatility_percentile"] <= low_percentile,
            result["volatility_percentile"] >= high_percentile,
        ],
        ["low_volatility", "high_volatility"],
        default="normal_volatility",
    )
    result.loc[result["volatility_percentile"].isna(), "volatility_regime"] = pd.NA
    return result


def add_trend_regime(
    df: pd.DataFrame,
    short_span: int = 20,
    long_span: int = 50,
    slope_window: int = 20,
    slope_threshold: float = 0.001,
) -> pd.DataFrame:
    """Classify trend using current price and trailing long-EMA slope."""
    _validate_dataframe(df, ["close"])
    if min(short_span, long_span, slope_window) < 2:
        raise ValueError("EMA spans and slope_window must be at least 2.")
    if short_span >= long_span:
        raise ValueError("short_span must be smaller than long_span.")
    if slope_threshold < 0:
        raise ValueError("slope_threshold must be non-negative.")

    result = df.copy()
    short_col = f"ema_{short_span}"
    long_col = f"ema_{long_span}"
    if short_col not in result.columns:
        result[short_col] = result["close"].ewm(span=short_span, adjust=False).mean()
    if long_col not in result.columns:
        result[long_col] = result["close"].ewm(span=long_span, adjust=False).mean()

    result["trend_slope"] = result[long_col] / result[long_col].shift(slope_window) - 1
    result["trend_regime"] = np.select(
        [
            (result["close"] > result[short_col])
            & (result[short_col] > result[long_col])
            & (result["trend_slope"] > slope_threshold),
            (result["close"] < result[short_col])
            & (result[short_col] < result[long_col])
            & (result["trend_slope"] < -slope_threshold),
        ],
        ["uptrend", "downtrend"],
        default="sideways",
    )
    result.loc[result["trend_slope"].isna(), "trend_regime"] = pd.NA
    return result


def evaluate_by_regime(
    df: pd.DataFrame,
    regime_col: str,
    return_col: str = "strategy_return",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Evaluate conditional strategy and classification performance by regime."""
    _validate_dataframe(df, [regime_col, return_col])
    valid = df.dropna(subset=[regime_col, return_col]).copy()
    if valid.empty:
        raise ValueError("No complete rows are available for regime evaluation.")

    records: list[dict[str, float | int | str]] = []
    for regime, group in valid.groupby(regime_col, sort=True):
        returns = group[return_col].astype(float)
        equity = (1 + returns).cumprod()
        record: dict[str, float | int | str] = {
            "regime": str(regime),
            "number_of_rows": int(len(group)),
            "average_strategy_return": float(returns.mean()),
            "total_strategy_return": float(equity.iloc[-1] - 1),
            "sharpe": calculate_sharpe_ratio(returns, periods_per_year),
            "max_drawdown": calculate_max_drawdown(equity),
            "win_rate": calculate_win_rate(returns[returns != 0]),
        }
        true_col = "true_target" if "true_target" in group.columns else "target"
        if true_col in group.columns and "prediction" in group.columns:
            record["accuracy"] = float(
                (group[true_col] == group["prediction"]).mean()
            )
        else:
            record["accuracy"] = float("nan")
        records.append(record)
    return pd.DataFrame(records).sort_values("regime").reset_index(drop=True)


def _last_percentile_rank(values: np.ndarray) -> float:
    return float(np.mean(values <= values[-1]))


def _validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    missing = set(required_columns).difference(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing columns: {sorted(missing)}")
