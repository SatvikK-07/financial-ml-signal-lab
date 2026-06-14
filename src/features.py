"""Leakage-safe feature engineering for OHLCV market data."""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
RETURN_FEATURES = ["return_1", "return_3", "return_5", "return_10", "log_return_1"]
VOLATILITY_FEATURES = [
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "atr_14",
    "atr_pct",
]
TREND_FEATURES = [
    "ema_10",
    "ema_20",
    "ema_50",
    "close_above_ema_10",
    "close_above_ema_20",
    "ema_10_20_ratio",
    "macd",
    "macd_signal",
    "macd_diff",
    "rsi_14",
]
CANDLE_FEATURES = ["candle_body", "high_low_range", "upper_wick", "lower_wick"]
TIME_FEATURES = ["day_of_week", "month", "quarter"]
FEATURE_COLUMNS = (
    RETURN_FEATURES
    + VOLATILITY_FEATURES
    + TREND_FEATURES
    + CANDLE_FEATURES
    + TIME_FEATURES
)


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing simple and log return features."""
    result = _validated_copy(df, ["close"])
    for period in (1, 3, 5, 10):
        result[f"return_{period}"] = result["close"].pct_change(period)
    result["log_return_1"] = np.log(result["close"] / result["close"].shift(1))
    return result


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing annualized volatility and average true range features."""
    result = _validated_copy(df, ["high", "low", "close"])
    daily_return = result["close"].pct_change()
    for window in (5, 10, 20):
        result[f"volatility_{window}"] = (
            daily_return.rolling(window=window, min_periods=window).std()
            * np.sqrt(252)
        )

    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()
    result["atr_pct"] = result["atr_14"] / result["close"]
    return result


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing exponential-moving-average, MACD, and RSI features."""
    result = _validated_copy(df, ["close"])
    for span in (10, 20, 50):
        result[f"ema_{span}"] = result["close"].ewm(span=span, adjust=False).mean()

    result["close_above_ema_10"] = (
        result["close"] > result["ema_10"]
    ).astype(int)
    result["close_above_ema_20"] = (
        result["close"] > result["ema_20"]
    ).astype(int)
    result["ema_10_20_ratio"] = result["ema_10"] / result["ema_20"] - 1

    ema_12 = result["close"].ewm(span=12, adjust=False).mean()
    ema_26 = result["close"].ewm(span=26, adjust=False).mean()
    result["macd"] = ema_12 - ema_26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macd_diff"] = result["macd"] - result["macd_signal"]

    close_change = result["close"].diff()
    average_gain = close_change.clip(lower=0).rolling(14, min_periods=14).mean()
    average_loss = -close_change.clip(upper=0).rolling(14, min_periods=14).mean()
    relative_strength = average_gain / average_loss
    result["rsi_14"] = 100 - (100 / (1 + relative_strength))
    result.loc[(average_loss == 0) & (average_gain > 0), "rsi_14"] = 100
    result.loc[(average_loss == 0) & (average_gain == 0), "rsi_14"] = 50
    return result


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features describing each observed price candle."""
    result = _validated_copy(df, ["open", "high", "low", "close"])
    body_top = result[["open", "close"]].max(axis=1)
    body_bottom = result[["open", "close"]].min(axis=1)
    result["candle_body"] = result["close"] - result["open"]
    result["high_low_range"] = result["high"] - result["low"]
    result["upper_wick"] = result["high"] - body_top
    result["lower_wick"] = body_bottom - result["low"]
    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features available at the observation time."""
    result = _validated_copy(df, ["date"])
    date = pd.to_datetime(result["date"], errors="coerce")
    if date.isna().any():
        raise ValueError("date contains values that cannot be parsed.")
    result["date"] = date
    result["day_of_week"] = date.dt.dayofweek
    result["month"] = date.dt.month
    result["quarter"] = date.dt.quarter
    return result


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all model features and return complete model-ready rows."""
    result = _validated_copy(df, OHLCV_COLUMNS)
    result = add_return_features(result)
    result = add_volatility_features(result)
    result = add_trend_features(result)
    result = add_candle_features(result)
    result = add_time_features(result)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return known model feature columns present in the dataframe."""
    return [column for column in FEATURE_COLUMNS if column in df.columns]


def _validated_copy(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    missing = set(required_columns).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.copy()

