"""Target creation for directional future-return classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_directional_target(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.005,
) -> pd.DataFrame:
    """Create long, neutral, and short labels from a future return horizon."""
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if "close" not in df.columns:
        raise ValueError("Input dataframe must contain a close column.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    result = df.copy()
    result["future_return"] = result["close"].shift(-horizon) / result["close"] - 1
    result["target"] = np.select(
        [
            result["future_return"] > threshold,
            result["future_return"] < -threshold,
        ],
        [1, -1],
        default=0,
    ).astype(int)

    return result.dropna(subset=["future_return"]).reset_index(drop=True)

