"""Shared fixtures for pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Return deterministic OHLCV data with enough rows for rolling features."""
    periods = 120
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    trend = np.linspace(100, 125, periods)
    cycle = np.sin(np.arange(periods) / 4) * 2
    close = trend + cycle
    open_price = close + np.cos(np.arange(periods) / 5) * 0.4
    high = np.maximum(open_price, close) + 1
    low = np.minimum(open_price, close) - 1
    volume = np.linspace(1_000_000, 1_500_000, periods)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def model_dataset() -> pd.DataFrame:
    """Return deterministic, learnable chronological classification data."""
    rows = 120
    feature = np.sin(np.arange(rows) / 5)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="D"),
            "feature_1": feature,
            "feature_2": np.cos(np.arange(rows) / 7),
            "target": np.select([feature > 0.3, feature < -0.3], [1, -1], default=0),
        }
    )
