"""Tests for future-return directional target creation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.targets import create_directional_target


def test_target_uses_future_close_and_drops_unknown_rows(sample_ohlcv):
    horizon = 5
    dataset = create_directional_target(sample_ohlcv, horizon=horizon, threshold=0.005)
    expected = sample_ohlcv["close"].iloc[horizon] / sample_ohlcv["close"].iloc[0] - 1

    assert len(dataset) == len(sample_ohlcv) - horizon
    assert dataset.loc[0, "future_return"] == pytest.approx(expected)
    assert set(dataset["target"].unique()).issubset({-1, 0, 1})


def test_target_threshold_classes_are_correct():
    df = pd.DataFrame(
        {
            "close": [100.0, 102.0, 101.0, 99.0],
        }
    )
    dataset = create_directional_target(df, horizon=1, threshold=0.005)

    assert dataset["target"].tolist() == [1, -1, -1]

