"""Tests for market data standardization and persistence."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    get_symbol_path,
    load_raw_data,
    save_raw_data,
    summarize_live_market_data,
    symbol_slug,
)


def test_raw_data_round_trip_is_clean_and_sorted(sample_ohlcv, tmp_path):
    dirty = pd.concat([sample_ohlcv.iloc[::-1], sample_ohlcv.iloc[[-1]]])
    path = save_raw_data(dirty, "spy", tmp_path)
    loaded = load_raw_data("SPY", tmp_path)

    assert path.endswith("SPY.csv")
    assert loaded.columns.tolist() == ["date", "open", "high", "low", "close", "volume"]
    assert loaded["date"].is_monotonic_increasing
    assert not loaded["date"].duplicated().any()
    assert not loaded.isna().any().any()


def test_symbol_paths_are_safe_for_forex_and_crypto(tmp_path):
    assert symbol_slug("EURUSD=X") == "EURUSD_X"
    assert get_symbol_path("EURUSD=X", tmp_path).name == "EURUSD_X.csv"
    assert get_symbol_path("BTC-USD", tmp_path).name == "BTC-USD.csv"


def test_live_market_summary_reports_latest_intraday_candle():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-06-10 09:30", "2026-06-10 09:35", "2026-06-10 09:40"]
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [101.5, 102.5, 103.5],
            "low": [99.5, 100.5, 101.5],
            "close": [101.0, 102.0, 103.0],
            "volume": [100, 200, 300],
        }
    )

    result = summarize_live_market_data(
        data,
        symbol="SPY",
        fetched_at="2026-06-10 09:42",
        stale_after_minutes=5,
    )

    assert result["current_price"] == 103
    assert result["session_high"] == 103.5
    assert result["session_low"] == 99.5
    assert result["session_volume"] == 600
    assert result["intraday_change"] == pytest.approx(0.03)
    assert result["data_age_minutes"] == pytest.approx(2)
    assert not result["is_stale"]


def test_live_market_summary_flags_stale_data():
    data = pd.DataFrame(
        {
            "date": ["2026-06-10 09:30"],
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100],
            "volume": [100],
        }
    )

    result = summarize_live_market_data(
        data,
        symbol="SPY",
        fetched_at="2026-06-10 10:00",
        stale_after_minutes=15,
    )

    assert result["is_stale"]
