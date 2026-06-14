"""Tests for interval-aware live-market freshness."""

from __future__ import annotations

import pandas as pd

from src.live_data import assess_market_data_freshness


def _market_data(timestamp: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [timestamp],
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100],
            "volume": [100],
        }
    )


def test_five_minute_data_marks_old_candle_stale():
    result = assess_market_data_freshness(
        _market_data("2026-06-10 10:00"),
        interval="5m",
        asset_type="equity",
        as_of="2026-06-10 10:20",
    )

    assert result["is_stale"]
    assert result["threshold_minutes"] == 15


def test_crypto_intraday_freshness_is_stricter():
    equity = assess_market_data_freshness(
        _market_data("2026-06-10 10:00"),
        interval="5m",
        asset_type="equity",
        as_of="2026-06-10 10:13",
    )
    crypto = assess_market_data_freshness(
        _market_data("2026-06-10 10:00"),
        interval="5m",
        asset_type="crypto",
        as_of="2026-06-10 10:13",
    )

    assert not equity["is_stale"]
    assert crypto["is_stale"]


def test_daily_equity_allows_closed_market_delay():
    result = assess_market_data_freshness(
        _market_data("2026-06-12"),
        interval="1d",
        asset_type="equity",
        as_of="2026-06-14 12:00",
    )

    assert not result["is_stale"]
    assert result["market_status"] == "closed"
    assert result["message"] == (
        "Market is likely closed; latest completed trading candle is from Friday."
    )


def test_daily_fx_explains_weekend_market_closure():
    result = assess_market_data_freshness(
        _market_data("2026-06-12"),
        interval="1d",
        asset_type="fx",
        as_of="2026-06-14 12:00",
    )

    assert not result["is_stale"]
    assert result["market_status"] == "closed"
    assert result["message"] == (
        "FX market is likely closed for the weekend; latest completed trading "
        "candle is from Friday."
    )


def test_daily_crypto_warns_when_candle_is_older_than_expected():
    result = assess_market_data_freshness(
        _market_data("2026-06-10"),
        interval="1d",
        asset_type="crypto",
        as_of="2026-06-14 12:00",
    )

    assert result["is_stale"]
    assert result["market_status"] == "open"
    assert result["message"] == (
        "Crypto trades 24/7; the latest candle is older than expected and may be stale."
    )
