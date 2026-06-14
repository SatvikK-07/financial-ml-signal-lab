"""Tests for market-data integrity checks."""

from __future__ import annotations

import pandas as pd

from src.data_integrity import (
    analyze_data_integrity,
    compare_reference_prices,
    infer_market_calendar,
    summarize_data_trust,
)


def test_data_integrity_flags_duplicates_invalid_candles_and_staleness():
    data = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-02", "2025-01-04"],
            "open": [100, 101, 101, 103],
            "high": [102, 100, 102, 104],
            "low": [99, 102, 100, 102],
            "close": [101, 101, 101, 103],
            "volume": [100, 100, 100, 100],
        }
    )

    result = analyze_data_integrity(
        data,
        calendar="continuous",
        stale_after_days=1,
        as_of="2025-01-10",
    )

    assert result["duplicate_timestamp_count"] == 1
    assert result["invalid_ohlc_count"] >= 1
    assert result["missing_candle_count"] == 1
    assert result["is_stale"]
    assert result["status"] == "critical"
    assert result["data_freshness_minutes"] > 0
    assert any(issue["severity"] == "critical" for issue in result["issues"])


def test_reference_price_comparison_reports_tolerance():
    result = compare_reference_prices(100.0, 100.1, tolerance=0.002)

    assert result["within_tolerance"]
    assert result["status"] == "ok"


def test_market_calendar_inference_distinguishes_asset_classes():
    assert infer_market_calendar("SPY") == "us_equity"
    assert infer_market_calendar("BTC-USD") == "continuous"
    assert infer_market_calendar("EURUSD=X") == "business"


def test_summarize_data_trust_groups_severity_levels():
    summary = summarize_data_trust(
        {
            "score": 70,
            "issues": [
                {"severity": "critical", "message": "bad candle"},
                {"severity": "warning", "message": "stale"},
                {"severity": "info", "message": "checked"},
            ],
        }
    )

    assert summary["status"] == "critical"
    assert summary["critical_issues"] == ["bad candle"]
    assert summary["warnings"] == ["stale"]
