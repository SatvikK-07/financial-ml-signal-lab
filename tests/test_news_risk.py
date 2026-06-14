"""Tests for manual economic-calendar risk scoring."""

from __future__ import annotations

import pandas as pd

from src.news_risk import (
    get_upcoming_events,
    infer_relevant_currencies,
    load_economic_calendar,
    score_news_risk,
)


def test_missing_manual_calendar_does_not_crash(tmp_path):
    calendar = load_economic_calendar(tmp_path / "missing.csv")
    upcoming = get_upcoming_events(
        calendar,
        now=pd.Timestamp("2026-01-01 10:00"),
        currencies=["USD"],
    )
    result = score_news_risk(upcoming)

    assert result["risk_level"] == "low"
    assert "No manual economic calendar file found." in result["warnings"]


def test_high_impact_event_within_hour_is_high_risk(tmp_path):
    path = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-01-01 10:30", "2026-01-01 13:00"],
            "currency": ["USD", "EUR"],
            "event": ["CPI", "ECB speech"],
            "impact": ["high", "medium"],
            "actual": ["", ""],
            "forecast": ["2.5", ""],
            "previous": ["2.4", ""],
        }
    ).to_csv(path, index=False)

    calendar = load_economic_calendar(path)
    upcoming = get_upcoming_events(
        calendar,
        now=pd.Timestamp("2026-01-01 10:00"),
        currencies=["USD", "EUR"],
    )
    result = score_news_risk(upcoming)

    assert result["risk_level"] == "high"
    assert result["news_risk_score"] == 100
    assert len(result["upcoming_events"]) == 1


def test_currency_inference_supports_yahoo_and_slash_fx_symbols():
    assert infer_relevant_currencies("EURUSD=X") == ["EUR", "USD"]
    assert infer_relevant_currencies("EUR/USD") == ["EUR", "USD"]
