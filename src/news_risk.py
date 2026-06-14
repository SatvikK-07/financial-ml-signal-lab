"""Manual economic-calendar loading and deterministic news-risk scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

CALENDAR_COLUMNS = [
    "datetime",
    "currency",
    "event",
    "impact",
    "actual",
    "forecast",
    "previous",
]


def load_economic_calendar(path: str | Path) -> pd.DataFrame:
    """Load a manual economic calendar, returning an empty frame when absent."""
    calendar_path = Path(path)
    if not calendar_path.exists():
        result = pd.DataFrame(columns=CALENDAR_COLUMNS)
        result.attrs["source_status"] = "missing"
        result.attrs["source_path"] = str(calendar_path)
        return result
    calendar = pd.read_csv(calendar_path)
    missing = set(CALENDAR_COLUMNS).difference(calendar.columns)
    if missing:
        raise ValueError(f"Economic calendar is missing columns: {sorted(missing)}")
    result = calendar[CALENDAR_COLUMNS].copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    result = result.dropna(subset=["datetime"]).sort_values("datetime").reset_index(
        drop=True
    )
    result["currency"] = result["currency"].astype(str).str.upper()
    result["impact"] = result["impact"].astype(str).str.lower()
    result.attrs["source_status"] = "loaded"
    result.attrs["source_path"] = str(calendar_path)
    return result


def get_upcoming_events(
    calendar: pd.DataFrame,
    now: pd.Timestamp,
    currencies: list[str],
    lookahead_minutes: int = 120,
) -> pd.DataFrame:
    """Return relevant events between now and the configured lookahead."""
    if lookahead_minutes < 0:
        raise ValueError("lookahead_minutes must be non-negative.")
    if calendar.empty:
        result = calendar.copy()
        result["minutes_until_event"] = pd.Series(dtype=float)
        result.attrs.update(calendar.attrs)
        return result
    current_time = pd.Timestamp(now)
    relevant_currencies = {currency.upper() for currency in currencies}
    event_times = pd.to_datetime(calendar["datetime"], errors="coerce")
    minutes = (event_times - current_time).dt.total_seconds() / 60
    mask = (
        calendar["currency"].astype(str).str.upper().isin(relevant_currencies)
        & minutes.between(0, lookahead_minutes, inclusive="both")
    )
    result = calendar.loc[mask].copy()
    result["minutes_until_event"] = minutes.loc[mask]
    result = result.sort_values("datetime").reset_index(drop=True)
    result.attrs.update(calendar.attrs)
    return result


def score_news_risk(
    upcoming_events: pd.DataFrame,
    high_impact_window_minutes: int = 60,
) -> dict[str, Any]:
    """Score upcoming manual-calendar events into low, medium, or high risk."""
    if high_impact_window_minutes <= 0:
        raise ValueError("high_impact_window_minutes must be positive.")
    warnings: list[str] = []
    source_status = upcoming_events.attrs.get("source_status", "loaded")
    if source_status == "missing":
        warnings.append("No manual economic calendar file found.")
        score = 20
        level = "low"
    elif upcoming_events.empty:
        score = 0
        level = "low"
    else:
        impact = upcoming_events["impact"].astype(str).str.lower()
        minutes = upcoming_events["minutes_until_event"].astype(float)
        high_close = (impact == "high") & (minutes <= high_impact_window_minutes)
        medium_close = (impact == "medium") & (
            minutes <= high_impact_window_minutes
        )
        high_later = (impact == "high") & (
            minutes > high_impact_window_minutes
        ) & (minutes <= 120)
        if high_close.any():
            score = 100
            level = "high"
            warnings.append("High-impact economic news is within 60 minutes.")
        elif medium_close.any() or high_later.any():
            score = 60
            level = "medium"
            warnings.append("Material economic news is within 120 minutes.")
        else:
            score = 20
            level = "low"
    return {
        "news_risk_score": score,
        "risk_level": level,
        "warnings": warnings,
        "source_status": source_status,
        "upcoming_events": upcoming_events,
    }


def infer_relevant_currencies(symbol: str) -> list[str]:
    """Infer economic-calendar currencies from common Yahoo-style symbols."""
    upper = symbol.upper().replace(" ", "")
    if upper.endswith("=X") and len(upper) >= 6:
        return [upper[:3], upper[3:6]]
    if "/" in upper:
        currencies = upper.split("/")
        if len(currencies) == 2 and all(len(currency) == 3 for currency in currencies):
            return currencies
    return ["USD"]
