"""Live-market data freshness and status utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

INTRADAY_STALE_THRESHOLDS = {
    "1m": 5.0,
    "5m": 15.0,
    "15m": 45.0,
    "1h": 120.0,
    "60m": 120.0,
}


def assess_market_data_freshness(
    df: pd.DataFrame,
    interval: str,
    asset_type: str,
    as_of: pd.Timestamp | str | None = None,
) -> dict[str, Any]:
    """Assess whether the latest available candle is reasonably current."""
    if df.empty or "date" not in df.columns:
        raise ValueError("Market data must contain at least one dated candle.")
    if asset_type not in {"equity", "crypto", "fx"}:
        raise ValueError("asset_type must be equity, crypto, or fx.")

    dates = pd.to_datetime(df["date"], errors="coerce", utc=True).dropna()
    if dates.empty:
        raise ValueError("Market data contains no valid candle timestamps.")
    latest = dates.max()
    resolved_as_of = (
        pd.Timestamp(as_of)
        if as_of is not None
        else pd.Timestamp.now(tz="UTC")
    )
    if resolved_as_of.tzinfo is None:
        resolved_as_of = resolved_as_of.tz_localize("UTC")
    else:
        resolved_as_of = resolved_as_of.tz_convert("UTC")
    age_minutes = max(0.0, float((resolved_as_of - latest).total_seconds() / 60))

    normalized_interval = interval.lower()
    is_daily = normalized_interval in {"1d", "1day", "d"}
    if is_daily:
        threshold_minutes = 3 * 24 * 60 if asset_type == "crypto" else 5 * 24 * 60
    else:
        threshold_minutes = INTRADAY_STALE_THRESHOLDS.get(normalized_interval, 120.0)
        if asset_type == "crypto":
            threshold_minutes *= 0.75
    is_stale = age_minutes > threshold_minutes
    market_status = _infer_market_status(resolved_as_of, asset_type, is_daily)
    message = _freshness_message(
        latest=latest,
        as_of=resolved_as_of,
        asset_type=asset_type,
        market_status=market_status,
        is_stale=is_stale,
    )
    return {
        "data_provider": "Yahoo Finance",
        "latest_candle": latest.tz_localize(None),
        "data_age_minutes": age_minutes,
        "threshold_minutes": threshold_minutes,
        "freshness_status": "stale" if is_stale else "fresh",
        "is_stale": is_stale,
        "market_status": market_status,
        "interval": interval,
        "asset_type": asset_type,
        "message": message,
    }


def _infer_market_status(
    as_of: pd.Timestamp,
    asset_type: str,
    is_daily: bool,
) -> str:
    if asset_type == "crypto":
        return "open"
    if as_of.weekday() >= 5:
        return "closed"
    if is_daily:
        return "unknown"
    if asset_type == "fx":
        return "open"
    eastern = as_of.tz_convert("America/New_York")
    minutes = eastern.hour * 60 + eastern.minute
    return "open" if 570 <= minutes < 960 else "closed"


def _freshness_message(
    *,
    latest: pd.Timestamp,
    as_of: pd.Timestamp,
    asset_type: str,
    market_status: str,
    is_stale: bool,
) -> str:
    """Explain freshness using the selected market's trading schedule."""
    is_weekend = as_of.weekday() >= 5
    latest_day = latest.day_name()
    if asset_type == "crypto":
        if is_stale:
            return (
                "Crypto trades 24/7; the latest candle is older than expected "
                "and may be stale."
            )
        return "Crypto trades 24/7; the latest candle is within the expected delay."
    if asset_type == "fx" and is_weekend:
        return (
            "FX market is likely closed for the weekend; latest completed trading "
            f"candle is from {latest_day}."
        )
    if asset_type == "equity" and is_weekend:
        return (
            "Market is likely closed; latest completed trading candle is from "
            f"{latest_day}."
        )
    if is_stale:
        return (
            "Latest data may be stale. Yahoo Finance may not have published a "
            "newer candle."
        )
    if market_status == "closed":
        return "Market is likely closed; latest candle is within the expected delay."
    return "Latest available candle is within the configured freshness threshold."
