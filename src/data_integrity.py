"""Market-data integrity checks and trust scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    USFederalHolidayCalendar,
)

REQUIRED_OHLCV = ["date", "open", "high", "low", "close", "volume"]


def analyze_data_integrity(
    data: pd.DataFrame,
    calendar: str = "business",
    stale_after_days: int = 3,
    jump_threshold: float = 0.08,
    as_of: pd.Timestamp | str | None = None,
    stale_after_minutes: float | None = None,
) -> dict[str, Any]:
    """Audit timestamps, candle consistency, freshness, and price anomalies."""
    if data.empty:
        raise ValueError("Market data is empty.")
    missing_columns = sorted(set(REQUIRED_OHLCV).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Market data is missing columns: {missing_columns}")
    if calendar not in {"business", "continuous", "us_equity"}:
        raise ValueError("calendar must be business, continuous, or us_equity.")
    if stale_after_days < 0:
        raise ValueError("stale_after_days must be non-negative.")
    if jump_threshold <= 0:
        raise ValueError("jump_threshold must be positive.")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    invalid_timestamp_count = int(frame["date"].isna().sum())
    duplicate_timestamp_count = int(frame["date"].duplicated().sum())
    missing_value_count = int(frame[REQUIRED_OHLCV].isna().sum().sum())
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_ohlc = (
        (numeric["high"] < numeric["low"])
        | (numeric["open"] > numeric["high"])
        | (numeric["open"] < numeric["low"])
        | (numeric["close"] > numeric["high"])
        | (numeric["close"] < numeric["low"])
    )
    invalid_ohlc_count = int(invalid_ohlc.fillna(True).sum())
    non_positive_price_count = int(
        (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    )
    ordered = frame.dropna(subset=["date"]).sort_values("date", kind="stable")
    returns = pd.to_numeric(ordered["close"], errors="coerce").pct_change()
    jump_anomaly_count = int((returns.abs() > jump_threshold).sum())
    zero_volume_fraction = float((numeric["volume"].fillna(0) == 0).mean())
    missing_timestamps = _find_missing_timestamps(ordered["date"], calendar)

    latest_timestamp = ordered["date"].max()
    resolved_as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    if resolved_as_of.tz is not None:
        resolved_as_of = resolved_as_of.tz_localize(None)
    data_freshness_minutes = max(
        0.0, float((resolved_as_of - latest_timestamp).total_seconds() / 60)
    )
    stale_days = data_freshness_minutes / 1440
    median_interval_minutes = _median_interval_minutes(ordered["date"])
    intraday_data = median_interval_minutes < 1440
    resolved_stale_minutes = (
        float(stale_after_minutes)
        if stale_after_minutes is not None
        else float(stale_after_days * 1440)
    )
    is_stale = (
        data_freshness_minutes > resolved_stale_minutes
        if intraday_data or stale_after_minutes is not None
        else stale_days > stale_after_days
    )

    deductions = {
        "missing_values": min(25.0, missing_value_count * 2.0),
        "invalid_timestamps": min(20.0, invalid_timestamp_count * 5.0),
        "duplicate_timestamps": min(20.0, duplicate_timestamp_count * 5.0),
        "invalid_ohlc": min(30.0, invalid_ohlc_count * 10.0),
        "non_positive_prices": min(30.0, non_positive_price_count * 10.0),
        "missing_candles": min(20.0, len(missing_timestamps) * 0.5),
        "jump_anomalies": min(10.0, jump_anomaly_count * 1.0),
        "stale_data": 25.0 if is_stale else 0.0,
        "zero_volume": 5.0 if zero_volume_fraction > 0.95 else 0.0,
    }
    score = max(0.0, 100.0 - sum(deductions.values()))
    critical = invalid_ohlc_count + non_positive_price_count + invalid_timestamp_count
    issues: list[dict[str, str]] = []
    if missing_timestamps:
        issues.append(
            _issue(
                "warning",
                f"{len(missing_timestamps)} expected session candles are missing.",
            )
        )
    if duplicate_timestamp_count:
        issues.append(
            _issue(
                "critical",
                f"{duplicate_timestamp_count} duplicate timestamps detected.",
            )
        )
    if invalid_ohlc_count:
        issues.append(
            _issue(
                "critical",
                f"{invalid_ohlc_count} candles violate OHLC boundaries.",
            )
        )
    if jump_anomaly_count:
        issues.append(
            _issue(
                "warning",
                f"{jump_anomaly_count} close-to-close moves exceed {jump_threshold:.1%}.",
            )
        )
    if is_stale:
        issues.append(
            _issue(
                "critical" if intraday_data else "warning",
                f"Latest candle is {data_freshness_minutes:.0f} minutes old.",
            )
        )
    if zero_volume_fraction > 0.95:
        issues.append(
            _issue("warning", "Volume is zero or unavailable for nearly all candles.")
        )
    if not issues:
        issues.append(_issue("info", "No automatic data-integrity issues detected."))
    warnings = [
        issue["message"] for issue in issues if issue["severity"] in {"warning", "critical"}
    ]

    report = {
        "score": score,
        "warnings": warnings,
        "issues": issues,
        "rows": int(len(frame)),
        "first_timestamp": ordered["date"].min(),
        "latest_timestamp": latest_timestamp,
        "stale_days": stale_days,
        "data_freshness_minutes": data_freshness_minutes,
        "median_interval_minutes": median_interval_minutes,
        "is_intraday": intraday_data,
        "is_stale": is_stale,
        "calendar": calendar,
        "missing_candle_count": len(missing_timestamps),
        "missing_candle_sample": missing_timestamps[:20],
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "missing_value_count": missing_value_count,
        "invalid_ohlc_count": invalid_ohlc_count,
        "non_positive_price_count": non_positive_price_count,
        "jump_anomaly_count": jump_anomaly_count,
        "zero_volume_fraction": zero_volume_fraction,
        "deductions": deductions,
    }
    report.update(summarize_data_trust(report))
    return report


def summarize_data_trust(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize integrity issues into trader-facing severity groups."""
    score = float(report["score"])
    issues = report.get("issues", [])
    critical_issues = [
        issue["message"] for issue in issues if issue.get("severity") == "critical"
    ]
    warnings = [
        issue["message"] for issue in issues if issue.get("severity") == "warning"
    ]
    info = [issue["message"] for issue in issues if issue.get("severity") == "info"]
    if critical_issues or score < 60:
        status = "critical"
    elif warnings or score < 80:
        status = "warning"
    elif score < 95:
        status = "good"
    else:
        status = "excellent"
    return {
        "score": score,
        "status": status,
        "critical_issues": critical_issues,
        "warnings": warnings,
        "info": info,
    }


def compare_reference_prices(
    primary_price: float,
    reference_price: float,
    tolerance: float = 0.002,
) -> dict[str, float | str | bool]:
    """Compare two source prices and flag excessive divergence."""
    if primary_price <= 0 or reference_price <= 0:
        raise ValueError("Source prices must be positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    difference = abs(primary_price / reference_price - 1)
    within_tolerance = difference <= tolerance
    return {
        "primary_price": float(primary_price),
        "reference_price": float(reference_price),
        "difference": float(difference),
        "tolerance": float(tolerance),
        "within_tolerance": within_tolerance,
        "status": "ok" if within_tolerance else "mismatch",
    }


def infer_market_calendar(symbol: str) -> str:
    """Infer whether daily candles should include weekends."""
    upper = symbol.upper()
    if upper.endswith("-USD"):
        return "continuous"
    if upper.endswith("=X"):
        return "business"
    return "us_equity"


def _find_missing_timestamps(dates: pd.Series, calendar: str) -> list[pd.Timestamp]:
    unique_dates = pd.DatetimeIndex(dates.dropna().dt.normalize().unique()).sort_values()
    if len(unique_dates) < 2:
        return []
    if calendar == "continuous":
        expected = pd.date_range(unique_dates.min(), unique_dates.max(), freq="D")
    elif calendar == "us_equity":
        federal_holidays = USFederalHolidayCalendar().holidays(
            unique_dates.min(), unique_dates.max()
        )
        good_friday_dates = _GoodFridayCalendar().holidays(
            unique_dates.min(), unique_dates.max()
        )
        expected = pd.bdate_range(unique_dates.min(), unique_dates.max()).difference(
            federal_holidays.union(good_friday_dates)
        )
    else:
        expected = pd.bdate_range(unique_dates.min(), unique_dates.max())
    return [timestamp for timestamp in expected.difference(unique_dates)]


class _GoodFridayCalendar(AbstractHolidayCalendar):
    rules = [GoodFriday]


def _median_interval_minutes(dates: pd.Series) -> float:
    differences = pd.Series(dates).sort_values().diff().dropna()
    if differences.empty:
        return float("inf")
    return float(differences.median().total_seconds() / 60)


def _issue(severity: str, message: str) -> dict[str, str]:
    return {"severity": severity, "message": message}
