"""Support, resistance, and reference-level detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_market_zones(
    data: pd.DataFrame,
    lookback: int = 120,
    swing_window: int = 5,
    proximity_pct: float = 0.01,
) -> dict[str, Any]:
    """Detect nearby swing, period, and round-number support/resistance levels."""
    required = {"date", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Market data is missing zone columns: {sorted(missing)}")
    if len(data) < swing_window * 2 + 1:
        raise ValueError("Not enough rows to detect market zones.")
    if lookback < 2 or swing_window < 1 or proximity_pct <= 0:
        raise ValueError("Zone parameters must be positive.")

    frame = data.sort_values("date", kind="stable").tail(lookback).reset_index(drop=True)
    current_price = float(frame["close"].iloc[-1])
    levels: list[dict[str, float | int | str]] = []
    for index in range(swing_window, len(frame) - swing_window):
        window = frame.iloc[index - swing_window : index + swing_window + 1]
        row = frame.iloc[index]
        if row["low"] <= window["low"].min():
            levels.append(
                _level_record(row["low"], "swing_support", current_price, 1)
            )
        if row["high"] >= window["high"].max():
            levels.append(
                _level_record(row["high"], "swing_resistance", current_price, 1)
            )

    reference_periods = {"previous_day": 1, "previous_week": 5, "previous_month": 20}
    history = frame.iloc[:-1]
    for name, period in reference_periods.items():
        subset = history.tail(period)
        if subset.empty:
            continue
        levels.append(
            _level_record(subset["low"].min(), f"{name}_low", current_price, period)
        )
        levels.append(
            _level_record(subset["high"].max(), f"{name}_high", current_price, period)
        )

    round_step = _round_number_step(current_price)
    lower_round = np.floor(current_price / round_step) * round_step
    upper_round = np.ceil(current_price / round_step) * round_step
    levels.append(_level_record(lower_round, "round_number", current_price, 2))
    if not np.isclose(lower_round, upper_round):
        levels.append(_level_record(upper_round, "round_number", current_price, 2))

    zone_table = pd.DataFrame(levels).drop_duplicates(subset=["level", "source"])
    zone_table = score_zones(
        zone_table,
        frame,
        current_price,
        tolerance_pct=min(proximity_pct, 0.002),
    )
    context = summarize_zone_context(
        zone_table, current_price, proximity_pct=proximity_pct
    )
    return {
        "current_price": current_price,
        "proximity_pct": proximity_pct,
        **context,
        "levels": zone_table,
    }


def score_zones(
    zones: pd.DataFrame,
    df: pd.DataFrame,
    current_price: float,
    tolerance_pct: float = 0.002,
) -> pd.DataFrame:
    """Score zones from touches, source strength, recency, and proximity."""
    if zones.empty:
        return zones.copy()
    if current_price <= 0 or tolerance_pct <= 0:
        raise ValueError("current_price and tolerance_pct must be positive.")
    required = {"date", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Market data is missing zone scoring columns: {sorted(missing)}")

    history = df.sort_values("date", kind="stable").copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    scored = zones.copy()
    scored["distance_pct_from_current_price"] = scored["level"] / current_price - 1
    scored["distance_pct"] = scored["distance_pct_from_current_price"]
    scored["absolute_distance_pct"] = scored[
        "distance_pct_from_current_price"
    ].abs()
    touch_counts: list[int] = []
    last_touched_dates: list[pd.Timestamp | pd.NaT] = []
    strength_scores: list[float] = []
    for _, zone in scored.iterrows():
        level = float(zone["level"])
        tolerance = abs(level) * tolerance_pct
        touched = (
            (history["low"] <= level + tolerance)
            & (history["high"] >= level - tolerance)
        )
        touch_count = int(touched.sum())
        touch_counts.append(touch_count)
        last_touched_dates.append(
            history.loc[touched, "date"].max() if touch_count else pd.NaT
        )
        source_strength = float(zone.get("source_strength", zone.get("strength", 1)))
        proximity_bonus = max(
            0.0,
            20.0
            * (1 - min(abs(level / current_price - 1) / (tolerance_pct * 5), 1)),
        )
        strength_scores.append(
            float(min(100, source_strength * 3 + min(touch_count, 10) * 6 + proximity_bonus))
        )
    scored["touch_count"] = touch_counts
    scored["last_touched_date"] = last_touched_dates
    scored["strength_score"] = strength_scores
    scored["strength"] = scored["strength_score"]
    scored["is_near_current_price"] = (
        scored["absolute_distance_pct"] <= tolerance_pct
    )
    return scored.sort_values(
        ["absolute_distance_pct", "strength_score"], ascending=[True, False]
    ).reset_index(drop=True)


def get_nearest_zones(
    zones: pd.DataFrame,
    current_price: float,
    n: int = 5,
) -> pd.DataFrame:
    """Return the nearest scored zones around current price."""
    if current_price <= 0 or n < 1:
        raise ValueError("current_price and n must be positive.")
    if zones.empty:
        return zones.copy()
    result = zones.copy()
    result["distance_pct_from_current_price"] = result["level"] / current_price - 1
    result["absolute_distance_pct"] = result[
        "distance_pct_from_current_price"
    ].abs()
    return result.sort_values("absolute_distance_pct").head(n).reset_index(drop=True)


def summarize_zone_context(
    zones: pd.DataFrame,
    current_price: float,
    proximity_pct: float = 0.01,
) -> dict[str, Any]:
    """Summarize nearest support/resistance and directional warnings."""
    prepared = zones.copy()
    prepared["distance_pct_from_current_price"] = (
        prepared["level"] / current_price - 1
    )
    prepared["distance_pct"] = prepared["distance_pct_from_current_price"]
    prepared["absolute_distance_pct"] = prepared[
        "distance_pct_from_current_price"
    ].abs()
    supports = prepared[prepared["level"] <= current_price]
    resistances = prepared[prepared["level"] >= current_price]
    nearest_support = _nearest_level(supports)
    nearest_resistance = _nearest_level(resistances)
    distance_to_support = (
        abs(float(nearest_support["level"]) / current_price - 1)
        if nearest_support
        else None
    )
    distance_to_resistance = (
        abs(float(nearest_resistance["level"]) / current_price - 1)
        if nearest_resistance
        else None
    )
    near_support = bool(
        distance_to_support is not None and distance_to_support <= proximity_pct
    )
    near_resistance = bool(
        distance_to_resistance is not None and distance_to_resistance <= proximity_pct
    )
    return {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "distance_to_support": distance_to_support,
        "distance_to_resistance": distance_to_resistance,
        "near_support": near_support,
        "near_resistance": near_resistance,
        "near_key_zone": near_support or near_resistance,
        "long_warning": (
            "Long signal is near resistance." if near_resistance else None
        ),
        "short_warning": "Short signal is near support." if near_support else None,
    }


def _level_record(
    level: float, level_type: str, current_price: float, strength: int
) -> dict[str, float | int | str]:
    distance = float(level / current_price - 1)
    zone_type = _normalize_zone_type(level_type)
    return {
        "level": float(level),
        "type": level_type,
        "zone_type": zone_type,
        "source": level_type,
        "distance_pct": distance,
        "distance_pct_from_current_price": distance,
        "absolute_distance_pct": abs(distance),
        "strength": int(strength),
        "source_strength": int(strength),
    }


def _nearest_level(levels: pd.DataFrame) -> dict[str, Any] | None:
    if levels.empty:
        return None
    return levels.sort_values("absolute_distance_pct").iloc[0].to_dict()


def _round_number_step(price: float) -> float:
    magnitude = 10 ** np.floor(np.log10(price))
    return float(magnitude / 100 if price < 10 else magnitude / 10)


def _normalize_zone_type(source: str) -> str:
    if source == "round_number":
        return "round_number"
    if source.endswith("_low"):
        return "previous_high_low"
    if source.endswith("_high"):
        return "previous_high_low"
    if "support" in source:
        return "support"
    if "resistance" in source:
        return "resistance"
    if "session" in source:
        return "session"
    return "previous_high_low"
