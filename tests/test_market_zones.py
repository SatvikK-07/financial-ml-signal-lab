"""Tests for support and resistance detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market_zones import (
    detect_market_zones,
    get_nearest_zones,
    summarize_zone_context,
)


def test_market_zones_return_nearest_support_and_resistance():
    close = 100 + np.sin(np.arange(80) / 3) * 5
    data = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=80),
            "high": close + 1,
            "low": close - 1,
            "close": close,
        }
    )

    zones = detect_market_zones(data, lookback=60, swing_window=3)

    assert zones["nearest_support"]["level"] <= zones["current_price"]
    assert zones["nearest_resistance"]["level"] >= zones["current_price"]
    assert {"swing_support", "swing_resistance"}.issubset(set(zones["levels"]["type"]))
    assert {
        "strength_score",
        "touch_count",
        "distance_pct_from_current_price",
        "last_touched_date",
        "source",
        "is_near_current_price",
    }.issubset(zones["levels"].columns)


def test_nearest_zone_context_calculates_distances():
    zones = pd.DataFrame(
        {
            "level": [98.0, 101.0, 105.0],
            "zone_type": ["support", "resistance", "resistance"],
        }
    )

    nearest = get_nearest_zones(zones, current_price=100.0, n=2)
    context = summarize_zone_context(zones, current_price=100.0, proximity_pct=0.02)

    assert nearest["level"].tolist() == [101.0, 98.0]
    assert context["nearest_support"]["level"] == 98.0
    assert context["nearest_resistance"]["level"] == 101.0
    assert context["distance_to_support"] == pytest.approx(0.02)
    assert context["distance_to_resistance"] == pytest.approx(0.01)
