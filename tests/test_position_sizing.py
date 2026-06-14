"""Tests for generic and forex position sizing."""

from __future__ import annotations

import pytest

from src.position_sizing import (
    calculate_fixed_fractional_position_size,
    calculate_fx_position_size,
)


def test_fx_position_size_matches_twelve_pip_example():
    result = calculate_fx_position_size(
        account_size=25_000,
        risk_pct=0.005,
        entry_price=1.0800,
        stop_price=1.0788,
    )

    assert result["risk_amount"] == pytest.approx(125)
    assert result["stop_pips"] == pytest.approx(12)
    assert result["standard_lots"] == pytest.approx(1.0416667)
    assert result["estimated_loss_at_stop"] == pytest.approx(125)
    assert result["valid"]


def test_position_sizing_returns_invalid_report_for_bad_inputs():
    fx = calculate_fx_position_size(0, 0, 1.08, 1.08)
    generic = calculate_fixed_fractional_position_size(10_000, 0, 100, 99)

    assert not fx["valid"]
    assert fx["warnings"]
    assert not generic["valid"]
    assert generic["warnings"]
