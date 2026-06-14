"""Tests for financial performance metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.metrics import (
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_total_return,
)


def test_max_drawdown_calculation():
    equity = pd.Series([100.0, 120.0, 90.0, 108.0])

    assert calculate_max_drawdown(equity) == pytest.approx(-0.25)


def test_sharpe_ratio_handles_zero_volatility():
    assert calculate_sharpe_ratio(pd.Series([0.01, 0.01, 0.01])) == 0.0


def test_total_return_calculation():
    assert calculate_total_return(pd.Series([100.0, 125.0])) == pytest.approx(0.25)


def test_profit_factor_handles_no_losing_trades():
    assert calculate_profit_factor(pd.Series([0.01, 0.02])) == float("inf")
    assert calculate_profit_factor(pd.Series(dtype=float)) == 0.0

