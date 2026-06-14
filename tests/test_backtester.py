"""Tests for delayed-execution, cost-adjusted backtesting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtester import (
    apply_execution_policy,
    compare_execution_policies,
    generate_positions,
    run_backtest,
    summarize_backtest,
)


def test_generate_positions_rejects_invalid_signals():
    with pytest.raises(ValueError, match="invalid signals"):
        generate_positions(pd.Series([0, 1, 2]))


def test_backtest_uses_shifted_position_and_transaction_costs():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "close": [100.0, 110.0, 99.0, 99.0, 108.9],
            "prediction": [1, -1, 0, 1, 1],
        }
    )
    result = run_backtest(data, transaction_cost=0.001)

    assert result["executed_position"].tolist() == [0.0, 1.0, -1.0, 0.0, 1.0]
    assert result.loc[1, "strategy_return_before_cost"] == pytest.approx(0.10)
    assert result.loc[2, "strategy_return_before_cost"] == pytest.approx(0.10)
    assert result["transaction_cost"].tolist() == pytest.approx(
        [0.0, 0.001, 0.002, 0.001, 0.001]
    )
    assert not result.isna().any().any()


def test_backtest_summary_contains_strategy_and_benchmark_metrics():
    data = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 100.0, 103.0],
            "prediction": [1, 1, -1, -1, 0],
        }
    )
    result = run_backtest(data)
    summary = summarize_backtest(result)

    assert "total_return" in summary
    assert "buy_hold_total_return" in summary
    assert "number_of_trades" in summary
    assert 0 <= summary["exposure"] <= 1


def test_hold_for_horizon_ignores_signals_during_holding_period():
    predictions = pd.Series([1, -1, 0, -1, 1, 0, 1])

    positions = apply_execution_policy(
        predictions,
        mode="hold_for_horizon",
        horizon=3,
    )

    assert positions.tolist() == [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0]


def test_confidence_filtered_policy_neutralizes_weak_signals():
    positions = apply_execution_policy(
        pd.Series([1, -1, 1]),
        mode="confidence_filtered",
        confidence=pd.Series([0.7, 0.5, 0.8]),
        minimum_confidence=0.6,
    )

    assert positions.tolist() == [1.0, 0.0, 1.0]


def test_execution_policy_is_delayed_in_backtest():
    data = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "prediction": [1, -1, -1, -1, 0],
        }
    )

    result = run_backtest(data, mode="hold_for_horizon", horizon=3)

    assert result["position"].tolist() == [1.0, 1.0, 1.0, -1.0, -1.0]
    assert result["executed_position"].tolist() == [0.0, 1.0, 1.0, 1.0, -1.0]


def test_compare_execution_policies_returns_required_metrics():
    data = pd.DataFrame(
        {
            "close": [100.0, 101.0, 100.0, 102.0, 101.0],
            "prediction": [1, -1, 1, -1, 1],
            "confidence": [0.7, 0.4, 0.8, 0.3, 0.9],
        }
    )

    comparison = compare_execution_policies(data, horizon=2)

    assert set(comparison["policy"]) == {
        "daily_rebalance",
        "hold_for_horizon_2",
        "signal_change_only",
        "confidence_filtered_60pct",
    }
    assert {
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_trades",
        "exposure",
        "transaction_cost_paid",
        "win_rate",
        "profit_factor",
    }.issubset(comparison.columns)
