"""Tests for Monte Carlo scenario simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import simulate_price_paths


def test_monte_carlo_is_deterministic_and_returns_risk_metrics():
    prices = pd.Series(100 * np.exp(np.cumsum(np.sin(np.arange(150)) * 0.01)))

    first = simulate_price_paths(
        prices,
        horizon=10,
        simulations=1_000,
        take_profit_level=float(prices.iloc[-1] * 1.02),
        stop_loss_level=float(prices.iloc[-1] * 0.98),
        random_seed=7,
    )
    second = simulate_price_paths(
        prices,
        horizon=10,
        simulations=1_000,
        take_profit_level=float(prices.iloc[-1] * 1.02),
        stop_loss_level=float(prices.iloc[-1] * 0.98),
        random_seed=7,
    )

    assert first["summary"] == second["summary"]
    assert first["displayed_paths"].shape == (11, 100)
    assert first["summary"]["expected_shortfall"] <= first["summary"]["value_at_risk"]
    probabilities = sum(
        first["summary"][key]
        for key in (
            "probability_take_profit_first",
            "probability_stop_loss_first",
            "probability_neither_level",
        )
    )
    assert probabilities == pytest.approx(1.0)


def test_monte_carlo_validates_long_barrier_order():
    with pytest.raises(ValueError, match="Long barriers"):
        simulate_price_paths(
            pd.Series([100.0, 101.0, 99.0, 102.0]),
            current_price=102.0,
            take_profit_level=100.0,
            stop_loss_level=99.0,
        )


def test_block_bootstrap_generates_finite_scenarios():
    prices = pd.Series(100 * np.exp(np.cumsum(np.sin(np.arange(80)) * 0.01)))

    result = simulate_price_paths(
        prices,
        horizon=12,
        simulations=200,
        method="block_bootstrap",
        block_size=4,
        displayed_paths=20,
    )

    assert result["displayed_paths"].shape == (13, 20)
    assert result["terminal_prices"].notna().all()
    assert np.isfinite(result["summary"]["value_at_risk"])
    assert np.isfinite(result["summary"]["expected_shortfall"])
    assert result["summary"]["method"] == "block_bootstrap"
    assert result["summary"]["forecast_horizon"] == 12
    assert result["summary"]["n_simulations"] == 200


def test_monte_carlo_rejects_invalid_method_and_block_size():
    prices = pd.Series([100.0, 101.0, 99.0, 102.0, 101.0, 103.0])

    with pytest.raises(ValueError, match="method"):
        simulate_price_paths(prices, method="unknown")
    with pytest.raises(ValueError, match="block_size"):
        simulate_price_paths(prices, method="block_bootstrap", block_size=0)
