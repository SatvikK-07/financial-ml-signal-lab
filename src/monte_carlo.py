"""Bootstrap Monte Carlo price-path simulation and risk metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def simulate_price_paths(
    prices: pd.Series,
    current_price: float | None = None,
    lookback: int = 100,
    horizon: int = 20,
    simulations: int = 10_000,
    confidence_level: float = 0.95,
    take_profit_level: float | None = None,
    stop_loss_level: float | None = None,
    side: int = 1,
    random_seed: int = 42,
    displayed_paths: int = 100,
    method: str = "iid_bootstrap",
    block_size: int = 5,
) -> dict[str, Any]:
    """Simulate empirical-return price paths and summarize scenario risk."""
    clean_prices = pd.Series(prices, dtype=float).replace([np.inf, -np.inf], np.nan)
    clean_prices = clean_prices.dropna()
    _validate_inputs(
        clean_prices,
        current_price,
        lookback,
        horizon,
        simulations,
        confidence_level,
        take_profit_level,
        stop_loss_level,
        side,
        displayed_paths,
        method,
        block_size,
    )
    resolved_price = float(current_price or clean_prices.iloc[-1])
    log_returns = np.log(clean_prices / clean_prices.shift(1)).dropna().tail(lookback)
    if log_returns.empty or np.isclose(log_returns.std(ddof=1), 0):
        raise ValueError("Price history must contain variable returns for simulation.")

    generator = np.random.default_rng(random_seed)
    sampled_returns = _sample_returns(
        log_returns.to_numpy(),
        simulations=simulations,
        horizon=horizon,
        method=method,
        block_size=block_size,
        generator=generator,
    )
    paths = np.empty((simulations, horizon + 1), dtype=float)
    paths[:, 0] = resolved_price
    paths[:, 1:] = resolved_price * np.exp(np.cumsum(sampled_returns, axis=1))

    terminal_returns = paths[:, -1] / resolved_price - 1
    path_returns = paths / resolved_price - 1
    path_minimum_returns = path_returns.min(axis=1)
    path_maximum_returns = path_returns.max(axis=1)
    value_at_risk = float(np.quantile(terminal_returns, 1 - confidence_level))
    tail_losses = terminal_returns[terminal_returns <= value_at_risk]
    expected_shortfall = (
        float(tail_losses.mean()) if tail_losses.size else value_at_risk
    )
    target_probability, stop_probability, neither_probability = _barrier_probabilities(
        paths,
        take_profit_level,
        stop_loss_level,
        side,
    )

    shown = min(displayed_paths, simulations)
    path_indices = np.linspace(0, simulations - 1, shown, dtype=int)
    displayed = pd.DataFrame(
        paths[path_indices].T,
        index=pd.RangeIndex(horizon + 1, name="step"),
    )
    displayed.columns = [f"path_{index + 1}" for index in range(shown)]

    lower_price, median_price, upper_price = np.quantile(
        paths[:, -1], [0.05, 0.50, 0.95]
    )
    return {
        "summary": {
            "current_price": resolved_price,
            "lookback": lookback,
            "horizon": horizon,
            "simulations": simulations,
            "method": method,
            "block_size": block_size,
            "forecast_horizon": horizon,
            "n_simulations": simulations,
            "confidence_level": confidence_level,
            "expected_return": float(terminal_returns.mean()),
            "median_return": float(np.median(terminal_returns)),
            "probability_positive_return": float((terminal_returns > 0).mean()),
            "value_at_risk": value_at_risk,
            "expected_shortfall": expected_shortfall,
            "expected_max_adverse_move": float(path_minimum_returns.mean()),
            "worst_5pct_adverse_move": float(
                np.quantile(path_minimum_returns, 0.05)
            ),
            "best_5pct_favorable_move": float(
                np.quantile(path_maximum_returns, 0.95)
            ),
            "terminal_price_5pct": float(lower_price),
            "terminal_price_median": float(median_price),
            "terminal_price_95pct": float(upper_price),
            "take_profit_level": take_profit_level,
            "stop_loss_level": stop_loss_level,
            "probability_take_profit_first": target_probability,
            "probability_stop_loss_first": stop_probability,
            "probability_neither_level": neither_probability,
            "side": side,
        },
        "displayed_paths": displayed,
        "terminal_returns": pd.Series(
            terminal_returns, name="terminal_return", dtype=float
        ),
        "terminal_prices": pd.Series(
            paths[:, -1], name="terminal_price", dtype=float
        ),
    }


def _sample_returns(
    returns: np.ndarray,
    simulations: int,
    horizon: int,
    method: str,
    block_size: int,
    generator: np.random.Generator,
) -> np.ndarray:
    if method == "iid_bootstrap":
        return generator.choice(returns, size=(simulations, horizon), replace=True)

    maximum_start = len(returns) - block_size
    blocks_needed = int(np.ceil(horizon / block_size))
    starts = generator.integers(
        0,
        maximum_start + 1,
        size=(simulations, blocks_needed),
    )
    offsets = np.arange(block_size)
    sampled_blocks = returns[starts[..., None] + offsets]
    return sampled_blocks.reshape(simulations, -1)[:, :horizon]


def _barrier_probabilities(
    paths: np.ndarray,
    take_profit_level: float | None,
    stop_loss_level: float | None,
    side: int,
) -> tuple[float, float, float]:
    if take_profit_level is None or stop_loss_level is None:
        return 0.0, 0.0, 1.0
    future_paths = paths[:, 1:]
    if side == 1:
        target_hits = future_paths >= take_profit_level
        stop_hits = future_paths <= stop_loss_level
    else:
        target_hits = future_paths <= take_profit_level
        stop_hits = future_paths >= stop_loss_level
    no_hit_index = future_paths.shape[1] + 1
    first_target = np.where(
        target_hits.any(axis=1), target_hits.argmax(axis=1), no_hit_index
    )
    first_stop = np.where(stop_hits.any(axis=1), stop_hits.argmax(axis=1), no_hit_index)
    target_first = first_target < first_stop
    stop_first = first_stop < first_target
    neither = (first_target == no_hit_index) & (first_stop == no_hit_index)
    return (
        float(target_first.mean()),
        float(stop_first.mean()),
        float(neither.mean()),
    )


def _validate_inputs(
    prices: pd.Series,
    current_price: float | None,
    lookback: int,
    horizon: int,
    simulations: int,
    confidence_level: float,
    take_profit_level: float | None,
    stop_loss_level: float | None,
    side: int,
    displayed_paths: int,
    method: str,
    block_size: int,
) -> None:
    if len(prices) < 3 or (prices <= 0).any():
        raise ValueError("Price history must contain at least three positive prices.")
    resolved_price = float(current_price or prices.iloc[-1])
    if resolved_price <= 0:
        raise ValueError("current_price must be positive.")
    for name, value in {
        "lookback": lookback,
        "horizon": horizon,
        "simulations": simulations,
        "displayed_paths": displayed_paths,
    }.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if side not in {-1, 1}:
        raise ValueError("side must be -1 or 1.")
    if method not in {"iid_bootstrap", "block_bootstrap"}:
        raise ValueError("method must be iid_bootstrap or block_bootstrap.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if method == "block_bootstrap" and block_size > min(lookback, len(prices) - 1):
        raise ValueError("block_size cannot exceed available lookback returns.")
    for name, value in {
        "take_profit_level": take_profit_level,
        "stop_loss_level": stop_loss_level,
    }.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when provided.")
    if take_profit_level is not None and stop_loss_level is not None:
        if side == 1 and not stop_loss_level < resolved_price < take_profit_level:
            raise ValueError("Long barriers must satisfy stop < current < target.")
        if side == -1 and not take_profit_level < resolved_price < stop_loss_level:
            raise ValueError("Short barriers must satisfy target < current < stop.")
