"""Numerically safe financial performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate compound annual growth rate from an equity curve."""
    values = _clean_series(equity)
    if len(values) < 2 or values.iloc[0] <= 0 or values.iloc[-1] <= 0:
        return 0.0
    years = (len(values) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)


def calculate_sharpe_ratio(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe ratio assuming a zero risk-free rate."""
    values = _clean_series(returns)
    if values.empty:
        return 0.0
    volatility = values.std(ddof=1)
    if not np.isfinite(volatility) or np.isclose(volatility, 0):
        return 0.0
    return float(values.mean() / volatility * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Calculate annualized Sortino ratio using downside return deviation."""
    values = _clean_series(returns)
    downside = values[values < 0]
    if downside.empty:
        return 0.0
    downside_deviation = np.sqrt(np.mean(np.square(downside)))
    if not np.isfinite(downside_deviation) or np.isclose(downside_deviation, 0):
        return 0.0
    return float(values.mean() / downside_deviation * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate the worst peak-to-trough drawdown as a negative decimal."""
    values = _clean_series(equity)
    if values.empty:
        return 0.0
    rolling_max = values.cummax()
    valid = rolling_max != 0
    drawdown = pd.Series(0.0, index=values.index)
    drawdown.loc[valid] = values.loc[valid] / rolling_max.loc[valid] - 1
    return float(drawdown.min())


def calculate_win_rate(trade_returns: pd.Series) -> float:
    """Calculate the proportion of completed trades with positive returns."""
    values = _clean_series(trade_returns)
    if values.empty:
        return 0.0
    return float((values > 0).mean())


def calculate_profit_factor(trade_returns: pd.Series) -> float:
    """Calculate gross winning returns divided by gross losing returns."""
    values = _clean_series(trade_returns)
    if values.empty:
        return 0.0
    gross_profit = values[values > 0].sum()
    gross_loss = -values[values < 0].sum()
    if np.isclose(gross_loss, 0):
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def calculate_total_return(equity: pd.Series) -> float:
    """Calculate total return from the first to final equity value."""
    values = _clean_series(equity)
    if len(values) < 2 or np.isclose(values.iloc[0], 0):
        return 0.0
    return float(values.iloc[-1] / values.iloc[0] - 1)


def _clean_series(values: pd.Series) -> pd.Series:
    series = pd.Series(values, dtype=float)
    return series.replace([np.inf, -np.inf], np.nan).dropna()

