"""Tests for leakage-safe regime analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.regimes import (
    add_trend_regime,
    add_volatility_regime,
    evaluate_by_regime,
)


def test_volatility_regime_does_not_change_when_future_rows_change():
    df = pd.DataFrame({"volatility_20": np.linspace(0.1, 0.5, 40)})
    baseline = add_volatility_regime(df, lookback=10)
    changed = df.copy()
    changed.loc[30:, "volatility_20"] *= 100
    changed_regime = add_volatility_regime(changed, lookback=10)

    pd.testing.assert_series_equal(
        baseline.loc[:29, "volatility_regime"],
        changed_regime.loc[:29, "volatility_regime"],
    )


def test_trend_regime_identifies_clear_uptrend_and_downtrend():
    uptrend = pd.DataFrame({"close": np.linspace(100, 200, 150)})
    downtrend = pd.DataFrame({"close": np.linspace(200, 100, 150)})

    assert add_trend_regime(uptrend)["trend_regime"].iloc[-1] == "uptrend"
    assert add_trend_regime(downtrend)["trend_regime"].iloc[-1] == "downtrend"


def test_evaluate_by_regime_returns_financial_and_accuracy_metrics():
    df = pd.DataFrame(
        {
            "regime": ["uptrend", "uptrend", "downtrend", "downtrend"],
            "strategy_return": [0.01, -0.005, -0.01, 0.002],
            "true_target": [1, 1, -1, -1],
            "prediction": [1, 0, -1, 1],
        }
    )
    result = evaluate_by_regime(df, "regime")

    assert set(result["regime"]) == {"uptrend", "downtrend"}
    assert {"sharpe", "max_drawdown", "win_rate", "accuracy"}.issubset(result.columns)
