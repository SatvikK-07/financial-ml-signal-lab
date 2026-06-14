"""Tests for automated research plausibility diagnostics."""

from __future__ import annotations

import pandas as pd

from src.diagnostics import analyze_research_plausibility


def test_diagnostics_flag_suspicious_results():
    research = {
        "dataset": pd.DataFrame(
            {
                "close": [100, 101, 102, 103],
                "volume": [0, 0, 0, 0],
                "feature": [0.0, 0.1, -0.1, 0.0],
            }
        ),
        "backtest": pd.DataFrame(
            {
                "executed_position": [0, 1, 1, 1],
                "asset_return": [0, 0.1, 0.1, 0.1],
                "strategy_return_before_cost": [0, 1.0, 1.0, 1.0],
            }
        ),
        "backtest_summary": {"sharpe_ratio": 4.0},
        "feature_cols": ["feature"],
    }

    diagnostics = analyze_research_plausibility(research)

    assert diagnostics["status"] == "review_required"
    assert len(diagnostics["warnings"]) >= 2


def test_diagnostics_accept_ordinary_results():
    research = {
        "dataset": pd.DataFrame(
            {
                "close": [100, 101, 100, 102],
                "volume": [100, 110, 90, 105],
                "feature": [0.0, 0.0, 0.0, 0.0],
            }
        ),
        "backtest": pd.DataFrame(
            {
                "executed_position": [0, 0, 0, 0],
                "asset_return": [0.01, -0.01, 0.02, -0.01],
                "strategy_return_before_cost": [0, -0.01, -0.02, -0.01],
            }
        ),
        "backtest_summary": {"sharpe_ratio": -1.0},
        "feature_cols": ["feature"],
    }

    diagnostics = analyze_research_plausibility(research)

    assert diagnostics["status"] == "no_automatic_flags"
    assert diagnostics["warnings"] == []


def test_diagnostics_flag_feature_correlated_with_next_return():
    research = {
        "dataset": pd.DataFrame(
            {
                "close": [100.0, 101.0, 99.0, 102.0, 98.0],
                "volume": [100, 100, 100, 100, 100],
                "leaky_feature": [0.01, -0.0198, 0.0303, -0.0392, 0.0],
            }
        ),
        "backtest": pd.DataFrame(
            {
                "executed_position": [0, 0, 0, 0, 0],
                "asset_return": [0.0, 0.01, -0.0198, 0.0303, -0.0392],
                "strategy_return_before_cost": [0, 0, 0, 0, 0],
            }
        ),
        "backtest_summary": {"sharpe_ratio": 0.0},
        "feature_cols": ["leaky_feature"],
    }

    diagnostics = analyze_research_plausibility(research)

    assert diagnostics["status"] == "review_required"
    assert diagnostics["strongest_feature_return_correlation"]["feature"] == (
        "leaky_feature"
    )
