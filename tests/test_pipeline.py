"""Tests for reproducible research artifact export."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_raw_data, save_raw_data
from src.pipeline import (
    build_research_summary,
    export_research_artifacts,
    get_configured_assets,
    prepare_configured_dataset,
)


@pytest.mark.slow
@pytest.mark.integration
def test_export_research_artifacts_writes_tables_summary_and_figures(tmp_path):
    dates = pd.date_range("2025-01-01", periods=4)
    research = {
        "symbol": "SPY",
        "selected_model": "Random Forest",
        "dataset": pd.DataFrame({"date": dates, "close": [100, 101, 102, 103]}),
        "feature_cols": ["return_1"],
        "comparison": pd.DataFrame(
            {"model": ["Random Forest"], "macro_f1": [0.35]}
        ),
        "walk_forward": pd.DataFrame(
            {"date": dates, "fold": [1, 1, 2, 2], "prediction": [1, 0, -1, 1]}
        ),
        "backtest": pd.DataFrame(
            {
                "date": dates,
                "strategy_return_before_cost": [0, 0.01, -0.01, 0.02],
                "equity": [10_000, 10_100, 9_999, 10_198.98],
                "buy_hold_equity": [10_000, 10_100, 10_200, 10_300],
            }
        ),
        "backtest_summary": {"total_return": 0.019898, "buy_hold_total_return": 0.03},
        "volatility_performance": pd.DataFrame(
            {"regime": ["low_volatility"], "sharpe": [0.5]}
        ),
        "trend_performance": pd.DataFrame({"regime": ["uptrend"], "sharpe": [0.4]}),
        "feature_importance": pd.DataFrame(
            {"feature": ["return_1"], "importance": [1.0]}
        ),
        "abstention_performance": pd.DataFrame(
            {
                "minimum_confidence": [0.0],
                "total_return": [0.019898],
                "sharpe_ratio": [0.2],
            }
        ),
        "execution_policy_performance": pd.DataFrame(
            {
                "policy": ["daily_rebalance"],
                "total_return": [0.019898],
                "sharpe_ratio": [0.2],
            }
        ),
        "latest_row": pd.DataFrame(
            {
                "date": [dates[-1]],
                "close": [103.0],
                "volatility_regime": ["low_volatility"],
                "trend_regime": ["uptrend"],
            }
        ),
        "latest_explanation": {"prediction": 0, "confidence": 0.4},
        "diagnostics": {"status": "no_automatic_flags", "warnings": []},
        "data_integrity": {"status": "good", "score": 100, "warnings": []},
        "monte_carlo": {
            "summary": {"expected_return": 0.01},
            "displayed_paths": pd.DataFrame({"path_1": [100, 101]}),
            "terminal_prices": pd.Series([101.0]),
            "terminal_returns": pd.Series([0.01]),
        },
        "market_zones": {
            "levels": pd.DataFrame(
                {"level": [99.0, 105.0], "type": ["support", "resistance"]}
            ),
            "nearest_support": {"level": 99.0},
            "nearest_resistance": {"level": 105.0},
            "near_support": False,
            "near_resistance": False,
        },
        "trade_intelligence": {"trust_score": 50, "warnings": []},
        "position_size": {"risk_amount": 0.0, "units": 0.0},
        "news_risk": {
            "news_risk_score": 0,
            "risk_level": "low",
            "warnings": [],
            "upcoming_events": pd.DataFrame(),
        },
    }

    paths = export_research_artifacts(research, tmp_path)

    assert all(Path(path).exists() for path in paths.values())
    with open(paths["summary"], encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    assert summary["selected_model"] == "Random Forest"
    assert summary["walk_forward_folds"] == 2


def test_build_research_summary_uses_best_holdout_model():
    research = {
        "symbol": "SPY",
        "selected_model": "Logistic Regression",
        "dataset": pd.DataFrame({"value": [1, 2]}),
        "feature_cols": ["return_1"],
        "comparison": pd.DataFrame(
            {"model": ["Random Forest", "Logistic Regression"], "macro_f1": [0.4, 0.3]}
        ),
        "walk_forward": pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=2),
                "fold": [1, 1],
            }
        ),
        "backtest_summary": {"total_return": -0.1},
        "abstention_performance": pd.DataFrame(
            {
                "minimum_confidence": [0.0],
                "total_return": [-0.1],
                "sharpe_ratio": [-0.2],
            }
        ),
        "execution_policy_performance": pd.DataFrame(
            {
                "policy": ["daily_rebalance"],
                "total_return": [-0.1],
                "sharpe_ratio": [-0.2],
            }
        ),
        "latest_row": pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-02")],
                "close": [100.0],
                "volatility_regime": ["normal_volatility"],
                "trend_regime": ["sideways"],
            }
        ),
        "latest_explanation": {"prediction": 0, "confidence": 0.4},
        "diagnostics": {"status": "no_automatic_flags", "warnings": []},
        "data_integrity": {"status": "good", "score": 100, "warnings": []},
        "monte_carlo": {"summary": {"expected_return": 0.01}},
        "market_zones": {
            "nearest_support": {"level": 99.0},
            "nearest_resistance": {"level": 105.0},
            "near_support": False,
            "near_resistance": False,
        },
        "trade_intelligence": {"trust_score": 50, "warnings": []},
        "position_size": {"risk_amount": 0.0, "units": 0.0},
        "news_risk": {
            "news_risk_score": 0,
            "risk_level": "low",
            "warnings": [],
            "upcoming_events": pd.DataFrame(),
        },
    }

    summary = build_research_summary(research)

    assert summary["best_holdout_model"] == "Random Forest"
    assert summary["selected_model"] == "Logistic Regression"


def test_prepare_configured_dataset_reuses_existing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processed_folder = tmp_path / "data" / "processed"
    processed_folder.mkdir(parents=True)
    existing = processed_folder / "SPY.csv"
    existing.write_text("date,close\n", encoding="utf-8")
    config = {
        "data": {"symbol": "SPY", "start_date": "2020-01-01", "interval": "1d"},
        "target": {"horizon": 5, "threshold": 0.005},
    }

    assert prepare_configured_dataset(config) == Path("data/processed/SPY.csv")


def test_refresh_merges_existing_and_recent_data(
    tmp_path, monkeypatch, sample_ohlcv
):
    monkeypatch.chdir(tmp_path)
    existing = sample_ohlcv.iloc[:100]
    older_download = sample_ohlcv.iloc[:90]
    recent_download = sample_ohlcv.iloc[95:105]
    save_raw_data(existing, "SPY")
    monkeypatch.setattr(
        "src.pipeline.download_data",
        lambda *args, **kwargs: older_download,
    )
    monkeypatch.setattr(
        "src.pipeline.download_recent_data",
        lambda *args, **kwargs: recent_download,
    )
    config = {
        "data": {"symbol": "SPY", "start_date": "2020-01-01", "interval": "1d"},
        "target": {"horizon": 5, "threshold": 0.005},
    }

    prepare_configured_dataset(config, refresh=True)
    refreshed = load_raw_data("SPY")

    assert refreshed["date"].max() == recent_download["date"].max()
    assert len(refreshed) == 105


def test_get_configured_assets_supports_labels_and_string_symbols():
    config = {
        "data": {
            "symbol": "SPY",
            "assets": [
                {"symbol": "SPY", "label": "US Equities"},
                "BTC-USD",
            ],
        }
    }

    assert get_configured_assets(config) == [
        {"symbol": "SPY", "label": "US Equities", "excluded_features": []},
        {"symbol": "BTC-USD", "label": "BTC-USD", "excluded_features": []},
    ]
