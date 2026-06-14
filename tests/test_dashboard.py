"""Tests for the dashboard research data service."""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard import (
    _clean_model_ready_dataset,
    build_dashboard_research,
    build_model_comparison,
)
from src.features import create_features, get_feature_columns
from src.models import train_logistic_regression, train_naive_baseline
from src.targets import create_directional_target


def test_model_comparison_returns_sorted_metrics(sample_ohlcv):
    dataset = create_directional_target(create_features(sample_ohlcv))
    comparison, models, predictions = build_model_comparison(
        dataset,
        get_feature_columns(dataset),
        test_size=0.2,
        purge_rows=5,
        trainers={
            "Naive": train_naive_baseline,
            "Logistic": train_logistic_regression,
        },
    )

    assert comparison["macro_f1"].is_monotonic_decreasing
    assert set(models) == {"Naive", "Logistic"}
    assert set(predictions) == {"Naive", "Logistic"}


def test_clean_model_ready_dataset_drops_corrupt_downloaded_rows():
    dataset = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "invalid", "2026-01-04"],
            "close": [100.0, 101.0, 102.0, 103.0],
            "feature": [1.0, 2.0, 3.0, float("inf")],
            "target": [1, 0, -1, 1],
            "future_return": [0.01, 0.0, -0.01, 0.02],
        }
    )

    clean = _clean_model_ready_dataset(dataset, ["feature"])

    assert len(clean) == 2
    assert clean["date"].notna().all()
    assert clean["feature"].notna().all()


@pytest.mark.slow
@pytest.mark.integration
def test_dashboard_research_bundle_contains_required_sections(tmp_path, sample_ohlcv):
    dataset = create_directional_target(create_features(sample_ohlcv))
    dataset.to_csv(tmp_path / "SPY.csv", index=False)

    research = build_dashboard_research(
        processed_folder=tmp_path,
        selected_model="Logistic Regression",
        test_size=0.2,
        target_horizon=3,
        volatility_lookback=20,
        train_window=40,
        test_window=10,
        step_size=10,
    )

    required = {
        "comparison",
        "walk_forward",
        "backtest",
        "backtest_summary",
        "execution_policy_performance",
        "abstention_performance",
        "volatility_performance",
        "trend_performance",
        "feature_importance",
        "current_signal",
        "research_signal",
        "latest_explanation",
        "monte_carlo",
        "data_integrity",
        "market_zones",
        "trade_intelligence",
        "position_size",
        "news_risk",
        "target_horizon",
    }
    assert required.issubset(research)
    assert research["walk_forward"]["source_index"].is_unique
    assert isinstance(research["backtest_summary"]["total_return"], float)
    assert research["current_signal"]["latest_date"] >= research["research_signal"]["date"]
    assert research["target_horizon"] == 3


@pytest.mark.slow
@pytest.mark.integration
def test_dashboard_research_applies_excluded_feature_policy(tmp_path, sample_ohlcv):
    dataset = create_directional_target(create_features(sample_ohlcv))
    dataset.to_csv(tmp_path / "SPY.csv", index=False)

    research = build_dashboard_research(
        processed_folder=tmp_path,
        selected_model="Logistic Regression",
        test_size=0.2,
        target_horizon=3,
        volatility_lookback=20,
        train_window=40,
        test_window=10,
        step_size=10,
        excluded_features=("upper_wick", "lower_wick"),
    )

    assert "upper_wick" not in research["feature_cols"]
    assert "lower_wick" not in research["feature_cols"]
    assert research["excluded_features"] == ["upper_wick", "lower_wick"]


def test_dashboard_research_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_dashboard_research(selected_model="Unknown")
