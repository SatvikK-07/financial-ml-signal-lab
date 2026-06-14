"""Tests for leakage-safe current market signal generation."""

from __future__ import annotations

import pandas as pd

from src.features import create_features, get_feature_columns
from src.latest_signal import build_latest_unlabeled_signal
from src.models import train_logistic_regression
from src.targets import create_directional_target


def test_latest_unlabeled_signal_uses_latest_feature_row(sample_ohlcv):
    full_features = create_features(sample_ohlcv)
    labeled = create_directional_target(full_features, horizon=5, threshold=0.005)

    result = build_latest_unlabeled_signal(
        sample_ohlcv,
        labeled,
        get_feature_columns(labeled),
        train_logistic_regression,
        as_of=full_features["date"].max(),
    )

    assert result["latest_date"] == full_features["date"].max()
    assert result["latest_labeled_date"] == labeled["date"].max()
    assert result["latest_date"] > result["latest_labeled_date"]
    assert result["latest_is_newer_than_labeled"]
    assert result["signal_source"] == "latest completed candle"


def test_latest_unlabeled_signal_does_not_require_future_target(sample_ohlcv):
    full_features = create_features(sample_ohlcv)
    labeled = create_directional_target(full_features, horizon=5, threshold=0.005)
    requested_features = get_feature_columns(labeled) + ["future_return", "target"]

    result = build_latest_unlabeled_signal(
        sample_ohlcv,
        labeled,
        requested_features,
        train_logistic_regression,
        as_of=full_features["date"].max(),
    )

    assert "future_return" not in result["feature_cols"]
    assert "target" not in result["feature_cols"]
    assert "future_return" not in result["feature_row"].columns
    assert "target" not in result["feature_row"].columns


def test_latest_unlabeled_signal_marks_old_completed_candle_stale(sample_ohlcv):
    full_features = create_features(sample_ohlcv)
    labeled = create_directional_target(full_features, horizon=5, threshold=0.005)

    result = build_latest_unlabeled_signal(
        sample_ohlcv,
        labeled,
        get_feature_columns(labeled),
        train_logistic_regression,
        as_of=full_features["date"].max() + pd.Timedelta(days=6),
    )

    assert result["is_stale"]
    assert result["freshness_status"] == "stale"
    assert result["freshness_message"]
    assert result["freshness_message"] in result["warnings"]


def test_latest_unlabeled_crypto_signal_uses_24_7_freshness_rules(sample_ohlcv):
    full_features = create_features(sample_ohlcv)
    labeled = create_directional_target(full_features, horizon=5, threshold=0.005)

    result = build_latest_unlabeled_signal(
        sample_ohlcv,
        labeled,
        get_feature_columns(labeled),
        train_logistic_regression,
        as_of=full_features["date"].max() + pd.Timedelta(days=4),
        asset_type="crypto",
    )

    assert result["is_stale"]
    assert "Crypto trades 24/7" in result["freshness_message"]
