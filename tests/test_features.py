"""Tests for leakage-safe feature engineering."""

from __future__ import annotations

import pandas as pd

from src.features import create_features, get_feature_columns
from src.targets import create_directional_target


def test_feature_creation_keeps_ohlcv_and_produces_complete_rows(sample_ohlcv):
    features = create_features(sample_ohlcv)

    assert {"date", "open", "high", "low", "close", "volume"}.issubset(features)
    assert not features[get_feature_columns(features)].isna().any().any()


def test_features_at_time_t_do_not_change_when_future_rows_change(sample_ohlcv):
    baseline = create_features(sample_ohlcv)
    changed = sample_ohlcv.copy()
    changed.loc[changed.index[-10:], ["open", "high", "low", "close"]] *= 10
    changed_features = create_features(changed)
    feature_cols = get_feature_columns(baseline)

    cutoff_date = sample_ohlcv.loc[sample_ohlcv.index[-11], "date"]
    baseline_past = baseline.loc[baseline["date"] <= cutoff_date, feature_cols]
    changed_past = changed_features.loc[
        changed_features["date"] <= cutoff_date, feature_cols
    ]
    pd.testing.assert_frame_equal(
        baseline_past.reset_index(drop=True),
        changed_past.reset_index(drop=True),
    )


def test_feature_columns_exclude_metadata_and_targets(sample_ohlcv):
    dataset = create_directional_target(create_features(sample_ohlcv))
    feature_cols = get_feature_columns(dataset)
    excluded = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "future_return",
        "target",
    }

    assert feature_cols
    assert excluded.isdisjoint(feature_cols)

