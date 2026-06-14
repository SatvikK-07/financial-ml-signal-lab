"""Purged walk-forward validation for strictly out-of-sample predictions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.models import LEAKAGE_COLUMNS, get_prediction_proba, get_predictions


def run_walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable[[pd.DataFrame, pd.Series], Any],
    target_col: str = "target",
    train_window: int = 1_000,
    test_window: int = 100,
    step_size: int = 100,
    purge_rows: int = 0,
    expanding: bool = False,
) -> pd.DataFrame:
    """Refit a model across historical folds and collect OOS predictions."""
    _validate_inputs(
        df,
        feature_cols,
        model_factory,
        target_col,
        train_window,
        test_window,
        step_size,
        purge_rows,
    )
    ordered = (
        df.sort_values("date", kind="stable").reset_index(names="source_index")
        if "date" in df.columns
        else df.reset_index(names="source_index")
    )
    first_test_start = train_window + purge_rows
    fold_results: list[pd.DataFrame] = []

    for fold, test_start in enumerate(
        range(first_test_start, len(ordered), step_size), start=1
    ):
        test_end = min(test_start + test_window, len(ordered))
        train_end = test_start - purge_rows
        train_start = 0 if expanding else train_end - train_window
        train = ordered.iloc[train_start:train_end]
        test = ordered.iloc[test_start:test_end]
        if test.empty:
            break

        model = model_factory(train[feature_cols], train[target_col])
        predictions = get_predictions(model, test[feature_cols])
        probabilities = get_prediction_proba(model, test[feature_cols])
        fold_result = _build_fold_result(
            train,
            test,
            target_col,
            predictions,
            probabilities,
            model,
            fold,
        )
        fold_results.append(fold_result)

    if not fold_results:
        raise ValueError("Configuration produced no walk-forward test folds.")
    results = pd.concat(fold_results, ignore_index=True)
    if results["source_index"].duplicated().any():
        raise ValueError("Walk-forward test windows produced duplicate predictions.")
    return results


def _build_fold_result(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    model: Any,
    fold: int,
) -> pd.DataFrame:
    output_columns = [
        column
        for column in ("source_index", "date", "close", target_col, "future_return")
        if column in test.columns
    ]
    result = test[output_columns].copy()
    if target_col != "true_target":
        result = result.rename(columns={target_col: "true_target"})
    result["prediction"] = predictions
    result["fold"] = fold
    result["train_rows"] = len(train)
    result["train_start_date"] = _boundary_value(train, "date", first=True)
    result["train_end_date"] = _boundary_value(train, "date", first=False)
    result["test_start_date"] = _boundary_value(test, "date", first=True)
    result["test_end_date"] = _boundary_value(test, "date", first=False)

    if probabilities is not None:
        classes = getattr(model, "classes_", np.arange(probabilities.shape[1]))
        if len(classes) != probabilities.shape[1]:
            raise ValueError("Model classes do not match probability columns.")
        for column_index, class_label in enumerate(classes):
            result[f"probability_{class_label}"] = probabilities[:, column_index]
        result["confidence"] = probabilities.max(axis=1)
    return result


def _boundary_value(
    frame: pd.DataFrame, column: str, first: bool
) -> pd.Timestamp | int:
    if column in frame.columns:
        return frame[column].iloc[0 if first else -1]
    return int(frame["source_index"].iloc[0 if first else -1])


def _validate_inputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable[[pd.DataFrame, pd.Series], Any],
    target_col: str,
    train_window: int,
    test_window: int,
    step_size: int,
    purge_rows: int,
) -> None:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature.")
    if not callable(model_factory):
        raise TypeError("model_factory must be callable.")
    leakage = set(feature_cols).intersection(LEAKAGE_COLUMNS.union({target_col}))
    if leakage:
        raise ValueError(f"Feature columns contain target leakage: {sorted(leakage)}")
    required = set(feature_cols).union({target_col})
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Model features and target must not contain missing values.")
    for name, value in {
        "train_window": train_window,
        "test_window": test_window,
        "step_size": step_size,
    }.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(purge_rows, int) or purge_rows < 0:
        raise ValueError("purge_rows must be a non-negative integer.")
    if step_size < test_window:
        raise ValueError("step_size must be at least test_window to avoid overlap.")
    if train_window + purge_rows >= len(df):
        raise ValueError("Not enough rows for the requested training window and purge.")
