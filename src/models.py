"""Chronological model training and classification evaluation utilities."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

LEAKAGE_COLUMNS = {"future_return", "target"}


class LabelEncodedClassifier:
    """Adapt classifiers that require zero-based integer target labels."""

    def __init__(self, estimator: Any) -> None:
        """Initialize the adapter with an unfitted classifier."""
        self.estimator = estimator
        self.label_encoder: Any | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LabelEncodedClassifier":
        """Encode target labels, fit the classifier, and return the adapter."""
        from sklearn.preprocessing import LabelEncoder

        self.label_encoder = LabelEncoder()
        encoded_target = self.label_encoder.fit_transform(y)
        self.estimator.fit(X, encoded_target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict labels in their original representation."""
        if self.label_encoder is None:
            raise ValueError("Classifier has not been fitted.")
        encoded_predictions = self.estimator.predict(X).astype(int)
        return self.label_encoder.inverse_transform(encoded_predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities ordered by the original class labels."""
        return self.estimator.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        """Return the original class labels learned during fitting."""
        if self.label_encoder is None:
            raise ValueError("Classifier has not been fitted.")
        return self.label_encoder.classes_


def chronological_train_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    test_size: float | int = 0.2,
    purge_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split chronologically and optionally purge overlapping training labels."""
    _validate_model_data(df, feature_cols, target_col)
    ordered = (
        df.sort_values("date", kind="stable") if "date" in df.columns else df.copy()
    )
    test_rows = _resolve_test_rows(len(ordered), test_size)
    split_index = len(ordered) - test_rows
    if not isinstance(purge_rows, int) or purge_rows < 0:
        raise ValueError("purge_rows must be a non-negative integer.")
    train_end = split_index - purge_rows
    if train_end < 1:
        raise ValueError("purge_rows must leave at least one training row.")

    X = ordered.loc[:, feature_cols]
    y = ordered.loc[:, target_col]
    return (
        X.iloc[:train_end],
        X.iloc[split_index:],
        y.iloc[:train_end],
        y.iloc[split_index:],
    )


def train_naive_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train a majority-class baseline classifier."""
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="most_frequent")
    return model.fit(X_train, y_train)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Any:
    """Train scaled, class-balanced logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return model.fit(X_train, y_train)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 100,
) -> Any:
    """Train a class-balanced random forest classifier."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    return model.fit(X_train, y_train)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LabelEncodedClassifier:
    """Train an XGBoost classifier when the optional dependency is installed."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ImportError(
            "XGBoost is unavailable. Install xgboost and its native runtime."
        ) from exc

    estimator = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=random_state,
    )
    return LabelEncodedClassifier(estimator).fit(X_train, y_train)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LabelEncodedClassifier:
    """Train a LightGBM classifier when the optional dependency is installed."""
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise ImportError(
            "LightGBM is unavailable. Install lightgbm and its native runtime."
        ) from exc

    estimator = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=random_state,
        verbosity=-1,
    )
    return LabelEncodedClassifier(estimator).fit(X_train, y_train)


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate a fitted classifier with robust multiclass metrics."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    predictions = get_predictions(model, X_test)
    labels = sorted(set(np.asarray(y_test)).union(predictions))
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "labels": labels,
    }


def get_predictions(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    """Return one-dimensional classifier predictions."""
    predictions = np.asarray(model.predict(X_test))
    if predictions.ndim != 1:
        raise ValueError("Classifier predictions must be one-dimensional.")
    return predictions


def get_prediction_proba(model: Any, X_test: pd.DataFrame) -> np.ndarray | None:
    """Return class probabilities when supported by the classifier."""
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = np.asarray(model.predict_proba(X_test))
    if probabilities.ndim != 2:
        raise ValueError("Classifier probabilities must be two-dimensional.")
    return probabilities


def _validate_model_data(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> None:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature.")
    if target_col not in df.columns:
        raise ValueError(f"Target column does not exist: {target_col}")
    missing = set(feature_cols).difference(df.columns)
    if missing:
        raise ValueError(f"Feature columns do not exist: {sorted(missing)}")
    leakage = set(feature_cols).intersection(LEAKAGE_COLUMNS.union({target_col}))
    if leakage:
        raise ValueError(f"Feature columns contain target leakage: {sorted(leakage)}")
    if df.loc[:, list(feature_cols) + [target_col]].isna().any().any():
        raise ValueError("Model features and target must not contain missing values.")


def _resolve_test_rows(number_of_rows: int, test_size: float | int) -> int:
    if number_of_rows < 2:
        raise ValueError("At least two rows are required for a train-test split.")
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("Float test_size must be between 0 and 1.")
        test_rows = int(np.ceil(number_of_rows * test_size))
    elif isinstance(test_size, int):
        test_rows = test_size
    else:
        raise TypeError("test_size must be a float or integer.")

    if not 0 < test_rows < number_of_rows:
        raise ValueError("test_size must leave at least one training and test row.")
    return test_rows
