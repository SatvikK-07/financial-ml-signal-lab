"""Reusable research pipeline for the Streamlit dashboard."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtester import compare_execution_policies, run_backtest, summarize_backtest
from src.data_loader import load_processed_data, load_raw_data
from src.data_integrity import analyze_data_integrity, infer_market_calendar
from src.diagnostics import analyze_research_plausibility
from src.explainability import get_feature_importance
from src.features import create_features, get_feature_columns
from src.latest_signal import build_latest_unlabeled_signal
from src.market_zones import detect_market_zones
from src.models import (
    chronological_train_test_split,
    evaluate_classifier,
    get_predictions,
    train_lightgbm,
    train_logistic_regression,
    train_naive_baseline,
    train_random_forest,
    train_xgboost,
)
from src.regimes import add_trend_regime, add_volatility_regime, evaluate_by_regime
from src.signal_policy import evaluate_abstention_thresholds
from src.monte_carlo import simulate_price_paths
from src.news_risk import (
    get_upcoming_events,
    infer_relevant_currencies,
    load_economic_calendar,
    score_news_risk,
)
from src.trade_intelligence import (
    build_trade_intelligence,
    calculate_position_size,
    infer_asset_type,
)
from src.walk_forward import run_walk_forward_validation

MODEL_TRAINERS: dict[str, Callable[[pd.DataFrame, pd.Series], Any]] = {
    "Naive Baseline": train_naive_baseline,
    "Logistic Regression": train_logistic_regression,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm,
}


def build_dashboard_research(
    symbol: str = "SPY",
    processed_folder: str | Path = "data/processed",
    raw_folder: str | Path = "data/raw",
    selected_model: str = "Random Forest",
    test_size: float = 0.2,
    target_horizon: int = 5,
    volatility_lookback: int = 252,
    train_window: int = 1_000,
    test_window: int = 100,
    step_size: int = 100,
    transaction_cost: float = 0.0005,
    position_size: float = 1.0,
    initial_capital: float = 10_000,
    excluded_features: tuple[str, ...] = (),
    abstention_thresholds: tuple[float, ...] = (0.0, 0.45, 0.50, 0.55, 0.60),
    intelligence_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete research bundle used by dashboard pages."""
    if selected_model not in MODEL_TRAINERS:
        raise ValueError(f"Unknown model: {selected_model}")
    dataset = load_processed_data(symbol, processed_folder)
    available_features = get_feature_columns(dataset)
    unknown_exclusions = set(excluded_features).difference(available_features)
    if unknown_exclusions:
        raise ValueError(
            f"Feature policy references unknown features: {sorted(unknown_exclusions)}"
        )
    feature_cols = [
        feature for feature in available_features if feature not in excluded_features
    ]
    if not feature_cols:
        raise ValueError("Feature policy excluded every available model feature.")
    regime_dataset = add_trend_regime(
        add_volatility_regime(dataset, lookback=volatility_lookback)
    )

    comparison, models, holdout_predictions = build_model_comparison(
        dataset,
        feature_cols,
        test_size=test_size,
        purge_rows=target_horizon,
    )
    selected_trainer = MODEL_TRAINERS[selected_model]
    walk_forward = run_walk_forward_validation(
        regime_dataset,
        feature_cols,
        selected_trainer,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        purge_rows=target_horizon,
    )
    walk_forward[["volatility_regime", "trend_regime"]] = regime_dataset.loc[
        walk_forward["source_index"], ["volatility_regime", "trend_regime"]
    ].reset_index(drop=True)
    backtest = run_backtest(
        walk_forward,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        position_size=position_size,
    )
    backtest_summary = summarize_backtest(backtest)
    execution_policy_performance = compare_execution_policies(
        walk_forward,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        position_size=position_size,
        horizon=target_horizon,
    )
    abstention_performance = evaluate_abstention_thresholds(
        walk_forward,
        abstention_thresholds,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        position_size=position_size,
    )
    comparison = comparison.copy()
    comparison["walk_forward_sharpe"] = pd.NA
    comparison["walk_forward_total_return"] = pd.NA
    comparison["walk_forward_max_drawdown"] = pd.NA
    comparison["walk_forward_win_rate"] = pd.NA
    selected_mask = comparison["model"] == selected_model
    comparison.loc[selected_mask, "walk_forward_sharpe"] = backtest_summary[
        "sharpe_ratio"
    ]
    comparison.loc[selected_mask, "walk_forward_total_return"] = backtest_summary[
        "total_return"
    ]
    comparison.loc[selected_mask, "walk_forward_max_drawdown"] = backtest_summary[
        "max_drawdown"
    ]
    comparison.loc[selected_mask, "walk_forward_win_rate"] = backtest_summary[
        "win_rate"
    ]

    raw_data = _load_raw_or_fallback(symbol, raw_folder, regime_dataset)
    current_signal = build_latest_unlabeled_signal(
        raw_data,
        regime_dataset,
        feature_cols,
        selected_trainer,
        excluded_features=excluded_features,
        asset_type=infer_asset_type(symbol),
    )
    latest_row = _load_latest_feature_row(
        raw_data,
        regime_dataset,
        volatility_lookback,
    )
    current_signal["feature_row"] = latest_row
    latest_explanation = {
        "prediction": current_signal["prediction"],
        "confidence": current_signal["confidence"],
        "probabilities": current_signal["probabilities"],
        "top_features": current_signal["top_features"],
    }
    research_signal = _build_research_signal(walk_forward)
    research = {
        "symbol": symbol,
        "selected_model": selected_model,
        "target_horizon": target_horizon,
        "dataset": regime_dataset,
        "feature_cols": feature_cols,
        "excluded_features": list(excluded_features),
        "comparison": comparison,
        "models": models,
        "holdout_predictions": holdout_predictions,
        "walk_forward": walk_forward,
        "backtest": backtest,
        "backtest_summary": backtest_summary,
        "execution_policy_performance": execution_policy_performance,
        "abstention_performance": abstention_performance,
        "volatility_performance": evaluate_by_regime(
            backtest, "volatility_regime"
        ),
        "trend_performance": evaluate_by_regime(backtest, "trend_regime"),
        "feature_importance": get_feature_importance(
            current_signal["model"], feature_cols
        ),
        "current_signal": current_signal,
        "research_signal": research_signal,
        "latest_explanation": latest_explanation,
        "latest_row": latest_row,
    }
    research["diagnostics"] = analyze_research_plausibility(research)
    research.update(
        _build_market_intelligence(
            research,
            raw_data,
            intelligence_config or {},
        )
    )
    return research


def build_model_comparison(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    test_size: float = 0.2,
    purge_rows: int = 5,
    trainers: dict[str, Callable[[pd.DataFrame, pd.Series], Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    """Train available models on one chronological holdout for comparison."""
    trainers = trainers or MODEL_TRAINERS
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        dataset,
        feature_cols,
        test_size=test_size,
        purge_rows=purge_rows,
    )
    models: dict[str, Any] = {}
    predictions: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, float | str]] = []
    for name, trainer in trainers.items():
        try:
            model = trainer(X_train, y_train)
        except ImportError:
            continue
        metrics = evaluate_classifier(model, X_test, y_test)
        models[name] = model
        prediction_frame = dataset.loc[
            X_test.index, ["date", "close", "target", "future_return"]
        ].copy()
        prediction_frame["prediction"] = get_predictions(model, X_test)
        predictions[name] = prediction_frame
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        "macro_f1", ascending=False
    ).reset_index(drop=True)
    return comparison, models, predictions


def _load_latest_feature_row(
    raw_data: pd.DataFrame,
    fallback_dataset: pd.DataFrame,
    volatility_lookback: int,
) -> pd.DataFrame:
    latest_features = add_trend_regime(
        add_volatility_regime(create_features(raw_data), lookback=volatility_lookback)
    )
    latest = latest_features.tail(1)
    return latest if not latest.empty else fallback_dataset.tail(1)


def _build_research_signal(walk_forward: pd.DataFrame) -> dict[str, Any]:
    """Return the latest strictly out-of-sample labeled signal."""
    latest = walk_forward.sort_values("date", kind="stable").iloc[-1]
    return {
        "date": pd.to_datetime(latest["date"]),
        "close": float(latest["close"]),
        "prediction": int(latest["prediction"]),
        "confidence": (
            float(latest["confidence"])
            if "confidence" in latest.index and pd.notna(latest["confidence"])
            else None
        ),
        "true_target": int(latest["true_target"]),
        "future_return": float(latest["future_return"]),
        "source": "latest strictly out-of-sample labeled row",
    }


def _load_raw_or_fallback(
    symbol: str,
    raw_folder: str | Path,
    fallback_dataset: pd.DataFrame,
) -> pd.DataFrame:
    try:
        return load_raw_data(symbol, raw_folder)
    except FileNotFoundError:
        return fallback_dataset[["date", "open", "high", "low", "close", "volume"]]


def _build_market_intelligence(
    research: dict[str, Any],
    raw_data: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    monte_carlo_config = config.get("monte_carlo", {})
    integrity_config = config.get("data_integrity", {})
    zone_config = config.get("zones", {})
    sizing_config = config.get("position_sizing", {})
    latest = research["latest_row"].iloc[0]
    explanation = research["latest_explanation"]
    current_price = float(latest["close"])
    prediction = int(explanation["prediction"])
    side = prediction if prediction in {-1, 1} else 1
    stop_loss_pct = float(monte_carlo_config.get("stop_loss_pct", 0.01))
    take_profit_pct = float(monte_carlo_config.get("take_profit_pct", 0.02))
    take_profit_level = current_price * (
        1 + take_profit_pct if side == 1 else 1 - take_profit_pct
    )
    stop_loss_level = current_price * (
        1 - stop_loss_pct if side == 1 else 1 + stop_loss_pct
    )
    monte_carlo = simulate_price_paths(
        raw_data["close"],
        current_price=current_price,
        lookback=int(monte_carlo_config.get("lookback", 100)),
        horizon=int(monte_carlo_config.get("horizon", 20)),
        simulations=int(monte_carlo_config.get("simulations", 10_000)),
        displayed_paths=int(monte_carlo_config.get("displayed_paths", 100)),
        confidence_level=float(monte_carlo_config.get("confidence_level", 0.95)),
        take_profit_level=take_profit_level,
        stop_loss_level=stop_loss_level,
        side=side,
        random_seed=int(monte_carlo_config.get("random_seed", 42)),
        method=str(monte_carlo_config.get("method", "iid_bootstrap")),
        block_size=int(monte_carlo_config.get("block_size", 5)),
    )
    monte_carlo["summary"]["signal_prediction"] = prediction
    monte_carlo["summary"]["target_label"] = (
        "Take profit" if prediction != 0 else "Upper barrier"
    )
    monte_carlo["summary"]["stop_label"] = (
        "Stop loss" if prediction != 0 else "Lower barrier"
    )
    integrity_lookback = int(integrity_config.get("lookback", 252))
    data_integrity = analyze_data_integrity(
        raw_data.tail(integrity_lookback),
        calendar=infer_market_calendar(research["symbol"]),
        stale_after_days=int(integrity_config.get("stale_after_days", 3)),
        jump_threshold=float(integrity_config.get("jump_threshold", 0.08)),
    )
    market_zones = detect_market_zones(
        raw_data,
        lookback=int(zone_config.get("lookback", 120)),
        swing_window=int(zone_config.get("swing_window", 5)),
        proximity_pct=float(zone_config.get("proximity_pct", 0.01)),
    )
    news_config = config.get("news_risk", {})
    calendar = load_economic_calendar(
        news_config.get("calendar_path", "data/manual/economic_calendar.csv")
    )
    upcoming_events = get_upcoming_events(
        calendar,
        now=pd.Timestamp.now(),
        currencies=infer_relevant_currencies(research["symbol"]),
        lookahead_minutes=int(news_config.get("lookahead_minutes", 120)),
    )
    news_risk = score_news_risk(
        upcoming_events,
        high_impact_window_minutes=int(
            news_config.get("high_impact_window_minutes", 60)
        ),
    )
    trade_intelligence = build_trade_intelligence(
        prediction=prediction,
        confidence=float(explanation["confidence"]),
        trend_regime=str(latest["trend_regime"]),
        volatility_regime=str(latest["volatility_regime"]),
        backtest_summary=research["backtest_summary"],
        trend_performance=research["trend_performance"],
        data_integrity=data_integrity,
        monte_carlo_summary=monte_carlo["summary"],
        market_zones=market_zones,
        diagnostic_warnings=research["diagnostics"]["warnings"],
        news_risk=news_risk,
    )
    position_size = calculate_position_size(
        account_size=float(sizing_config.get("account_size", 25_000)),
        risk_pct=float(trade_intelligence["suggested_risk_pct"]),
        entry_price=current_price,
        stop_price=stop_loss_level,
        asset_type=infer_asset_type(research["symbol"]),
        contract_size=float(sizing_config.get("contract_size", 100_000)),
    )
    return {
        "monte_carlo": monte_carlo,
        "data_integrity": data_integrity,
        "market_zones": market_zones,
        "trade_intelligence": trade_intelligence,
        "position_size": position_size,
        "news_risk": news_risk,
    }
