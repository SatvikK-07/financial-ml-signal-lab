"""Command-line research pipeline and reproducible artifact export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.dashboard import MODEL_TRAINERS, build_dashboard_research
from src.data_loader import (
    download_data,
    download_recent_data,
    get_symbol_path,
    load_raw_data,
    save_processed_data,
    save_raw_data,
    symbol_slug,
)
from src.features import create_features
from src.targets import create_directional_target


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load project YAML configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")
    return config


def run_research_pipeline(
    config_path: str | Path = "config.yaml",
    reports_folder: str | Path = "reports",
    selected_model: str = "Random Forest",
    refresh_data: bool = False,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Build configured research results and export portfolio artifacts."""
    config = load_config(config_path)
    data_config = config["data"]
    selected_symbol = symbol or data_config["symbol"]
    asset_config = get_asset_config(config, selected_symbol)
    prepare_configured_dataset(config, selected_symbol, refresh=refresh_data)
    target_config = config["target"]
    backtest_config = config["backtest"]
    model_config = config["model"]
    walk_forward_config = config["walk_forward"]
    signal_policy_config = config.get("signal_policy", {})

    research = build_dashboard_research(
        symbol=selected_symbol,
        selected_model=selected_model,
        test_size=model_config["test_size"],
        target_horizon=target_config["horizon"],
        train_window=walk_forward_config["train_window"],
        test_window=walk_forward_config["test_window"],
        step_size=walk_forward_config["step_size"],
        transaction_cost=backtest_config["transaction_cost"],
        position_size=backtest_config["position_size"],
        initial_capital=backtest_config["initial_capital"],
        excluded_features=tuple(asset_config.get("excluded_features", [])),
        abstention_thresholds=tuple(
            signal_policy_config.get(
                "abstention_thresholds", [0.0, 0.45, 0.50, 0.55, 0.60]
            )
        ),
        intelligence_config=config.get("market_intelligence", {}),
    )
    export_research_artifacts(research, reports_folder)
    return research


def prepare_configured_dataset(
    config: dict[str, Any],
    symbol: str | None = None,
    refresh: bool = False,
) -> Path:
    """Download and process configured market data when needed."""
    data_config = config["data"]
    target_config = config["target"]
    selected_symbol = symbol or data_config["symbol"]
    processed_path = get_symbol_path(selected_symbol, "data/processed")
    if processed_path.exists() and not refresh:
        return processed_path

    interval = data_config.get("interval", "1d")
    downloaded = download_data(
        selected_symbol,
        data_config["start_date"],
        data_config.get("end_date"),
        interval,
    )
    raw_parts = [downloaded]
    raw_path = get_symbol_path(selected_symbol, "data/raw")
    if refresh and raw_path.exists():
        raw_parts.insert(0, load_raw_data(selected_symbol))
    if refresh:
        try:
            recent = download_recent_data(
                selected_symbol,
                period="10d" if interval == "1d" else "5d",
                interval=interval,
                include_prepost=False,
            )
        except Exception:
            recent = pd.DataFrame()
        if not recent.empty:
            raw_parts.append(recent)
    raw = pd.concat(raw_parts, ignore_index=True)
    save_raw_data(raw, selected_symbol)
    clean_raw = load_raw_data(selected_symbol)
    dataset = create_directional_target(
        create_features(clean_raw),
        horizon=target_config["horizon"],
        threshold=target_config["threshold"],
    )
    return Path(save_processed_data(dataset, selected_symbol))


def get_configured_assets(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return configured symbols and display labels."""
    data_config = config["data"]
    configured = data_config.get("assets")
    if not configured:
        symbol = data_config["symbol"]
        return [{"symbol": symbol, "label": symbol, "excluded_features": []}]
    assets: list[dict[str, Any]] = []
    for asset in configured:
        if isinstance(asset, str):
            assets.append(
                {"symbol": asset, "label": asset, "excluded_features": []}
            )
        elif isinstance(asset, dict) and asset.get("symbol"):
            normalized = dict(asset)
            normalized["symbol"] = str(asset["symbol"])
            normalized["label"] = str(asset.get("label", asset["symbol"]))
            normalized["excluded_features"] = list(asset.get("excluded_features", []))
            assets.append(normalized)
        else:
            raise ValueError("Each configured asset must contain a symbol.")
    return assets


def get_asset_config(config: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Return the normalized configuration for one symbol."""
    for asset in get_configured_assets(config):
        if asset["symbol"] == symbol:
            return asset
    return {"symbol": symbol, "label": symbol, "excluded_features": []}


def run_multi_asset_pipeline(
    config_path: str | Path = "config.yaml",
    reports_folder: str | Path = "reports",
    selected_model: str = "Random Forest",
    refresh_data: bool = False,
) -> pd.DataFrame:
    """Run and export configured research independently for every asset."""
    config = load_config(config_path)
    reports_path = Path(reports_folder)
    summaries: list[dict[str, Any]] = []
    for asset in get_configured_assets(config):
        symbol = asset["symbol"]
        asset_folder = reports_path / "assets" / symbol_slug(symbol)
        research = run_research_pipeline(
            config_path,
            asset_folder,
            selected_model,
            refresh_data,
            symbol,
        )
        summary = build_research_summary(research)
        summaries.append(
            {
                "symbol": symbol,
                "label": asset["label"],
                "dataset_rows": summary["dataset_rows"],
                "walk_forward_rows": summary["walk_forward_rows"],
                "best_holdout_model": summary["best_holdout_model"],
                "best_holdout_macro_f1": summary["best_holdout_macro_f1"],
                "strategy_total_return": summary["backtest"]["total_return"],
                "strategy_sharpe": summary["backtest"]["sharpe_ratio"],
                "strategy_max_drawdown": summary["backtest"]["max_drawdown"],
                "buy_hold_total_return": summary["backtest"]["buy_hold_total_return"],
                "latest_signal": summary["latest_signal"]["prediction"],
                "latest_confidence": summary["latest_signal"]["confidence"],
                "latest_date": summary["latest_signal"]["date"],
                "diagnostic_status": summary["diagnostics"]["status"],
                "diagnostic_warnings": " | ".join(
                    summary["diagnostics"]["warnings"]
                ),
                "data_integrity_score": summary["data_integrity"]["score"],
                "trust_score": summary["trade_intelligence"]["trust_score"],
                "trade_decision": summary["trade_intelligence"]["decision"],
                "monte_carlo_tp_first": summary["monte_carlo"][
                    "probability_take_profit_first"
                ],
                "monte_carlo_sl_first": summary["monte_carlo"][
                    "probability_stop_loss_first"
                ],
            }
        )
    cross_asset = pd.DataFrame(summaries)
    results_path = reports_path / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    cross_asset.to_csv(results_path / "cross_asset_summary.csv", index=False)
    return cross_asset


def export_research_artifacts(
    research: dict[str, Any], reports_folder: str | Path = "reports"
) -> dict[str, str]:
    """Export research tables, summary JSON, and figures."""
    reports_path = Path(reports_folder)
    results_path = reports_path / "results"
    figures_path = reports_path / "figures"
    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "model_comparison": results_path / "model_comparison.csv",
        "walk_forward_predictions": results_path / "walk_forward_predictions.csv",
        "backtest": results_path / "walk_forward_backtest.csv",
        "volatility_performance": results_path / "volatility_regimes.csv",
        "trend_performance": results_path / "trend_regimes.csv",
        "feature_importance": results_path / "feature_importance.csv",
        "abstention_performance": results_path / "abstention_performance.csv",
        "execution_policy_performance": results_path
        / "execution_policy_performance.csv",
        "market_zones": results_path / "market_zones.csv",
        "monte_carlo_paths": results_path / "monte_carlo_paths.csv",
        "monte_carlo_distribution": results_path
        / "monte_carlo_terminal_distribution.csv",
        "summary": results_path / "summary.json",
        "diagnostics": results_path / "diagnostics.json",
        "data_integrity": results_path / "data_integrity.json",
        "monte_carlo_summary": results_path / "monte_carlo_summary.json",
        "trade_intelligence": results_path / "trade_intelligence.json",
        "position_size": results_path / "position_size.json",
        "news_risk": results_path / "news_risk.json",
        "upcoming_events": results_path / "upcoming_events.csv",
        "equity_figure": figures_path / "walk_forward_equity.png",
        "regime_figure": figures_path / "regime_sharpe.png",
        "importance_figure": figures_path / "feature_importance.png",
    }
    research["comparison"].to_csv(paths["model_comparison"], index=False)
    research["walk_forward"].to_csv(paths["walk_forward_predictions"], index=False)
    research["backtest"].to_csv(paths["backtest"], index=False)
    research["volatility_performance"].to_csv(
        paths["volatility_performance"], index=False
    )
    research["trend_performance"].to_csv(paths["trend_performance"], index=False)
    research["feature_importance"].to_csv(paths["feature_importance"], index=False)
    research["abstention_performance"].to_csv(
        paths["abstention_performance"], index=False
    )
    research["execution_policy_performance"].to_csv(
        paths["execution_policy_performance"], index=False
    )
    research["market_zones"]["levels"].to_csv(paths["market_zones"], index=False)
    research["monte_carlo"]["displayed_paths"].to_csv(paths["monte_carlo_paths"])
    pd.DataFrame(
        {
            "terminal_price": research["monte_carlo"]["terminal_prices"],
            "terminal_return": research["monte_carlo"]["terminal_returns"],
        }
    ).to_csv(paths["monte_carlo_distribution"], index=False)
    research["news_risk"]["upcoming_events"].to_csv(
        paths["upcoming_events"], index=False
    )

    summary = build_research_summary(research)
    with paths["summary"].open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, default=_json_default)
    with paths["diagnostics"].open("w", encoding="utf-8") as diagnostics_file:
        json.dump(
            research["diagnostics"],
            diagnostics_file,
            indent=2,
            default=_json_default,
        )
    for path_name, payload in {
        "data_integrity": research["data_integrity"],
        "monte_carlo_summary": research["monte_carlo"]["summary"],
        "trade_intelligence": research["trade_intelligence"],
        "position_size": research["position_size"],
        "news_risk": _serializable_news_risk(research["news_risk"]),
    }.items():
        with paths[path_name].open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, default=_json_default)

    _plot_equity(research["backtest"], paths["equity_figure"])
    _plot_regimes(research, paths["regime_figure"])
    _plot_feature_importance(research["feature_importance"], paths["importance_figure"])
    return {name: str(path) for name, path in paths.items()}


def build_research_summary(research: dict[str, Any]) -> dict[str, Any]:
    """Create a compact, JSON-serializable research summary."""
    latest = research["latest_row"].iloc[0]
    explanation = research["latest_explanation"]
    current_signal = research.get("current_signal", {})
    best_holdout = research["comparison"].iloc[0]
    return {
        "symbol": research["symbol"],
        "selected_model": research["selected_model"],
        "dataset_rows": len(research["dataset"]),
        "feature_count": len(research["feature_cols"]),
        "excluded_features": research.get("excluded_features", []),
        "walk_forward_rows": len(research["walk_forward"]),
        "walk_forward_folds": int(research["walk_forward"]["fold"].nunique()),
        "walk_forward_start": research["walk_forward"]["date"].min(),
        "walk_forward_end": research["walk_forward"]["date"].max(),
        "best_holdout_model": best_holdout["model"],
        "best_holdout_macro_f1": best_holdout["macro_f1"],
        "backtest": research["backtest_summary"],
        "abstention_scenarios": research["abstention_performance"].to_dict(
            orient="records"
        ),
        "diagnostics": research["diagnostics"],
        "data_integrity": research["data_integrity"],
        "monte_carlo": research["monte_carlo"]["summary"],
        "trade_intelligence": research["trade_intelligence"],
        "position_size": research["position_size"],
        "news_risk": _serializable_news_risk(research["news_risk"]),
        "market_zones": {
            "nearest_support": research["market_zones"]["nearest_support"],
            "nearest_resistance": research["market_zones"]["nearest_resistance"],
            "near_support": research["market_zones"]["near_support"],
            "near_resistance": research["market_zones"]["near_resistance"],
        },
        "execution_policy_scenarios": research[
            "execution_policy_performance"
        ].to_dict(orient="records"),
        "latest_signal": {
            "date": latest["date"],
            "close": latest["close"],
            "prediction": explanation["prediction"],
            "confidence": explanation["confidence"],
            "volatility_regime": latest["volatility_regime"],
            "trend_regime": latest["trend_regime"],
        },
        "current_market_signal": {
            "date": current_signal.get("latest_date", latest["date"]),
            "close": current_signal.get("latest_close", latest["close"]),
            "prediction": current_signal.get(
                "prediction", explanation["prediction"]
            ),
            "confidence": current_signal.get(
                "confidence", explanation["confidence"]
            ),
            "signal_source": current_signal.get(
                "signal_source", "latest completed candle"
            ),
            "freshness_status": current_signal.get("freshness_status"),
            "data_freshness_minutes": current_signal.get(
                "data_freshness_minutes"
            ),
        },
        "latest_research_signal": research.get("research_signal"),
    }


def _plot_equity(backtest: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    first_return = (
        backtest["strategy_return"].iloc[0]
        if "strategy_return" in backtest.columns
        else 0.0
    )
    initial_capital = backtest["equity"].iloc[0] / (1 + first_return)
    pre_cost = initial_capital * (
        1 + backtest["strategy_return_before_cost"]
    ).cumprod()
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(backtest["date"], backtest["equity"], label="Strategy after costs")
    axis.plot(backtest["date"], pre_cost, label="Strategy before costs", alpha=0.8)
    axis.plot(backtest["date"], backtest["buy_hold_equity"], label="Buy and hold")
    axis.set_title("Walk-Forward Out-of-Sample Equity")
    axis.set_ylabel("Equity ($)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_regimes(research: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    volatility = research["volatility_performance"]
    trend = research["trend_performance"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(volatility["regime"], volatility["sharpe"], color="#315a84")
    axes[0].set_title("Sharpe by Volatility Regime")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].bar(trend["regime"], trend["sharpe"], color="#c7772f")
    axes[1].set_title("Sharpe by Trend Regime")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].axhline(0, color="black", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    top = importance.head(15).sort_values("importance")
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(top["feature"], top["importance"], color="#315a84")
    axis.set_title("Random Forest Global Feature Importance")
    axis.set_xlabel("Normalized importance")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _serializable_news_risk(news_risk: dict[str, Any]) -> dict[str, Any]:
    result = dict(news_risk)
    events = result.pop("upcoming_events", pd.DataFrame())
    result["upcoming_events"] = events.to_dict(orient="records")
    return result


def main() -> None:
    """Run the configured research pipeline from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--reports-folder", default="reports")
    parser.add_argument("--model", choices=MODEL_TRAINERS, default="Random Forest")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--symbol")
    parser.add_argument("--all-assets", action="store_true")
    args = parser.parse_args()
    if args.all_assets:
        summary = run_multi_asset_pipeline(
            args.config,
            args.reports_folder,
            args.model,
            refresh_data=args.refresh_data,
        )
        print(summary.to_string(index=False))
        return
    research = run_research_pipeline(
        args.config,
        args.reports_folder,
        args.model,
        refresh_data=args.refresh_data,
        symbol=args.symbol,
    )
    summary = research["backtest_summary"]
    print(
        f"Exported {research['selected_model']} research: "
        f"{summary['total_return']:.2%} strategy return, "
        f"{summary['buy_hold_total_return']:.2%} buy-and-hold return."
    )


if __name__ == "__main__":
    main()
