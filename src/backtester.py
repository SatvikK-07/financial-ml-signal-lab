"""Cost-adjusted, delayed-execution backtesting utilities."""

from __future__ import annotations

import pandas as pd

from src.metrics import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_win_rate,
)

VALID_SIGNALS = {-1, 0, 1}
EXECUTION_MODES = {
    "daily_rebalance",
    "hold_for_horizon",
    "signal_change_only",
    "confidence_filtered",
}


def generate_positions(predictions: pd.Series) -> pd.Series:
    """Map long, neutral, and short predictions directly to positions."""
    positions = pd.to_numeric(predictions, errors="coerce")
    if positions.isna().any():
        raise ValueError("Predictions must contain only numeric signals.")
    invalid = set(positions.unique()).difference(VALID_SIGNALS)
    if invalid:
        raise ValueError(f"Predictions contain invalid signals: {sorted(invalid)}")
    return positions.astype(float).rename("position")


def apply_execution_policy(
    predictions: pd.Series,
    mode: str = "daily_rebalance",
    horizon: int = 5,
    confidence: pd.Series | None = None,
    minimum_confidence: float | None = None,
) -> pd.Series:
    """Convert raw predictions into intended positions without lookahead."""
    signals = generate_positions(predictions).reset_index(drop=True)
    if mode not in EXECUTION_MODES:
        raise ValueError(f"Unknown execution mode: {mode}")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    if mode == "confidence_filtered":
        if confidence is None or minimum_confidence is None:
            raise ValueError(
                "confidence_filtered requires confidence and minimum_confidence."
            )
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1.")
        confidence_values = pd.to_numeric(confidence, errors="coerce").reset_index(
            drop=True
        )
        if len(confidence_values) != len(signals) or confidence_values.isna().any():
            raise ValueError("confidence must align with predictions and contain numbers.")
        if ((confidence_values < 0) | (confidence_values > 1)).any():
            raise ValueError("confidence values must be between 0 and 1.")
        signals.loc[confidence_values < minimum_confidence] = 0.0
        return signals.rename("position")

    if mode in {"daily_rebalance", "signal_change_only"}:
        return signals.rename("position")

    intended = pd.Series(0.0, index=signals.index, name="position")
    index = 0
    while index < len(signals):
        signal = float(signals.iloc[index])
        if signal == 0:
            index += 1
            continue
        holding_end = min(index + horizon, len(signals))
        intended.iloc[index:holding_end] = signal
        index = holding_end
    return intended


def run_backtest(
    df: pd.DataFrame,
    prediction_col: str = "prediction",
    price_col: str = "close",
    initial_capital: float = 10_000,
    transaction_cost: float = 0.0005,
    position_size: float = 1.0,
    mode: str = "daily_rebalance",
    horizon: int = 5,
    confidence_col: str | None = None,
    minimum_confidence: float | None = None,
) -> pd.DataFrame:
    """Run a delayed-execution strategy backtest with proportional costs."""
    _validate_backtest_inputs(
        df,
        prediction_col,
        price_col,
        initial_capital,
        transaction_cost,
        position_size,
    )
    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="raise")
        result = result.sort_values("date", kind="stable").reset_index(drop=True)

    result[price_col] = pd.to_numeric(result[price_col], errors="raise")
    confidence = result[confidence_col] if confidence_col else None
    result["position"] = apply_execution_policy(
        result[prediction_col],
        mode=mode,
        horizon=horizon,
        confidence=confidence,
        minimum_confidence=minimum_confidence,
    ) * position_size
    result["executed_position"] = result["position"].shift(1).fillna(0.0)
    result["asset_return"] = result[price_col].pct_change().fillna(0.0)
    result["strategy_return_before_cost"] = (
        result["executed_position"] * result["asset_return"]
    )
    result["turnover"] = result["executed_position"].diff().abs()
    result.loc[result.index[0], "turnover"] = abs(
        result.loc[result.index[0], "executed_position"]
    )
    result["transaction_cost"] = result["turnover"] * transaction_cost
    result["strategy_return"] = (
        result["strategy_return_before_cost"] - result["transaction_cost"]
    )
    result["equity"] = initial_capital * (1 + result["strategy_return"]).cumprod()
    result["buy_hold_equity"] = initial_capital * (1 + result["asset_return"]).cumprod()
    result["rolling_max_equity"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["rolling_max_equity"] - 1
    return result


def compare_execution_policies(
    df: pd.DataFrame,
    prediction_col: str = "prediction",
    policies: list[dict[str, object]] | None = None,
    price_col: str = "close",
    initial_capital: float = 10_000,
    transaction_cost: float = 0.0005,
    position_size: float = 1.0,
    horizon: int = 5,
) -> pd.DataFrame:
    """Compare predefined execution policies on the same predictions."""
    configured_policies = policies or [
        {"policy": "daily_rebalance", "mode": "daily_rebalance"},
        {
            "policy": f"hold_for_horizon_{horizon}",
            "mode": "hold_for_horizon",
            "horizon": horizon,
        },
        {"policy": "signal_change_only", "mode": "signal_change_only"},
        {
            "policy": "confidence_filtered_60pct",
            "mode": "confidence_filtered",
            "confidence_col": "confidence",
            "minimum_confidence": 0.60,
        },
    ]
    rows: list[dict[str, float | int | str]] = []
    for policy in configured_policies:
        policy_name = str(policy.get("policy", policy.get("mode", "policy")))
        mode = str(policy.get("mode", "daily_rebalance"))
        confidence_col = policy.get("confidence_col")
        if mode == "confidence_filtered" and (
            not isinstance(confidence_col, str) or confidence_col not in df.columns
        ):
            continue
        backtest = run_backtest(
            df,
            prediction_col=prediction_col,
            price_col=price_col,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            position_size=position_size,
            mode=mode,
            horizon=int(policy.get("horizon", horizon)),
            confidence_col=confidence_col if isinstance(confidence_col, str) else None,
            minimum_confidence=(
                float(policy["minimum_confidence"])
                if policy.get("minimum_confidence") is not None
                else None
            ),
        )
        summary = summarize_backtest(backtest)
        rows.append(
            {
                "policy": policy_name,
                "total_return": summary["total_return"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "max_drawdown": summary["max_drawdown"],
                "number_of_trades": summary["number_of_trades"],
                "exposure": summary["exposure"],
                "transaction_cost_paid": summary["total_transaction_cost"],
                "win_rate": summary["win_rate"],
                "profit_factor": summary["profit_factor"],
            }
        )
    return pd.DataFrame(rows)


def summarize_backtest(
    backtest_df: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Summarize strategy and buy-and-hold financial performance."""
    required = {
        "equity",
        "buy_hold_equity",
        "strategy_return",
        "executed_position",
    }
    missing = required.difference(backtest_df.columns)
    if missing:
        raise ValueError(f"Backtest dataframe is missing columns: {sorted(missing)}")
    if backtest_df.empty:
        raise ValueError("Backtest dataframe is empty.")

    trade_returns = _extract_trade_returns(backtest_df)
    return {
        "total_return": calculate_total_return(backtest_df["equity"]),
        "cagr": calculate_cagr(backtest_df["equity"], periods_per_year),
        "sharpe_ratio": calculate_sharpe_ratio(
            backtest_df["strategy_return"], periods_per_year
        ),
        "sortino_ratio": calculate_sortino_ratio(
            backtest_df["strategy_return"], periods_per_year
        ),
        "max_drawdown": calculate_max_drawdown(backtest_df["equity"]),
        "win_rate": calculate_win_rate(trade_returns),
        "profit_factor": calculate_profit_factor(trade_returns),
        "number_of_trades": int(len(trade_returns)),
        "average_trade_return": (
            float(trade_returns.mean()) if not trade_returns.empty else 0.0
        ),
        "exposure": float((backtest_df["executed_position"] != 0).mean()),
        "total_transaction_cost": float(backtest_df["transaction_cost"].sum()),
        "buy_hold_total_return": calculate_total_return(
            backtest_df["buy_hold_equity"]
        ),
        "buy_hold_sharpe_ratio": calculate_sharpe_ratio(
            backtest_df["asset_return"], periods_per_year
        ),
        "buy_hold_max_drawdown": calculate_max_drawdown(
            backtest_df["buy_hold_equity"]
        ),
    }


def _extract_trade_returns(backtest_df: pd.DataFrame) -> pd.Series:
    active = backtest_df["executed_position"] != 0
    new_trade = active & (
        backtest_df["executed_position"]
        != backtest_df["executed_position"].shift(1).fillna(0)
    )
    trade_id = new_trade.cumsum()
    active_returns = backtest_df.loc[active, ["strategy_return"]].copy()
    if active_returns.empty:
        return pd.Series(dtype=float, name="trade_return")
    active_returns["trade_id"] = trade_id.loc[active]
    trade_returns = active_returns.groupby("trade_id")["strategy_return"].apply(
        lambda returns: (1 + returns).prod() - 1
    )
    return trade_returns.rename("trade_return")


def _validate_backtest_inputs(
    df: pd.DataFrame,
    prediction_col: str,
    price_col: str,
    initial_capital: float,
    transaction_cost: float,
    position_size: float,
) -> None:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    missing = {prediction_col, price_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing columns: {sorted(missing)}")
    if df[[prediction_col, price_col]].isna().any().any():
        raise ValueError("Prediction and price columns must not contain missing values.")
    prices = pd.to_numeric(df[price_col], errors="coerce")
    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("Prices must be positive.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative.")
    if not 0 < position_size <= 1:
        raise ValueError("position_size must be between 0 and 1.")
