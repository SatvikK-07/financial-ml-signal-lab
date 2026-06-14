"""Persistent local paper-trading journal and candle-based outcome tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

PAPER_TRADE_COLUMNS = [
    "trade_id",
    "created_at",
    "symbol",
    "asset_type",
    "timeframe",
    "signal",
    "confidence",
    "trust_score",
    "trade_quality_score",
    "decision",
    "entry_price",
    "stop_price",
    "target_price",
    "risk_pct",
    "account_size",
    "position_size",
    "status",
    "exit_price",
    "exit_time",
    "exit_reason",
    "pnl_pct",
    "pnl_amount",
    "notes",
    "model_name",
    "data_timestamp",
]
DATETIME_COLUMNS = ["created_at", "exit_time", "data_timestamp"]
NUMERIC_COLUMNS = [
    "signal",
    "confidence",
    "trust_score",
    "trade_quality_score",
    "entry_price",
    "stop_price",
    "target_price",
    "risk_pct",
    "account_size",
    "position_size",
    "exit_price",
    "pnl_pct",
    "pnl_amount",
]
TEXT_COLUMNS = [
    "trade_id",
    "symbol",
    "asset_type",
    "timeframe",
    "decision",
    "status",
    "exit_reason",
    "notes",
    "model_name",
]


def create_paper_trade(
    symbol: str,
    signal: int,
    confidence: float,
    trust_score: float,
    trade_quality_score: float,
    decision: str,
    entry_price: float,
    stop_price: float | None,
    target_price: float | None,
    risk_pct: float,
    account_size: float,
    position_size: float | None,
    model_name: str,
    data_timestamp: Any,
    notes: str = "",
    storage_path: str = "data/paper_trades/paper_trades.csv",
    asset_type: str | None = None,
    timeframe: str = "1d",
) -> dict[str, Any]:
    """Persist a new hypothetical trade without placing a real order."""
    if signal not in {-1, 1}:
        raise ValueError("Paper trades require a long or short signal.")
    if entry_price <= 0 or account_size <= 0:
        raise ValueError("entry_price and account_size must be positive.")
    if not 0 <= confidence <= 1 or not 0 <= risk_pct <= 1:
        raise ValueError("confidence and risk_pct must be between 0 and 1.")
    created_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    trade = {
        "trade_id": uuid4().hex,
        "created_at": created_at,
        "symbol": symbol,
        "asset_type": asset_type or _infer_asset_type(symbol),
        "timeframe": timeframe,
        "signal": signal,
        "confidence": confidence,
        "trust_score": trust_score,
        "trade_quality_score": trade_quality_score,
        "decision": decision,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_pct": risk_pct,
        "account_size": account_size,
        "position_size": position_size,
        "status": "open",
        "exit_price": None,
        "exit_time": None,
        "exit_reason": None,
        "pnl_pct": None,
        "pnl_amount": None,
        "notes": notes,
        "model_name": model_name,
        "data_timestamp": pd.Timestamp(data_timestamp),
    }
    trades = load_paper_trades(storage_path)
    if trades.empty:
        trades = pd.DataFrame([trade], columns=PAPER_TRADE_COLUMNS)
    else:
        trades.loc[len(trades), PAPER_TRADE_COLUMNS] = [
            trade[column] for column in PAPER_TRADE_COLUMNS
        ]
    _save_paper_trades(trades, storage_path)
    return trade


def load_paper_trades(
    storage_path: str = "data/paper_trades/paper_trades.csv",
) -> pd.DataFrame:
    """Load the local paper-trading journal."""
    path = Path(storage_path)
    if not path.exists():
        return pd.DataFrame(columns=PAPER_TRADE_COLUMNS)
    trades = pd.read_csv(path)
    for column in PAPER_TRADE_COLUMNS:
        if column not in trades.columns:
            trades[column] = pd.NA
    for column in DATETIME_COLUMNS:
        trades[column] = pd.to_datetime(trades[column], errors="coerce")
    for column in NUMERIC_COLUMNS:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")
    for column in TEXT_COLUMNS:
        trades[column] = trades[column].astype("object")
    return trades[PAPER_TRADE_COLUMNS].sort_values(
        "created_at", ascending=False, kind="stable"
    ).reset_index(drop=True)


def update_open_paper_trades(
    latest_data: pd.DataFrame,
    storage_path: str = "data/paper_trades/paper_trades.csv",
) -> pd.DataFrame:
    """Update open trades from later candles using conservative barrier ordering."""
    required = {"date", "high", "low", "close"}
    missing = required.difference(latest_data.columns)
    if missing:
        raise ValueError(f"Latest data is missing columns: {sorted(missing)}")
    candles = latest_data.copy()
    candles["date"] = pd.to_datetime(candles["date"], errors="coerce")
    candles = candles.dropna(subset=["date"]).sort_values("date", kind="stable")
    trades = load_paper_trades(storage_path)
    symbol = latest_data.attrs.get("symbol")
    for index, trade in trades.loc[trades["status"] == "open"].iterrows():
        if symbol and trade["symbol"] != symbol:
            continue
        eligible = candles.loc[candles["date"] > pd.Timestamp(trade["data_timestamp"])]
        outcome = _first_trade_outcome(trade, eligible)
        if outcome is None:
            continue
        _apply_close(
            trades,
            index,
            outcome["exit_price"],
            outcome["exit_reason"],
            outcome["exit_time"],
            outcome.get("warning"),
        )
    _save_paper_trades(trades, storage_path)
    return load_paper_trades(storage_path)


def close_paper_trade(
    trade_id: str,
    exit_price: float,
    exit_reason: str,
    storage_path: str = "data/paper_trades/paper_trades.csv",
) -> dict[str, Any]:
    """Manually close an open paper trade."""
    if exit_price <= 0:
        raise ValueError("exit_price must be positive.")
    if exit_reason not in {"target_hit", "stop_hit", "manual_close", "expired"}:
        raise ValueError("Unknown paper-trade exit reason.")
    trades = load_paper_trades(storage_path)
    matches = trades.index[trades["trade_id"] == trade_id]
    if len(matches) != 1:
        raise ValueError("Paper trade was not found.")
    index = matches[0]
    if trades.at[index, "status"] != "open":
        raise ValueError("Only open paper trades can be closed.")
    _apply_close(
        trades,
        index,
        exit_price,
        exit_reason,
        pd.Timestamp.now(tz="UTC").tz_localize(None),
    )
    result = trades.loc[index].to_dict()
    _save_paper_trades(trades, storage_path)
    return result


def summarize_paper_trading(trades: pd.DataFrame) -> dict[str, float | int]:
    """Summarize hypothetical paper-trading outcomes."""
    if trades.empty:
        return {
            "total_trades": 0,
            "open_trades": 0,
            "closed_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "average_pnl": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }
    closed = trades.loc[trades["status"] == "closed"].copy()
    pnl = pd.to_numeric(closed["pnl_amount"], errors="coerce").dropna()
    pnl_pct = pd.to_numeric(closed["pnl_pct"], errors="coerce").dropna()
    return {
        "total_trades": int(len(trades)),
        "open_trades": int((trades["status"] == "open").sum()),
        "closed_trades": int(len(closed)),
        "win_rate": float((pnl_pct > 0).mean()) if not pnl_pct.empty else 0.0,
        "total_pnl": float(pnl.sum()) if not pnl.empty else 0.0,
        "average_pnl": float(pnl.mean()) if not pnl.empty else 0.0,
        "best_trade": float(pnl.max()) if not pnl.empty else 0.0,
        "worst_trade": float(pnl.min()) if not pnl.empty else 0.0,
    }


def _first_trade_outcome(
    trade: pd.Series,
    candles: pd.DataFrame,
) -> dict[str, Any] | None:
    signal = int(trade["signal"])
    stop = _optional_float(trade["stop_price"])
    target = _optional_float(trade["target_price"])
    for _, candle in candles.iterrows():
        high = float(candle["high"])
        low = float(candle["low"])
        if signal == 1:
            stop_hit = stop is not None and low <= stop
            target_hit = target is not None and high >= target
        else:
            stop_hit = stop is not None and high >= stop
            target_hit = target is not None and low <= target
        if stop_hit:
            return {
                "exit_price": stop,
                "exit_reason": "stop_hit",
                "exit_time": candle["date"],
                "warning": (
                    "Both stop and target touched in same candle; conservative "
                    "stop-first assumption used."
                    if target_hit
                    else None
                ),
            }
        if target_hit:
            return {
                "exit_price": target,
                "exit_reason": "target_hit",
                "exit_time": candle["date"],
            }
    return None


def _apply_close(
    trades: pd.DataFrame,
    index: int,
    exit_price: float,
    exit_reason: str,
    exit_time: Any,
    warning: str | None = None,
) -> None:
    signal = int(trades.at[index, "signal"])
    entry_price = float(trades.at[index, "entry_price"])
    pnl_pct = signal * (float(exit_price) / entry_price - 1)
    position_size = _optional_float(trades.at[index, "position_size"])
    pnl_amount = (
        signal * (float(exit_price) - entry_price) * position_size
        if position_size is not None
        else float(trades.at[index, "account_size"]) * pnl_pct
    )
    trades.at[index, "status"] = "closed"
    trades.at[index, "exit_price"] = float(exit_price)
    trades.at[index, "exit_time"] = pd.Timestamp(exit_time)
    trades.at[index, "exit_reason"] = exit_reason
    trades.at[index, "pnl_pct"] = pnl_pct
    trades.at[index, "pnl_amount"] = pnl_amount
    if warning:
        current_notes = trades.at[index, "notes"]
        existing = "" if pd.isna(current_notes) else str(current_notes).strip()
        trades.at[index, "notes"] = f"{existing} {warning}".strip()


def _save_paper_trades(trades: pd.DataFrame, storage_path: str) -> None:
    path = Path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = trades.copy()
    for column in PAPER_TRADE_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    output[PAPER_TRADE_COLUMNS].to_csv(path, index=False)


def _optional_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def _infer_asset_type(symbol: str) -> str:
    upper = symbol.upper()
    if upper.endswith("=X") or "/" in upper:
        return "fx"
    if upper.endswith("-USD"):
        return "crypto"
    return "equity"
