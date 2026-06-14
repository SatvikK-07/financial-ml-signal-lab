"""Tests for local hypothetical paper-trading outcome tracking."""

from __future__ import annotations

import pandas as pd
import pytest

from src.paper_trading import (
    close_paper_trade,
    create_paper_trade,
    load_paper_trades,
    summarize_paper_trading,
    update_open_paper_trades,
)


def _create_trade(path, signal=1, stop=95.0, target=105.0):
    return create_paper_trade(
        symbol="SPY",
        signal=signal,
        confidence=0.7,
        trust_score=60,
        trade_quality_score=60,
        decision="Paper research setup",
        entry_price=100,
        stop_price=stop,
        target_price=target,
        risk_pct=0.005,
        account_size=25_000,
        position_size=10,
        model_name="Test Model",
        data_timestamp="2026-01-01",
        storage_path=str(path),
    )


def _update(path, high, low, close=100):
    candles = pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "open": [100],
            "high": [high],
            "low": [low],
            "close": [close],
        }
    )
    candles.attrs["symbol"] = "SPY"
    return update_open_paper_trades(candles, str(path))


def test_create_and_load_paper_trade(tmp_path):
    path = tmp_path / "paper.csv"
    created = _create_trade(path)
    trades = load_paper_trades(str(path))

    assert created["status"] == "open"
    assert trades.loc[0, "trade_id"] == created["trade_id"]
    assert trades.loc[0, "symbol"] == "SPY"


def test_manual_close_paper_trade(tmp_path):
    path = tmp_path / "paper.csv"
    created = _create_trade(path)
    closed = close_paper_trade(created["trade_id"], 102, "manual_close", str(path))

    assert closed["status"] == "closed"
    assert closed["pnl_pct"] == pytest.approx(0.02)
    assert closed["pnl_amount"] == pytest.approx(20)


@pytest.mark.parametrize(
    ("signal", "stop", "target", "high", "low", "reason", "pnl_pct"),
    [
        (1, 95, 105, 106, 99, "target_hit", 0.05),
        (1, 95, 105, 101, 94, "stop_hit", -0.05),
        (-1, 105, 95, 101, 94, "target_hit", 0.05),
        (-1, 105, 95, 106, 99, "stop_hit", -0.05),
    ],
)
def test_long_and_short_barrier_outcomes(
    tmp_path, signal, stop, target, high, low, reason, pnl_pct
):
    path = tmp_path / f"{signal}_{reason}.csv"
    _create_trade(path, signal=signal, stop=stop, target=target)
    trades = _update(path, high=high, low=low)

    assert trades.loc[0, "exit_reason"] == reason
    assert trades.loc[0, "pnl_pct"] == pytest.approx(pnl_pct)


def test_both_barriers_use_conservative_stop_first(tmp_path):
    path = tmp_path / "both.csv"
    _create_trade(path, signal=1, stop=95, target=105)
    trades = _update(path, high=106, low=94)

    assert trades.loc[0, "exit_reason"] == "stop_hit"
    assert "conservative stop-first" in trades.loc[0, "notes"]


def test_paper_trading_summary_metrics(tmp_path):
    path = tmp_path / "summary.csv"
    first = _create_trade(path)
    close_paper_trade(first["trade_id"], 105, "target_hit", str(path))
    second = _create_trade(path)
    close_paper_trade(second["trade_id"], 95, "stop_hit", str(path))
    _create_trade(path)

    summary = summarize_paper_trading(load_paper_trades(str(path)))

    assert summary["total_trades"] == 3
    assert summary["open_trades"] == 1
    assert summary["closed_trades"] == 2
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["best_trade"] == pytest.approx(50)
    assert summary["worst_trade"] == pytest.approx(-50)
