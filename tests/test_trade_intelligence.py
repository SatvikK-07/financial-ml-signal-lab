"""Tests for trader-facing trust scoring and position sizing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.trade_intelligence import (
    build_trade_intelligence,
    calculate_position_size,
    infer_asset_type,
)


def test_trade_intelligence_is_explainable_and_conservative():
    result = build_trade_intelligence(
        prediction=1,
        confidence=0.65,
        trend_regime="uptrend",
        volatility_regime="normal_volatility",
        backtest_summary={"sharpe_ratio": 0.5},
        trend_performance=pd.DataFrame({"regime": ["uptrend"], "sharpe": [0.8]}),
        data_integrity={"score": 100.0, "warnings": []},
        monte_carlo_summary={
            "probability_take_profit_first": 0.6,
            "probability_stop_loss_first": 0.3,
        },
        market_zones={"near_support": True, "near_resistance": False},
        diagnostic_warnings=[],
    )

    assert 0 <= result["trust_score"] <= 100
    assert result["reasons"]
    assert result["news_status"] == "unavailable"
    assert any("news" in warning.lower() for warning in result["warnings"])
    assert {
        "model_confidence_component",
        "data_integrity_component",
        "regime_component",
        "backtest_component",
        "monte_carlo_component",
        "zone_component",
        "volatility_component",
        "diagnostic_penalty",
        "freshness_penalty",
    }.issubset(result["score_breakdown"])
    assert result["trust_level"] in {"Low", "Moderate", "High"}
    assert len(result["trust_components"]) == 10
    assert {
        "component",
        "score",
        "impact",
        "status",
        "reason",
    }.issubset(result["trust_components"][0])


def test_position_size_calculates_fx_lots():
    result = calculate_position_size(
        account_size=25_000,
        risk_pct=0.005,
        entry_price=1.08,
        stop_price=1.0788,
        asset_type="fx",
    )

    assert result["risk_amount"] == 125
    assert result["stop_pips"] == pytest.approx(12)
    assert result["lots"] == pytest.approx(1.0416667)
    assert result["estimated_loss_at_stop"] == pytest.approx(125)


def test_asset_type_inference_recognizes_common_fx_symbols():
    assert infer_asset_type("EURUSD=X") == "fx"
    assert infer_asset_type("EUR/USD") == "fx"
    assert infer_asset_type("SPY") == "equity"


def test_high_news_risk_forces_wait_and_zero_risk():
    result = build_trade_intelligence(
        prediction=1,
        confidence=0.8,
        trend_regime="uptrend",
        volatility_regime="normal_volatility",
        backtest_summary={"sharpe_ratio": 1.0},
        trend_performance=pd.DataFrame({"regime": ["uptrend"], "sharpe": [1.0]}),
        data_integrity={"score": 100.0, "warnings": []},
        monte_carlo_summary={
            "probability_take_profit_first": 0.7,
            "probability_stop_loss_first": 0.2,
        },
        market_zones={"near_support": True, "near_resistance": False},
        diagnostic_warnings=[],
        news_risk={
            "risk_level": "high",
            "warnings": ["High-impact event soon."],
        },
    )

    assert result["decision"] == "Wait — high-impact news risk"
    assert result["suggested_risk_pct"] == 0


def test_stale_intraday_data_reduces_trust_and_forces_wait():
    base_arguments = {
        "prediction": 1,
        "confidence": 0.8,
        "trend_regime": "uptrend",
        "volatility_regime": "normal_volatility",
        "backtest_summary": {"sharpe_ratio": 1.0},
        "trend_performance": pd.DataFrame(
            {"regime": ["uptrend"], "sharpe": [1.0]}
        ),
        "monte_carlo_summary": {
            "probability_take_profit_first": 0.7,
            "probability_stop_loss_first": 0.2,
        },
        "market_zones": {"near_support": True, "near_resistance": False},
        "diagnostic_warnings": [],
        "news_risk": {"risk_level": "low", "warnings": []},
    }
    fresh = build_trade_intelligence(
        **base_arguments,
        data_integrity={
            "score": 100.0,
            "warnings": [],
            "status": "excellent",
            "is_stale": False,
            "is_intraday": True,
            "data_freshness_minutes": 2,
        },
    )
    stale = build_trade_intelligence(
        **base_arguments,
        data_integrity={
            "score": 75.0,
            "warnings": ["Latest candle is stale."],
            "status": "warning",
            "is_stale": True,
            "is_intraday": True,
            "data_freshness_minutes": 60,
        },
    )

    assert stale["trust_score"] < fresh["trust_score"]
    assert stale["score_breakdown"]["freshness_penalty"] == -20
    assert stale["decision"] == "Wait — intraday market data is stale"
    assert stale["suggested_risk_pct"] == 0
    assert any(
        detail["severity"] == "critical"
        and detail["source"] == "Data freshness"
        for detail in stale["warning_details"]
    )
