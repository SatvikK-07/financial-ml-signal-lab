"""Trader-facing trust, confluence, and position-sizing decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.position_sizing import (
    calculate_fixed_fractional_position_size,
    calculate_fx_position_size,
)


def build_trade_intelligence(
    prediction: int,
    confidence: float,
    trend_regime: str,
    volatility_regime: str,
    backtest_summary: dict[str, Any],
    trend_performance: pd.DataFrame,
    data_integrity: dict[str, Any],
    monte_carlo_summary: dict[str, Any],
    market_zones: dict[str, Any],
    diagnostic_warnings: list[str],
    news_status: str = "unavailable",
    news_risk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an explainable signal trust and trade-quality assessment."""
    if prediction not in {-1, 0, 1}:
        raise ValueError("prediction must be -1, 0, or 1.")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")

    reasons: list[str] = []
    warnings: list[str] = []
    confidence_score = confidence * 100
    data_score = float(data_integrity["score"])
    regime_sharpe = _regime_sharpe(trend_performance, trend_regime)
    regime_score = float(np.clip(50 + regime_sharpe * 20, 0, 100))
    backtest_score = float(
        np.clip(50 + float(backtest_summary["sharpe_ratio"]) * 20, 0, 100)
    )
    monte_carlo_score = _monte_carlo_alignment(prediction, monte_carlo_summary)
    zone_score = _zone_alignment(prediction, market_zones)
    is_stale = bool(data_integrity.get("is_stale", False))
    is_intraday = bool(data_integrity.get("is_intraday", False))

    raw_component_scores = {
        "model_confidence": confidence_score,
        "data_integrity": data_score,
        "regime": regime_score,
        "backtest": backtest_score,
        "monte_carlo": monte_carlo_score,
        "zone": zone_score,
    }
    weights = {
        "model_confidence": 0.20,
        "data_integrity": 0.20,
        "regime": 0.15,
        "backtest": 0.15,
        "monte_carlo": 0.20,
        "zone": 0.10,
    }
    score_breakdown = {
        f"{name}_component": raw_component_scores[name] * weights[name]
        for name in raw_component_scores
    }
    volatility_component = -5.0 if volatility_regime == "high_volatility" else 0.0
    diagnostic_penalty = -float(min(20, len(diagnostic_warnings) * 5))
    resolved_news_level = (
        str(news_risk.get("risk_level", "low")) if news_risk else news_status
    )
    news_penalty = {
        "high": -20.0,
        "critical": -20.0,
        "medium": -10.0,
        "unavailable": -5.0,
    }.get(resolved_news_level, 0.0)
    freshness_penalty = -20.0 if is_stale and is_intraday else (-10.0 if is_stale else 0.0)
    score_breakdown.update(
        {
            "volatility_component": volatility_component,
            "diagnostic_penalty": diagnostic_penalty,
            "news_penalty": news_penalty,
            "freshness_penalty": freshness_penalty,
        }
    )
    trust_score = sum(score_breakdown.values())
    if diagnostic_warnings:
        warnings.extend(diagnostic_warnings)
    if resolved_news_level == "unavailable":
        warnings.append("Live economic-news risk is not connected; confirm manually.")
    elif news_risk:
        warnings.extend(news_risk.get("warnings", []))
    if volatility_component < 0:
        warnings.append("Volatility is elevated; use reduced size.")
    if is_stale:
        warnings.append("Latest data may be stale. Signal trust reduced.")
    trust_score = float(np.clip(trust_score, 0, 100))

    if confidence >= 0.6:
        reasons.append("Model confidence clears the 60% research threshold.")
    else:
        warnings.append("Model confidence is moderate or low.")
    if data_score >= 90:
        reasons.append("Data integrity checks are strong.")
    else:
        warnings.extend(data_integrity["warnings"])
    if regime_sharpe > 0:
        reasons.append(f"Historical performance is positive in {trend_regime}.")
    else:
        warnings.append(f"Historical performance is weak in {trend_regime}.")
    if monte_carlo_score > 55:
        reasons.append("Monte Carlo scenarios favor the signal direction.")
    else:
        warnings.append("Monte Carlo scenarios do not strongly favor the signal.")
    _append_zone_context(prediction, market_zones, reasons, warnings)
    reasons = list(dict.fromkeys(reasons))
    warnings = list(dict.fromkeys(warnings))

    if prediction == 0:
        decision = "Wait — model is neutral"
    elif trust_score >= 75:
        decision = "High-quality setup — still require execution confirmation"
    elif trust_score >= 60:
        decision = "Selective setup — consider reduced risk"
    elif trust_score >= 45:
        decision = "Low-quality setup — wait for stronger confluence"
    else:
        decision = "Avoid — trust and confluence are insufficient"
    usefulness = "high" if trust_score >= 75 else ("moderate" if trust_score >= 55 else "low")
    suggested_risk_pct = 0.0075 if trust_score >= 75 else (0.005 if trust_score >= 60 else 0.0025)
    if trust_score < 45 or prediction == 0:
        suggested_risk_pct = 0.0
    if resolved_news_level in {"high", "critical"}:
        suggested_risk_pct = 0.0
        warnings.append("High-impact news risk requires avoiding new trades.")
        decision = "Wait — high-impact news risk"
    if is_stale and is_intraday:
        suggested_risk_pct = 0.0
        decision = "Wait — intraday market data is stale"
        warnings.append("Stale intraday data requires waiting for a fresh candle.")

    reasons = list(dict.fromkeys(reasons))
    warnings = list(dict.fromkeys(warnings))
    trust_components = _build_trust_components(
        confidence_score=confidence_score,
        data_score=data_score,
        data_integrity=data_integrity,
        backtest_score=backtest_score,
        backtest_summary=backtest_summary,
        monte_carlo_score=monte_carlo_score,
        monte_carlo_summary=monte_carlo_summary,
        zone_score=zone_score,
        market_zones=market_zones,
        prediction=prediction,
        regime_score=regime_score,
        regime_sharpe=regime_sharpe,
        trend_regime=trend_regime,
        volatility_regime=volatility_regime,
        resolved_news_level=resolved_news_level,
        diagnostic_warnings=diagnostic_warnings,
        is_stale=is_stale,
        data_freshness_minutes=float(data_integrity.get("data_freshness_minutes", 0.0)),
        score_breakdown=score_breakdown,
    )
    warning_details = _build_warning_details(
        warnings,
        is_stale=is_stale,
        is_intraday=is_intraday,
        resolved_news_level=resolved_news_level,
    )
    trust_level = _trust_level(trust_score)

    return {
        "trust_score": trust_score,
        "trade_quality_score": trust_score,
        "usefulness": usefulness,
        "decision": decision,
        "suggested_risk_pct": suggested_risk_pct,
        "components": raw_component_scores,
        "raw_component_scores": raw_component_scores,
        "score_breakdown": score_breakdown,
        "weights": weights,
        "trust_level": trust_level,
        "trust_components": trust_components,
        "trust_summary": {
            "answer": trust_level,
            "supporting_reasons": reasons[:4],
            "risk_flags": warnings[:5],
        },
        "reasons": reasons,
        "warnings": warnings,
        "warning_details": warning_details,
        "news_status": resolved_news_level,
    }


def calculate_position_size(
    account_size: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    asset_type: str = "equity",
    contract_size: float = 100_000,
) -> dict[str, Any]:
    """Dispatch to FX pip sizing or generic fixed-fractional sizing."""
    if asset_type not in {"equity", "crypto", "fx"}:
        raise ValueError("asset_type must be equity, crypto, or fx.")
    if asset_type == "fx":
        result = calculate_fx_position_size(
            account_size,
            risk_pct,
            entry_price,
            stop_price,
        )
    else:
        result = calculate_fixed_fractional_position_size(
            account_size,
            risk_pct,
            entry_price,
            stop_price,
        )
        result["asset_type"] = asset_type
        result["display_unit"] = "shares" if asset_type == "equity" else "units"
    return result


def infer_asset_type(symbol: str) -> str:
    """Infer a sizing convention from a Yahoo-style symbol."""
    upper = symbol.upper().replace(" ", "")
    if upper.endswith("=X") or (
        "/" in upper
        and len(upper.split("/")) == 2
        and all(len(currency) == 3 for currency in upper.split("/"))
    ):
        return "fx"
    if upper.endswith("-USD"):
        return "crypto"
    return "equity"


def _regime_sharpe(performance: pd.DataFrame, regime: str) -> float:
    match = performance.loc[performance["regime"] == regime, "sharpe"]
    return float(match.iloc[0]) if not match.empty else 0.0


def _monte_carlo_alignment(prediction: int, summary: dict[str, Any]) -> float:
    if prediction == 0:
        return 50.0
    target = float(summary["probability_take_profit_first"])
    stop = float(summary["probability_stop_loss_first"])
    decisive = target + stop
    return float(np.clip(target / decisive * 100 if decisive else 50, 0, 100))


def _zone_alignment(prediction: int, zones: dict[str, Any]) -> float:
    if prediction == 1:
        return 70.0 if zones["near_support"] else (30.0 if zones["near_resistance"] else 50.0)
    if prediction == -1:
        return 70.0 if zones["near_resistance"] else (30.0 if zones["near_support"] else 50.0)
    return 50.0


def _append_zone_context(
    prediction: int,
    zones: dict[str, Any],
    reasons: list[str],
    warnings: list[str],
) -> None:
    if prediction == 1 and zones["near_support"]:
        reasons.append("Long signal is near detected support.")
    elif prediction == 1 and zones["near_resistance"]:
        warnings.append("Long signal is close to detected resistance.")
    elif prediction == -1 and zones["near_resistance"]:
        reasons.append("Short signal is near detected resistance.")
    elif prediction == -1 and zones["near_support"]:
        warnings.append("Short signal is close to detected support.")


def _build_trust_components(
    *,
    confidence_score: float,
    data_score: float,
    data_integrity: dict[str, Any],
    backtest_score: float,
    backtest_summary: dict[str, Any],
    monte_carlo_score: float,
    monte_carlo_summary: dict[str, Any],
    zone_score: float,
    market_zones: dict[str, Any],
    prediction: int,
    regime_score: float,
    regime_sharpe: float,
    trend_regime: str,
    volatility_regime: str,
    resolved_news_level: str,
    diagnostic_warnings: list[str],
    is_stale: bool,
    data_freshness_minutes: float,
    score_breakdown: dict[str, float],
) -> list[dict[str, Any]]:
    """Return ordered, trader-facing trust component explanations."""
    target_probability = float(monte_carlo_summary["probability_take_profit_first"])
    stop_probability = float(monte_carlo_summary["probability_stop_loss_first"])
    zone_reason = _zone_reason(prediction, market_zones)
    news_score = {
        "low": 100.0,
        "medium": 50.0,
        "high": 0.0,
        "critical": 0.0,
        "unavailable": 40.0,
    }.get(resolved_news_level, 75.0)
    diagnostic_score = max(0.0, 100.0 - len(diagnostic_warnings) * 25.0)
    freshness_score = 0.0 if is_stale else 100.0
    volatility_score = 35.0 if volatility_regime == "high_volatility" else 100.0
    rows = [
        _component(
            "Model confidence",
            confidence_score,
            score_breakdown["model_confidence_component"],
            f"Selected model confidence is {confidence_score:.1f}%.",
        ),
        _component(
            "Data integrity",
            data_score,
            score_breakdown["data_integrity_component"],
            f"Integrity status is {data_integrity.get('status', 'unknown')}.",
        ),
        _component(
            "Backtest support",
            backtest_score,
            score_breakdown["backtest_component"],
            f"Strictly out-of-sample Sharpe is {float(backtest_summary['sharpe_ratio']):.2f}.",
        ),
        _component(
            "Monte Carlo alignment",
            monte_carlo_score,
            score_breakdown["monte_carlo_component"],
            f"Target-first {target_probability:.1%}; stop-first {stop_probability:.1%}.",
        ),
        _component(
            "Zone alignment",
            zone_score,
            score_breakdown["zone_component"],
            zone_reason,
        ),
        _component(
            "Regime support",
            regime_score,
            score_breakdown["regime_component"],
            f"{trend_regime} historical Sharpe is {regime_sharpe:.2f}.",
        ),
        _component(
            "News risk",
            news_score,
            score_breakdown["news_penalty"],
            f"Economic-calendar risk is {resolved_news_level}.",
        ),
        _component(
            "Diagnostics",
            diagnostic_score,
            score_breakdown["diagnostic_penalty"],
            (
                f"{len(diagnostic_warnings)} automated diagnostic warning(s)."
                if diagnostic_warnings
                else "No automated plausibility warnings."
            ),
        ),
        _component(
            "Data freshness",
            freshness_score,
            score_breakdown["freshness_penalty"],
            (
                f"Latest data is stale at {data_freshness_minutes:.0f} minutes old."
                if is_stale
                else f"Latest data age is {data_freshness_minutes:.0f} minutes."
            ),
        ),
        _component(
            "Volatility",
            volatility_score,
            score_breakdown["volatility_component"],
            f"Current volatility regime is {volatility_regime}.",
        ),
    ]
    return rows


def _component(
    component: str,
    score: float,
    impact: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "component": component,
        "score": float(np.clip(score, 0, 100)),
        "impact": float(impact),
        "status": _component_status(score),
        "reason": reason,
    }


def _component_status(score: float) -> str:
    if score >= 70:
        return "pass"
    if score >= 45:
        return "warning"
    return "fail"


def _trust_level(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 55:
        return "Moderate"
    return "Low"


def _zone_reason(prediction: int, zones: dict[str, Any]) -> str:
    if prediction == 1 and zones["near_support"]:
        return "Long signal is aligned with nearby support."
    if prediction == 1 and zones["near_resistance"]:
        return "Long signal is close to nearby resistance."
    if prediction == -1 and zones["near_resistance"]:
        return "Short signal is aligned with nearby resistance."
    if prediction == -1 and zones["near_support"]:
        return "Short signal is close to nearby support."
    return "No strong nearby zone alignment."


def _build_warning_details(
    warnings: list[str],
    *,
    is_stale: bool,
    is_intraday: bool,
    resolved_news_level: str,
) -> list[dict[str, str]]:
    """Attach consistent severity and source labels to warning messages."""
    details: list[dict[str, str]] = []
    for warning in warnings:
        lower = warning.lower()
        if "news" in lower or "event" in lower:
            source = "News risk"
        elif "stale" in lower or "fresh" in lower or "candle" in lower:
            source = "Data freshness"
        elif "volatility" in lower:
            source = "Volatility"
        elif "monte carlo" in lower:
            source = "Monte Carlo"
        elif "resistance" in lower or "support" in lower:
            source = "Market zones"
        elif "confidence" in lower:
            source = "Model confidence"
        elif "performance" in lower or "sharpe" in lower:
            source = "Backtest"
        else:
            source = "Diagnostics"
        severity = "warning"
        if (is_stale and is_intraday and source == "Data freshness") or (
            resolved_news_level in {"high", "critical"} and source == "News risk"
        ):
            severity = "critical"
        details.append({"severity": severity, "source": source, "message": warning})
    return details
