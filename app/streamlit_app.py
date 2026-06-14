"""Interactive dashboard for Financial ML Signal Lab."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard import MODEL_TRAINERS, build_dashboard_research
from src.data_loader import download_recent_data
from src.live_data import assess_market_data_freshness
from src.paper_trading import (
    close_paper_trade,
    create_paper_trade,
    load_paper_trades,
    summarize_paper_trading,
    update_open_paper_trades,
)
from src.pipeline import get_configured_assets, load_config, prepare_configured_dataset

SIGNAL_LABELS = {-1: "Short", 0: "Neutral", 1: "Long"}
CONFIG = load_config(PROJECT_ROOT / "config.yaml")
ASSETS = get_configured_assets(CONFIG)
ASSET_LABELS = {asset["symbol"]: asset["label"] for asset in ASSETS}
ASSET_EXCLUDED_FEATURES = {
    asset["symbol"]: tuple(asset.get("excluded_features", [])) for asset in ASSETS
}
ABSTENTION_THRESHOLDS = tuple(
    CONFIG.get("signal_policy", {}).get(
        "abstention_thresholds", [0.0, 0.45, 0.50, 0.55, 0.60]
    )
)
INTELLIGENCE_CONFIG = CONFIG.get("market_intelligence", {})
LIVE_CONFIG = INTELLIGENCE_CONFIG.get("live_market", {})
LIVE_AUTO_REFRESH_SECONDS = max(
    15, int(LIVE_CONFIG.get("auto_refresh_seconds", 60))
)
LIVE_CACHE_TTL_SECONDS = max(5, LIVE_AUTO_REFRESH_SECONDS - 5)
PAPER_TRADES_PATH = str(PROJECT_ROOT / "data/paper_trades/paper_trades.csv")
RESEARCH_CACHE_VERSION = "cloud-deploy-v1"


@st.cache_resource(show_spinner=False)
def load_research(
    symbol: str,
    selected_model: str,
    transaction_cost: float,
    train_window: int,
    test_window: int,
    excluded_features: tuple[str, ...],
    abstention_thresholds: tuple[float, ...],
    intelligence_config: dict,
    cache_version: str,
) -> dict:
    """Build and cache the complete dashboard research bundle."""
    del cache_version
    return build_dashboard_research(
        symbol=symbol,
        selected_model=selected_model,
        transaction_cost=transaction_cost,
        train_window=train_window,
        test_window=test_window,
        step_size=test_window,
        excluded_features=excluded_features,
        abstention_thresholds=abstention_thresholds,
        intelligence_config=intelligence_config,
    )


@st.cache_data(ttl=LIVE_CACHE_TTL_SECONDS, show_spinner=False)
def load_live_chart_data(
    symbol: str,
    period: str,
    interval: str,
    include_prepost: bool,
) -> pd.DataFrame:
    """Download and cache recent candles for the dedicated live chart."""
    return download_recent_data(
        symbol,
        period=period,
        interval=interval,
        include_prepost=include_prepost,
    )


def format_percent(value: float) -> str:
    """Format a decimal value as a percentage."""
    return f"{value:.2%}"


def format_confidence(value: float | None) -> str:
    """Format model confidence safely when probability output is unavailable."""
    return f"{value:.1%}" if value is not None else "Unavailable"


def format_freshness(minutes: float) -> str:
    """Format data age in a compact trader-facing form."""
    if minutes < 120:
        return f"{minutes:.0f} min"
    if minutes < 2880:
        return f"{minutes / 60:.1f} hours"
    return f"{minutes / 1440:.1f} days"


def signal_card_class(prediction: int) -> str:
    """Return the visual class for a directional signal."""
    return {1: "signal-long", -1: "signal-short"}.get(prediction, "signal-neutral")


def tab_order_for_mode(app_mode: str) -> list[str]:
    """Return the focused dashboard workspace for the selected mode."""
    workspaces = {
        "Research Mode": [
            "Overview",
            "Backtest",
            "Model Comparison",
            "Signal Policy",
            "Regimes",
            "Explainability",
            "Data Integrity",
        ],
        "Live Mode": [
            "Live Market",
            "Overview",
            "Risk Simulator",
            "Data Integrity",
            "Explainability",
        ],
        "Paper Trading Mode": [
            "Paper Trading",
            "Live Market",
            "Overview",
            "Risk Simulator",
            "Data Integrity",
        ],
    }
    return workspaces.get(app_mode, workspaces["Research Mode"])


def render_mode_badge(app_mode: str) -> None:
    """Render the active dashboard workflow clearly."""
    descriptions = {
        "Research Mode": "Historical validation and strictly out-of-sample evidence",
        "Live Mode": "Current completed candles, freshness, zones, and risk",
        "Paper Trading Mode": "Hypothetical trade logging and outcome review",
    }
    st.markdown(
        f"""
        <div class="mode-banner">
            <span class="mode-badge">{app_mode.upper()}</span>
            <span>{descriptions[app_mode]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trust_reliability(intelligence: dict) -> None:
    """Render the trader-facing trust answer and component evidence."""
    trust_level = intelligence["trust_level"]
    trust_class = {
        "High": "trust-high",
        "Moderate": "trust-moderate",
        "Low": "trust-low",
    }[trust_level]
    st.markdown("### Can I trust this signal?")
    st.markdown(
        f"""
        <div class="trust-card {trust_class}">
            <div class="trust-answer">Answer: {trust_level}</div>
            <div>{intelligence['decision']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(round(intelligence["trust_score"])), text="Signal trust score")
    components = pd.DataFrame(intelligence["trust_components"]).rename(
        columns={
            "component": "Component",
            "score": "Score",
            "impact": "Trust Impact",
            "status": "Status",
            "reason": "Reason",
        }
    )
    status_icons = {"pass": "✅ Pass", "warning": "⚠️ Warning", "fail": "❌ Fail"}
    components["Status"] = components["Status"].map(status_icons)
    components["Score"] = components["Score"].map(lambda value: f"{value:.0f}/100")
    components["Trust Impact"] = components["Trust Impact"].map(
        lambda value: f"{value:+.1f}"
    )
    st.dataframe(components, hide_index=True, width="stretch")

    supporting, risks = st.columns(2)
    with supporting:
        st.markdown("**Why it may be useful**")
        reasons = intelligence["trust_summary"]["supporting_reasons"]
        st.markdown(
            "\n".join(f"- {reason}" for reason in reasons)
            if reasons
            else "- No strong supporting evidence detected."
        )
    with risks:
        st.markdown("**Structured risk flags**")
        warning_details = intelligence["warning_details"]
        if warning_details:
            flags = pd.DataFrame(warning_details).rename(
                columns={
                    "severity": "Severity",
                    "source": "Source",
                    "message": "Warning",
                }
            )
            st.dataframe(flags, hide_index=True, width="stretch")
        else:
            st.success("No automated risk flags.")


def add_page_style() -> None:
    """Apply portfolio-oriented dashboard styling."""
    st.markdown(
        """
        <style>
        .stApp { background: #f8fafc; color: #0f172a; }
        [data-testid="stSidebar"] { background: #0f172a; }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #e5e7eb !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border-color: #475569 !important;
        }
        [data-testid="stSidebar"] input {
            color: #f8fafc !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="menu"] {
            background-color: #ffffff !important;
            color: #111827 !important;
        }
        [data-baseweb="menu"] li,
        [role="option"] {
            color: #111827 !important;
        }
        [data-testid="stSidebar"] [data-testid="stCheckbox"] label span,
        [data-testid="stSidebar"] [data-testid="stSlider"] label,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
            color: #e5e7eb !important;
        }
        [data-testid="stSidebar"] .stButton button {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border-color: #475569 !important;
        }
        [data-testid="stSidebar"] .stButton button p {
            color: #f8fafc !important;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }
        .signal-card {
            padding: 22px 24px;
            border-radius: 18px;
            color: white;
            margin-bottom: 18px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
        }
        .signal-long { background: linear-gradient(135deg, #14532d, #16a34a); }
        .signal-short { background: linear-gradient(135deg, #7f1d1d, #dc2626); }
        .signal-neutral { background: linear-gradient(135deg, #1e293b, #2563eb); }
        .signal-meta {
            font-size: 13px;
            opacity: 0.86;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .signal-value {
            font-size: 36px;
            font-weight: 750;
            margin: 5px 0;
        }
        .research-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-left: 5px solid #2563eb;
            border-radius: 14px;
            padding: 16px 18px;
            margin: 8px 0 18px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .decision-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 18px;
            margin: 10px 0 18px;
        }
        .research-note {
            background: #eef4fa;
            border-left: 4px solid #2563eb;
            padding: 12px 16px;
            border-radius: 8px;
        }
        .mode-banner {
            display: flex;
            align-items: center;
            gap: 12px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 10px 14px;
            margin: 4px 0 18px;
            color: #475569;
        }
        .mode-badge {
            background: #0f172a;
            color: #f8fafc;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.06em;
        }
        .trust-card {
            border-radius: 14px;
            padding: 16px 18px;
            margin: 8px 0 12px;
            border: 1px solid;
        }
        .trust-answer {
            font-size: 22px;
            font-weight: 750;
            margin-bottom: 4px;
        }
        .trust-high { background: #f0fdf4; border-color: #86efac; color: #14532d; }
        .trust-moderate { background: #fffbeb; border-color: #fcd34d; color: #78350f; }
        .trust-low { background: #fef2f2; border-color: #fca5a5; color: #7f1d1d; }
        .signal-date-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0 14px;
        }
        .signal-date-badge {
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            color: #3730a3;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 12px;
            font-weight: 650;
        }
        .explanation-card {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            padding: 14px 16px;
            margin: -8px 0 18px;
            color: #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_overview(research: dict) -> None:
    """Render the dashboard overview tab."""
    dataset = research["dataset"]
    current = research["current_signal"]
    research_signal = research["research_signal"]
    latest = current["feature_row"].iloc[0]
    summary = research["backtest_summary"]
    signal = SIGNAL_LABELS.get(current["prediction"], str(current["prediction"]))
    historical_signal = SIGNAL_LABELS.get(
        research_signal["prediction"], str(research_signal["prediction"])
    )
    current_freshness = assess_market_data_freshness(
        current["feature_row"],
        interval="1d",
        asset_type=research["position_size"]["asset_type"],
    )
    target_horizon = int(
        research.get("target_horizon", CONFIG.get("target", {}).get("horizon", 5))
    )
    freshness_message = current.get("freshness_message", current_freshness["message"])
    current_is_stale = bool(current.get("is_stale", current_freshness["is_stale"]))

    st.markdown(
        f"""
        <div class="signal-card {signal_card_class(current['prediction'])}">
            <div class="signal-meta">Current Latest Signal · Outcome Not Known Yet · Latest Completed Candle</div>
            <div class="signal-value">{signal}</div>
            <div>{research['selected_model']} · {format_confidence(current['confidence'])} confidence ·
            {pd.to_datetime(current['latest_date']).strftime('%Y-%m-%d')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    intelligence = research["trade_intelligence"]
    columns[0].metric("Current Close", f"${current['latest_close']:,.2f}")
    columns[1].metric("Signal Trust", f"{intelligence['trust_score']:.0f}/100")
    columns[2].metric("Trade Quality", f"{intelligence['trade_quality_score']:.0f}/100")
    columns[3].metric(
        "Daily Data Freshness",
        format_freshness(current["data_freshness_minutes"]),
    )
    if current_is_stale:
        st.warning(freshness_message)
    else:
        st.info(freshness_message)
    for warning in current["warnings"]:
        if warning != freshness_message:
            st.warning(
                warning.replace(
                    "historically evaluable research row",
                    "backtest-evaluable row because its future outcome is not known yet",
                )
            )

    st.markdown(
        f"""
        <div class="signal-date-badges">
            <span class="signal-date-badge">Latest candle: {pd.to_datetime(current['latest_date']).strftime('%Y-%m-%d')}</span>
            <span class="signal-date-badge">Backtest-evaluable candle: {pd.to_datetime(research_signal['date']).strftime('%Y-%m-%d')}</span>
            <span class="signal-date-badge">Target horizon: {target_horizon} candles</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="research-card">
            <b>Latest Backtest-Evaluable Signal Date:</b>
            {pd.to_datetime(research_signal['date']).strftime('%Y-%m-%d')}<br>
            <b>Strictly out-of-sample signal:</b> {historical_signal} ·
            {format_confidence(research_signal['confidence'])} confidence<br>
            <span style="color:#64748b;">
            This signal has a known future outcome and can therefore be evaluated
            inside the historical walk-forward backtest.
            </span>
        </div>
        <div class="explanation-card">
            <b>Why is this older than the current signal?</b><br>
            The current signal is generated from the latest completed candle, but its
            future outcome is not known yet. The backtest-evaluable signal must be at
            least {target_horizon} candles behind the latest data because
            the model predicts <code>future_return</code> over the next horizon. This
            prevents lookahead bias.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_trust_reliability(intelligence)

    monte_carlo = research["monte_carlo"]["summary"]
    decision_columns = st.columns(4)
    decision_columns[0].metric("Walk-Forward Return", format_percent(summary["total_return"]))
    decision_columns[1].metric("Max Drawdown", format_percent(summary["max_drawdown"]))
    decision_columns[2].metric(
        f"{monte_carlo['target_label']} First",
        format_percent(monte_carlo["probability_take_profit_first"]),
    )
    decision_columns[3].metric(
        "Suggested Risk", format_percent(intelligence["suggested_risk_pct"])
    )
    st.markdown(
        f'<div class="decision-card"><b>Decision</b><br>{intelligence["decision"]}</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Current market signal: the selected model is trained only on labeled "
        "historical data, then applied to the latest completed feature row without "
        "using a future return. Walk-forward research remains strictly out-of-sample."
    )

    with st.expander("Advanced research details"):
        breakdown = pd.DataFrame(
            {
                "Component": list(intelligence["score_breakdown"]),
                "Score Contribution": list(intelligence["score_breakdown"].values()),
            }
        )
        breakdown_chart = px.bar(
            breakdown,
            x="Score Contribution",
            y="Component",
            orientation="h",
            title="Trust Score Breakdown",
            template="plotly_white",
        )
        breakdown_chart.update_traces(marker_color="#2563eb")
        st.plotly_chart(breakdown_chart, width="stretch")
        diagnostics = research["diagnostics"]
        if diagnostics["warnings"]:
            st.warning(
                "Automated plausibility review required:\n\n- "
                + "\n- ".join(diagnostics["warnings"])
            )

    price = px.line(
        dataset,
        x="date",
        y="close",
        title=f"{research['symbol']} Adjusted Close",
        template="plotly_white",
    )
    price.update_traces(line_color="#315a84", line_width=2)
    nearest_support = research["market_zones"]["nearest_support"]
    nearest_resistance = research["market_zones"]["nearest_resistance"]
    if nearest_support:
        price.add_hline(
            y=nearest_support["level"],
            line_dash="dash",
            line_color="#2f855a",
            annotation_text="Nearest support",
        )
    if nearest_resistance:
        price.add_hline(
            y=nearest_resistance["level"],
            line_dash="dash",
            line_color="#b43c4c",
            annotation_text="Nearest resistance",
        )
    st.plotly_chart(price, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Current Market Context")
        context = pd.DataFrame(
            {
                "Metric": ["Volatility Regime", "Trend Regime", "Signal", "Confidence"],
                "Value": [
                    latest["volatility_regime"],
                    latest["trend_regime"],
                    signal,
                    format_confidence(current["confidence"]),
                ],
            }
        )
        st.dataframe(context, hide_index=True, width="stretch")
    with right:
        st.subheader("Research Scope")
        excluded_features = research["excluded_features"]
        policy_note = (
            f"<br><b>{len(excluded_features)}</b> features excluded by asset policy"
            if excluded_features
            else ""
        )
        st.markdown(
            f"""
            <div class="research-note">
            <b>{len(dataset):,}</b> model-ready rows<br>
            <b>{len(research['feature_cols'])}</b> leakage-safe features<br>
            <b>{research['walk_forward']['fold'].nunique()}</b> walk-forward folds<br>
            <b>{len(research['walk_forward']):,}</b> strictly out-of-sample predictions
            {policy_note}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_live_market(research: dict) -> None:
    """Render the dedicated live/recent candlestick workspace."""
    st.subheader("Live Market")
    st.caption(
        "Recent Yahoo Finance candles for research and demonstration only. "
        "This is not broker-grade data or live execution."
    )
    controls = st.columns([1, 1, 1, 1])
    interval = controls[0].selectbox(
        "Interval",
        ["1m", "5m", "15m", "1h", "1d"],
        index=1,
        key="live_chart_interval",
    )
    period = controls[1].selectbox(
        "Period",
        ["1d", "5d", "1mo", "3mo"],
        index=1,
        key="live_chart_period",
    )
    auto_refresh = controls[2].toggle(
        "Auto-refresh every 60s",
        value=False,
        key="live_chart_auto_refresh",
    )
    if controls[3].button("Refresh Live Chart", width="stretch"):
        load_live_chart_data.clear()
        st.rerun()
    if interval == "1m" and period not in {"1d", "5d"}:
        st.warning("Yahoo limits one-minute history; requesting the latest five days.")
        period = "5d"
    if auto_refresh:
        render_auto_live_chart(research, interval, period)
    else:
        render_live_chart_content(research, interval, period)


@st.fragment(run_every=60)
def render_auto_live_chart(research: dict, interval: str, period: str) -> None:
    """Auto-refresh the live chart when explicitly enabled."""
    render_live_chart_content(research, interval, period)


def render_live_chart_content(research: dict, interval: str, period: str) -> None:
    """Download and render recent candles with decision-useful overlays."""
    try:
        candles = load_live_chart_data(
            research["symbol"],
            period,
            interval,
            bool(LIVE_CONFIG.get("include_prepost", True)),
        )
    except Exception as exc:
        st.error(
            "No live data is available for this interval. Yahoo Finance may be "
            f"delayed or rejecting the request. Details: {exc}"
        )
        return
    if candles.empty:
        st.warning("No live data available for this interval.")
        return

    freshness = assess_market_data_freshness(
        candles,
        interval,
        research["position_size"]["asset_type"],
    )
    chart_data = candles.copy()
    chart_data["ema_20"] = chart_data["close"].ewm(span=20, adjust=False).mean()
    chart_data["ema_50"] = chart_data["close"].ewm(span=50, adjust=False).mean()
    previous_close = chart_data["close"].shift(1)
    true_range = pd.concat(
        [
            chart_data["high"] - chart_data["low"],
            (chart_data["high"] - previous_close).abs(),
            (chart_data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.rolling(14, min_periods=1).mean().iloc[-1])
    latest = chart_data.iloc[-1]
    session = chart_data.loc[
        chart_data["date"].dt.normalize() == pd.Timestamp(latest["date"]).normalize()
    ]
    intraday_return = float(latest["close"] / session.iloc[0]["open"] - 1)
    primary_metrics = st.columns(3)
    primary_metrics[0].metric("Latest Price", f"${latest['close']:,.4f}")
    primary_metrics[1].metric(
        "Latest Candle", pd.Timestamp(latest["date"]).strftime("%b %d · %H:%M")
    )
    primary_metrics[2].metric(
        "Data Freshness", format_freshness(freshness["data_age_minutes"])
    )
    context_metrics = st.columns(3)
    context_metrics[0].metric("Market Status", freshness["market_status"].upper())
    context_metrics[1].metric("Intraday Return", format_percent(intraday_return))
    context_metrics[2].metric("ATR", f"{atr:,.4f}")
    if freshness["is_stale"]:
        st.warning(freshness["message"])
    else:
        st.success(freshness["message"])

    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=chart_data["date"],
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="OHLC",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["ema_20"],
            name="EMA 20",
            line={"color": "#2563eb", "width": 1.5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["ema_50"],
            name="EMA 50",
            line={"color": "#d97706", "width": 1.5},
        )
    )
    figure.add_hline(
        y=float(latest["close"]),
        line_dash="dot",
        line_color="#0f172a",
        annotation_text="Current price",
    )
    for zone_key, label, color in (
        ("nearest_support", "Nearest support", "#16a34a"),
        ("nearest_resistance", "Nearest resistance", "#dc2626"),
    ):
        zone = research["market_zones"].get(zone_key)
        if zone:
            figure.add_hline(
                y=float(zone["level"]),
                line_dash="dash",
                line_color=color,
                annotation_text=label,
            )
    current = research["current_signal"]
    figure.add_annotation(
        x=chart_data["date"].iloc[-1],
        y=float(latest["close"]),
        text=f"Current signal: {SIGNAL_LABELS[current['prediction']]}",
        showarrow=True,
        arrowhead=2,
        bgcolor="#ffffff",
    )
    figure.update_layout(
        title=f"{research['symbol']} · {interval} Candlestick Chart",
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        height=650,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    st.plotly_chart(figure, width="stretch")
    status = pd.DataFrame(
        {
            "Field": [
                "Data Provider",
                "Latest Candle",
                "Data Delay",
                "Market Status",
                "Freshness",
                "Current Signal",
                "Trust Score",
            ],
            "Value": [
                freshness["data_provider"],
                pd.Timestamp(freshness["latest_candle"]).strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
                format_freshness(freshness["data_age_minutes"]),
                freshness["market_status"],
                freshness["freshness_status"],
                SIGNAL_LABELS[current["prediction"]],
                f"{research['trade_intelligence']['trust_score']:.0f}/100",
            ],
        }
    )
    with st.expander("Advanced live data-source details"):
        st.dataframe(status, hide_index=True, width="stretch")


def render_paper_trading(research: dict) -> None:
    """Render persistent hypothetical trade logging and outcome tracking."""
    st.subheader("Paper Trading")
    st.warning(
        "Paper trading is hypothetical. No real broker connection or order execution exists."
    )
    current = research["current_signal"]
    intelligence = research["trade_intelligence"]
    monte_carlo = research["monte_carlo"]["summary"]
    sizing = research["position_size"]
    signal = int(current["prediction"])
    setup = pd.DataFrame(
        {
            "Metric": [
                "Symbol",
                "Signal",
                "Entry price",
                "Suggested stop",
                "Suggested target",
                "Trust score",
                "Risk",
                "Position size",
            ],
            "Value": [
                research["symbol"],
                SIGNAL_LABELS[signal],
                f"{current['latest_close']:,.5f}",
                f"{monte_carlo['stop_loss_level']:,.5f}",
                f"{monte_carlo['take_profit_level']:,.5f}",
                f"{intelligence['trust_score']:.0f}/100",
                format_percent(intelligence["suggested_risk_pct"]),
                f"{sizing['units']:,.4f} units",
            ],
        }
    )
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Current Hypothetical Setup")
        st.dataframe(setup, hide_index=True, width="stretch")
    with right:
        notes = st.text_area(
            "Paper trade notes",
            placeholder="Optional setup rationale or review note",
        )
        if st.button(
            "Log Paper Trade",
            type="primary",
            disabled=signal == 0,
            width="stretch",
        ):
            create_paper_trade(
                symbol=research["symbol"],
                signal=signal,
                confidence=float(current["confidence"] or 0),
                trust_score=float(intelligence["trust_score"]),
                trade_quality_score=float(intelligence["trade_quality_score"]),
                decision=str(intelligence["decision"]),
                entry_price=float(current["latest_close"]),
                stop_price=float(monte_carlo["stop_loss_level"]),
                target_price=float(monte_carlo["take_profit_level"]),
                risk_pct=float(intelligence["suggested_risk_pct"]),
                account_size=float(sizing["account_size"]),
                position_size=float(sizing["units"]),
                model_name=research["selected_model"],
                data_timestamp=current["latest_date"],
                notes=notes,
                storage_path=PAPER_TRADES_PATH,
                asset_type=str(sizing["asset_type"]),
                timeframe="1d",
            )
            st.success("Hypothetical paper trade logged locally.")
        if signal == 0:
            st.caption("Neutral signals cannot be logged as paper trades.")

    if st.button("Refresh Paper Trade Outcomes"):
        try:
            latest_data = download_recent_data(
                research["symbol"],
                period="1y",
                interval="1d",
                include_prepost=False,
            )
            latest_data.attrs["symbol"] = research["symbol"]
            update_open_paper_trades(latest_data, PAPER_TRADES_PATH)
            st.success("Open paper trades updated from available daily candles.")
        except Exception as exc:
            st.error(f"Paper trade outcome refresh failed: {exc}")

    trades = load_paper_trades(PAPER_TRADES_PATH)
    summary = summarize_paper_trading(trades)
    metrics = st.columns(4)
    metrics[0].metric("Total Paper Trades", summary["total_trades"])
    metrics[1].metric("Open Trades", summary["open_trades"])
    metrics[2].metric("Closed Trades", summary["closed_trades"])
    metrics[3].metric("Win Rate", format_percent(summary["win_rate"]))
    pnl_metrics = st.columns(4)
    pnl_metrics[0].metric("Total Paper PnL", f"${summary['total_pnl']:,.2f}")
    pnl_metrics[1].metric("Average PnL", f"${summary['average_pnl']:,.2f}")
    pnl_metrics[2].metric("Best Trade", f"${summary['best_trade']:,.2f}")
    pnl_metrics[3].metric("Worst Trade", f"${summary['worst_trade']:,.2f}")

    open_trades = trades.loc[trades["status"] == "open"]
    closed_trades = trades.loc[trades["status"] == "closed"]
    st.markdown("#### Open Paper Trades")
    if open_trades.empty:
        st.caption("No open paper trades.")
    else:
        st.dataframe(open_trades, hide_index=True, width="stretch")
        with st.expander("Manual paper trade close"):
            trade_id = st.selectbox(
                "Open trade",
                open_trades["trade_id"].tolist(),
                format_func=lambda value: f"{open_trades.loc[open_trades['trade_id'] == value, 'symbol'].iloc[0]} · {value[:8]}",
            )
            exit_price = st.number_input(
                "Manual exit price",
                min_value=0.00001,
                value=float(current["latest_close"]),
                format="%.5f",
            )
            if st.button("Close Selected Paper Trade"):
                close_paper_trade(
                    trade_id,
                    exit_price,
                    "manual_close",
                    PAPER_TRADES_PATH,
                )
                st.success("Paper trade closed manually.")
    st.markdown("#### Closed Paper Trades")
    if closed_trades.empty:
        st.caption("No closed paper trades.")
    else:
        st.dataframe(closed_trades, hide_index=True, width="stretch")


def render_model_comparison(research: dict) -> None:
    """Render chronological holdout model comparison."""
    st.subheader("Chronological Holdout Classification")
    comparison = research["comparison"].copy()
    display_columns = [
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "precision_macro",
        "recall_macro",
    ]
    chart_data = comparison.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="score",
    )
    chart = px.bar(
        chart_data,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        template="plotly_white",
        title="Holdout Model Metrics",
    )
    st.plotly_chart(chart, width="stretch")
    with st.expander("Advanced holdout metric table"):
        st.dataframe(
            comparison[display_columns].style.format(
                {column: "{:.3f}" for column in display_columns[1:]}
            ),
            hide_index=True,
            width="stretch",
        )
    st.info(
        "Raw accuracy can be misleading when one target class dominates. "
        "Macro F1 weights long, neutral, and short classes equally."
    )


def render_backtest(research: dict) -> None:
    """Render walk-forward backtest and risk metrics."""
    backtest = research["backtest"].copy()
    summary = research["backtest_summary"]
    pre_cost = 10_000 * (1 + backtest["strategy_return_before_cost"]).cumprod()

    metrics = st.columns(5)
    metrics[0].metric("After-Cost Return", format_percent(summary["total_return"]))
    metrics[1].metric("Buy & Hold Return", format_percent(summary["buy_hold_total_return"]))
    metrics[2].metric("Sharpe", f"{summary['sharpe_ratio']:.2f}")
    metrics[3].metric("Trades", f"{summary['number_of_trades']:,}")
    metrics[4].metric("Exposure", format_percent(summary["exposure"]))
    st.markdown("#### Execution Policy Comparison")
    policy_comparison = research["execution_policy_performance"].copy()
    st.dataframe(
        policy_comparison.style.format(
            {
                "total_return": "{:.2%}",
                "sharpe_ratio": "{:.2f}",
                "max_drawdown": "{:.2%}",
                "exposure": "{:.1%}",
                "transaction_cost_paid": "{:.4f}",
                "win_rate": "{:.1%}",
                "profit_factor": "{:.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Daily rebalance is the default. Hold-for-horizon and signal-change-only "
        "evaluate turnover and target-horizon mismatch without lookahead."
    )

    equity_chart = go.Figure()
    equity_chart.add_trace(
        go.Scatter(x=backtest["date"], y=backtest["equity"], name="Strategy after costs")
    )
    equity_chart.add_trace(
        go.Scatter(x=backtest["date"], y=pre_cost, name="Strategy before costs")
    )
    equity_chart.add_trace(
        go.Scatter(x=backtest["date"], y=backtest["buy_hold_equity"], name="Buy and hold")
    )
    equity_chart.update_layout(
        title="Strictly Out-of-Sample Walk-Forward Equity",
        yaxis_title="Equity ($)",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(equity_chart, width="stretch")

    left, right = st.columns(2)
    with left:
        drawdown = px.area(
            backtest,
            x="date",
            y="drawdown",
            title="Strategy Drawdown",
            template="plotly_white",
        )
        drawdown.update_traces(line_color="#b43c4c", fillcolor="rgba(180,60,76,0.35)")
        st.plotly_chart(drawdown, width="stretch")
    with right:
        distribution = px.histogram(
            backtest,
            x="strategy_return",
            nbins=60,
            title="Daily Strategy Return Distribution",
            template="plotly_white",
        )
        distribution.update_traces(marker_color="#315a84")
        st.plotly_chart(distribution, width="stretch")


def render_regimes(research: dict) -> None:
    """Render conditional strategy performance by market regime."""
    st.subheader("Conditional Walk-Forward Performance")
    volatility = research["volatility_performance"]
    trend = research["trend_performance"]
    left, right = st.columns(2)
    with left:
        st.markdown("#### Volatility Regimes")
        vol_chart = px.bar(
            volatility,
            x="regime",
            y="sharpe",
            color="sharpe",
            color_continuous_scale="RdYlGn",
            title="Sharpe by Volatility Regime",
            template="plotly_white",
        )
        st.plotly_chart(vol_chart, width="stretch")
    with right:
        st.markdown("#### Trend Regimes")
        trend_chart = px.bar(
            trend,
            x="regime",
            y="sharpe",
            color="sharpe",
            color_continuous_scale="RdYlGn",
            title="Sharpe by Trend Regime",
            template="plotly_white",
        )
        st.plotly_chart(trend_chart, width="stretch")
    with st.expander("Advanced regime metric tables"):
        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.dataframe(
                volatility.style.format(
                    {
                        "average_strategy_return": "{:.4%}",
                        "total_strategy_return": "{:.2%}",
                        "sharpe": "{:.2f}",
                        "max_drawdown": "{:.2%}",
                        "win_rate": "{:.2%}",
                        "accuracy": "{:.2%}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        with detail_right:
            st.dataframe(
                trend.style.format(
                    {
                        "average_strategy_return": "{:.4%}",
                        "total_strategy_return": "{:.2%}",
                        "sharpe": "{:.2f}",
                        "max_drawdown": "{:.2%}",
                        "win_rate": "{:.2%}",
                        "accuracy": "{:.2%}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
    st.warning(
        "Regime results are diagnostic. Selecting only successful regimes after "
        "observing outcomes would introduce research bias."
    )


def render_signal_policy(research: dict) -> None:
    """Render predefined confidence-threshold abstention scenarios."""
    scenarios = research["abstention_performance"].copy()
    st.subheader("Confidence-Aware Abstention Scenarios")
    st.info(
        "These thresholds are predefined research scenarios. The dashboard does "
        "not select or promote the best result from this out-of-sample period."
    )
    display = scenarios[
        [
            "minimum_confidence",
            "confidence_coverage",
            "active_signal_rate",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "number_of_trades",
            "exposure",
        ]
    ]
    left, right = st.columns(2)
    with left:
        return_chart = px.line(
            scenarios,
            x="minimum_confidence",
            y="total_return",
            markers=True,
            title="After-Cost Return by Minimum Confidence",
            template="plotly_white",
        )
        return_chart.update_yaxes(tickformat=".0%")
        return_chart.update_xaxes(tickformat=".0%")
        st.plotly_chart(return_chart, width="stretch")
    with right:
        coverage_chart = px.line(
            scenarios,
            x="minimum_confidence",
            y=["confidence_coverage", "active_signal_rate", "exposure"],
            markers=True,
            title="Coverage and Exposure Trade-Off",
            template="plotly_white",
        )
        coverage_chart.update_yaxes(tickformat=".0%")
        coverage_chart.update_xaxes(tickformat=".0%")
        st.plotly_chart(coverage_chart, width="stretch")
    with st.expander("Advanced abstention scenario table"):
        st.dataframe(
            display.style.format(
                {
                    "minimum_confidence": "{:.0%}",
                    "confidence_coverage": "{:.1%}",
                    "active_signal_rate": "{:.1%}",
                    "total_return": "{:.2%}",
                    "sharpe_ratio": "{:.2f}",
                    "max_drawdown": "{:.2%}",
                    "exposure": "{:.1%}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    st.warning(
        "Classifier confidence is not calibrated probability. A threshold that "
        "looks favorable here requires a new untouched validation period."
    )


def render_risk_simulator(research: dict) -> None:
    """Render Monte Carlo paths, tail risk, and position sizing."""
    monte_carlo = research["monte_carlo"]
    summary = monte_carlo["summary"]
    position_size = research["position_size"]
    target_label = summary["target_label"]
    stop_label = summary["stop_label"]
    st.subheader("Monte Carlo Risk Simulator")
    st.caption(
        "Monte Carlo is a scenario simulator, not a price forecast. It assumes "
        "the recent return distribution remains relevant."
    )
    metrics = st.columns(6)
    metrics[0].metric(
        f"{target_label} First",
        format_percent(summary["probability_take_profit_first"]),
    )
    metrics[1].metric(
        f"{stop_label} First",
        format_percent(summary["probability_stop_loss_first"]),
    )
    metrics[2].metric("Expected Return", format_percent(summary["expected_return"]))
    metrics[3].metric("95% VaR", format_percent(summary["value_at_risk"]))
    metrics[4].metric(
        "95% Expected Shortfall", format_percent(summary["expected_shortfall"])
    )
    metrics[5].metric(
        "Worst 5% Adverse Move",
        format_percent(summary["worst_5pct_adverse_move"]),
    )

    paths = monte_carlo["displayed_paths"].reset_index().melt(
        id_vars="step", var_name="simulation", value_name="price"
    )
    path_chart = px.line(
        paths,
        x="step",
        y="price",
        color="simulation",
        title=f"{summary['simulations']:,} Simulations · Displaying {paths['simulation'].nunique()} Paths",
        template="plotly_white",
    )
    path_chart.update_traces(line_width=1, opacity=0.2)
    path_chart.update_layout(showlegend=False)
    path_chart.add_hline(
        y=summary["take_profit_level"],
        line_dash="dash",
        line_color="#2f855a",
        annotation_text=target_label,
    )
    path_chart.add_hline(
        y=summary["stop_loss_level"],
        line_dash="dash",
        line_color="#b43c4c",
        annotation_text=stop_label,
    )
    st.plotly_chart(path_chart, width="stretch")

    left, right = st.columns(2)
    with left:
        distribution = px.histogram(
            monte_carlo["terminal_prices"],
            nbins=60,
            title="Terminal Price Distribution",
            template="plotly_white",
        )
        distribution.update_traces(marker_color="#315a84")
        st.plotly_chart(distribution, width="stretch")
    with right:
        st.markdown("#### Position Sizing")
        sizing = pd.DataFrame(
            {
                "Metric": [
                    "Account size",
                    "Suggested risk",
                    "Risk amount",
                    "Entry",
                    "Stop",
                    "Stop distance",
                    "Stop pips",
                    "Units",
                    "Standard FX lots",
                    "Estimated loss at stop",
                ],
                "Value": [
                    f"${position_size['account_size']:,.2f}",
                    format_percent(position_size["risk_pct"]),
                    f"${position_size['risk_amount']:,.2f}",
                    f"{position_size['entry_price']:,.5f}",
                    f"{position_size['stop_price']:,.5f}",
                    f"{position_size['stop_distance']:,.5f}",
                    (
                        f"{position_size['stop_pips']:,.1f}"
                        if "stop_pips" in position_size
                        else "N/A"
                    ),
                    f"{position_size['units']:,.4f}",
                    f"{position_size['lots']:,.4f}",
                    f"${position_size['estimated_loss_at_stop']:,.2f}",
                ],
            }
        )
        st.dataframe(sizing, hide_index=True, width="stretch")
        st.warning(
            "Confirm broker contract specifications, pip value, spread, and leverage."
        )
        if position_size["warnings"]:
            st.warning("\n".join(f"- {warning}" for warning in position_size["warnings"]))

    news_risk = research["news_risk"]
    st.markdown("#### Economic Calendar Risk")
    news_columns = st.columns(2)
    news_columns[0].metric("News Risk", news_risk["risk_level"].upper())
    news_columns[1].metric("News Risk Score", f"{news_risk['news_risk_score']}/100")
    if news_risk["warnings"]:
        st.warning("\n".join(f"- {warning}" for warning in news_risk["warnings"]))
    if news_risk["upcoming_events"].empty:
        st.caption("No relevant manual-calendar events are currently listed.")
    else:
        with st.expander("Advanced upcoming event details"):
            st.dataframe(news_risk["upcoming_events"], hide_index=True, width="stretch")


def render_data_integrity(research: dict) -> None:
    """Render data quality audit and detected market zones."""
    integrity = research["data_integrity"]
    st.subheader("Data Integrity Engine")
    metrics = st.columns(5)
    metrics[0].metric("Integrity Score", f"{integrity['score']:.0f}/100")
    metrics[1].metric("Status", integrity["status"].upper())
    metrics[2].metric("Missing Candles", integrity["missing_candle_count"])
    metrics[3].metric("Invalid OHLC", integrity["invalid_ohlc_count"])
    metrics[4].metric(
        "Freshness",
        (
            f"{integrity['data_freshness_minutes']:.0f} min"
            if integrity["is_intraday"]
            else f"{integrity['stale_days']:.1f} days"
        ),
    )
    if integrity["warnings"]:
        st.warning("\n".join(f"- {warning}" for warning in integrity["warnings"]))
    else:
        st.success("No automatic data-integrity warnings.")
    st.markdown("#### Nearby Support and Resistance")
    zones = research["market_zones"]["levels"].head(10)
    st.dataframe(
        zones.style.format(
            {
                "level": "{:.5f}",
                "distance_pct": "{:.2%}",
                "absolute_distance_pct": "{:.2%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Advanced raw data-integrity details"):
        issue_table = pd.DataFrame(integrity["issues"])
        st.dataframe(issue_table, hide_index=True, width="stretch")
        deductions = pd.DataFrame(
            {
                "Check": list(integrity["deductions"]),
                "Score Deduction": list(integrity["deductions"].values()),
            }
        )
        st.dataframe(deductions, hide_index=True, width="stretch")
        st.info(
            "Cross-source validation is not active because no second market-data "
            "provider is configured."
        )


def render_explainability(research: dict) -> None:
    """Render global feature importance and latest signal explanation."""
    explanation = research["latest_explanation"]
    signal = SIGNAL_LABELS.get(explanation["prediction"], str(explanation["prediction"]))
    st.subheader(f"Current Unlabeled Signal Explanation: {signal}")
    left, right = st.columns([1, 2])
    with left:
        probability = pd.DataFrame(
            {
                "Signal": [
                    SIGNAL_LABELS.get(int(label), label)
                    for label in explanation["probabilities"]
                ],
                "Probability": explanation["probabilities"].values(),
            }
        )
        probability_chart = px.bar(
            probability,
            x="Signal",
            y="Probability",
            color="Signal",
            title="Prediction Probabilities",
            template="plotly_white",
        )
        st.plotly_chart(probability_chart, width="stretch")
    with right:
        top_importance = research["feature_importance"].head(15).sort_values("importance")
        importance_chart = px.bar(
            top_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Global Feature Importance",
            template="plotly_white",
        )
        importance_chart.update_traces(marker_color="#315a84")
        st.plotly_chart(importance_chart, width="stretch")

    with st.expander("Advanced current feature values"):
        latest_features = pd.DataFrame(explanation["top_features"])
        st.dataframe(
            latest_features[["feature", "importance", "value", "method"]].style.format(
                {"importance": "{:.3f}", "value": "{:.4f}"}
            ),
            hide_index=True,
            width="stretch",
        )
    st.info(
        "Tree importance is global and does not prove causal impact on the latest prediction."
    )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Financial ML Signal Lab",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    add_page_style()
    st.title("AI Trading Intelligence Command Center")
    st.caption(
        "Research-backed market context, scenario risk, data integrity, and explainable trade-quality assessment."
    )
    if not ASSETS:
        st.error("No assets are configured. Add at least one asset to config.yaml.")
        st.stop()
    model_names = list(MODEL_TRAINERS)
    if not model_names:
        st.error("No model trainers are available. Check the model configuration.")
        st.stop()
    default_model = "Random Forest"
    default_index = model_names.index(default_model) if default_model in model_names else 0

    with st.sidebar:
        st.header("Command Center")
        app_mode = st.radio(
            "App Mode",
            ["Research Mode", "Live Mode", "Paper Trading Mode"],
            help="Shows a focused workspace while preserving the same underlying finance and validation rules.",
        )
        st.divider()
        st.markdown("#### Research Controls")
        selected_symbol = st.selectbox(
            "Asset",
            list(ASSET_LABELS),
            format_func=ASSET_LABELS.get,
        )
        selected_model = st.selectbox(
            "Walk-forward model",
            model_names,
            index=default_index,
        )
        transaction_bps = st.slider("One-way transaction cost (bps)", 0, 20, 5)
        train_window = st.select_slider(
            "Rolling training window",
            options=[500, 750, 1_000, 1_250, 1_500],
            value=1_000,
        )
        test_window = st.select_slider(
            "Test block size",
            options=[50, 100, 150, 200],
            value=100,
        )
        st.divider()
        st.markdown("#### Data Updates")
        refresh_latest_data = st.button(
            "Refresh latest market data",
            help="Downloads daily history again, rebuilds features, and refreshes current and research signals.",
        )
        if refresh_latest_data:
            with st.spinner("Refreshing latest market data and features..."):
                try:
                    prepare_configured_dataset(CONFIG, selected_symbol, refresh=True)
                except Exception as exc:
                    st.error(f"Latest market data refresh failed: {exc}")
                else:
                    load_live_chart_data.clear()
                    load_research.clear()
                    st.session_state["daily_refresh_message"] = (
                        f"Latest market data refreshed for {selected_symbol}."
                    )
                    st.rerun()
        if st.session_state.get("daily_refresh_message"):
            st.success(st.session_state.pop("daily_refresh_message"))
        st.caption(
            "Current signals use the latest completed feature row. Historical "
            "research only uses labeled rows with known future outcomes."
        )
        st.divider()
        st.caption(
            "Signals execute one period later. Walk-forward folds purge the five-day target horizon."
        )
        with st.expander("Scenario Simulator Controls"):
            monte_carlo_defaults = INTELLIGENCE_CONFIG.get("monte_carlo", {})
            sizing_defaults = INTELLIGENCE_CONFIG.get("position_sizing", {})
            scenario_lookback = st.select_slider(
                "Simulation lookback",
                options=[60, 100, 252, 500],
                value=int(monte_carlo_defaults.get("lookback", 100)),
            )
            scenario_horizon = st.select_slider(
                "Forecast horizon",
                options=[5, 10, 20, 40, 60],
                value=int(monte_carlo_defaults.get("horizon", 20)),
            )
            scenario_simulations = st.select_slider(
                "Number of simulations",
                options=[1_000, 5_000, 10_000, 25_000],
                value=int(monte_carlo_defaults.get("simulations", 10_000)),
            )
            method_labels = {
                "iid_bootstrap": "IID bootstrap",
                "block_bootstrap": "Block bootstrap",
            }
            configured_method = str(
                monte_carlo_defaults.get("method", "iid_bootstrap")
            )
            method_names = list(method_labels)
            method_index = (
                method_names.index(configured_method)
                if configured_method in method_names
                else 0
            )
            scenario_method = st.selectbox(
                "Monte Carlo method",
                method_names,
                index=method_index,
                format_func=method_labels.get,
            )
            scenario_block_size = st.slider(
                "Block size",
                min_value=1,
                max_value=20,
                value=int(monte_carlo_defaults.get("block_size", 5)),
                disabled=scenario_method != "block_bootstrap",
            )
            scenario_stop_pct = st.slider(
                "Stop distance (%)",
                min_value=0.25,
                max_value=5.0,
                value=float(monte_carlo_defaults.get("stop_loss_pct", 0.01)) * 100,
                step=0.25,
            )
            scenario_target_pct = st.slider(
                "Target distance (%)",
                min_value=0.25,
                max_value=10.0,
                value=float(monte_carlo_defaults.get("take_profit_pct", 0.02)) * 100,
                step=0.25,
            )
            account_size = st.number_input(
                "Account size",
                min_value=1_000.0,
                max_value=10_000_000.0,
                value=float(sizing_defaults.get("account_size", 25_000)),
                step=1_000.0,
            )

    render_mode_badge(app_mode)
    runtime_intelligence_config = {
        **INTELLIGENCE_CONFIG,
        "monte_carlo": {
            **INTELLIGENCE_CONFIG.get("monte_carlo", {}),
            "lookback": scenario_lookback,
            "horizon": scenario_horizon,
            "simulations": scenario_simulations,
            "method": scenario_method,
            "block_size": scenario_block_size,
            "stop_loss_pct": scenario_stop_pct / 100,
            "take_profit_pct": scenario_target_pct / 100,
        },
        "position_sizing": {
            **INTELLIGENCE_CONFIG.get("position_sizing", {}),
            "account_size": account_size,
        },
    }

    with st.spinner("Preparing market data for this asset..."):
        try:
            prepare_configured_dataset(CONFIG, selected_symbol)
        except Exception as exc:
            st.error(
                "Market data could not be prepared. Yahoo Finance may be unavailable "
                f"or rate-limiting this deployment. Details: {exc}"
            )
            st.stop()

    with st.spinner("Running cached research pipeline..."):
        research = load_research(
            selected_symbol,
            selected_model,
            transaction_bps / 10_000,
            train_window,
            test_window,
            ASSET_EXCLUDED_FEATURES[selected_symbol],
            ABSTENTION_THRESHOLDS,
            runtime_intelligence_config,
            RESEARCH_CACHE_VERSION,
        )

    renderers = {
        "Overview": render_overview,
        "Live Market": render_live_market,
        "Paper Trading": render_paper_trading,
        "Model Comparison": render_model_comparison,
        "Backtest": render_backtest,
        "Signal Policy": render_signal_policy,
        "Risk Simulator": render_risk_simulator,
        "Data Integrity": render_data_integrity,
        "Regimes": render_regimes,
        "Explainability": render_explainability,
    }
    tab_order = tab_order_for_mode(app_mode)
    tab_key = f"mode_tabs_{app_mode.lower().replace(' ', '_')}"
    if st.session_state.get("_active_app_mode") != app_mode:
        st.session_state.pop(tab_key, None)
        st.session_state["_active_app_mode"] = app_mode
    tabs = st.tabs(
        tab_order,
        default=tab_order[0],
        key=tab_key,
    )
    for tab, label in zip(tabs, tab_order):
        with tab:
            renderers[label](research)

    st.divider()
    st.caption(
        "Educational research only. Historical results may not generalize and are not financial advice."
    )


if __name__ == "__main__":
    main()
