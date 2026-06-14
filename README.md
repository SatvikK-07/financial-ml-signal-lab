# Financial ML Signal Lab

## What This Project Is

Financial ML Signal Lab is a financial machine-learning and AI trading
intelligence research platform. It combines leakage-safe model evaluation with
scenario risk simulation, data-quality diagnostics, execution-policy analysis,
support/resistance context, news-risk placeholders, and explainable trade
quality scoring.

The platform is designed to answer:

- What is the model signal?
- Is the signal historically trustworthy?
- What market regime is active?
- How sensitive is the result to execution policy and transaction costs?
- What could happen under simulated risk scenarios?
- Is the source data reliable enough to use?
- Where are nearby support and resistance zones?
- How much capital would be at risk at the proposed stop?

## What This Project Is Not

- Not a guaranteed trading bot
- Not financial advice
- Not live broker execution
- Not proof of profitability
- Not a replacement for broker-grade data, execution controls, or human review

The system deliberately reports weak and losing strategies. It does not convert
an unreliable model prediction into a false recommendation.

## Current Strategy Performance

The configured primary experiment uses SPY daily adjusted OHLCV data, a
five-day future-return target, Random Forest walk-forward predictions, delayed
execution, and five-basis-point one-way transaction costs.

| Metric | Current Result |
|---|---:|
| Model-ready SPY rows | 4,108 |
| Leakage-safe features | 27 |
| Strictly out-of-sample predictions | 3,103 |
| Random Forest holdout macro F1 | 33.21% |
| Strategy total return after costs | -63.31% |
| Strategy Sharpe ratio | -0.39 |
| Strategy maximum drawdown | -75.24% |
| SPY buy-and-hold total return | +421.74% |

The current strategy is not robust enough for deployment. Bitcoin also loses
materially under the configured methodology. The corrected EUR/USD experiment
remains negative and carries source-data quality warnings.

## Why Losing Results Are Valuable

Negative results prevent false confidence. This project demonstrates that:

- Classification accuracy does not guarantee profitable execution.
- Weak model confidence can disappear after transaction costs.
- Data-vendor anomalies can create implausible apparent alpha.
- Daily rebalancing may not match a multi-day prediction target.
- Attractive thresholds or regimes observed after testing require new,
  untouched validation data.

A research platform that detects weak signals and invalid assumptions is more
useful than one that hides losses.

## Validation Methodology

The research pipeline applies:

- Chronological train/test splits
- Purged walk-forward validation
- Training-only scaling and model fitting
- One-period-delayed execution
- Proportional transaction costs
- Buy-and-hold comparison
- Regime-level evaluation
- Automated plausibility diagnostics
- Data-integrity diagnostics
- Monte Carlo scenario simulation

Future returns and target labels are excluded from model features.

## Backtest Assumption Clarification

The default target predicts a future return over a configurable horizon, while
the default backtest rebalances daily using the latest signal. Additional
execution policies such as hold-for-horizon and signal-change-only are included
to evaluate turnover and horizon mismatch.

Available execution policies:

- `daily_rebalance`: existing delayed daily signal behavior
- `hold_for_horizon`: ignores new signals while holding for the configured
  horizon
- `signal_change_only`: changes intended position only when the prediction
  changes
- `confidence_filtered`: neutralizes signals below a fixed confidence threshold

All final positions remain delayed by one period, and transaction costs use
executed-position turnover.

## Monte Carlo Scenario Simulator

The Monte Carlo engine is a scenario simulator, not a price forecast. It
assumes the recent empirical return distribution remains relevant.

Supported methods:

- IID bootstrap: samples individual recent log returns
- Block bootstrap: samples contiguous return blocks to preserve short-term
  sequences

Outputs include:

- Simulated price paths and terminal-price distribution
- Probability of take profit before stop loss
- Probability of stop loss before take profit
- Expected return and expected range
- Value at Risk: the loss threshold exceeded in the worst configured tail
- Expected Shortfall: the average loss within that worst tail

## Trader Usefulness

### Trust and Trade-Quality Score

Each signal receives an explainable score based on:

- Model confidence
- Data integrity
- Historical regime support
- Walk-forward backtest support
- Monte Carlo alignment
- Support/resistance alignment
- Volatility adjustment
- Diagnostic penalties
- Manual economic-calendar risk

The dashboard displays the score contribution of every component and explains
why the setup is strong, weak, or should be avoided.

### Data Integrity Engine

The integrity engine checks:

- Missing expected sessions
- Duplicate timestamps
- Invalid or impossible OHLC candles
- Missing values and non-positive prices
- Large close-to-close jumps
- Data freshness in minutes
- Stale intraday/daily data
- Missing or zero volume

Yahoo FX volume may be zero or unavailable. EUR/USD also has documented vendor
candle issues, so asset-specific unsafe features are excluded.

### Position Sizing

The system includes:

- Generic fixed-fractional sizing
- Dedicated FX pip-risk sizing
- Risk amount and stop distance
- Standard, mini, and micro lot estimates
- Estimated loss at stop

Sizing remains an estimate. Broker contract specifications, pip values, spread,
leverage, and execution rules must be confirmed independently.

### Support, Resistance, and News Risk

Market zones include swing levels, prior-period highs/lows, and round numbers.
Zones receive touch, recency, proximity, and source-strength scores.

News risk currently uses an optional manual CSV:

`data/manual/economic_calendar.csv`

It does not use a paid live calendar API. If the file is missing, the app
continues safely and reports that no manual calendar file was found.

## Project Structure

```text
app/                    Streamlit command-center dashboard
data/manual/            Optional manual economic calendar
data/raw/               Saved raw market data
data/processed/         Feature and target datasets
reports/                Exported tables, JSON summaries, and figures
src/                    Research, risk, integrity, and intelligence modules
tests/                  Fast unit and marked integration tests
config.yaml             Reproducible experiment configuration
pytest.ini              Slow/integration test markers
```

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, optional XGBoost and LightGBM native dependencies may require:

```bash
brew install libomp
```

## Run the Research Pipeline

Run the primary configured asset:

```bash
python -m src.pipeline
```

Run all configured assets:

```bash
python -m src.pipeline --all-assets
```

Refresh downloaded Yahoo data:

```bash
python -m src.pipeline --all-assets --refresh-data
```

## Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard includes:

- Dedicated Yahoo live/recent candlestick chart
- Explicit daily-data refresh and research retraining control
- Latest daily signal and clear latest-vs-walk-forward distinction
- Persistent local paper-trading journal
- Trust score and component breakdown
- Execution-policy comparison
- Confidence-abstention research
- Monte Carlo IID/block bootstrap simulator
- VaR and Expected Shortfall
- Generic and FX position sizing
- Data-integrity severity reporting
- Scored support/resistance zones
- Manual economic-calendar risk
- Model comparison, regimes, backtest, and explainability

### Research Signal vs Current Market Signal

The dashboard deliberately separates two different concepts:

- **Latest Backtest-Evaluable Signal** uses a labeled historical row where the future
  return is already known. It belongs to strictly out-of-sample walk-forward
  evaluation and can lag the latest market candle by the target horizon.
- **Current Market Signal** trains only on labeled historical rows, then
  predicts the latest available completed feature row where the future outcome
  is unknown.

The current signal does not use `future_return` or any future target. This
separation prevents the latest displayed signal from being artificially stale
without introducing lookahead bias.

### UI Modes

The sidebar provides three focused workflow modes. Each mode opens its primary
workspace first and hides unrelated tabs without changing the underlying
finance or validation rules:

- **Research Mode** prioritizes backtests, model comparison, signal policy, and
  regime diagnostics. It opens on Overview.
- **Live Mode** shows the current chart, data freshness, market context, and
  risk. It opens directly on Live Market.
- **Paper Trading Mode** shows hypothetical trade logging, live context, and
  outcome review. It opens directly on Paper Trading.

### Trust Score and Data Freshness

The Overview answers **“Can I trust this signal?”** with a Low, Moderate, or
High assessment. Every component displays its raw score, weighted trust impact,
pass/warning/fail status, and reason.

Components include model confidence, data integrity, backtest support, Monte
Carlo alignment, zone alignment, regime support, news risk, diagnostics,
volatility, and data freshness. Stale daily data explicitly reduces trust.
Stale intraday data also forces a wait decision and zero suggested risk until a
fresh completed candle is available.

### Dashboard Screenshots

Screenshot placeholders and capture guidance are documented in
`docs/screenshots/README.md`. Add final demo captures there after deployment so
the README does not show outdated market data.

## Run Tests

Fast test suite:

```bash
python -m pytest -q -m "not slow"
```

Full test suite:

```bash
python -m pytest -q
```

Compile-check Python files:

```bash
python -m compileall -q src app tests
```

Slow integration tests are marked with `slow` and `integration` in
`pytest.ini`. They cover full dashboard-research construction and report/figure
generation.

## Reproducible Artifacts

Pipeline output includes:

- Model comparison and walk-forward predictions
- Walk-forward backtest and execution-policy comparison
- Confidence-abstention scenarios
- Volatility and trend regime performance
- Feature importance
- Data-integrity report
- Scored market zones
- Monte Carlo summary, paths, and terminal distribution
- Trade-intelligence score and position sizing
- Manual news-risk summary
- Cross-asset comparison

Artifacts are stored under `reports/results/`, `reports/figures/`, and
`reports/assets/`.

## Limitations

- Yahoo data may be delayed, incomplete, or vendor-specific.
- The latest labeled research row can lag because the target horizon requires
  future data; the current signal is generated separately from the latest
  unlabeled completed candle.
- FX volume from Yahoo may be zero or missing.
- Monte Carlo relies on recent distribution assumptions.
- Transaction costs are simplified.
- No live spread, order book, or slippage feed is connected.
- No live broker execution is implemented.
- No paid live news or economic-calendar API is connected.
- Position sizing does not replace broker contract verification.
- Support/resistance detection is heuristic.
- Historical and simulated results do not guarantee future profitability.
- Paper-trading outcomes are approximate and local Streamlit Cloud storage is
  not durable.

### Live Update Behavior

The dashboard separates live market context from audited daily research:

- The **Live Market** tab requests recent Yahoo candles and displays a Plotly
  candlestick chart with selectable interval and period.
- The chart includes EMA 20, EMA 50, current price, nearest support/resistance,
  current signal, freshness, market status, intraday return, and ATR.
- **Refresh Live Chart** performs an explicit refresh. Optional 60-second
  auto-refresh is disabled by default.
- The **Refresh latest market data** sidebar button downloads daily history,
  rebuilds features, and retrains the latest research signal.
- The daily research signal may still show the prior completed trading day
  before a new daily candle is available.
- A stale live-data warning commonly means the selected market is closed or
  Yahoo has not published a newer candle.

Live Yahoo data may be delayed and is not broker-grade.

### Paper Trading

The **Paper Trading** tab stores hypothetical trades locally in
`data/paper_trades/paper_trades.csv`. No real orders are submitted.

- Current signal, stop, target, trust score, risk, and position size can be
  logged as a paper trade.
- Open trades can be updated from later available candles.
- If stop and target both touch within one candle, the outcome uses the
  conservative stop-first assumption.
- Open and closed trades, win rate, total hypothetical PnL, average PnL, best
  trade, and worst trade are displayed.
- Candle-based outcomes are approximate and do not model fill quality, gaps,
  spread, slippage, or broker execution.
- On Streamlit Community Cloud, local paper-trade CSV storage is ephemeral and
  may reset when the app restarts or redeploys.

### Streamlit Community Cloud

Use `app/streamlit_app.py` as the app entry point. Python dependencies are
listed in `requirements.txt`, Linux OpenMP support for optional tree libraries
is listed in `packages.txt`, and the app theme/server defaults are stored in
`.streamlit/config.toml`.

On first launch, the dashboard downloads and prepares the selected asset when
ignored local raw/processed CSV files are unavailable. The dashboard handles
missing optional live data and missing paper-trade files without crashing.
Yahoo requests can still be throttled or delayed by the data provider.

## Future Roadmap

- Live five-minute lightweight update mode
- Broker-grade primary and reference data feeds
- Spread, slippage, and order-book inputs
- DXY correlation for EUR/USD context
- Trade journal and setup review
- Historical replay mode
- Probability calibration inside walk-forward folds
- LLM research assistant for explaining audited results

See `reports/final_report.md` and `reports/eurusd_audit.md` for detailed research
discussion.
