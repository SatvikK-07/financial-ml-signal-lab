# Financial ML Signal Lab: Final Research Report

## Executive Summary

This project evaluates whether machine-learning classifiers can generate useful
long, short, or neutral SPY signals after leakage-safe validation and realistic
trading frictions.

The central result is negative. Random Forest produced the strongest
chronological holdout macro F1 among the tested machine-learning models, but its
purged walk-forward strategy lost **63.31%** after transaction costs from
January 30, 2014 through June 2, 2026. SPY buy-and-hold gained **421.74%** over
the same out-of-sample period.

This failure is the most important result. It demonstrates that modest
classification performance does not imply an economically useful trading
strategy, and that transaction costs, market regimes, and strictly
out-of-sample testing can invalidate an apparently promising model.

The same fixed methodology was also run on Bitcoin and EUR/USD. Bitcoin lost
98.58% after costs. The initial EUR/USD result produced an implausible 3,321.15%
return and 4.09 Sharpe. An independent audit traced that result to Yahoo FX wick
fields with strong forward-return correlation. Excluding those fields reduced
EUR/USD performance to **-11.43%** with a **-0.09 Sharpe**.

## Research Design

### Problem

The model predicts the direction of SPY's next five-day return:

- `1`: future return above +0.5%
- `0`: future return between -0.5% and +0.5%
- `-1`: future return below -0.5%

The target predicts return direction rather than exact future price.

### Data

The configured experiment uses adjusted daily SPY OHLCV data from Yahoo
Finance. The processed dataset contains 4,108 complete labeled rows and 27
features.

### Leakage Controls

The pipeline applies the following controls:

1. Features at date `t` use only observations at or before `t`.
2. `future_return` and `target` are never included as model features.
3. Chronological splits are used instead of random splits.
4. Five rows are purged before each test boundary because targets use a
   five-day future horizon.
5. Models and scalers are refitted inside every walk-forward fold.
6. Backtest positions use a one-period execution delay.
7. Future return is used only for target construction and evaluation.

## Models

The experiment compares:

- Naive majority-class baseline
- Logistic Regression with training-only scaling
- Random Forest
- XGBoost
- LightGBM

### Chronological Holdout Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 34.67% | 33.21% |
| Logistic Regression | 31.02% | 30.77% |
| LightGBM | 33.09% | 27.50% |
| XGBoost | 33.94% | 27.35% |
| Naive baseline | 51.09% | 22.54% |

The majority baseline's high accuracy shows why accuracy alone is misleading.
Random Forest predicts all classes more evenly and achieves the strongest macro
F1, so it is selected for the configured walk-forward experiment.

## Walk-Forward Evaluation

The walk-forward design uses:

- 1,000-row rolling training windows
- 100-row non-overlapping test windows
- Five purged rows before every test window
- 32 model refits
- 3,103 strictly out-of-sample predictions

### Financial Results

| Metric | Strategy | Buy and Hold |
|---|---:|---:|
| Total return | -63.31% | +421.74% |
| Sharpe ratio | -0.39 | 0.86 |
| Maximum drawdown | -75.24% | -33.72% |

Additional strategy metrics:

- CAGR: -7.82%
- Sortino ratio: -0.37
- Win rate: 52.55%
- Profit factor: 0.73
- Trades: 392
- Average trade return: -0.22%
- Exposure: 84.18%

The strategy's win rate exceeds 50%, but the average losing impact is larger
than the average winning impact. Profit factor below one confirms that gross
losses exceed gross gains.

![Walk-forward equity](figures/walk_forward_equity.png)

## Transaction-Cost Impact

On the fixed latest holdout, the Random Forest strategy gained approximately
7.46% before costs and lost 4.69% after costs. The model changes positions too
frequently for its weak edge to survive a five basis-point one-way cost.

Across the broader walk-forward period, performance is negative even before
costs. Transaction costs worsen an already unstable signal.

## Regime Analysis

Regimes are classified using only trailing/current information.

### Volatility Regimes

| Regime | Total Strategy Return | Sharpe |
|---|---:|---:|
| High volatility | -39.89% | -0.36 |
| Low volatility | +16.78% | 0.42 |
| Normal volatility | -47.73% | -1.39 |

### Trend Regimes

| Regime | Total Strategy Return | Sharpe |
|---|---:|---:|
| Downtrend | -62.10% | -2.07 |
| Sideways | +22.33% | 0.38 |
| Uptrend | -20.85% | -0.28 |

![Regime Sharpe](figures/regime_sharpe.png)

The strongest failure occurs during downtrends. Classification accuracy is
actually highest in downtrends, reinforcing that correct direction labels do
not necessarily translate into profitable one-period-delayed positions.

These regime results are diagnostic only. A strategy that filters regimes after
observing these outcomes would require a new untouched validation period.

## Explainability

The top Random Forest global feature importances are:

1. `volatility_10`
2. `atr_pct`
3. `volatility_5`
4. `volatility_20`
5. `rsi_14`

![Feature importance](figures/feature_importance.png)

The model relies heavily on volatility-related features. Feature importance is
not causal and does not explain the direction of an individual prediction.

## Latest Research Signal

Using all available labeled rows for training and the newest raw-data feature
row for inference:

- Date: June 9, 2026
- SPY adjusted close: $729.88
- Signal: Long
- Confidence: 49.4%
- Volatility regime: High volatility
- Trend regime: Sideways

This is a research output, not a trading recommendation. The low confidence and
negative historical walk-forward performance do not support deployment.

## Confidence-Aware Abstention

The project evaluates a predefined minimum-confidence grid of 0%, 45%, 50%,
55%, and 60%. Predictions below each threshold are converted to neutral before
the delayed-execution backtest. The raw baseline remains the configured result;
no threshold is selected using the same out-of-sample period.

At a 60% minimum confidence, active signal rates fall to 12.5% for SPY, 19.5%
for Bitcoin, and 9.7% for EUR/USD. SPY and Bitcoin losses shrink substantially,
while EUR/USD remains negative. This pattern suggests weak predictions account
for much of the loss, but it does not validate the 60% threshold. Model
confidence is not calibrated probability, and any threshold chosen after
observing these results requires a new untouched validation period.

Scenario tables are exported as `abstention_performance.csv` for each asset.

## Trading Intelligence Command Center

The project now converts the research signal into a trader-facing decision
brief rather than presenting Long, Short, or Neutral in isolation.

### Scenario Risk

The Monte Carlo engine bootstraps recent empirical returns into 10,000 possible
20-candle paths. It reports take-profit-before-stop probability, stop-first
probability, terminal-price range, Value at Risk, Expected Shortfall, and
adverse-move scenarios. This is a risk simulator, not a price forecast.

For the current SPY long signal, the configured simulation estimates:

- Take profit before stop: 46.67%
- Stop before take profit: 52.06%
- 95% Value at Risk: -5.29%
- 95% Expected Shortfall: -6.89%

### Integrity and Trust

The Data Integrity Engine audits missing sessions, duplicates, invalid OHLC
candles, stale data, price jumps, and volume availability. SPY currently scores
100/100. EUR/USD remains critical because Yahoo reports unavailable volume and
several inconsistent daily candle boundaries.

The trust engine combines model confidence, data quality, historical regime
support, backtest support, Monte Carlo alignment, and zone alignment. It also
applies explicit warnings when news risk or a second validation feed is not
connected. The current SPY setup is classified as low quality and recommends
waiting for stronger confluence.

Position sizing is risk-tiered from the trust score and remains a simplified
estimate that must be checked against broker contract and leverage rules.

## Conclusions

The current feature set and model family do not generate a robust SPY trading
signal under the configured assumptions.

The project still succeeds as a research platform because it:

- Prevents common time-series leakage errors
- Distinguishes classification metrics from financial usefulness
- Quantifies the impact of transaction costs
- Exposes performance instability across market regimes
- Produces reproducible artifacts and an interactive dashboard
- Reports negative results without hiding them
- Converts model outputs into explainable risk and trade-quality decisions

## Cross-Asset Expansion

| Asset | Strategy Return | Sharpe | Buy and Hold | Diagnostic Status |
|---|---:|---:|---:|---|
| SPY | -63.31% | -0.39 | +421.74% | No automatic flags |
| Bitcoin | -98.58% | -0.31 | +2,381.26% | No automatic flags |
| EUR/USD | -11.43% | -0.09 | -15.32% | Review required |

The EUR/USD audit found that `upper_wick` correlated 0.62 and `lower_wick`
correlated -0.24 with the following period's return. Removing both fields
collapsed strategy Sharpe from 4.09 to -0.09 and reduced delayed-position return
correlation from 0.32 to 0.02. The corrected result remains flagged because
Yahoo's FX dataset has effectively no usable volume.

The full investigation is documented in `reports/eurusd_audit.md`.

Each asset exports separate tables, figures, summary, and diagnostic JSON under
`reports/assets/`. The consolidated comparison is stored in
`reports/results/cross_asset_summary.csv`.

## Limitations and Next Experiments

The experiment covers three assets and one daily target definition. Further
research should:

1. Source institutional-grade FX data with reliable session and volume fields.
2. Test additional assets and frequencies without changing methodology after
   observing results.
3. Use nested time-series validation for hyperparameter selection.
4. Calibrate probabilities inside every training fold before testing abstention.
5. Penalize turnover during model selection.
6. Add signal smoothing and minimum holding periods.
7. Model short borrowing costs, slippage, and portfolio constraints.
8. Compare advanced neural models only after improving simple baselines.

## Reproduction

Run:

```bash
python -m src.pipeline --refresh-data
python -m src.pipeline --all-assets
python -m pytest -q
python -m streamlit run app/streamlit_app.py
```

Machine-readable results are stored under `reports/results/`.
