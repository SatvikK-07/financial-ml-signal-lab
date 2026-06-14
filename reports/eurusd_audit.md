# EUR/USD Result Audit

## Trigger

The initial EUR/USD walk-forward result returned 3,321.15% after costs with a
4.09 Sharpe. Automated diagnostics also measured a 0.32 correlation between
delayed positions and next-period asset returns. These values required an
independent leakage and data-quality audit.

## Findings

Target construction, purged walk-forward boundaries, and one-period execution
delay were correctly aligned. The anomaly originated in Yahoo Finance's daily
FX candle fields:

- `upper_wick` correlation with next-period return: 0.62
- `lower_wick` correlation with next-period return: -0.24
- Next open gap correlation with next-period close return: 0.97
- Reported volume is zero for all EUR/USD rows

The wick fields therefore contain a vendor/session-specific forward association
that is unsafe for this experiment.

## Remediation

The EUR/USD asset configuration now explicitly excludes `upper_wick` and
`lower_wick`. The same feature definitions remain available for SPY and Bitcoin.
Diagnostics now automatically flag any selected model feature whose absolute
correlation with the following period's return exceeds 0.20.

## Corrected Result

| Metric | Before Audit | After Audit |
|---|---:|---:|
| Strategy total return | +3,321.15% | -11.43% |
| Sharpe ratio | 4.09 | -0.09 |
| Maximum drawdown | -6.79% | -24.09% |
| Position/return correlation | 0.32 | 0.02 |
| Strongest selected feature/next-return correlation | 0.62 | -0.03 |

The corrected result does not demonstrate a robust trading edge. It remains
`review_required` because the source has no usable volume field. Future FX
research should use data with explicit and reliable session definitions.
