# Regime Detection and Factor Timing

A non-parametric framework for identifying the current macroeconomic regime and using that information to time equity factor premia.

---

## Overview

Rather than imposing discrete regime labels, the method identifies which historical periods were most similar to the present by computing Euclidean distances across seven standardised financial state variables. The intuition: if we can find months in history that looked like today, the factor returns that followed those months carry predictive information for factor returns going forward.

The framework also exploits *anti-regimes* — periods that are maximally dissimilar to the current environment — as a short signal. The spread between the most-similar and most-dissimilar quintile portfolios delivers positive risk-adjusted returns across the 1985–2024 sample with near-zero correlation to the long-only factor benchmark.

---

## State Variables

Seven monthly series form the state vector:

| # | Variable | Source |
|---|----------|--------|
| 1 | S&P 500 level | Yahoo Finance (`^GSPC`) |
| 2 | 10Y − 3M Treasury yield spread | FRED: `GS10`, `TB3MS` |
| 3 | WTI crude oil price | FRED: `MCOILWTICO` (spliced with `WPU0561` pre-1986) |
| 4 | Copper price | FRED: `PCOPPUSDM` (spliced with `WPU102502` pre-1992) |
| 5 | 3-month T-bill yield | FRED: `TB3MS` |
| 6 | Equity volatility | VIX post-1990 (`^VIX`); 21-day realised vol pre-1990 |
| 7 | Rolling 3-year stock-bond correlation | Computed from daily returns |

Each series is transformed via: 12-month difference → rolling 10-year z-score → winsorise at ±3.

---

## Methodology

**Distance metric.** For evaluation month *T* and historical month *i*:

```
d(T, i) = sqrt( sum_v (z_{T,v} - z_{i,v})^2 )
```

**Exclusion mask.** The 36 months immediately preceding *T* are excluded to avoid look-ahead bias from momentum-like contamination.

**Factor timing.** Historical months are sorted into quintiles by distance. The average factor return in the month following each quintile determines the signal direction (+1 / −1) for each of the six Fama-French factors.

**Portfolio construction.** Signals are applied with a one-month lag. The spread portfolio (Q1 long, Q5 short) is volatility-scaled to 15% annualised using a trailing 12-month realised volatility estimate.

**EWMA regime-shift detection.** An exponentially weighted average of distance scores — with half-lives of 8, 16, 25, and 33 months — flags periods when the current environment is unusually different from recent history.

---

## Factors

Six Fama-French / Carhart factors from Ken French's data library:
`Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `Mom`

---

## Project Structure

```
regime/
├── main.py                  # End-to-end pipeline runner
├── src/
│   ├── data_loader.py       # FRED + Yahoo Finance data acquisition
│   ├── transforms.py        # Z-score transformation and descriptive stats
│   ├── similarity.py        # Distance matrix and EWMA regime-shift detection
│   ├── backtest.py          # Signal construction and portfolio building
│   ├── evaluation.py        # Performance metrics
│   └── exhibits.pyy         # All figures (Exhibits 1–13, A1–A2)
├── figures/                 # Output figures (PNG + PDF)
├── paper/                   # Compiled PDF
└── data/                    # Pickle cache (git-ignored)
```

---

## Setup

```bash
# Create environment (Python 3.11)
uv venv .venv311 --python 3.11
uv pip install --python .venv311/Scripts/python.exe \
    pandas numpy scipy matplotlib seaborn yfinance fredapi statsmodels reportlab

# Run (set your FRED API key first)
export FRED_API_KEY=your_key_here
.venv311/Scripts/python.exe main.py
# or
.venv311/Scripts/python.exe main.py --fred-key your_key_here
```

A free FRED API key can be obtained at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

Data is cached to `data/` after the first run. Use `--force` to re-download.

---

## Robustness

- **Quantile count:** Results hold across 2, 3, 4, 5, 10, and 20 quantile splits (Exhibit 12)
- **Lookback window:** Consistent performance with 1-year, 3-year, and 5-year z-score lookbacks (Exhibit 13)
- **Individual factors:** Both quintile and long-short results shown per factor (Exhibits A1–A2)
