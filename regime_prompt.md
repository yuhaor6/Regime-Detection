# Regime Detection & Factor Timing — Full Plan

## Objective

Replicate the methodology from Mulliner, Harvey, Xia, Fang & van Hemert (2025) "Regimes" (SSRN 5164863). Build an end-to-end Python pipeline that:

1. Sources and transforms 7 macroeconomic state variables
2. Computes Euclidean-distance-based regime similarity scores
3. Constructs quintile-sorted factor timing portfolios
4. Evaluates performance with a full suite of metrics and robustness checks
5. **Produces every exhibit from the paper** (charts, tables, heatmaps)
6. **Generates a polished PDF research paper** documenting the replication

Results do not need to match the paper exactly (we use public data vs. Bloomberg/Man Group internals), but directional findings should hold.

---

## Environment Setup

```
Python 3.10+
```

### Required packages

```
pip install pandas numpy scipy matplotlib seaborn yfinance pandas_datareader fredapi statsmodels reportlab pdfplumber
```

### API keys needed

- **FRED API key** — free from https://fred.stlouisfed.org/docs/api/api_key.html (set as env variable `FRED_API_KEY`)

### Directory structure

```
regime_replication/
├── data/              # Raw and processed data
├── src/               # Source code modules
│   ├── data_loader.py
│   ├── transforms.py
│   ├── similarity.py
│   ├── backtest.py
│   └── evaluation.py
├── figures/           # All generated exhibits (PNG + PDF)
├── paper/             # Final PDF paper
├── notebooks/         # Exploration notebooks (optional)
└── main.py            # Master pipeline script
```

---

## STAGE 1: Data Acquisition

### 1.1 Seven State Variables

Source monthly data going back as far as possible (target: 1950s–1960s start dates). Use FRED API as primary source, Yahoo Finance as fallback.

| # | Variable | Paper Label | Source | FRED Ticker / Notes |
|---|----------|-------------|--------|---------------------|
| 1 | S&P 500 price level | Market | FRED or Yahoo | `SP500` (FRED, monthly) or `^GSPC` (Yahoo, daily → resample) |
| 2 | 10Y minus 3M yield spread | Yield curve | FRED | Compute: `GS10` minus `TB3MS`. Or use `T10Y3M` (starts 1982 only — less history). Paper says 10Y minus 3Y in some chart labels but means 10Y minus 3M based on Exhibit 2 title |
| 3 | WTI crude oil price | Oil | FRED | `MCOILWTICO` (monthly WTI). Or `DCOILWTICO` (daily) resampled |
| 4 | Copper price | Copper | FRED | `PCOPPUSDM` (monthly world copper price, USD). Or Yahoo `HG=F` for futures |
| 5 | US 3-month T-bill yield | Monetary policy | FRED | `TB3MS` |
| 6 | Equity volatility (VIX post-1990, realized vol pre-1990) | Volatility | Yahoo + computed | `^VIX` from Yahoo (daily, resample to month-end). Pre-1990: compute 21-day rolling std of daily S&P 500 log returns × sqrt(252), then take month-end value |
| 7 | Stock-bond correlation | Stock-bond | Computed | Rolling 3-year correlation of daily S&P 500 returns vs. daily long-term Treasury returns. For Treasury returns: approximate using `−ModifiedDuration × ΔYield` on the 10-year (`DGS10` daily from FRED). Assume duration ≈ 8. Compute on daily data, take month-end value |

### 1.2 Fama-French Factor Data

Download from Kenneth French's data library:
- **Fama-French 5 Factors (monthly)** — Market (Mkt-RF), SMB, HML, RMW, CMA
- **Momentum Factor (Mom, monthly)** — 12-month momentum (UMD)

Use `pandas_datareader.famafrench` or direct CSV download.

### 1.3 Data Validation Checks

After loading, verify:
- [ ] Date ranges: each series starts before 1975 (ideally 1960s)
- [ ] No large gaps (interpolate or flag missing months)
- [ ] Merge all 7 series into a single DataFrame indexed by month-end date
- [ ] Print start/end dates, count of observations, any NaNs

---

## STAGE 2: Transformations

### 2.1 Z-Score Construction

For each of the 7 variables:

1. **12-month difference**: `Δx_t = x_t − x_{t−12}`
2. **Rolling 10-year (120-month) standard deviation** of the 12-month differences: `σ_t = std(Δx_{t−119}, ..., Δx_t)`
3. **Z-score**: `z_t = Δx_t / σ_t`
4. **Winsorize** at ±3: `z_t = clip(z_t, −3, 3)`

Important: use **expanding window for the first 120 months** if you don't want to lose early data, or simply start your z-score series once you have 120 months of 12-month differences. The paper starts analysis from 1963 (Exhibit 3), which implies they begin the z-score once enough history is available.

### 2.2 Point-in-Time Discipline

All transformations must be strictly causal. At month T, you may only use data from months ≤ T. No future leakage.

---

## STAGE 3: Similarity Engine

### 3.1 Euclidean Distance Computation

For each investment month T (from 1985-01 to 2024-12):

```
For every historical month i where i ≤ T − 36:
    d(T, i) = sqrt( Σ_{v=1}^{7} (z_{T,v} − z_{i,v})^2 )
```

**Critical: the 36-month exclusion mask.** Exclude the 36 months immediately before T (i.e., months T−35 through T−1) to avoid momentum contamination. The paper is explicit about this.

Store the full distance vector for each T.

### 3.2 Quintile Assignment

For each month T:
1. Sort all valid historical months by distance (ascending)
2. Assign to quintiles: Q1 = 20% most similar (lowest distance), Q5 = 20% most dissimilar (highest distance)

### 3.3 Anti-Regime Concept

Q5 (most dissimilar) represents "anti-regimes." These carry information too — returns subsequent to anti-regime months tend to predict *opposite* performance for today.

---

## STAGE 4: Factor Timing Backtest

### 4.1 Signal Construction

For each month T and each factor f:

1. Take Q1 months (most similar). Look up factor f's return in the month *following* each Q1 month (i.e., if Q1 includes month i, use factor return at month i+1).
2. Compute the average of those subsequent returns: `signal_Q1(T, f) = mean(r_{i+1,f} for i in Q1)`
3. If `signal_Q1(T, f) > 0`, go **long** factor f at month T+1. If < 0, go **short**.
4. Repeat for Q2, Q3, Q4, Q5.

### 4.2 Portfolio Construction

- **Quintile portfolios**: For each quintile q, form an equally-weighted portfolio across all 6 timed factors. The return at month T+1 is `R_q(T+1) = (1/6) Σ_f direction_q(T,f) × r_{T+1,f}`
- **Long-only benchmark**: Always long all 6 factors equally weighted: `R_LO(T+1) = (1/6) Σ_f r_{T+1,f}`
- **Spread portfolio**: `R_spread(T+1) = R_Q1(T+1) − R_Q5(T+1)`

### 4.3 Volatility Targeting

The paper targets 15% annualized volatility. Implement:
- Compute trailing 12-month realized volatility of each portfolio
- Scale position by `0.15 / realized_vol` each month
- Apply this to the spread portfolio and to the quintile portfolios shown in Exhibit 1

---

## STAGE 5: Exhibits to Reproduce

**Reproduce every exhibit below. Save each as both PNG (300 dpi) and PDF in `figures/`.**

### Exhibit 1: Yearly Returns Bar Chart
- Bar chart of yearly returns from the spread (Q1 long, Q5 short) portfolio at 15% vol target
- X-axis: year (1985–2024), Y-axis: % return
- Annotate: 80% positive years, +13.3% avg conditional on up, −5.1% avg conditional on down
- Style: clean, professional, muted blue bars

### Exhibit 2: Raw State Variables (7 panels)
- 7 separate time series plots showing the raw data for each state variable
- Match the paper's layout: 2 columns, 4 rows (last row has 1 panel)
- Include proper axis labels, titles, source annotations

### Exhibit 3: Transformed State Variables (7 panels)
- 7 panels showing both the raw z-score and the winsorized z-score overlaid
- Orange for winsorized, blue/grey for raw
- Highlight where winsorization bites (oil 1970s, volatility 1987)

### Exhibit 4: Autocorrelation Table
- Table with rows = 7 variables, columns = autocorrelations at 1-month, 3-month, 12-month, 3-year, 10-year lags + monthly mean + std + frequency
- Format as a clean table in the paper

### Exhibit 5: Cross-Correlation Heatmap
- 7×7 correlation matrix of the transformed state variables
- Color-coded heatmap (red = positive, blue = negative)
- Numbers displayed in each cell

### Exhibit 6: Global Similarity Score — January 2009 (GFC)
- Time series of global distance score for all historical months relative to Jan 2009
- Highlight the 15% most similar months (green bars or shading)
- Grey shading for the 36-month exclusion mask
- Left axis: global score; annotate "More similar" at bottom, "Less similar" at top
- Should visually show similarity to 1980s recessions

### Exhibit 7: Global Similarity Score — Feb 2020 & Apr 2020 (COVID)
- Two panels (one for Feb 2020, one for Apr 2020)
- Same format as Exhibit 6
- Expect: less clear patterns (COVID was unique)

### Exhibit 8: Global Similarity Score — Aug 2022 (Inflation Surge)
- Same format as Exhibit 6, reference month = Aug 2022
- Should show similarity to 1970s–1980s inflation episodes
- Include a sub-table of historical inflationary periods (paper provides this data — recreate it):

| Period | Start | End | Total CPI Change | Length (months) |
|--------|-------|-----|------------------|-----------------|
| US enters WW2 | Apr-41 | May-42 | 15% | 14 |
| End of WW2 | Mar-46 | Mar-47 | 21% | 13 |
| Korean War | Aug-50 | Feb-51 | 7% | 7 |
| End of Bretton Woods | Feb-66 | Jan-70 | 19% | 48 |
| OPEC oil embargo | Jul-72 | Dec-74 | 24% | 30 |
| Iranian revolution | Feb-77 | Mar-80 | 37% | 38 |
| Reagan's boom | Feb-87 | Nov-90 | 20% | 46 |
| China demand boom | Sep-07 | Jul-08 | 6% | 11 |
| Post-COVID | Jan-22 | Oct-22 | 6% | 10 |

### Exhibit 9: EWMA Regime Shift Detection
- Plot of EWMA of global scores with 1-year, 2-year, 3-year, 4-year lookbacks + their mean
- Annotate key regime shift dates (Oct 82, May 83, Dec 90, etc.)
- Include small table of half-lives used

EWMA calculation:
```
β = 1 − 1/n  (where n = lookback in months)
half_life = −ln(2) / ln(β)
```

| Lookback | Half-life |
|----------|-----------|
| 1-year (12 months) | 8 months |
| 2-year (24 months) | 16 months |
| 3-year (36 months) | 25 months |
| 4-year (48 months) | 33 months |

### Exhibit 10: Quintile Performance — Cumulative Returns (2 panels)
- **Left panel**: Cumulative returns for Q1 through Q5 + long-only benchmark
  - Legend shows Sharpe ratio and correlation to LO for each
  - Q1 should have highest SR (~0.95), Q5 lowest (~0.17)
- **Right panel**: Cumulative return of Q1−Q5 spread
  - Show SR and correlation to LO

### Exhibit 11: Drawdown Comparison
- Drawdown chart comparing LO model vs. Q1−Q5 spread model
- Y-axis: % of capital (negative), X-axis: time
- Key insight: when LO is down ~50%, spread is only down ~10%

### Exhibit 12: Quantile Robustness — Similar-Dissimilar Spread
- Overlay cumulative returns of the top-minus-bottom spread for different quantile choices (2, 3, 4, 5, 10, 20 quantiles)
- Legend shows SR for each

### Exhibit 13: Z-Score Lookback Robustness
- Overlay cumulative returns of the spread for 1-year, 3-year, 5-year lookback periods (vs. default 10-year)

### Exhibit A1: Individual Factor Quintile Performance (6 panels)
- One panel per factor (Market, Size, Value, Profitability, Investment, Momentum)
- Each panel shows Q1–Q5 cumulative returns + LO benchmark
- Legend: SR and correlation for each quintile

### Exhibit A2: Individual Factor Long-Short Spread (6 panels)
- One panel per factor
- Each shows Q1−Q5 spread cumulative return + LO benchmark
- Legend: SR and correlation

---

## STAGE 6: Additional Performance Metrics

Compute and report in the paper:

- **Sharpe ratios** for all quintile portfolios, spread, and LO
- **Alpha and t-stat** from regression: `R_spread = α + β × R_LO + ε`. Report α and t(α). Paper claims α is 3 standard deviations from zero.
- **Maximum drawdown** for spread and LO
- **Yearly return distribution**: % positive years, average conditional on positive, average conditional on negative
- **Skewness** of yearly returns for the spread portfolio
- **Correlation matrix** of quintile returns vs. LO

---

## STAGE 7: Paper Generation

### Paper Structure

Produce a PDF research paper (~20 pages) with the following sections:

1. **Title Page**
   - Title: " Non-Parametric Regime Detection for Factor Timing"
   - Author: Yuhao Ren
   - Date, abstract (~150 words)

2. **Introduction** (~1.5 pages)
   - Motivation: real-time regime identification is hard; discretionary classification is subjective
   - Preview of results

3. **Economic State Variables** (~3 pages)
   - Description of the 7 variables and data sources
   - Transformation methodology (12-month change, rolling z-score, winsorization)
   - Include Exhibits 2, 3, 4, 5

4. **Regime Similarity Methodology** (~3 pages)
   - Euclidean distance definition
   - Global score aggregation
   - 36-month exclusion mask rationale
   - Case studies: GFC (Exhibit 6), COVID (Exhibit 7), Inflation (Exhibit 8)
   - Regime shift detection via EWMA (Exhibit 9)

5. **Factor Timing Results** (~4 pages)
   - Quintile portfolio construction
   - Main results (Exhibit 10)
   - Drawdown analysis (Exhibit 11)
   - Alpha regression results

6. **Robustness** (~2 pages)
   - Quantile choice sensitivity (Exhibit 12)
   - Lookback period sensitivity (Exhibit 13)

7. **Individual Factor Analysis** (~2 pages)
   - Exhibits A1, A2
   - Discussion of which factors benefit most from timing

8. **Conclusion** (~1 page)
   - Summary of findings

** No References**

### Paper Formatting Requirements

- **Font**: Times New Roman or similar serif, 11pt body
- **Margins**: 1 inch all sides
- **Figures**: embedded inline, numbered "Exhibit N", with descriptive captions
- **Tables**: clean formatting, no vertical lines, horizontal rules only at top/bottom/header
- **Equations**: properly typeset (use LaTeX-style rendering if possible, or clean Unicode)
- **Page numbers**: bottom center
- **Generate using**: `reportlab` (Python) or LaTeX via subprocess. If using reportlab, use Platypus for structured layout.

---

## Implementation Notes & Pitfalls

### Critical

1. **36-month mask**: Without this, the most "similar" months are just recent months and you're building a momentum strategy, not a regime strategy. Always exclude months T−35 through T−1.

2. **Off-by-one in returns**: When you identify similar month i, look at the factor return at month **i+1** (the subsequent month). When you form position at month T, trade at month **T+1**. Double-check alignment.

3. **Point-in-time z-scores**: The rolling 10-year std at month T uses only data through month T. No look-ahead.

4. **Stock-bond correlation**: This is the hardest series to construct. If daily bond return data is unavailable pre-1962, either start the correlation series later or use a simpler proxy (e.g., negative of yield changes as a directional proxy). Document any approximations.

5. **VIX prepending**: VIX starts Jan 1990. For earlier months, use realized volatility from daily S&P 500 returns (annualized 21-day rolling std). Splice at Jan 1990. The two series should be roughly comparable in level.

### Computational

- The similarity engine is O(T²) per month — for ~480 months (1985–2024) with ~600 historical months each, this is ~300K distance computations. Vectorize with numpy; it should run in seconds.
- Store all distances in a matrix for reuse across quantile robustness checks.

### Data Quality

- FRED copper series (`PCOPPUSDM`) is in USD/metric ton. The paper uses futures-adjusted prices. Levels don't matter because we take 12-month changes and z-score, but be aware the raw plots (Exhibit 2) will look different.
- Fama-French factors are in percent (e.g., 1.5 means 1.5%). Convert as needed for cumulative return calculations.

---

## Execution Order

Run these tasks sequentially. After each task, validate output before proceeding.

```
Task 1: Download and validate all 7 state variable series. Plot raw series (→ Exhibit 2).
Task 2: Implement z-score transformation. Plot raw vs. winsorized (→ Exhibit 3).
Task 3: Compute autocorrelations and cross-correlations (→ Exhibits 4, 5).
Task 4: Build similarity engine. Test on Jan 2009, Feb 2020, Apr 2020, Aug 2022 (→ Exhibits 6, 7, 8).
Task 5: Implement EWMA regime shift detection (→ Exhibit 9).
Task 6: Download Fama-French factors. Implement quintile backtest (→ Exhibit 10).
Task 7: Compute drawdowns (→ Exhibit 11).
Task 8: Run quantile and lookback robustness (→ Exhibits 12, 13).
Task 9: Run individual factor analysis (→ Exhibits A1, A2).
Task 10: Compute all summary statistics, alpha regressions.
Task 11: Generate final PDF paper with all exhibits embedded.
```

---

## Success Criteria

- [ ] All 7 state variables sourced and transformed correctly
- [ ] Similarity scores computed with proper 36-month mask
- [ ] Q1 portfolio outperforms Q5 portfolio
- [ ] Q1−Q5 spread has Sharpe ratio > 0.5 (paper reports 0.82; ours may differ with public data)
- [ ] Alpha of spread vs. LO is statistically significant (t-stat > 2)
- [ ] All 15+ exhibits produced at publication quality
- [ ] Final PDF paper is 15–25 pages, properly formatted, with all exhibits
- [ ] Robustness checks show results are not highly sensitive to parameter choices
