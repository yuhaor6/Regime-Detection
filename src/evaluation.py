"""Performance metrics."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def sharpe_ratio(returns, annualise=12):
    if returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(annualise)


def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()


def drawdown_series(returns):
    cum = (1 + returns).cumprod()
    return (cum - cum.cummax()) / cum.cummax() * 100


def cumulative_return(returns, base=100.0):
    return base * (1 + returns).cumprod()


def alpha_tstat(strategy_returns, benchmark_returns):
    """OLS alpha and t-stat of strategy vs benchmark."""
    common = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(common) < 10:
        return np.nan, np.nan, np.nan
    y   = common.iloc[:, 0]
    x   = sm.add_constant(common.iloc[:, 1])
    res = sm.OLS(y, x).fit()
    alpha = res.params.iloc[0] * 12
    beta  = res.params.iloc[1]
    tstat = res.tvalues.iloc[0]
    return alpha, beta, tstat


def yearly_returns(returns):
    return returns.groupby(returns.index.year).apply(lambda r: (1 + r).prod() - 1)


def performance_summary(returns, benchmark=None):
    yr = yearly_returns(returns)
    pos_years = yr[yr > 0]
    neg_years = yr[yr <= 0]
    corr_to_bm = returns.corr(benchmark) if benchmark is not None else np.nan
    alpha, beta, tstat = (alpha_tstat(returns, benchmark)
                          if benchmark is not None else (np.nan, np.nan, np.nan))
    return {
        "Sharpe (annualised)":        sharpe_ratio(returns),
        "Ann. return (%)":            returns.mean() * 12 * 100,
        "Ann. volatility (%)":        returns.std() * np.sqrt(12) * 100,
        "Max drawdown (%)":           max_drawdown(returns) * 100,
        "% positive years":           len(pos_years) / len(yr) * 100 if len(yr) > 0 else np.nan,
        "Avg return | positive (%)":  pos_years.mean() * 100 if len(pos_years) > 0 else np.nan,
        "Avg return | negative (%)":  neg_years.mean() * 100 if len(neg_years) > 0 else np.nan,
        "Skewness":                   yr.skew(),
        "Correlation to LO":          corr_to_bm,
        "Alpha (ann., %)":            alpha * 100 if not np.isnan(alpha) else np.nan,
        "Alpha t-stat":               tstat,
    }
