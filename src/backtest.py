"""
Factor timing backtest using regime similarity quintiles.

For each month T and each factor:
  - Find the Q1 months (most similar to T from the distance matrix)
  - Average the factor return at i+1 across those months -> timing signal
  - Long if positive, short if negative
"""

import numpy as np
import pandas as pd


FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def _next_month_returns(ff, hist_months):
    next_months = hist_months + pd.offsets.MonthEnd(1)
    available   = [m for m in next_months if m in ff.index]
    if not available:
        return pd.Series(np.nan, index=FACTOR_COLS)
    return ff.loc[available, FACTOR_COLS].mean()


def build_signals(dist_matrix, ff, n_quantiles=5,
                  eval_start="1985-01-01", eval_end="2024-12-31"):
    """
    For each evaluation month and quantile, compute +1/-1 factor-timing signals.
    Returns dict mapping quantile -> DataFrame(dates x factors).
    """
    eval_dates = dist_matrix.index[
        (dist_matrix.index >= pd.Timestamp(eval_start)) &
        (dist_matrix.index <= pd.Timestamp(eval_end))
    ]

    signal_dfs = {q: pd.DataFrame(np.nan, index=eval_dates, columns=FACTOR_COLS)
                  for q in range(1, n_quantiles + 1)}

    for T in eval_dates:
        row = dist_matrix.loc[T].dropna().sort_values()
        n_valid = len(row)
        if n_valid < n_quantiles:
            continue

        q_size = n_valid / n_quantiles
        for q in range(1, n_quantiles + 1):
            lo = int(round((q - 1) * q_size))
            hi = min(int(round(q * q_size)), n_valid)
            q_months = row.iloc[lo:hi].index
            avg_ret  = _next_month_returns(ff, q_months)
            signal_dfs[q].loc[T] = np.sign(avg_ret).where(avg_ret != 0, 0)

    return signal_dfs


def build_portfolios(signals, ff, vol_target=0.15, vol_lookback=12):
    """
    Construct quintile portfolios and the Q1-Q5 spread.
    Signal at T is applied to the return at T+1.
    Returns dict: q1..q5, spread, lo, spread_scaled.
    """
    portfolios = {}

    for q, sig_df in signals.items():
        sig_shifted = sig_df.shift(1)
        ff_rets     = ff[FACTOR_COLS].reindex(sig_shifted.index)
        port_ret    = (sig_shifted * ff_rets / 100.0).mean(axis=1)
        port_ret.name = f"q{q}"
        portfolios[f"q{q}"] = port_ret

    ff_rets_lo = ff[FACTOR_COLS].reindex(portfolios["q1"].index) / 100.0
    lo = ff_rets_lo.mean(axis=1)
    lo.name = "lo"
    portfolios["lo"] = lo

    spread = portfolios["q1"] - portfolios[f"q{max(signals.keys())}"]
    spread.name = "spread"
    portfolios["spread"] = spread

    spread_scaled = _vol_scale(spread, vol_target=vol_target, lookback=vol_lookback)
    spread_scaled.name = "spread_scaled"
    portfolios["spread_scaled"] = spread_scaled

    return portfolios


def _vol_scale(returns, vol_target=0.15, lookback=12):
    realized_vol = returns.rolling(lookback, min_periods=3).std() * np.sqrt(12)
    scale = (vol_target / realized_vol).shift(1).clip(0.0, 3.0)
    return (returns * scale).dropna()


def robustness_quantiles(dist_matrix, ff, quantile_choices=None,
                         eval_start="1985-01-01", eval_end="2024-12-31"):
    """Spread series for different numbers of quantiles."""
    if quantile_choices is None:
        quantile_choices = [2, 3, 4, 5, 10, 20]

    results = {}
    for nq in quantile_choices:
        sigs  = build_signals(dist_matrix, ff, n_quantiles=nq,
                              eval_start=eval_start, eval_end=eval_end)
        ports = build_portfolios(sigs, ff)
        results[nq] = ports["q1"] - ports[f"q{nq}"]
    return results


def robustness_lookback(state_vars, ff, lookbacks=None, exclude_recent=36,
                        eval_start="1985-01-01", eval_end="2024-12-31"):
    """Spread series for different z-score lookback windows."""
    from src.transforms import transform_all
    from src.similarity import compute_distance_matrix

    if lookbacks is None:
        lookbacks = [12, 36, 60]

    results = {}
    for lb in lookbacks:
        _, winsor_df = transform_all(state_vars, lookback=lb)
        dist_mat = compute_distance_matrix(winsor_df, exclude_recent=exclude_recent)
        sigs  = build_signals(dist_mat, ff, n_quantiles=5,
                              eval_start=eval_start, eval_end=eval_end)
        ports = build_portfolios(sigs, ff)
        label = {12: "1-year lookback", 36: "3-year lookback", 60: "5-year lookback"}.get(lb, f"{lb}m")
        results[label] = ports["q1"] - ports["q5"]
    return results
