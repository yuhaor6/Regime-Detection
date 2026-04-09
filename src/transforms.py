"""
Transformation pipeline for the macroeconomic state variables.
Each series goes through: 12-month diff, rolling 10yr std, z-score, winsorise at +-3.
"""

import numpy as np
import pandas as pd


def compute_zscore(series, diff_lag=12, lookback=120, cap=3.0):
    """Z-score the 12-month changes of a series, then winsorise."""
    diff = series.diff(diff_lag)
    roll_std = diff.rolling(lookback, min_periods=lookback).std()
    raw_z = diff / roll_std
    winsorised = raw_z.clip(-cap, cap)
    return raw_z, winsorised


def transform_all(df, diff_lag=12, lookback=120, cap=3.0):
    """Apply compute_zscore to every column. Returns (raw_z_df, winsor_df)."""
    raw_dict    = {}
    winsor_dict = {}
    for col in df.columns:
        raw_z, winsor = compute_zscore(df[col], diff_lag=diff_lag, lookback=lookback, cap=cap)
        raw_dict[col]    = raw_z
        winsor_dict[col] = winsor
    return pd.DataFrame(raw_dict, index=df.index), pd.DataFrame(winsor_dict, index=df.index)


_VAR_LABELS = {
    "sp500":           "Market",
    "yield_curve":     "Yield curve",
    "oil":             "Oil",
    "copper":          "Copper",
    "tbill":           "Monetary policy",
    "volatility":      "Volatility",
    "stock_bond_corr": "Stock-bond",
}


def autocorrelation_table(df):
    """Autocorrelations at lags 1, 3, 12, 36, 120 months."""
    lags       = [1, 3, 12, 36, 120]
    lag_labels = ["1-month", "3-month", "12-month", "3-year", "10-year"]

    rows = []
    for col in df.columns:
        s = df[col].dropna()
        acfs = {lbl: (s.autocorr(lag=lag) if len(s) > lag else np.nan)
                for lag, lbl in zip(lags, lag_labels)}
        row = {**acfs, "monthly mean": s.mean(), "std": s.std(), "frequency": "monthly"}
        rows.append((_VAR_LABELS.get(col, col), row))

    result = pd.DataFrame({name: data for name, data in rows}).T
    result.index.name = None
    return result


def correlation_matrix(df):
    """Cross-correlation matrix, ordered as in Exhibit 5."""
    renamed = df.rename(columns=_VAR_LABELS)
    order = ["Copper", "Monetary policy", "Oil", "Yield curve", "Stock-bond", "Volatility", "Market"]
    order = [c for c in order if c in renamed.columns]
    return renamed[order].corr().loc[order, order]
