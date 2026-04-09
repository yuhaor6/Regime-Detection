"""
Euclidean-distance regime similarity engine.

For evaluation month T, the global score to historical month i is:
    d(T, i) = sqrt( sum_v (z_{T,v} - z_{i,v})^2 )

Recent months (T-35 to T-1) are excluded to avoid momentum contamination.
"""

import numpy as np
import pandas as pd


def compute_distance_matrix(winsor_df, exclude_recent=36):
    """
    Full T x T distance matrix with a recency exclusion mask.
    dist[T, i] is NaN when i is within the last `exclude_recent` months of T.
    """
    mat = winsor_df.dropna(how="all").values
    dates = winsor_df.dropna(how="all").index
    N = len(dates)

    # vectorised pairwise: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    sq_norms = np.nansum(mat ** 2, axis=1)
    mat_nz   = np.where(np.isnan(mat), 0, mat)
    dots     = mat_nz @ mat_nz.T
    sq_dist  = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2 * dots, 0.0)
    dist_full = np.sqrt(sq_dist)

    idx   = np.arange(N)
    valid = idx[None, :] <= idx[:, None] - exclude_recent
    return pd.DataFrame(np.where(valid, dist_full, np.nan), index=dates, columns=dates)


def get_similarity_scores(dist_matrix, eval_date):
    if eval_date not in dist_matrix.index:
        raise KeyError(f"{eval_date} not in distance matrix index.")
    return dist_matrix.loc[eval_date].dropna()


def assign_quintiles(dist_matrix, n_quantiles=5):
    def _quantile_row(row):
        valid = row.dropna()
        if len(valid) == 0:
            return row
        labels = pd.qcut(valid, q=n_quantiles, labels=False, duplicates="drop") + 1
        return labels.reindex(row.index)
    return dist_matrix.apply(_quantile_row, axis=1)


def compute_ewma_regime_shift(winsor_df, lookbacks_months=None):
    """
    EWMA of global distance scores, using beta = 1 - 1/n.
    A rising EWMA suggests the current environment is unusual relative to history.
    """
    if lookbacks_months is None:
        lookbacks_months = [12, 24, 36, 48]

    mat = winsor_df.dropna(how="all").values
    dates = winsor_df.dropna(how="all").index
    N = len(dates)

    sq_norms  = np.nansum(mat ** 2, axis=1)
    mat_nz    = np.where(np.isnan(mat), 0, mat)
    dots      = mat_nz @ mat_nz.T
    dist_full = np.sqrt(np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2 * dots, 0.0))

    results = {}
    for n in lookbacks_months:
        beta = 1.0 - 1.0 / n
        ewma_vals = []
        for T_idx in range(N):
            hist = dist_full[T_idx, :T_idx]
            if len(hist) == 0:
                ewma_vals.append(np.nan)
                continue
            age = np.arange(len(hist) - 1, -1, -1)   # 0 = most recent
            ewma_vals.append(np.average(hist, weights=beta ** age))
        label = {12: "1-year", 24: "2-year", 36: "3-year", 48: "4-year"}.get(n, f"{n}m")
        results[label] = pd.Series(ewma_vals, index=dates)

    return pd.DataFrame(results)
