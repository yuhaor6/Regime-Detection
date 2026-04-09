"""
End-to-end regime detection and factor timing pipeline.

Usage:
    python main.py [--fred-key YOUR_KEY]
    # or set FRED_API_KEY in environment
"""

import argparse
import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader   import load_all_state_variables, load_fama_french
from src.transforms    import transform_all, autocorrelation_table, correlation_matrix
from src.similarity    import compute_distance_matrix, compute_ewma_regime_shift
from src.backtest      import build_signals, build_portfolios, robustness_quantiles, robustness_lookback
from src.evaluation    import alpha_tstat, performance_summary
from src.paper_writer  import build_paper
from src.exhibits      import (
    exhibit1_yearly_returns,
    exhibit2_raw_variables,
    exhibit3_transformed_variables,
    exhibit4_autocorrelation_table,
    exhibit5_correlation_heatmap,
    exhibit6_gfc,
    exhibit7_covid,
    exhibit8_inflation,
    exhibit9_ewma,
    exhibit10_quintile_returns,
    exhibit11_drawdown,
    exhibit12_quantile_robustness,
    exhibit13_lookback_robustness,
    exhibit_a1_individual_quintiles,
    exhibit_a2_individual_longshort,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR   = os.path.join(os.path.dirname(__file__), "figures")


def _cached(name, func, *args, force=False, **kwargs):
    path = os.path.join(CACHE_DIR, f"{name}.pkl")
    if not force and os.path.exists(path):
        print(f"  [cache] Loading {name}")
        with open(path, "rb") as f:
            return pickle.load(f)
    result = func(*args, **kwargs)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def main(fred_key=None, force_download=False):
    t0 = time.time()
    os.makedirs(FIG_DIR, exist_ok=True)

    print("\n=== Stage 1: Data Acquisition ===")
    state_vars = _cached("state_vars", load_all_state_variables, fred_key, force=force_download)
    ff = _cached("fama_french", load_fama_french, force=force_download)

    print(f"\nState variables shape: {state_vars.shape}")
    print(f"FF factors shape: {ff.shape}, cols: {list(ff.columns)}")
    print(f"FF sample: {ff.index[0].date()} to {ff.index[-1].date()}")

    print("\n=== Stage 2: Transformations ===")
    raw_z_df, winsor_df = transform_all(state_vars)

    acf_table = autocorrelation_table(winsor_df)
    corr_mat  = correlation_matrix(winsor_df)

    print("\nAutocorrelation table:")
    print(acf_table[["1-month", "3-month", "12-month", "3-year", "10-year"]].round(2).to_string())
    print("\nCross-correlation matrix:")
    print(corr_mat.round(2).to_string())

    print("\n=== Stage 3: Similarity Engine ===")
    print("  Computing distance matrix (may take ~30s) ...")
    dist_matrix = _cached("dist_matrix",
                           lambda: compute_distance_matrix(winsor_df, exclude_recent=36),
                           force=force_download)
    print(f"  Distance matrix shape: {dist_matrix.shape}")

    print("  Computing EWMA regime shift indicators ...")
    ewma_df = _cached("ewma_df",
                       lambda: compute_ewma_regime_shift(winsor_df),
                       force=force_download)

    print("\n=== Stage 4: Backtest ===")
    print("  Building quintile signals ...")
    signals = _cached("signals",
                       lambda: build_signals(dist_matrix, ff, n_quantiles=5),
                       force=force_download)

    print("  Constructing portfolios ...")
    portfolios = build_portfolios(signals, ff)

    lo_returns    = portfolios["lo"]
    spread        = portfolios["spread"]
    spread_scaled = portfolios["spread_scaled"]

    print("\n=== Stage 5: Performance Summary ===")
    for label, ret in [("Q1", portfolios["q1"]), ("Q5", portfolios["q5"]),
                        ("Spread (Q1-Q5)", spread), ("Long-only", lo_returns)]:
        bm = lo_returns if label != "Long-only" else None
        summ = performance_summary(ret.dropna(), bm.dropna() if bm is not None else None)
        print(f"\n  {label}:")
        for k, v in summ.items():
            print(f"    {k:35s}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")

    alpha, beta, tstat = alpha_tstat(spread.dropna(), lo_returns.dropna())
    print(f"\n  Spread vs LO: alpha={alpha*100:.2f}% p.a., beta={beta:.2f}, t-stat={tstat:.2f}")

    print("\n=== Stage 6: Robustness ===")
    print("  Quantile robustness ...")
    spread_by_nq = _cached("spread_by_nq",
                             lambda: robustness_quantiles(dist_matrix, ff),
                             force=force_download)

    print("  Lookback robustness ...")
    spread_by_lb = _cached("spread_by_lb",
                             lambda: robustness_lookback(state_vars, ff),
                             force=force_download)

    print("\n=== Stage 7: Generating Exhibits ===")

    print("  Exhibit 1 ...")
    exhibit1_yearly_returns(spread_scaled, fig_dir=FIG_DIR)

    print("  Exhibit 2 ...")
    exhibit2_raw_variables(state_vars, fig_dir=FIG_DIR)

    print("  Exhibit 3 ...")
    exhibit3_transformed_variables(raw_z_df, winsor_df, fig_dir=FIG_DIR)

    print("  Exhibit 4 ...")
    exhibit4_autocorrelation_table(acf_table, fig_dir=FIG_DIR)

    print("  Exhibit 5 ...")
    exhibit5_correlation_heatmap(corr_mat, fig_dir=FIG_DIR)

    print("  Exhibit 6 (GFC) ...")
    exhibit6_gfc(dist_matrix, fig_dir=FIG_DIR)

    print("  Exhibit 7 (COVID) ...")
    exhibit7_covid(dist_matrix, fig_dir=FIG_DIR)

    print("  Exhibit 8 (Inflation 2022) ...")
    exhibit8_inflation(dist_matrix, fig_dir=FIG_DIR)

    print("  Exhibit 9 (EWMA) ...")
    exhibit9_ewma(ewma_df, fig_dir=FIG_DIR)

    print("  Exhibit 10 (Quintile returns) ...")
    exhibit10_quintile_returns(portfolios, lo_returns, ff, fig_dir=FIG_DIR)

    print("  Exhibit 11 (Drawdown) ...")
    exhibit11_drawdown(spread, lo_returns, fig_dir=FIG_DIR)

    print("  Exhibit 12 (Quantile robustness) ...")
    exhibit12_quantile_robustness(spread_by_nq, fig_dir=FIG_DIR)

    print("  Exhibit 13 (Lookback robustness) ...")
    exhibit13_lookback_robustness(spread_by_lb, fig_dir=FIG_DIR)

    print("  Exhibit A1 (Individual factor quintiles) ...")
    exhibit_a1_individual_quintiles(signals, ff, fig_dir=FIG_DIR)

    print("  Exhibit A2 (Individual factor long-short) ...")
    exhibit_a2_individual_longshort(signals, ff, fig_dir=FIG_DIR)

    print("\n=== Stage 8: Building Paper PDF ===")
    paper_dir  = os.path.join(os.path.dirname(__file__), "paper")
    paper_path = os.path.join(paper_dir, "regime_paper.pdf")
    os.makedirs(paper_dir, exist_ok=True)
    build_paper(portfolios=portfolios, lo_returns=lo_returns,
                spread=spread, spread_scaled=spread_scaled,
                fig_dir=FIG_DIR, output_path=paper_path)
    print(f"  Paper saved to: {paper_path}")

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"Figures: {FIG_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fred-key", default=None)
    parser.add_argument("--force",    action="store_true", help="Force re-download (ignore cache)")
    args = parser.parse_args()

    key = args.fred_key or os.environ.get("FRED_API_KEY")
    if not key:
        print("ERROR: No FRED API key. Set FRED_API_KEY or pass --fred-key.")
        sys.exit(1)

    main(fred_key=key, force_download=args.force)
