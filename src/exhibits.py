"""
Publication-quality figures for the regime detection and factor timing analysis.
All exhibits saved as both PNG (300 dpi) and PDF to the figures/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.frameon":     True,
    "legend.framealpha":  0.8,
})

BLUE   = "#1f5fa6"
BLUE2  = "#5b9bd5"
ORANGE = "#f4a261"
GREEN  = "#52b788"
RED    = "#e63946"
GREY   = "#adb5bd"


def _save(fig, name, fig_dir="figures"):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def exhibit1_yearly_returns(spread_scaled, fig_dir="figures"):
    from src.evaluation import yearly_returns
    yr = yearly_returns(spread_scaled) * 100

    pos_mask = yr > 0
    neg_mask = ~pos_mask
    pct_pos  = pos_mask.sum() / len(yr) * 100
    avg_pos  = yr[pos_mask].mean()
    avg_neg  = yr[neg_mask].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(yr.index, yr.values,
           color=[BLUE if v > 0 else BLUE2 for v in yr.values],
           edgecolor="none", width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Yearly returns", fontsize=11, pad=6)
    ax.set_ylabel("% return")
    ax.set_xlabel("Year")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    textstr = (f"Positive years: {pct_pos:.0f}%\n"
               f"Avg | positive: {avg_pos:.1f}%\n"
               f"Avg | negative: {avg_neg:.1f}%")
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.tight_layout()
    _save(fig, "exhibit1_yearly_returns", fig_dir)


def exhibit2_raw_variables(raw_df, fig_dir="figures"):
    titles = {
        "sp500":           ("S&P500 Prices",                        "log(price)",       True),
        "yield_curve":     ("US 10-year yield minus 3-month yield", "%",                False),
        "oil":             ("Oil Prices",                           "US$ / barrel",     False),
        "copper":          ("Copper Prices",                        "Scaled price",     False),
        "tbill":           ("US 3-month yield",                     "%",                False),
        "volatility":      ("Volatility",                           "%",                False),
        "stock_bond_corr": ("Stock-Bond Correlation",               "3-yr Correlation", False),
    }

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes_flat = axes.flatten()

    for i, (col, (title, ylabel, log_scale)) in enumerate(titles.items()):
        ax = axes_flat[i]
        s  = raw_df[col].dropna()
        y  = np.log(s) if log_scale else s
        ax.plot(s.index, y.values, color=BLUE, linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(mpl.dates.YearLocator(10))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=0)

    axes_flat[-1].set_visible(False)
    fig.tight_layout(h_pad=2.5)
    _save(fig, "exhibit2_raw_variables", fig_dir)


def exhibit3_transformed_variables(raw_z_df, winsor_df, start_date="1963-01-01", fig_dir="figures"):
    titles = {
        "sp500":           "S&P500 Prices",
        "yield_curve":     "US 10-year yield minus 3-year yield",
        "oil":             "Oil Prices",
        "copper":          "Copper Prices",
        "tbill":           "US 3-month yield",
        "volatility":      "Volatility",
        "stock_bond_corr": "Stock-Bond Correlation",
    }

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes_flat = axes.flatten()

    for i, (col, title) in enumerate(titles.items()):
        ax = axes_flat[i]
        rz = raw_z_df[col].dropna().loc[start_date:]
        wz = winsor_df[col].dropna().loc[start_date:]
        ax.plot(rz.index, rz.values, color=BLUE,   linewidth=0.6, label="raw z-score", zorder=2)
        ax.plot(wz.index, wz.values, color=ORANGE, linewidth=0.8, linestyle="--",
                label="winsorized", zorder=3)
        ax.set_title(title)
        ax.set_ylabel("z-score")
        ax.legend(loc="upper right", fontsize=7)
        ax.xaxis.set_major_locator(mpl.dates.YearLocator(10))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=0)

    axes_flat[-1].set_visible(False)
    fig.tight_layout(h_pad=2.5)
    _save(fig, "exhibit3_transformed_variables", fig_dir)


def exhibit4_autocorrelation_table(acf_table, fig_dir="figures"):
    cols_display = ["1-month", "3-month", "12-month", "3-year", "10-year",
                    "monthly mean", "std", "frequency"]
    table_data = acf_table[cols_display].copy()

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")

    col_labels = ["", "1-mo", "3-mo", "12-mo", "3-yr", "10-yr", "Mean", "Std", "Freq"]
    rows = []
    for var, row in table_data.iterrows():
        r = [var]
        for c in cols_display:
            v = row[c]
            r.append(str(v) if c == "frequency" else (f"{v:.2f}" if not pd.isna(v) else "—"))
        rows.append(r)

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.6)

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#d0d8e8")
        cell.set_text_props(weight="bold")

    for i, row in enumerate(rows, start=1):
        for j, c in enumerate(cols_display, start=1):
            try:
                val = float(row[j])
                if val > 0.7:
                    tbl[i, j].set_facecolor("#d4e9c9")
                elif val < -0.2:
                    tbl[i, j].set_facecolor("#ffd6d6")
            except ValueError:
                pass

    ax.set_title("Exhibit 4. Persistence of the Economic State Variables",
                 fontsize=10, pad=6, loc="left")
    fig.tight_layout()
    _save(fig, "exhibit4_autocorrelation_table", fig_dir)


def exhibit5_correlation_heatmap(corr_matrix, fig_dir="figures"):
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = sns.diverging_palette(220, 20, s=80, l=50, as_cmap=True)
    sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f", cmap=cmap,
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlations", fontsize=11, pad=8, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, "exhibit5_correlation_heatmap", fig_dir)


def _similarity_chart(dist_row, ref_date, pct_selected=0.15, exclude_recent=36,
                      title="", inflation_periods=None, fig_dir="figures", save_name="similarity_chart"):
    valid     = dist_row.dropna()
    n_select  = max(1, int(len(valid) * pct_selected))
    threshold = valid.nsmallest(n_select).max()
    selected_mask = (dist_row <= threshold) & dist_row.notna()
    mask_start    = ref_date - pd.DateOffset(months=exclude_recent)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dist_row.index, dist_row.values, color=BLUE, linewidth=0.7,
            linestyle="--", label="Global Score", zorder=3)

    for date in dist_row[selected_mask].index:
        ax.axvline(date, color=GREEN, alpha=0.5, linewidth=2.5, zorder=2)
    ax.axvspan(mask_start, ref_date, alpha=0.2, color=GREY, zorder=1, label="Masked period")

    if inflation_periods:
        first = True
        for start, end in inflation_periods:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.25, color="#e9c46a", zorder=0,
                       label="Inflation periods" if first else None)
            first = False

    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Global score")

    ax.annotate("Less similar", xy=(0, 1), xycoords="axes fraction",
                fontsize=7, color=GREY, ha="left", va="top",
                xytext=(-38, 0), textcoords="offset points")
    ax.annotate("More similar", xy=(0, 0), xycoords="axes fraction",
                fontsize=7, color=GREY, ha="left", va="bottom",
                xytext=(-38, 0), textcoords="offset points")

    handles = [
        mpatches.Patch(color=GREEN, alpha=0.6, label="Selected similar months"),
        mpatches.Patch(color=GREY,  alpha=0.4, label="Masked period"),
        plt.Line2D([0], [0], color=BLUE, linestyle="--", linewidth=0.8, label="Global Score"),
    ]
    if inflation_periods:
        handles.insert(0, mpatches.Patch(color="#e9c46a", alpha=0.5, label="Inflation periods"))
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=2)

    ax.xaxis.set_major_locator(mpl.dates.YearLocator(5))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, save_name, fig_dir)


def exhibit6_gfc(dist_matrix, fig_dir="figures"):
    ref = pd.Timestamp("2009-01-31")
    if ref not in dist_matrix.index:
        ref = dist_matrix.index[dist_matrix.index.get_indexer([ref], method="nearest")[0]]
    _similarity_chart(dist_matrix.loc[ref], ref,
                      title="Global similarity score calculated for January 2009",
                      save_name="exhibit6_gfc", fig_dir=fig_dir)


def exhibit7_covid(dist_matrix, fig_dir="figures"):
    for label, date_str in [("February 2020", "2020-02-29"), ("April 2020", "2020-04-30")]:
        ref = pd.Timestamp(date_str)
        if ref not in dist_matrix.index:
            ref = dist_matrix.index[dist_matrix.index.get_indexer([ref], method="nearest")[0]]
        safe = label.lower().replace(" ", "_")
        _similarity_chart(dist_matrix.loc[ref], ref,
                          title=f"Global similarity score calculated for {label}",
                          save_name=f"exhibit7_covid_{safe}", fig_dir=fig_dir)


def exhibit8_inflation(dist_matrix, fig_dir="figures"):
    ref = pd.Timestamp("2022-08-31")
    if ref not in dist_matrix.index:
        ref = dist_matrix.index[dist_matrix.index.get_indexer([ref], method="nearest")[0]]

    inflation_periods = [
        ("1941-04-01", "1942-05-31"),
        ("1946-03-01", "1947-03-31"),
        ("1950-08-01", "1951-02-28"),
        ("1966-02-01", "1970-01-31"),
        ("1972-07-01", "1974-12-31"),
        ("1977-02-01", "1980-03-31"),
        ("1987-02-01", "1990-11-30"),
        ("2007-09-01", "2008-07-31"),
        ("2022-01-01", "2022-10-31"),
    ]
    _similarity_chart(dist_matrix.loc[ref], ref,
                      title="Global similarity score calculated for August 2022",
                      inflation_periods=inflation_periods,
                      save_name="exhibit8_inflation", fig_dir=fig_dir)


def exhibit9_ewma(ewma_df, fig_dir="figures"):
    fig, ax = plt.subplots(figsize=(11, 4))
    colours  = [BLUE, GREEN, ORANGE, RED]
    mean_ewma = ewma_df.mean(axis=1)

    for col, colour in zip(ewma_df.columns, colours):
        ax.plot(ewma_df.index, ewma_df[col].values, color=colour, linewidth=0.9, label=col)
    ax.plot(mean_ewma.index, mean_ewma.values, color="black", linewidth=1.2,
            linestyle="--", label="Mean of four approaches")

    key_dates = {
        "May 83": "1983-05-31", "Dec 90": "1990-12-31",
        "Jan 09": "2009-01-31", "May 20": "2020-05-31",
        "Oct 22": "2022-10-31", "May 23": "2023-05-31",
    }
    for label, date_str in key_dates.items():
        dt  = pd.Timestamp(date_str)
        idx = mean_ewma.index.get_indexer([dt], method="nearest")[0]
        ax.annotate(label, xy=(mean_ewma.index[idx], mean_ewma.iloc[idx]),
                    xytext=(0, 12), textcoords="offset points", fontsize=7, ha="center",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))

    ax.set_ylabel("X-year EWMA of global score")
    ax.set_title("Plot of EWMAs")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(5))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, "exhibit9_ewma", fig_dir)


def exhibit10_quintile_returns(portfolios, lo_returns, ff, fig_dir="figures"):
    from src.evaluation import sharpe_ratio, cumulative_return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    q_colours = [BLUE, GREEN, ORANGE, RED, GREY]

    for q, colour in zip(range(1, 6), q_colours):
        ret  = portfolios[f"q{q}"].dropna()
        sr   = sharpe_ratio(ret)
        corr = ret.corr(lo_returns.reindex(ret.index))
        cum  = cumulative_return(ret)
        ax1.plot(cum.index, cum.values, color=colour, linewidth=0.9,
                 label=f"Q{q} (SR: {sr:.2f}, Corr: {corr:.2f})")

    cum_lo = cumulative_return(lo_returns.dropna())
    sr_lo  = sharpe_ratio(lo_returns.dropna())
    ax1.plot(cum_lo.index, cum_lo.values, color="black", linestyle="--",
             linewidth=1.0, label=f"LO Model (SR: {sr_lo:.2f})")
    ax1.set_title("Similarity quintiles for Fama-French Factors")
    ax1.set_ylabel("Cumulative return")
    ax1.legend(fontsize=7)
    ax1.xaxis.set_major_locator(mpl.dates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax1.tick_params(axis="x", rotation=30)

    spread = (portfolios["q1"] - portfolios["q5"]).dropna()
    sr_sp  = sharpe_ratio(spread)
    corr_sp = spread.corr(lo_returns.reindex(spread.index))
    cum_sp  = cumulative_return(spread)
    ax2.plot(cum_sp.index, cum_sp.values, color="black", linewidth=1.0,
             label=f"1st - 5th (SR: {sr_sp:.2f}, Corr to LO: {corr_sp:.2f})")
    ax2.set_title("Similar minus dissimilar spread: Fama-French Factors")
    ax2.set_ylabel("Cumulative return")
    ax2.legend(fontsize=7)
    ax2.xaxis.set_major_locator(mpl.dates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    _save(fig, "exhibit10_quintile_returns", fig_dir)


def exhibit11_drawdown(spread_returns, lo_returns, fig_dir="figures"):
    from src.evaluation import drawdown_series

    dd_spread = drawdown_series(spread_returns.dropna())
    dd_lo     = drawdown_series(lo_returns.dropna())

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd_lo.index,     dd_lo.values,     0, color=BLUE, alpha=0.4, label="LO Drawdown")
    ax.fill_between(dd_spread.index, dd_spread.values, 0, color=RED,  alpha=0.5, label="Model Drawdown")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("% of capital")
    ax.set_title("Drawdown profile")
    ax.legend()
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(5))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, "exhibit11_drawdown", fig_dir)


def exhibit12_quantile_robustness(spread_by_nq, fig_dir="figures"):
    from src.evaluation import sharpe_ratio, cumulative_return

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(spread_by_nq)))

    for (nq, ret), colour in zip(sorted(spread_by_nq.items()), colours):
        ret = ret.dropna()
        sr  = sharpe_ratio(ret)
        cum = cumulative_return(ret)
        ax.plot(cum.index, cum.values, color=colour, linewidth=0.9,
                label=f"{nq} quantiles, SR: {sr:.2f}")

    ax.set_title("Fama-French quantile analysis (top minus bottom)")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(5))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, "exhibit12_quantile_robustness", fig_dir)


def exhibit13_lookback_robustness(spread_by_lookback, fig_dir="figures"):
    from src.evaluation import cumulative_return

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = [BLUE, RED, GREEN]

    for (label, ret), colour in zip(spread_by_lookback.items(), colours):
        ret = ret.dropna()
        cum = cumulative_return(ret)
        ax.plot(cum.index, cum.values, color=colour, linewidth=1.0, label=label)

    ax.set_title("Sensitivity to different z-score lookbacks")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(5))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, "exhibit13_lookback_robustness", fig_dir)


def exhibit_a1_individual_quintiles(signals, ff, fig_dir="figures"):
    from src.backtest import FACTOR_COLS
    from src.evaluation import sharpe_ratio, cumulative_return

    factor_labels = {
        "Mkt-RF": "Market (excess) returns",
        "SMB":    "Size returns",
        "HML":    "Value returns",
        "RMW":    "Profitability returns",
        "CMA":    "Investment returns",
        "Mom":    "Momentum returns",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()
    q_colours = [BLUE, GREEN, ORANGE, RED, GREY]

    for i, factor in enumerate(FACTOR_COLS):
        ax = axes_flat[i]
        factor_ret = ff[factor].dropna() / 100.0

        for q, colour in zip(range(1, 6), q_colours):
            sig = signals[q][factor].dropna().shift(1)
            ret = (sig * factor_ret.reindex(sig.index)).dropna()
            sr   = sharpe_ratio(ret)
            corr = ret.corr(factor_ret.reindex(ret.index))
            cum  = cumulative_return(ret)
            ax.plot(cum.index, cum.values, color=colour, linewidth=0.7,
                    label=f"Q{q} (SR: {sr:.2f}, corr: {corr:.2f})")

        cum_lo = cumulative_return(factor_ret.dropna())
        sr_lo  = sharpe_ratio(factor_ret.dropna())
        ax.plot(cum_lo.index, cum_lo.values, color="black", linestyle="--",
                linewidth=0.8, label=f"LO model (SR: {sr_lo:.2f})")
        ax.set_title(factor_labels.get(factor, factor))
        ax.legend(fontsize=5.5)
        ax.xaxis.set_major_locator(mpl.dates.YearLocator(10))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    _save(fig, "exhibit_a1_individual_quintiles", fig_dir)


def exhibit_a2_individual_longshort(signals, ff, fig_dir="figures"):
    from src.backtest import FACTOR_COLS
    from src.evaluation import sharpe_ratio, cumulative_return

    factor_labels = {
        "Mkt-RF": "Market (excess) returns",
        "SMB":    "Size returns",
        "HML":    "Value returns",
        "RMW":    "Profitability returns",
        "CMA":    "Investment returns",
        "Mom":    "Momentum returns",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat   = axes.flatten()
    n_quantiles = max(signals.keys())

    for i, factor in enumerate(FACTOR_COLS):
        ax         = axes_flat[i]
        factor_ret = ff[factor].dropna() / 100.0

        def _q_ret(q):
            sig = signals[q][factor].dropna().shift(1)
            return (sig * factor_ret.reindex(sig.index)).dropna()

        q1  = _q_ret(1)
        qN  = _q_ret(n_quantiles)
        idx = q1.index.intersection(qN.index)
        spread  = q1.loc[idx] - qN.loc[idx]
        sr_sp   = sharpe_ratio(spread)
        corr_sp = spread.corr(factor_ret.reindex(spread.index))
        cum_sp  = cumulative_return(spread)

        ax.plot(cum_sp.index, cum_sp.values, color=BLUE, linewidth=0.9,
                label=f"1st - 5th (SR: {sr_sp:.2f}, corr: {corr_sp:.2f})")

        cum_lo = cumulative_return(factor_ret.dropna())
        sr_lo  = sharpe_ratio(factor_ret.dropna())
        ax.plot(cum_lo.index, cum_lo.values, color="black", linestyle="--",
                linewidth=0.8, label=f"LO model (SR: {sr_lo:.2f})")

        ax.axhline(0, color=GREY, linewidth=0.5)
        ax.set_title(factor_labels.get(factor, factor))
        ax.legend(fontsize=7)
        ax.xaxis.set_major_locator(mpl.dates.YearLocator(10))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    _save(fig, "exhibit_a2_individual_longshort", fig_dir)
