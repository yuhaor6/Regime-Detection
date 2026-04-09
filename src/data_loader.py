"""
Data acquisition for macroeconomic state variables and Fama-French factors.

State variables:
    sp500, yield_curve, oil, copper, tbill, volatility, stock_bond_corr
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred


def _fred(api_key=None):
    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        raise ValueError("Set FRED_API_KEY environment variable or pass api_key.")
    return Fred(api_key=key)


def _to_month_end(s):
    return s.resample("ME").last().dropna()


def _to_month_end_mean(s):
    return s.resample("ME").mean().dropna()


def load_sp500(fred):
    """S&P 500 index level, monthly. Yahoo Finance goes back to 1927."""
    try:
        raw = yf.download("^GSPC", start="1920-01-01", auto_adjust=True, progress=False)["Close"]
        raw = raw.squeeze().dropna()
        s = _to_month_end(raw)
        if len(s) < 200:
            raise ValueError("Insufficient data from Yahoo")
    except Exception:
        raw = fred.get_series("SP500")
        s = _to_month_end(raw)
    s.name = "sp500"
    return s


def load_yield_curve(fred):
    """10-year minus 3-month Treasury yield spread."""
    gs10  = fred.get_series("GS10")
    tb3ms = fred.get_series("TB3MS")
    spread = (_to_month_end(gs10) - _to_month_end(tb3ms)).dropna()
    spread.name = "yield_curve"
    return spread


def load_oil(fred):
    """
    WTI crude oil price, USD/barrel.
    Post-1986: MCOILWTICO. Pre-1986: WPU0561 (PPI crude, corr=0.997 with WTI),
    scaled to match at the splice point.
    """
    wti = fred.get_series("MCOILWTICO").dropna()
    wti.index = pd.to_datetime(wti.index) + pd.offsets.MonthEnd(0)

    ppi = fred.get_series("WPU0561").dropna()
    ppi.index = pd.to_datetime(ppi.index) + pd.offsets.MonthEnd(0)

    splice_start = pd.Timestamp("1986-01-31")
    overlap = pd.concat([ppi, wti], axis=1).dropna()
    overlap = overlap.loc[splice_start:"1987-12-31"]
    scale = overlap.iloc[:, 1].mean() / overlap.iloc[:, 0].mean() if len(overlap) > 0 else 1.0
    ppi_scaled = ppi * scale

    pre  = ppi_scaled[ppi_scaled.index < splice_start]
    post = wti[wti.index >= splice_start]
    s = pd.concat([pre, post]).sort_index()
    s.name = "oil"
    return s


def load_copper(fred):
    """
    Copper price, monthly.
    Post-1992: PCOPPUSDM (World Bank). Pre-1992: WPU102502 (PPI copper, from 1954),
    scaled to match at 1992.
    """
    modern = fred.get_series("PCOPPUSDM").dropna()
    modern.index = pd.to_datetime(modern.index) + pd.offsets.MonthEnd(0)

    ppi = fred.get_series("WPU102502").dropna()
    ppi.index = pd.to_datetime(ppi.index) + pd.offsets.MonthEnd(0)

    splice_start = pd.Timestamp("1992-01-31")
    overlap = pd.concat([ppi, modern], axis=1).dropna()
    overlap = overlap.loc[splice_start:"1993-12-31"]
    scale = overlap.iloc[:, 1].mean() / overlap.iloc[:, 0].mean() if len(overlap) > 0 else 1.0
    ppi_scaled = ppi * scale

    pre  = ppi_scaled[ppi_scaled.index < splice_start]
    post = modern[modern.index >= splice_start]
    s = pd.concat([pre, post]).sort_index()
    s.name = "copper"
    return s


def load_tbill(fred):
    """US 3-month T-bill yield."""
    raw = fred.get_series("TB3MS")
    s = _to_month_end(raw)
    s.name = "tbill"
    return s


def load_volatility(fred):
    """
    Equity volatility: VIX post-Jan 1990, annualised 21-day realised vol pre-1990.
    """
    vix_daily = yf.download("^VIX", start="1989-12-01", auto_adjust=True, progress=False)["Close"]
    vix_daily = vix_daily.squeeze().dropna()
    vix_monthly = _to_month_end(vix_daily)

    sp_daily = yf.download("^GSPC", start="1920-01-01", auto_adjust=True, progress=False)["Close"]
    sp_daily = sp_daily.squeeze().dropna()
    log_ret  = np.log(sp_daily / sp_daily.shift(1)).dropna()
    rvol_daily = log_ret.rolling(21).std() * np.sqrt(252) * 100
    rvol_monthly = _to_month_end(rvol_daily)

    cutoff = pd.Timestamp("1990-01-31")
    pre  = rvol_monthly[rvol_monthly.index < cutoff]
    post = vix_monthly[vix_monthly.index >= cutoff]
    s = pd.concat([pre, post]).sort_index()
    s.name = "volatility"
    return s


def load_stock_bond_correlation(fred):
    """
    Rolling 36-month correlation of daily S&P 500 returns vs. duration-approximated
    bond returns (-8 * delta_yield), resampled to month-end.
    """
    dgs10 = fred.get_series("DGS10").dropna()
    dgs10.index = pd.to_datetime(dgs10.index)

    sp_daily = yf.download("^GSPC", start="1950-01-01", auto_adjust=True, progress=False)["Close"]
    sp_daily = sp_daily.squeeze().dropna()
    sp_ret = np.log(sp_daily / sp_daily.shift(1))

    bond_ret = -8.0 * dgs10.diff() / 100.0
    combined = pd.DataFrame({"stock": sp_ret, "bond": bond_ret}).dropna()

    roll_corr = combined["stock"].rolling(756).corr(combined["bond"]).dropna()
    s = _to_month_end(roll_corr)
    s.name = "stock_bond_corr"
    return s


def load_fama_french():
    """
    Download Fama-French 5 factors + Momentum from Ken French's data library.
    Returns monthly DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    Values in percent (e.g. 1.0 = 1%).
    """
    url5 = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    )
    url_mom = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Momentum_Factor_CSV.zip"
    )

    def _parse_ff_zip(url, skip_rows=0):
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            fname = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
            with z.open(fname) as f:
                raw = f.read().decode("utf-8", errors="replace")

        lines = raw.splitlines()
        data_lines = []
        header = None
        started = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if started:
                    break
                continue
            if stripped.startswith("2") or stripped.startswith("1"):
                started = True
                data_lines.append(stripped)
            elif started:
                break
            else:
                if "Mkt" in stripped or "SMB" in stripped or "Mom" in stripped or "WML" in stripped:
                    header = stripped

        if header is None:
            for line in lines:
                s = line.strip()
                if "Mkt" in s or "SMB" in s or "Mom" in s or "WML" in s:
                    header = s
                    break

        cols = [c.strip() for c in header.split(",") if c.strip()]
        rows = []
        for line in data_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= len(cols) + 1:
                rows.append(parts[: len(cols) + 1])

        df = pd.DataFrame(rows, columns=["date"] + cols)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m")
        df = df.set_index("date")
        df.index = df.index + pd.offsets.MonthEnd(0)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(how="all")

    ff5 = _parse_ff_zip(url5)
    mom = _parse_ff_zip(url_mom)

    mom_col = [c for c in mom.columns if c.upper() in ("MOM", "WML", "UMD", "PR1YR")][0]
    mom = mom[[mom_col]].rename(columns={mom_col: "Mom"})

    ff = ff5.join(mom, how="left")

    rename = {}
    for c in ff.columns:
        cu = c.upper().replace("-", "").replace(" ", "")
        if cu in ("MKTRF", "RMRF"):
            rename[c] = "Mkt-RF"
        elif cu == "SMB":
            rename[c] = "SMB"
        elif cu == "HML":
            rename[c] = "HML"
        elif cu == "RMW":
            rename[c] = "RMW"
        elif cu == "CMA":
            rename[c] = "CMA"
        elif cu == "RF":
            rename[c] = "RF"
        elif cu == "MOM":
            rename[c] = "Mom"
    ff = ff.rename(columns=rename)

    keep = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
    return ff[[c for c in keep if c in ff.columns]]


def load_all_state_variables(api_key=None):
    """Load and align all 7 state variables at monthly frequency."""
    fred = _fred(api_key)

    print("Loading S&P 500 ...", flush=True)
    sp   = load_sp500(fred)
    print("Loading yield curve ...", flush=True)
    yc   = load_yield_curve(fred)
    print("Loading oil prices ...", flush=True)
    oil  = load_oil(fred)
    print("Loading copper prices ...", flush=True)
    cop  = load_copper(fred)
    print("Loading T-bill yield ...", flush=True)
    tb   = load_tbill(fred)
    print("Loading volatility ...", flush=True)
    vol  = load_volatility(fred)
    print("Loading stock-bond correlation ...", flush=True)
    sbc  = load_stock_bond_correlation(fred)

    df = pd.DataFrame({
        "sp500":           sp,
        "yield_curve":     yc,
        "oil":             oil,
        "copper":          cop,
        "tbill":           tb,
        "volatility":      vol,
        "stock_bond_corr": sbc,
    })

    print("\n=== State Variable Validation ===")
    for col in df.columns:
        s = df[col].dropna()
        print(f"  {col:20s}  start={s.index[0].date()}  end={s.index[-1].date()}  n={len(s)}  NaN={df[col].isna().sum()}")

    return df
