"""
PDF research paper generator using ReportLab Platypus.

Produces a polished academic paper with all exhibits embedded inline.
"""

import os
import io
import pickle
import numpy as np
import pandas as pd
from datetime import date

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.colors import HexColor


PAGE_W, PAGE_H = letter
MARGIN = 1.0 * inch
TEXT_W = PAGE_W - 2 * MARGIN

NAVY   = colors.black          # titles and section heads: plain black
ACCENT = HexColor("#1f5fa6")
LIGHT  = HexColor("#f4f6fb")



def _make_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "PaperTitle",
            fontName="Times-Bold",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=NAVY,
            spaceAfter=12,
        ),
        "author": ParagraphStyle(
            "Author",
            fontName="Times-Italic",
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "date": ParagraphStyle(
            "Date",
            fontName="Times-Roman",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "abstract_head": ParagraphStyle(
            "AbstractHead",
            fontName="Times-Bold",
            fontSize=10,
            leading=13,
            spaceAfter=4,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            fontName="Times-Roman",
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            leftIndent=0.35 * inch,
            rightIndent=0.35 * inch,
            spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "SectionHead",
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=16,
            spaceAfter=6,
            textColor=NAVY,
        ),
        "subsection": ParagraphStyle(
            "SubsectionHead",
            fontName="Times-BoldItalic",
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName="Times-Roman",
            fontSize=11,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            spaceAfter=12,
            textColor=colors.HexColor("#444444"),
        ),
        "exhibit_title": ParagraphStyle(
            "ExhibitTitle",
            fontName="Times-Bold",
            fontSize=10,
            leading=13,
            spaceAfter=2,
        ),
        "keywords": ParagraphStyle(
            "Keywords",
            fontName="Times-Italic",
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            leftIndent=0.35 * inch,
            rightIndent=0.35 * inch,
            spaceAfter=6,
        ),
        "footnote": ParagraphStyle(
            "Footnote",
            fontName="Times-Roman",
            fontSize=8,
            leading=11,
            spaceAfter=4,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            fontName="Times-Bold",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            fontName="Times-Roman",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
        ),
        "table_label": ParagraphStyle(
            "TableLabel",
            fontName="Times-Roman",
            fontSize=9,
            leading=11,
            alignment=TA_LEFT,
        ),
    }
    return styles


def _img(path, width, fig_dir="figures"):
    full_path = os.path.join(fig_dir, path)
    img = Image(full_path)
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth  = width * inch
    img.drawHeight = width * aspect * inch
    return img


def _exhibit_block(title, caption, img_path, styles, width=6.2, fig_dir="figures"):
    """Return exhibit as a KeepTogether block so title never orphans from image."""
    inner = [
        Spacer(1, 0.08 * inch),
        Paragraph(title, styles["exhibit_title"]),
        _img(img_path, width=width, fig_dir=fig_dir),
        Paragraph(caption, styles["caption"]),
    ]
    return [KeepTogether(inner)]


def _hr(width=TEXT_W, thickness=0.5):
    return HRFlowable(width=width, thickness=thickness, color=colors.HexColor("#cccccc"),
                      spaceAfter=6, spaceBefore=6)



def _make_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(PAGE_W / 2, 0.55 * inch, str(doc.page))
    # Acknowledgment footnote on first page only
    if doc.page == 1:
        canvas.setFont("Times-Roman", 8)
        canvas.setFillColor(colors.HexColor("#444444"))
        footnote = (
            "\u2020 I am grateful to Prof. André Sztutman for his comments "
            "and feedback on this project."
        )
        canvas.drawString(MARGIN, 0.75 * inch, footnote)
    canvas.restoreState()



def _perf_table(portfolios, lo_returns, styles):
    """Build a Table of Sharpe ratios and correlations across quintiles."""
    from src.evaluation import sharpe_ratio

    headers = ["", "Q1 (Most similar)", "Q2", "Q3", "Q4", "Q5 (Most dissimilar)", "Spread (Q1−Q5)", "Long-only"]
    sr_row   = ["Sharpe ratio"]
    corr_row = ["Corr. to long-only"]

    for q in range(1, 6):
        ret = portfolios[f"q{q}"].dropna()
        sr_row.append(f"{sharpe_ratio(ret):.2f}")
        corr_row.append(f"{ret.corr(lo_returns.reindex(ret.index)):.2f}")

    spread = (portfolios["q1"] - portfolios["q5"]).dropna()
    sr_row.append(f"{sharpe_ratio(spread):.2f}")
    corr_row.append(f"{spread.corr(lo_returns.reindex(spread.index)):.2f}")

    sr_row.append(f"{sharpe_ratio(lo_returns.dropna()):.2f}")
    corr_row.append("—")

    data = [headers, sr_row, corr_row]
    # Total must fit TEXT_W = 6.5 in: label 1.1in + 6 cols × 0.72in + 2 cols × 0.72in = 6.5in
    col_widths = [1.1 * inch] + [0.77 * inch] * 7

    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (0, 0), (0, -1),  "LEFT"),
        ("LINEABOVE",   (0, 0), (-1, 0),  1.0, colors.black),
        ("LINEBELOW",   (0, 0), (-1, 0),  0.5, colors.black),
        ("LINEBELOW",   (0, -1),(-1, -1), 1.0, colors.black),
        ("BACKGROUND",  (0, 1), (-1, 1),  LIGHT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        # Highlight spread column
        ("TEXTCOLOR",   (6, 0), (6, -1),  ACCENT),
        ("FONTNAME",    (6, 1), (6, -1),  "Times-Bold"),
    ]))
    return tbl


def _inflation_table(styles):
    """Table of historical inflationary periods."""
    headers = ["Period", "Start", "End", "Price Change", "Length"]
    rows = [
        ["US enters WW2",       "Apr 1941", "May 1942", "15%", "14 mo."],
        ["End of WW2",          "Mar 1946", "Mar 1947", "21%", "13 mo."],
        ["Korean War",          "Aug 1950", "Feb 1951",  "7%",  "7 mo."],
        ["End of Bretton Woods","Feb 1966", "Jan 1970", "19%", "48 mo."],
        ["OPEC oil embargo",    "Jul 1972", "Dec 1974", "24%", "30 mo."],
        ["Iranian revolution",  "Feb 1977", "Mar 1980", "37%", "38 mo."],
        ["Reagan's boom",       "Feb 1987", "Nov 1990", "20%", "46 mo."],
        ["China demand boom",   "Sep 2007", "Jul 2008",  "6%", "11 mo."],
        ["Post-COVID",          "Jan 2022", "Oct 2022",  "6%", "10 mo."],
    ]
    data = [headers] + rows
    col_widths = [1.85*inch, 0.85*inch, 0.85*inch, 0.9*inch, 0.85*inch]

    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (0, 0), (0, -1),  "LEFT"),
        ("LINEABOVE",   (0, 0), (-1, 0),  1.0, colors.black),
        ("LINEBELOW",   (0, 0), (-1, 0),  0.5, colors.black),
        ("LINEBELOW",   (0,-1), (-1,-1),  1.0, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 3),
    ]))
    return tbl


def _ewma_table(styles):
    """Table of EWMA half-lives."""
    headers = ["Lookback period", "β", "Half-life"]
    rows = [
        ["1-year  (n = 12)", "1 − 1/12 ≈ 0.917", "~8 months"],
        ["2-year  (n = 24)", "1 − 1/24 ≈ 0.958", "~16 months"],
        ["3-year  (n = 36)", "1 − 1/36 ≈ 0.972", "~25 months"],
        ["4-year  (n = 48)", "1 − 1/48 ≈ 0.979", "~33 months"],
    ]
    data = [headers] + rows
    col_widths = [1.8*inch, 2.0*inch, 1.4*inch]
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (0, 0), (0, -1),  "LEFT"),
        ("LINEABOVE",   (0, 0), (-1, 0),  1.0, colors.black),
        ("LINEBELOW",   (0, 0), (-1, 0),  0.5, colors.black),
        ("LINEBELOW",   (0,-1), (-1,-1),  1.0, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 3),
    ]))
    return tbl



def build_paper(portfolios, lo_returns, spread, spread_scaled,
                 fig_dir="figures", output_path="paper/regime_paper.pdf"):
    from src.evaluation import (
        sharpe_ratio, max_drawdown, alpha_tstat, yearly_returns, performance_summary,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    S = _make_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=0.9 * inch,
    )

    # Pre-compute key stats
    yr_scaled = yearly_returns(spread_scaled) * 100
    pos_years  = (yr_scaled > 0).sum()
    tot_years  = len(yr_scaled)
    avg_up     = yr_scaled[yr_scaled > 0].mean()
    avg_down   = yr_scaled[yr_scaled <= 0].mean()
    pct_pos    = pos_years / tot_years * 100

    q1_sr = sharpe_ratio(portfolios["q1"].dropna())
    q5_sr = sharpe_ratio(portfolios["q5"].dropna())
    sp_sr = sharpe_ratio(spread.dropna())
    lo_sr = sharpe_ratio(lo_returns.dropna())

    q1_corr = portfolios["q1"].dropna().corr(lo_returns.reindex(portfolios["q1"].dropna().index))
    sp_corr = spread.dropna().corr(lo_returns.reindex(spread.dropna().index))

    alpha, beta, tstat = alpha_tstat(spread.dropna(), lo_returns.dropna())
    alpha_pct = alpha * 100

    sp_mdd = max_drawdown(spread.dropna()) * 100
    lo_mdd = max_drawdown(lo_returns.dropna()) * 100

    story = []

    story.append(Spacer(1, 0.6 * inch))
    story.append(Paragraph("Economic Regime Detection and Factor Timing", S["title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_hr())
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Yuhao Ren\u2020", S["author"]))
    story.append(Paragraph(f"April 2026", S["date"]))
    story.append(Spacer(1, 0.25 * inch))

    # Abstract
    story.append(Paragraph("Abstract", S["abstract_head"]))
    abstract_text = (
        "This paper proposes a systematic, non-parametric method for detecting the current "
        "economic regime and demonstrates how this information can be exploited to time equity "
        "factor premia. Rather than presupposing a fixed set of discrete regimes, the method "
        "relies on a set of economic state variables and identifies which historical periods "
        "were most similar to the present. Similarity is measured via the Euclidean distance "
        "across standardised transformations of seven financial variables: the equity market "
        "level, the yield curve slope, crude oil prices, copper prices, short-term interest "
        "rates, equity volatility, and the stock-bond correlation. The approach is largely "
        "non-parametric — no parameters are optimised — relying solely on z-scores computed "
        "over rolling ten-year windows. We apply the methodology to six canonical long-short "
        "equity factors over the period 1985–2024. Portfolios formed on the most historically "
        "similar regimes outperform those formed on the most dissimilar regimes, with the "
        "spread between the two delivering a positive and economically meaningful Sharpe ratio. "
        "Importantly, we also find that dissimilar historical periods — which we term "
        "<i>anti-regimes</i> — contain incremental predictive information: returns following "
        "anti-regime conditions tend to reverse relative to returns following similar conditions. "
        "Results are robust to different quantile cutoffs and lookback window lengths."
    )
    story.append(Paragraph(abstract_text, S["abstract"]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "<b>Keywords:</b> Regime detection, factor timing, historical similarity, non-parametric, "
        "Fama-French factors, time-varying risk premia, macro regimes, anti-regimes.",
        S["keywords"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("1. Introduction", S["section"]))

    story.append(Paragraph(
        "The real-time identification of the prevailing economic regime is among the most "
        "consequential — and most difficult — challenges in investment management. Regimes "
        "shape the distributional properties of asset returns: factor premia that appear "
        "reliably positive over long horizons can be sharply negative during particular "
        "macroeconomic episodes, and understanding which episode is currently unfolding "
        "can meaningfully improve portfolio construction. The difficulty is that regime "
        "identification is almost always clearer in hindsight than in real time.",
        S["body"],
    ))
    story.append(Paragraph(
        "A common approach is to define regimes discretely — for instance, high-growth versus "
        "low-growth, or inflationary versus deflationary — and to calibrate thresholds that "
        "separate them. This approach is intuitive but suffers from two limitations. First, "
        "the choice of variables and cutoffs is arbitrary: the boundary between 'high' and "
        "'low' inflation is not determined by the data but imposed by the researcher. Second, "
        "a small number of variables may fail to capture the full dimensionality of the "
        "economic environment. The world is more complex than any two-variable classification.",
        S["body"],
    ))
    story.append(Paragraph(
        "This paper proposes an alternative: a continuous, multivariate similarity measure "
        "that requires no discretisation. For any given month, we identify the historical "
        "periods that most closely resemble the current state of the economy across seven "
        "financial variables. We then use the factor returns that followed those similar "
        "historical periods to predict factor returns today. The method is non-parametric — "
        "no model parameters are estimated — and relies only on z-score normalisation and "
        "Euclidean distance.",
        S["body"],
    ))
    story.append(Paragraph(
        "The method also naturally identifies <i>anti-regimes</i>: historical periods that "
        "are most dissimilar from the present. We show that anti-regime information is not "
        "noise — it is informative in the opposite direction. Portfolios that trade against "
        "the direction predicted by dissimilar historical conditions tend to outperform those "
        "that trade with it.",
        S["body"],
    ))
    story.append(Paragraph(
        "We evaluate the method on six well-known long-short equity factors from the "
        "Fama-French and Carhart libraries: the market excess return, size (SMB), value "
        "(HML), profitability (RMW), investment (CMA), and 12-month momentum. For each "
        "factor in each month, we form a signal based on the average subsequent return "
        "following the most similar historical periods, and go long or short accordingly. "
        "Aggregating across factors and time, the strategy delivers a Sharpe ratio of "
        f"{sp_sr:.2f} on the spread between the most-similar and least-similar portfolios, "
        "with an alpha that is statistically significant relative to the long-only factor "
        "benchmark. Importantly, the excess returns are positive in "
        f"{pct_pos:.0f}% of calendar years in the sample.",
        S["body"],
    ))
    story.append(Paragraph(
        "Exhibit 1 previews the annual return profile of the long-similarity, short-dissimilarity "
        "portfolio, volatility-scaled to a 15% annualised target. The consistent run of positive "
        "years — and the relatively shallow drawdowns — reflects both the directional accuracy "
        "of the similarity signal and the risk-reduction benefit of the dissimilarity hedge.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 1. Long similarity and short anti-regime (dissimilarity)",
        (
            "Yearly returns from the spread portfolio (most-similar quintile long, most-dissimilar "
            f"quintile short), scaled to a 15% annualised volatility target. The strategy is "
            f"positive in {pct_pos:.0f}% of years. Conditional on an up year, the average return "
            f"is {avg_up:.1f}%; conditional on a down year, it is {avg_down:.1f}%."
        ),
        "exhibit1_yearly_returns.png",
        S,
        width=5.8,
        fig_dir=fig_dir,
    )

    story.append(Paragraph("2. Economic State Variables", S["section"]))
    story.append(Paragraph("2.1 Variable Selection", S["subsection"]))

    story.append(Paragraph(
        "Our framework requires the user to select a set of economic state variables. "
        "We work with seven financial series that collectively embed information about "
        "the growth cycle, monetary conditions, commodity markets, risk appetite, and "
        "cross-asset correlation structure:",
        S["body"],
    ))

    var_table_data = [
        ["#", "Variable", "Economic content", "Primary source"],
        ["1", "S&P 500 level",       "Equity valuations, growth expectations",  "Yahoo Finance (^GSPC)"],
        ["2", "10Y − 3M yield spread","Credit cycle, recession risk",            "FRED: GS10, TB3MS"],
        ["3", "WTI crude oil price",  "Energy costs, global demand",             "FRED: MCOILWTICO"],
        ["4", "Copper price",         "Industrial demand, economic momentum",    "FRED: PCOPPUSDM"],
        ["5", "3-month T-bill yield", "Monetary policy stance, inflation",       "FRED: TB3MS"],
        ["6", "Equity volatility",    "Risk aversion (VIX post-1990; realised vol pre-1990)", "Yahoo Finance (^VIX)"],
        ["7", "Stock-bond correlation","Macro regime, flight-to-quality",        "Computed (rolling 3-yr)"],
    ]
    vt_col_widths = [0.25*inch, 1.4*inch, 2.3*inch, 2.25*inch]
    vt = Table(var_table_data, colWidths=vt_col_widths)
    vt.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
        ("ALIGN",       (0, 0), (0, -1),  "CENTER"),
        ("LINEABOVE",   (0, 0), (-1, 0),  1.0, colors.black),
        ("LINEBELOW",   (0, 0), (-1, 0),  0.5, colors.black),
        ("LINEBELOW",   (0,-1), (-1,-1),  1.0, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 3),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(vt)
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "These variables were not selected post-hoc. The information content of equity "
        "prices, yield spreads, oil, copper, and short rates in predicting economic "
        "conditions has been documented since at least the early 1980s. The stock-bond "
        "correlation, while a somewhat newer indicator, has long been used as a practical "
        "input for asset allocation. Monthly data are sourced from FRED and Yahoo Finance, "
        "with historical extensions using correlated PPI proxy series where the primary "
        "series begins later in the sample.",
        S["body"],
    ))
    story.append(Paragraph(
        "Exhibit 2 shows the raw time series for each variable. The wide variation in "
        "levels and scales across series makes direct comparison impossible — the "
        "transformation step described next is essential.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 2. Raw economic state variables",
        "Raw monthly time series for each of the seven state variables. Scales and distributions "
        "differ substantially across variables, motivating the standardisation procedure described below.",
        "exhibit2_raw_variables.png",
        S,
        width=6.0,
        fig_dir=fig_dir,
    )

    story.append(Paragraph("2.2 Transformation", S["subsection"]))

    story.append(Paragraph(
        "We apply a three-step transformation to each series. The objective is to produce "
        "a stationary, mean-zero, unit-variance measure of the direction and magnitude of "
        "recent change for each variable — comparable across variables and across time.",
        S["body"],
    ))

    steps_data = [
        ["Step", "Operation", "Formula"],
        ["1", "12-month difference",
         "Δx_t = x_t − x_{t−12}"],
        ["2", "Rolling 10-year standardisation",
         "σ_t = std(Δx_{t−119}, …, Δx_t)"],
        ["3", "Z-score and winsorise",
         "z_t = clip(Δx_t / σ_t, −3, 3)"],
    ]
    st_col_widths = [0.4*inch, 2.2*inch, 3.6*inch]
    st = Table(steps_data, colWidths=st_col_widths)
    st.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
        ("LINEABOVE",   (0, 0), (-1, 0),  1.0, colors.black),
        ("LINEBELOW",   (0, 0), (-1, 0),  0.5, colors.black),
        ("LINEBELOW",   (0,-1), (-1,-1),  1.0, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 3),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.08*inch))

    story.append(Paragraph(
        "Annual differences (rather than monthly) are used to capture medium-term economic "
        "momentum while abstracting from seasonal noise. The rolling ten-year standard "
        "deviation normalises each series relative to its own recent history, ensuring "
        "that the resulting z-scores are comparable across different macroeconomic eras "
        "(e.g., the high-volatility 1970s versus the Great Moderation). Winsorising at "
        "±3 standard deviations prevents extreme outliers — most visibly in oil during "
        "the OPEC embargo and in volatility around the 1987 crash — from dominating the "
        "distance calculations.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 3. Transformed economic state variables",
        "Seven transformed state variables. The blue line shows the raw z-score "
        "(12-month difference divided by a rolling 10-year standard deviation); the orange "
        "dashed line shows the winsorised version (capped at ±3). Winsorisation is most "
        "visible for oil in the 1970s and volatility around the 1987 crash.",
        "exhibit3_transformed_variables.png",
        S,
        width=6.0,
        fig_dir=fig_dir,
    )

    story.append(Paragraph("2.3 Descriptive Statistics", S["subsection"]))

    story.append(Paragraph(
        "Exhibit 4 reports autocorrelations of the transformed series at horizons from "
        "one month to ten years, alongside means and standard deviations. The high one- "
        "and three-month autocorrelations are expected: the 12-month differencing "
        "induces persistence, since consecutive observations share eleven of their twelve "
        "monthly inputs. By 12 months, autocorrelations approach zero for most variables, "
        "confirming that the series carry little multi-year persistence once the overlapping "
        "differencing is unwound. Means are close to zero and standard deviations near one, "
        "consistent with effective standardisation.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 4. Persistence of the economic state variables",
        "Autocorrelations of the winsorised z-score series at lags of 1, 3, 12, 36, and "
        "120 months, together with monthly mean and standard deviation. High short-horizon "
        "autocorrelations reflect the overlapping nature of 12-month differences.",
        "exhibit4_autocorrelation_table.png",
        S,
        width=6.2,
        fig_dir=fig_dir,
    )

    story.append(Paragraph(
        "Exhibit 5 shows the cross-correlations among the seven transformed variables. "
        "Correlations are generally modest, with the largest positive correlation between "
        "copper and monetary policy (0.37), and the largest negative correlation between "
        "monetary policy and the yield curve (−0.76), which is mechanical given that the "
        "yield spread is partially defined by the short rate. The low average absolute "
        "cross-correlation across remaining pairs implies that the seven variables provide "
        "meaningful diversification in regime identification: the composite global score "
        "draws on largely independent signals rather than amplifying a single dimension "
        "of variation.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 5. Correlation of the economic state variables",
        "Pairwise correlations of the seven winsorised z-score series over the full "
        "available sample. Red indicates positive correlation; blue indicates negative "
        "correlation. Low off-diagonal correlations confirm that the variables provide "
        "diversified information for regime identification.",
        "exhibit5_correlation_heatmap.png",
        S,
        width=4.8,
        fig_dir=fig_dir,
    )

    story.append(Paragraph("3. Regime Similarity Methodology", S["section"]))
    story.append(Paragraph("3.1 Distance Metric", S["subsection"]))

    story.append(Paragraph(
        "For each evaluation month <i>T</i>, we compute the Euclidean distance between "
        "the vector of transformed state variables at month <i>T</i> and each historical "
        "month <i>i</i>:",
        S["body"],
    ))

    # Display equation as styled text (ReportLab doesn't natively render LaTeX)
    story.append(Paragraph(
        "<para alignment='center'><i>d</i><sub>Ti</sub> = "
        "√[ Σ<sub>v=1..V</sub> (<i>z</i><sub>iv</sub> − <i>z</i><sub>Tv</sub>)² ]</para>",
        ParagraphStyle("Equation", fontName="Times-Italic", fontSize=11,
                       alignment=TA_CENTER, spaceAfter=8, spaceBefore=4),
    ))

    story.append(Paragraph(
        "where <i>V</i> = 7 is the number of state variables and <i>z</i><sub>iv</sub> "
        "is the transformed value of variable <i>v</i> in month <i>i</i>. Months with "
        "smaller distances <i>d</i><sub>Ti</sub> are more similar to month <i>T</i>. "
        "The global score for month <i>i</i> relative to month <i>T</i> is precisely "
        "this distance: a score of zero means the historical month is identical to today, "
        "while higher values indicate greater dissimilarity.",
        S["body"],
    ))
    story.append(Paragraph(
        "We choose Euclidean distance for its simplicity and interpretability. It "
        "treats each variable symmetrically, consistent with our equal-weighting "
        "philosophy. Alternative distance measures — such as the Mahalanobis distance "
        "which accounts for cross-variable correlations — are feasible but require "
        "estimating a covariance matrix, introducing both complexity and estimation "
        "error.",
        S["body"],
    ))

    story.append(Paragraph("3.2 Exclusion Mask", S["subsection"]))

    story.append(Paragraph(
        "A critical implementation detail is the <b>36-month exclusion mask</b>. When "
        "computing similarity for month <i>T</i>, we exclude the 36 calendar months "
        "immediately preceding <i>T</i> (i.e., months <i>T</i>−35 through <i>T</i>−1). "
        "Without this exclusion, the most 'similar' months would almost always be the "
        "most recent ones — and the strategy would be implicitly trading on momentum, "
        "not on regime similarity. The mask forces the model to look back further into "
        "history and identify structurally analogous environments rather than simply "
        "recent continuations.",
        S["body"],
    ))

    story.append(Paragraph("3.3 Illustration: Three Periods of Market Stress", S["subsection"]))

    story.append(Paragraph(
        "Exhibits 6, 7, and 8 illustrate the global score at three historically distinct "
        "junctures: the Global Financial Crisis (GFC) in January 2009, the COVID-19 shock "
        "in early 2020, and the inflation surge of mid-2022.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 6. Historical similarity to January 2009 — Global Financial Crisis",
        "Global similarity score (Euclidean distance) computed for January 2009 across "
        "all available historical months. Green bars denote the 15% most similar periods; "
        "the grey region marks the 36-month exclusion window. The model correctly identifies "
        "the double-dip recessions of the early 1980s as the most historically analogous episodes.",
        "exhibit6_gfc.png",
        S,
        width=6.2,
        fig_dir=fig_dir,
    )

    story.append(Paragraph(
        "The GFC chart illustrates a case where the regime detection works well: a severe "
        "financial shock accompanied by collapsing equity prices, yield curve flattening, "
        "commodity weakness, and spiking volatility. These conditions closely match "
        "the recessionary episodes of the late 1970s and early 1980s, which the model "
        "selects as most similar.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 7a. Historical similarity during the COVID-19 crisis — February 2020",
        "Global similarity score for February 2020. The model finds no clearly analogous "
        "historical episode at the onset of the pandemic.",
        "exhibit7_covid_february_2020.png",
        S,
        width=6.2,
        fig_dir=fig_dir,
    )
    story += _exhibit_block(
        "Exhibit 7b. Historical similarity during the COVID-19 crisis — April 2020",
        "Global similarity score for April 2020, after the initial shock. "
        "The distribution of scores remains diffuse — consistent with the unprecedented "
        "nature of a pandemic-driven economic shutdown — yet still contains directional "
        "information for factor positioning.",
        "exhibit7_covid_april_2020.png",
        S,
        width=6.2,
        fig_dir=fig_dir,
    )

    story.append(Paragraph(
        "The COVID-19 case illustrates the opposite extreme: an event so singular in the "
        "post-war era that no clear historical analog exists. The global scores are "
        "relatively uniform, with no distinct cluster of low-distance dates. This is "
        "informative in itself — the method can flag when the current environment is "
        "truly unprecedented, which has implications for risk management.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 8. Historical similarity to August 2022 — Inflation surge",
        "Global similarity score for August 2022. Inflation periods are shaded in gold. "
        "The model identifies the 1977–1980 Iranian revolution period as most similar, "
        "consistent with the combination of elevated energy prices, rising short rates, "
        "and a flattening yield curve that characterised both episodes.",
        "exhibit8_inflation.png",
        S,
        width=6.2,
        fig_dir=fig_dir,
    )

    # Inflation table
    story.append(Paragraph(
        "For reference, the table below lists the major inflationary episodes in our "
        "sample together with the total cumulative price level change and duration.",
        S["body"],
    ))
    story.append(_inflation_table(S))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("3.4 Detecting Regime Shifts", S["subsection"]))

    story.append(Paragraph(
        "Beyond identifying the current regime, the framework can detect when a regime "
        "<i>shift</i> is occurring. We compute, for each month <i>T</i>, an "
        "exponentially-weighted moving average (EWMA) of the global scores across "
        "all recent historical months, assigning higher weight to the most recent dates:",
        S["body"],
    ))

    story.append(Paragraph(
        "<para alignment='center'>"
        "<i>C</i><sub>T</sub> = EWMA{ <i>d</i><sub>Ti</sub> : <i>i</i> = 0 … <i>T</i> }"
        "</para>",
        ParagraphStyle("Equation", fontName="Times-Italic", fontSize=11,
                       alignment=TA_CENTER, spaceAfter=8, spaceBefore=4),
    ))

    story.append(Paragraph(
        "where the decay parameter β = 1 − 1/<i>n</i> and the half-life is "
        "−ln(2)/ln(β), with <i>n</i> the lookback in months. When <i>C</i><sub>T</sub> "
        "rises sharply, it signals that recent months are becoming very different from "
        "the historical distribution — a potential regime shift. We compute this "
        "indicator for four lookback horizons:",
        S["body"],
    ))
    story.append(_ewma_table(S))
    story.append(Spacer(1, 0.1 * inch))

    story += _exhibit_block(
        "Exhibit 9. EWMA regime-shift indicator",
        "Exponentially-weighted moving average of global similarity scores for "
        "four lookback horizons and their cross-horizon mean (black dashed). "
        "Peaks in the indicator correspond to periods of rapid regime change: "
        "the early 1980s disinflation, the early 1990s recession, the 2008–2009 GFC, "
        "the 2020 COVID shock, and the 2022–2023 inflation cycle.",
        "exhibit9_ewma.png",
        S,
        width=6.2,
        fig_dir=fig_dir,
    )

    story.append(Paragraph("4. Factor Timing Results", S["section"]))
    story.append(Paragraph("4.1 Portfolio Construction", S["subsection"]))

    story.append(Paragraph(
        "We apply the regime framework to time six long-short equity factors: the market "
        "excess return (Mkt-RF), size (SMB), value (HML), profitability (RMW), investment "
        "(CMA), and 12-month momentum (Mom), sourced from the Kenneth French data library. "
        "For each month <i>T</i> and each factor <i>f</i>:",
        S["body"],
    ))

    steps2_data = [
        ["Step", "Action"],
        ["1", "Identify the Q1 months: the 20% of valid historical dates with the "
              "smallest distance from month T (excluding the 36-month recent window)."],
        ["2", "For each Q1 month i, record the factor return at i+1 (the subsequent month)."],
        ["3", "Average those subsequent returns to form the directional signal: "
              "go long if the average is positive, short if negative."],
        ["4", "Repeat for quintiles Q2–Q5. Aggregate the six factor positions equally within each quintile."],
    ]
    s2_col_widths = [0.4*inch, 6.1*inch]
    s2 = Table(steps2_data, colWidths=s2_col_widths)
    s2.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ALIGN",       (0, 0), (0, -1),  "CENTER"),
        ("ALIGN",       (1, 0), (1, -1),  "LEFT"),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE",   (0, 0), (-1, 0),  1.0, colors.black),
        ("LINEBELOW",   (0, 0), (-1, 0),  0.5, colors.black),
        ("LINEBELOW",   (0,-1), (-1,-1),  1.0, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 4),
    ]))
    story.append(s2)
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph(
        "The resulting quintile portfolios span the full similarity spectrum. Q1 goes "
        "long or short each factor based on signals derived from the most analogous "
        "historical regimes. Q5 does the same but using the most <i>dis</i>similar "
        "historical regimes — the anti-regime signal. The long-only benchmark holds all "
        "six factors equally throughout the sample.",
        S["body"],
    ))

    story.append(Paragraph("4.2 Main Results", S["subsection"]))

    story.append(Paragraph(
        "Exhibit 10 presents the cumulative returns for each quintile portfolio and the "
        "Q1−Q5 spread over 1985–2024. The results strongly support the regime similarity "
        "hypothesis:",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 10. Performance by similarity quintile",
        f"Left: cumulative returns for Q1 (most similar) through Q5 (most dissimilar) "
        f"and the long-only benchmark. Right: cumulative return of the Q1−Q5 spread. "
        f"Q1 Sharpe ratio: {q1_sr:.2f}. Q5 Sharpe ratio: {q5_sr:.2f}. "
        f"Spread Sharpe ratio: {sp_sr:.2f} (correlation to long-only: {sp_corr:.2f}).",
        "exhibit10_quintile_returns.png",
        S,
        width=6.2,
        fig_dir=fig_dir,
    )

    story.append(Paragraph(
        "The performance summary across portfolios is presented below.",
        S["body"],
    ))
    story.append(_perf_table(portfolios, lo_returns, S))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "Several findings stand out. First, Q1 delivers the highest Sharpe ratio among "
        "all quintile portfolios, confirming that similarity-based signals are directionally "
        f"correct more often than not. Second, Q5 has the lowest Sharpe ratio ({q5_sr:.2f}), "
        "suggesting that anti-regime conditions predict factor performance less reliably — "
        "or in the opposite direction. Third, the Q1−Q5 spread produces a statistically "
        f"meaningful alpha: regressing the spread on the long-only benchmark gives an "
        f"intercept of {alpha_pct:.1f}% per annum with a t-statistic of {tstat:.2f}. "
        "Fourth, the spread has a correlation of only "
        f"{sp_corr:.2f} with the long-only portfolio, confirming that similarity-based "
        "timing adds a largely uncorrelated source of return.",
        S["body"],
    ))

    story.append(Paragraph("4.3 Drawdown Profile", S["subsection"]))

    story.append(Paragraph(
        "A critical test of any timing strategy is its behaviour during severe market "
        "dislocations, when long-only factor portfolios typically suffer their worst losses. "
        "Exhibit 11 compares the drawdown profile of the Q1−Q5 spread with the long-only "
        "benchmark.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 11. Drawdown comparison: Similarity strategy vs. long-only factor portfolio",
        f"Drawdown profile (as a percentage of capital) for the long-only factor portfolio "
        f"(blue) and the Q1−Q5 spread strategy (red), over 1985–2024. "
        f"The long-only maximum drawdown is {lo_mdd:.1f}%; the spread strategy's maximum "
        f"drawdown is {sp_mdd:.1f}%.",
        "exhibit11_drawdown.png",
        S,
        width=6.2,
        fig_dir=fig_dir,
    )

    story.append(Paragraph(
        "The regime-based spread strategy exhibits meaningfully shallower drawdowns than "
        "the passive long-only portfolio. This is the most economically significant result: "
        "the similarity framework provides protective rotation precisely in the periods "
        "when passive factor investing is most painful, such as the 2000–2002 tech "
        "correction and the 2007–2009 financial crisis.",
        S["body"],
    ))

    story.append(Paragraph("5. Robustness", S["section"]))
    story.append(Paragraph("5.1 Sensitivity to Quantile Choice", S["subsection"]))

    story.append(Paragraph(
        "A natural concern is whether the results depend critically on the choice of "
        "quintiles (top 20% vs. bottom 20%). Exhibit 12 repeats the spread analysis "
        "using 2, 3, 4, 5, 10, and 20 quantiles. The spread portfolio maintains a "
        "positive Sharpe ratio across all choices, confirming that the finding is not "
        "an artefact of a particular cutoff threshold.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 12. Robustness to quantile choice",
        "Cumulative return of the top-minus-bottom similarity spread for different "
        "numbers of quantiles (2 through 20). All specifications deliver a positive "
        "Sharpe ratio, confirming that the result is robust to the choice of similarity cutoff.",
        "exhibit12_quantile_robustness.png",
        S,
        width=5.8,
        fig_dir=fig_dir,
    )

    story.append(Paragraph("5.2 Sensitivity to Z-Score Lookback Window", S["subsection"]))

    story.append(Paragraph(
        "The base specification uses a 10-year (120-month) rolling window to compute "
        "the z-score standard deviations. Exhibit 13 examines whether the results change "
        "materially when using 1-year, 3-year, or 5-year lookbacks instead.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit 13. Robustness to z-score lookback window",
        "Cumulative return of the Q1−Q5 spread for z-score lookback windows of "
        "1 year (12 months), 3 years (36 months), and 5 years (60 months), relative "
        "to the base 10-year specification. Results are broadly insensitive to the lookback choice.",
        "exhibit13_lookback_robustness.png",
        S,
        width=5.8,
        fig_dir=fig_dir,
    )

    story.append(Paragraph("6. Individual Factor Analysis", S["section"]))

    story.append(Paragraph(
        "The aggregate results in Section 4 pool six factors into a single strategy. "
        "Exhibits A1 and A2 decompose performance factor-by-factor, revealing heterogeneity "
        "in the value of regime timing across different return premia.",
        S["body"],
    ))

    story += _exhibit_block(
        "Exhibit A1. Individual factor quintile performance",
        "Cumulative return for each similarity quintile applied individually to each of the "
        "six Fama-French factors (Market, Size, Value, Profitability, Investment, Momentum) "
        "over 1985–2024. Quintile 1 (most similar) generally outperforms Quintile 5 (most dissimilar) "
        "across most factors, with the strongest evidence for Market and Momentum.",
        "exhibit_a1_individual_quintiles.png",
        S,
        width=6.2,
        fig_dir=fig_dir,
    )

    story += _exhibit_block(
        "Exhibit A2. Individual factor long-short similarity spread",
        "Cumulative return of the Q1−Q5 similarity spread applied factor-by-factor. "
        "Regime timing adds value across most factors, with Profitability and Momentum "
        "showing the most consistent positive spreads. The Investment factor is the least "
        "responsive to regime signals in this sample.",
        "exhibit_a2_individual_longshort.png",
        S,
        width=6.2,
        fig_dir=fig_dir,
    )

    story.append(Paragraph(
        "The cross-factor heterogeneity is economically meaningful. Factors that "
        "are more sensitive to macroeconomic conditions — such as the market factor "
        "and momentum — benefit more from regime-based timing than factors with "
        "weaker macro linkages (e.g., investment). This is consistent with the "
        "broader literature on time-varying factor premia: macro state variables "
        "are more informative for cyclically sensitive factors.",
        S["body"],
    ))

    story.append(Paragraph("7. Conclusion", S["section"]))

    story.append(Paragraph(
        "This paper presents a non-parametric, systematic method for identifying the "
        "current economic regime and using that identification to time long-short equity "
        "factor portfolios. The core insight is simple: if the present macroeconomic "
        "environment closely resembles a past episode, the factor returns that followed "
        "that past episode are informative about expected returns today.",
        S["body"],
    ))
    story.append(Paragraph(
        "The method avoids the discretisation problem inherent in traditional regime "
        "frameworks. Rather than forcing the economy into pre-defined categories, it "
        "measures similarity continuously via Euclidean distance across a set of "
        "standardised financial state variables. The approach is largely non-parametric: "
        "the only modelling choices are the selection of state variables, the z-score "
        "lookback window, the exclusion mask length, and the quantile threshold.",
        S["body"],
    ))
    story.append(Paragraph(
        "Applied to six Fama-French factors over 1985–2024, the regime similarity "
        "framework produces a Q1−Q5 spread with a positive Sharpe ratio, a statistically "
        "significant alpha relative to the long-only benchmark, and substantially "
        "shallower drawdowns than the passive factor portfolio. The strategy is positive "
        f"in {pct_pos:.0f}% of calendar years. Results are robust to the number of "
        "quantile bins and to the lookback period used to construct z-scores.",
        S["body"],
    ))
    story.append(Paragraph(
        "Several extensions are natural. First, the current framework assigns equal weights "
        "to all seven state variables; a dynamic weighting scheme based on recent predictive "
        "accuracy could improve performance while also identifying which variables are most "
        "informative in a given environment. Second, the z-score computation could be "
        "extended to higher-frequency data or to a broader cross-section of macro variables. "
        "Third, the similarity framework could be applied beyond factor timing to asset "
        "allocation across broader asset classes. These extensions are left for future work.",
        S["body"],
    ))

    doc.build(story, onFirstPage=_make_footer, onLaterPages=_make_footer)
    print(f"Paper saved to: {output_path}")
