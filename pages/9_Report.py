# pages/9_Report.py — Report Builder (Markdown & Standalone HTML)
# ============================================================================
# This is a re-issued version (safe & self-contained) so you can drop it into
# /pages/9_Report.py. It builds an executive report from AppState artefacts and
# exports both Markdown and a single-file HTML with embedded Plotly figures.
# ============================================================================

from __future__ import annotations

import json
from datetime import datetime
import typing as t

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------------------------------
# State (robust import with fallback)
# ----------------------------------------------------------------------------
try:
    from utils.state import AppState  # type: ignore
except Exception:
    class _State:
        def __init__(self) -> None:
            self.y_monthly: pd.Series | None = None
            self.panel_monthly: pd.DataFrame | None = None
            self.bt_results: dict[str, pd.Series] = {}
            self.bt_metrics: pd.DataFrame | None = None
            self.news_monthly: pd.DataFrame | None = None
    class AppState:  # type: ignore
        @staticmethod
        def init() -> _State:
            if "_app" not in st.session_state:
                st.session_state["_app"] = _State()
            return st.session_state["_app"]  # type: ignore
        @staticmethod
        def get() -> _State:
            return AppState.init()

state = AppState.init()

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    ya, pa = y.align(p, join="inner")
    e = ya - pa
    if len(e) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "SMAPE": np.nan, "MASE": np.nan, "obs": 0.0}
    mae = float(e.abs().mean())
    rmse = float(np.sqrt((e ** 2).mean()))
    smape = float((200.0 * (e.abs() / (ya.abs() + pa.abs() + 1e-12))).mean())
    denom = float((ya.diff().abs()).mean())
    mase = float(mae / (denom + 1e-12)) if np.isfinite(denom) else float("nan")
    return {"MAE": mae, "RMSE": rmse, "SMAPE": smape, "MASE": mase, "obs": float(len(ya))}


def build_metrics_table(y: pd.Series, preds: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, s in preds.items():
        m = _metrics(y, s); m.update({"model": name})
        rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = ["model", "MAE", "RMSE", "SMAPE", "MASE", "obs"]
    return df[cols].sort_values("MAE")


def _corr_top(y: pd.Series, X: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    try:
        ya, Xa = y.align(X.select_dtypes(include=[np.number]), join="inner")
        c = Xa.corrwith(ya).dropna().sort_values(ascending=False)
        return c.head(k).to_frame("corr").reset_index().rename(columns={"index":"feature"})
    except Exception:
        return pd.DataFrame(columns=["feature","corr"])

# ----------------------------------------------------------------------------
# UI — Meta
# ----------------------------------------------------------------------------

st.title("📝 Report Builder")
st.caption("Export an executive report as Markdown and standalone HTML. Works even with partial data.")

with st.sidebar:
    st.header("Report meta")
    title = st.text_input("Title", value="ISTAT Unemployment Nowcasting — Research Report")
    author = st.text_input("Author", value="Research Team")
    org = st.text_input("Organization", value="ISTAT / Internship Lab")
    date_str = st.text_input("Date", value=datetime.today().date().isoformat())

    st.markdown("---")
    st.header("Sections")
    sec_exec = st.checkbox("Executive Summary", value=True)
    sec_data = st.checkbox("Data Summary", value=True)
    sec_perf = st.checkbox("Model Performance", value=True)
    sec_charts = st.checkbox("Charts", value=True)
    sec_news = st.checkbox("News Impact", value=True)
    sec_events = st.checkbox("Events Timeline", value=True)
    sec_methods = st.checkbox("Methods & Config", value=True)

    st.markdown("---")
    st.header("Charts options")
    topk = st.slider("Top‑k models", 1, 8, 4)
    roll_w = st.slider("Rolling MAE window", 6, 36, 12, step=6)
    fig_h = st.slider("Figure height", 280, 640, 420, step=20)

# Guards (allow partial)
y_exists = state.y_monthly is not None and not state.y_monthly.empty
pred_exists = bool(state.bt_results)
metric_exists = state.bt_metrics is not None and not state.bt_metrics.empty
news_exists = state.news_monthly is not None and not state.news_monthly.empty

Y = state.y_monthly if y_exists else pd.Series(dtype=float)
M = build_metrics_table(Y, state.bt_results) if y_exists and pred_exists else pd.DataFrame()

# ----------------------------------------------------------------------------
# Figures (built on the fly)
# ----------------------------------------------------------------------------

fig_ts = None; fig_err = None; fig_news = None; fig_corr = None

if y_exists and pred_exists:
    order = (M.sort_values("MAE")["model"].tolist() if not M.empty else list(state.bt_results.keys()))[:topk]
    fig_ts = go.Figure(); fig_ts.add_trace(go.Scatter(x=Y.index, y=Y.values, name="Target", mode="lines", line=dict(width=3)))
    for name in order:
        s = state.bt_results.get(name)
        if s is not None:
            fig_ts.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines"))
    fig_ts.update_layout(template="plotly_white", height=fig_h, title="Target vs forecasts (top‑k)")

    fig_err = go.Figure()
    def _roll_mae(y: pd.Series, p: pd.Series, w: int) -> pd.Series:
        ya, pa = y.align(p, join="inner"); return (ya - pa).abs().rolling(w, min_periods=max(1, w//2)).mean()
    for name in order:
        s = state.bt_results.get(name)
        if s is not None:
            r = _roll_mae(Y, s, roll_w)
            fig_err.add_trace(go.Scatter(x=r.index, y=r.values, name=name, mode="lines"))
    fig_err.update_layout(template="plotly_white", height=fig_h, title=f"Rolling MAE (w={roll_w})")

if y_exists and news_exists:
    cols = [c for c in state.news_monthly.columns if c.endswith("news_count") or c.endswith("news_sent")][:2]
    if cols:
        y, X = Y.align(state.news_monthly[cols], join="inner")
        if len(y) > 0:
            fig_news = go.Figure(); fig_news.add_trace(go.Scatter(x=y.index, y=(y-y.mean())/(y.std(ddof=0)+1e-9), name="Target (z)", mode="lines", line=dict(width=3)))
            for c in X.columns:
                z = (X[c] - X[c].mean())/(X[c].std(ddof=0)+1e-9)
                fig_news.add_trace(go.Scatter(x=X.index, y=z, name=c, mode="lines"))
            fig_news.update_layout(template="plotly_white", height=fig_h, title="News signals vs Target (z‑scores)")

if y_exists and state.panel_monthly is not None and not state.panel_monthly.empty:
    corr_df = _corr_top(Y, state.panel_monthly, k=15)
    if not corr_df.empty:
        fig_corr = px.bar(corr_df[::-1], x="corr", y="feature", orientation="h", title="Top correlations with target")
        fig_corr.update_layout(template="plotly_white", height=fig_h)

# Preview (inline)
if fig_ts is not None: st.plotly_chart(fig_ts, use_container_width=True)
if fig_err is not None: st.plotly_chart(fig_err, use_container_width=True)
if fig_news is not None: st.plotly_chart(fig_news, use_container_width=True)
if fig_corr is not None: st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------------------------------------------------------------
# Compose Markdown
# ----------------------------------------------------------------------------

def compose_markdown() -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Author:** {author}  ")
    lines.append(f"**Organization:** {org}  ")
    lines.append(f"**Date:** {date_str}")
    lines.append("")

    if sec_exec:
        lines.append("## Executive Summary")
        if not M.empty:
            b = M.iloc[0]
            lines.append(f"- Best model by MAE: **{b['model']}** (MAE={b['MAE']:.3f}, RMSE={b['RMSE']:.3f}, obs={int(b['obs'])}).")
        if y_exists:
            ymin, ymax = Y.index.min().date(), Y.index.max().date()
            lines.append(f"- Target coverage: **{ymin} → {ymax}** ({len(Y)} months). Latest: **{float(Y.iloc[-1]):.3f}**.")
        if news_exists:
            lines.append("- News signals constructed and analyzed (cross/rolling corr).")
        lines.append("")

    if sec_data and state.panel_monthly is not None and not state.panel_monthly.empty:
        lines.append("## Data Summary")
        lines.append(f"Panel monthly shape: **{state.panel_monthly.shape[0]} × {state.panel_monthly.shape[1]}**.")
        lines.append("")

    if sec_perf and not M.empty:
        lines.append("## Model Performance")
        tbl = M.copy()
        for k in ["MAE","RMSE","SMAPE","MASE"]:
            tbl[k] = tbl[k].map(lambda x: f"{x:.3f}")
        lines.append(tbl.to_markdown(index=False))
        lines.append("")

    if sec_charts:
        lines.append("## Charts")
        if fig_ts is not None: lines.append("**Figure 1.** Target vs forecasts (top‑k).")
        if fig_err is not None: lines.append(f"**Figure 2.** Rolling MAE (w={roll_w}).")
        if fig_news is not None: lines.append("**Figure 3.** News vs Target (z‑scores).")
        if fig_corr is not None: lines.append("**Figure 4.** Top correlations with target.")
        lines.append("")

    if sec_news and news_exists:
        lines.append("## News Impact")
        lines.append("Signals derived from RSS/NewsAPI/GDELT or uploaded CSV; aligned to EOM; optional smoothing and lead/lag.")
        lines.append("")

    if sec_events and st.session_state.get("events"):
        lines.append("## Events Timeline")
        lines.append("Events were overlayed on the target; pre/post MAE windows are available on page 7.")
        lines.append("")

    if sec_methods:
        lines.append("## Methods & Config (snapshot)")
        lines.append("Walk‑forward backtesting with baselines, Ridge‑ARX, U‑MIDAS, and (S)ARIMAX (h≤1).")
        lines.append("Metrics: MAE, RMSE, SMAPE, MASE; DM test in Results page.")
        lines.append("")

    return "\n".join(lines)

md_text = compose_markdown()

st.markdown("---")
st.subheader("Export")
st.download_button("report.md", data=md_text.encode("utf-8"), file_name="report.md")

# HTML single file

def build_html(md: str) -> str:
    try:
        import markdown  # type: ignore
        md_html = markdown.markdown(md, extensions=["tables", "fenced_code"])  # type: ignore
    except Exception:
        md_html = f"<pre>{md}</pre>"

    def frag(fig):
        return pio.to_html(fig, full_html=False, include_plotlyjs="inline", default_width="100%", default_height="100%")

    parts = []
    if fig_ts is not None: parts.append("<h3>Figure 1 — Target vs forecasts</h3>" + frag(fig_ts))
    if fig_err is not None: parts.append(f"<h3>Figure 2 — Rolling MAE (w={roll_w})</h3>" + frag(fig_err))
    if fig_news is not None: parts.append("<h3>Figure 3 — News vs Target (z)</h3>" + frag(fig_news))
    if fig_corr is not None: parts.append("<h3>Figure 4 — Top correlations</h3>" + frag(fig_corr))

    css = """
    body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; margin:24px;}
    h1,h2,h3{color:#111827}
    .meta{color:#374151;margin-bottom:12px}
    hr{border:none;border-top:1px solid #E5E7EB;margin:24px 0}
    .section{margin:20px 0}
    """

    html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
      <meta charset='utf-8'>
      <meta name='viewport' content='width=device-width, initial-scale=1'>
      <title>{title}</title>
      <style>{css}</style>
    </head>
    <body>
      <h1>{title}</h1>
      <div class='meta'><b>Author:</b> {author} &nbsp; | &nbsp; <b>Org:</b> {org} &nbsp; | &nbsp; <b>Date:</b> {date_str}</div>
      <hr/>
      <div class='section'>{md_html}</div>
      <hr/>
      {''.join(parts)}
      <hr/>
      <footer style='color:#6B7280;font-size:12px;margin-top:32px'>Generated by Nowcasting Lab — pages/9_Report.py</footer>
    </body>
    </html>
    """
    return html

html_text = build_html(md_text)
st.download_button("report.html", data=html_text.encode("utf-8"), file_name="report.html")

st.caption("Place this file at /pages/9_Report.py. If you see a 'Missing page' error from app.py, reboot the app after commit or use a safe page_link wrapper.")
