# pages/1_Dashboard.py — Professional Overview Dashboard (EN)
# ============================================================================
# Role
#   • Executive overview for the Nowcasting Lab
#   • KPIs for data readiness and (when available) model performance
#   • Interactive target chart with overlays (moving averages & predictions)
#   • Top feature correlations snapshot
#   • Data quality panel (missing/duplicates/outliers)
#
# Notes
#   • This page does NOT set page_config (centralized in app.py)
#   • Works even if utils/state.py is not present (safe fallback)
# ============================================================================

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
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
            self.google_trends: pd.DataFrame | None = None
            self.raw_daily: list[pd.DataFrame] = []

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
# Helpers & metrics
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def pct_change(s: pd.Series, m: int) -> float:
    if s is None or s.empty or len(s) <= m:
        return float("nan")
    s = s.dropna()
    if len(s) <= m:
        return float("nan")
    try:
        return float(100.0 * (s.iloc[-1] - s.iloc[-1 - m]) / (abs(s.iloc[-1 - m]) + 1e-12))
    except Exception:
        return float("nan")


@st.cache_data(show_spinner=False)
def moving_avg(s: pd.Series, k: int) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.sort_index().rolling(k, min_periods=max(1, k // 2)).mean().rename(f"MA{k}")


def _metrics_against(y: pd.Series, yhat: pd.Series) -> dict[str, float]:
    ya, pa = y.align(yhat, join="inner")
    e = ya - pa
    mae = float(e.abs().mean())
    rmse = float(np.sqrt((e ** 2).mean()))
    smape = float((200.0 * (e.abs() / (ya.abs() + pa.abs() + 1e-12))).mean())
    denom = float((ya.diff().abs()).mean())
    mase = float(mae / (denom + 1e-12)) if np.isfinite(denom) else float("nan")
    return {"MAE": mae, "RMSE": rmse, "SMAPE": smape, "MASE": mase}


def _continuous_month_index(s: pd.Series) -> pd.DatetimeIndex:
    idx = s.sort_index().index
    if len(idx) == 0:
        return idx
    full = pd.date_range(idx.min(), idx.max(), freq="M")
    return full


@st.cache_data(show_spinner=False)
def data_quality(y: pd.Series) -> dict[str, t.Any]:
    if y is None or y.empty:
        return {"missing": 0, "duplicate": 0, "missing_dates": [], "outliers": pd.DataFrame()}
    s = y.dropna()
    # missing by calendar
    full = _continuous_month_index(s)
    miss_idx = full.difference(s.index)
    # duplicates
    duplicate = int(s.index.duplicated().sum())
    # outliers via z-score (robust)
    v = s.values.astype(float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) + 1e-9
    z = 0.6745 * (v - med) / mad
    out_mask = np.abs(z) > 3.5
    out_df = pd.DataFrame({"date": s.index[out_mask], "value": s.values[out_mask], "|z|": np.abs(z[out_mask])})
    return {"missing": int(len(miss_idx)), "duplicate": duplicate, "missing_dates": list(miss_idx), "outliers": out_df}


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.title("📊 Dashboard")
st.caption("High‑level overview: data readiness, quick trends, feature snapshot, and quality checks.")

# Guard: no target
if state.y_monthly is None or state.y_monthly.empty:
    st.warning("No monthly target loaded yet. Go to **Data & Aggregation** or use Quick Start on Home.")
    st.stop()

# ----------------------------------------------------------------------------
# KPI strip
# ----------------------------------------------------------------------------

y = state.y_monthly.dropna().sort_index()
latest = y.iloc[-1]
range_txt = f"{y.index.min().date()} → {y.index.max().date()}"

m1 = pct_change(y, 1)
m12 = pct_change(y, 12)
qoq = pct_change(y, 3)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Latest value", f"{latest:,.3f}", help=f"Sample: {range_txt}")
with c2:
    st.metric("m/m %", f"{m1:,.2f}%")
with c3:
    st.metric("q/q %", f"{qoq:,.2f}%")
with c4:
    st.metric("y/y %", f"{m12:,.2f}%")

# ----------------------------------------------------------------------------
# Interactive chart (target + overlays)
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("Target timeline & overlays")

colA, colB, colC = st.columns([1.1, 1, 1])
with colA:
    win_years = st.slider("Window (years)", 1, max(1, min(10, max(1, len(y) // 12))), 3)
with colB:
    ma_short = st.number_input("Short MA (months)", 2, 24, 3)
with colC:
    ma_long = st.number_input("Long MA (months)", 3, 48, 12)

start_cut = y.index.max() - pd.DateOffset(years=win_years)
plot_y = y[y.index >= start_cut]

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_y.index, y=plot_y.values, mode="lines", name="Target", line=dict(width=3)))

ma_s = moving_avg(y, int(ma_short))
ma_l = moving_avg(y, int(ma_long))
if not ma_s.empty:
    ms = ma_s[ma_s.index >= start_cut]
    fig.add_trace(go.Scatter(x=ms.index, y=ms.values, mode="lines", name=f"MA{ma_short}"))
if not ma_l.empty:
    ml = ma_l[ma_l.index >= start_cut]
    fig.add_trace(go.Scatter(x=ml.index, y=ml.values, mode="lines", name=f"MA{ma_long}"))

# Overlay model predictions if present
if state.bt_results:
    st.caption("Overlay backtest predictions (if any):")
    names = sorted(list(state.bt_results.keys()))
    chosen = st.multiselect("Models to overlay", names, default=names[: min(3, len(names))])
    for name in chosen:
        s = state.bt_results[name]
        s = s[s.index >= start_cut]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name))

fig.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# Quick metrics for overlays
if state.bt_results and len(state.bt_results) > 0:
    rows = []
    for name, s in state.bt_results.items():
        m = _metrics_against(y, s)
        m.update({"model": name})
        rows.append(m)
    dfm = pd.DataFrame(rows).sort_values("MAE")[["model", "MAE", "RMSE", "SMAPE", "MASE"]]
    st.dataframe(dfm.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "SMAPE": "{:.2f}", "MASE": "{:.3f}"}), use_container_width=True)

# ----------------------------------------------------------------------------
# Feature snapshot (correlations)
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("Top feature correlations (last N months)")

if state.panel_monthly is None or state.panel_monthly.empty:
    st.info("Monthly panel is empty. Build it in **Data & Aggregation**.")
else:
    lookback = st.slider("Lookback (months)", 12, min(120, max(12, len(state.panel_monthly))), 36, step=12)
    # align & slice
    y_al, X_al = y.align(state.panel_monthly.select_dtypes(include=[np.number]), join="inner")
    y_al = y_al.tail(lookback)
    X_al = X_al.loc[y_al.index]
    if X_al.shape[1] == 0:
        st.info("Panel has no numeric columns.")
    else:
        corr = X_al.corrwith(y_al).sort_values(ascending=False)
        top = corr.head(15).dropna()
        st.dataframe(top.to_frame("corr_with_target").style.format({"corr_with_target": "{:.3f}"}), use_container_width=True)
        # Horizontal bar chart
        figc = px.bar(top[::-1], x=top[::-1].values, y=top[::-1].index, orientation="h", title="Top correlations")
        figc.update_layout(template="plotly_white", height=520, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(figc, use_container_width=True)

# ----------------------------------------------------------------------------
# Data Quality panel
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("Data quality")

q = data_quality(y)
colq1, colq2, colq3 = st.columns(3)
with colq1:
    st.metric("Missing months", f"{q['missing']}")
with colq2:
    st.metric("Duplicate index", f"{q['duplicate']}")
with colq3:
    st.metric("Outliers (>|z|>3.5)", f"{0 if q['outliers'] is None else len(q['outliers'])}")

with st.expander("Details", expanded=False):
    if q["missing_dates"]:
        miss_df = pd.DataFrame({"missing_month": pd.to_datetime(q["missing_dates"]).date})
        st.dataframe(miss_df, use_container_width=True)
    else:
        st.write("No missing months in the current sample range.")
    if q["outliers"] is not None and not q["outliers"].empty:
        st.write("Potential outliers (robust z-score):")
        st.dataframe(q["outliers"], use_container_width=True)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------

st.caption("Dashboard provides an at-a-glance view. For modeling, proceed to **Backtesting** and **Results** pages.")
