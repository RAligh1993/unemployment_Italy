# pages/5_Results.py — Results Explorer & Model Comparison (EN)
# ============================================================================
# Role
#   • Compare models quantitatively and visually after backtesting
#   • Tables: global metrics; ranks; pairwise Diebold–Mariano tests
#   • Charts: Actual vs forecasts, rolling errors, cumulative errors, hist/box, scatter (R²)
#   • Exports: merged predictions/residuals; metrics; selected plots as JSON
#
# Notes
#   • Does NOT set page_config (centralized in app.py)
#   • Works with AppState produced by page 4 (bt_results & bt_metrics)
#   • Stats: DM test implemented with HAC (Bartlett) variance; SciPy optional
# ============================================================================

from __future__ import annotations

import io
import json
import typing as t

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
            self.bt_results: dict[str, pd.Series] = {}
            self.bt_metrics: pd.DataFrame | None = None
            self.panel_monthly: pd.DataFrame | None = None
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
# Guards
# ----------------------------------------------------------------------------

st.title("📈 Results & Comparison")
st.caption("Inspect forecasts, errors, and statistical comparisons across models.")

if state.y_monthly is None or state.y_monthly.empty:
    st.warning("Monthly target not found. Run pages 2–4 first.")
    st.stop()

if not state.bt_results:
    st.warning("No backtest results yet. Go to **Backtesting** and run at least one model.")
    st.stop()

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def merge_predictions(y: pd.Series, preds: dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame({"y": y}).sort_index()
    for name, s in preds.items():
        df[name] = s
    return df.dropna(how="all").sort_index()


def _metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    ya, pa = y.align(p, join="inner")
    e = ya - pa
    mae = float(e.abs().mean())
    rmse = float(np.sqrt((e ** 2).mean()))
    smape = float((200.0 * (e.abs() / (ya.abs() + pa.abs() + 1e-12))).mean())
    denom = float((ya.diff().abs()).mean())
    mase = float(mae / (denom + 1e-12)) if np.isfinite(denom) else float("nan")
    r2 = float(1.0 - (e.var(ddof=0) / (ya.var(ddof=0) + 1e-12))) if len(ya) > 1 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "SMAPE": smape, "MASE": mase, "R2": r2, "obs": float(len(ya))}


@st.cache_data(show_spinner=False)
def build_metrics_table(y: pd.Series, preds: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, s in preds.items():
        m = _metrics(y, s); m.update({"model": name})
        rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = ["model", "MAE", "RMSE", "SMAPE", "MASE", "R2", "obs"]
    return df[cols].sort_values("MAE")


def _rolling_abs_err(y: pd.Series, p: pd.Series, w: int) -> pd.Series:
    ya, pa = y.align(p, join="inner")
    e = (ya - pa).abs()
    return e.rolling(w, min_periods=max(1, w // 2)).mean().rename(p.name)


def _cumulative_abs_err(y: pd.Series, p: pd.Series) -> pd.Series:
    ya, pa = y.align(p, join="inner")
    return (ya - pa).abs().cumsum().rename(p.name)


def _r2_scatter(y: pd.Series, p: pd.Series) -> float:
    ya, pa = y.align(p, join="inner")
    if len(ya) < 3:
        return float("nan")
    e = ya - pa
    return float(1.0 - (e.var(ddof=0) / (ya.var(ddof=0) + 1e-12)))


# Diebold–Mariano test with HAC variance (Bartlett weights)
@st.cache_data(show_spinner=False)
def dm_test(y: pd.Series, p1: pd.Series, p2: pd.Series, h: int = 1, power: int = 1) -> tuple[float, float, int]:
    """Return (t_stat, p_value, T_eff). H0: equal predictive accuracy.
    power=1 → AE loss; power=2 → SE loss.
    """
    y1, f1 = y.align(p1, join="inner")
    y2, f2 = y.align(p2, join="inner")
    idx = y1.index.intersection(y2.index)
    y, f1, f2 = y1.loc[idx], f1.loc[idx], f2.loc[idx]
    e1 = y - f1
    e2 = y - f2
    if power == 1:
        d = (e1.abs() - e2.abs()).values
    else:
        d = ((e1 ** 2) - (e2 ** 2)).values
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 8:
        return float("nan"), float("nan"), T
    d = d - d.mean()
    # lag length q: for monthly h=1, set q=1; otherwise use  floor(1.5*T^(1/3))
    q = 1 if h <= 1 else int(np.floor(1.5 * (T ** (1 / 3))))
    # Newey–West HAC variance (Bartlett kernel)
    gamma0 = np.dot(d, d) / T
    cov = gamma0
    for k in range(1, q + 1):
        gk = np.dot(d[:-k], d[k:]) / T
        cov += 2 * (1 - k / (q + 1)) * gk
    var_dbar = cov / T
    if var_dbar <= 0:
        return float("nan"), float("nan"), T
    t_stat = (d.mean()) / np.sqrt(var_dbar)
    # Two‑sided normal approximation
    try:
        from math import erf, sqrt
        p_val = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / np.sqrt(2))))
    except Exception:
        p_val = float("nan")
    return float(t_stat), float(p_val), int(T)


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters & Options")
    df_all = merge_predictions(state.y_monthly, state.bt_results)
    available_models = [c for c in df_all.columns if c != "y"]
    default_sel = available_models[: min(4, len(available_models))]
    chosen = st.multiselect("Models to display", options=available_models, default=default_sel)

    # Date range filter
    smin, smax = df_all.index.min().date(), df_all.index.max().date()
    dr = st.slider("Date range", min_value=smin, max_value=smax, value=(smin, smax))
    mask = (df_all.index.date >= dr[0]) & (df_all.index.date <= dr[1])

    # Rolling window
    roll_w = st.slider("Rolling window (months)", 6, 48, 12, step=6)

    # Sort metric
    sort_metric = st.selectbox("Sort by metric", ["MAE", "RMSE", "SMAPE", "MASE", "R2"], index=0)

    # DM test pair
    st.markdown("---")
    st.header("Diebold–Mariano test")
    dm_m1 = st.selectbox("Model 1", options=available_models, index=0)
    dm_m2 = st.selectbox("Model 2", options=available_models, index=1 if len(available_models) > 1 else 0)
    dm_loss = st.radio("Loss", ["AE (power=1)", "SE (power=2)"], horizontal=True, index=0)
    dm_run = st.button("Run DM test", use_container_width=True)

# Filtered slice
DF = df_all.loc[mask].copy()

# ----------------------------------------------------------------------------
# Global metrics table + KPIs
# ----------------------------------------------------------------------------

st.subheader("Global metrics (filtered range)")
M = build_metrics_table(DF["y"], {c: DF[c] for c in DF.columns if c != "y"})
if not M.empty:
    M = M.sort_values(sort_metric if sort_metric in M.columns else "MAE")
    best = M.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best model", best["model"])
    c2.metric("MAE", f"{best['MAE']:.3f}")
    c3.metric("RMSE", f"{best['RMSE']:.3f}")
    c4.metric("SMAPE", f"{best['SMAPE']:.2f}%")
    c5.metric("R²", f"{best['R2']:.3f}")

    st.dataframe(M.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","SMAPE":"{:.2f}","MASE":"{:.3f}","R2":"{:.3f}"}), use_container_width=True)
else:
    st.info("No overlapping observations between target and selected models in this range.")

st.markdown("---")

# ----------------------------------------------------------------------------
# Chart 1 — Actual vs forecasts
# ----------------------------------------------------------------------------

st.subheader("Actual vs forecasts")
fig = go.Figure(); fig.add_trace(go.Scatter(x=DF.index, y=DF["y"], name="Target", mode="lines", line=dict(width=3)))
for name in chosen:
    if name in DF:
        fig.add_trace(go.Scatter(x=DF.index, y=DF[name], name=name, mode="lines"))
fig.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Chart 2 — Rolling MAE (or chosen rolling metric)
# ----------------------------------------------------------------------------

st.subheader("Rolling absolute error (mean)")
figr = go.Figure()
for name in chosen:
    if name in DF:
        r = _rolling_abs_err(DF["y"], DF[name], roll_w)
        figr.add_trace(go.Scatter(x=r.index, y=r.values, name=name, mode="lines"))
figr.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(figr, use_container_width=True)

# ----------------------------------------------------------------------------
# Chart 3 — Cumulative absolute error (regret)
# ----------------------------------------------------------------------------

st.subheader("Cumulative absolute error")
figc = go.Figure()
for name in chosen:
    if name in DF:
        c = _cumulative_abs_err(DF["y"], DF[name])
        figc.add_trace(go.Scatter(x=c.index, y=c.values, name=name, mode="lines"))
figc.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(figc, use_container_width=True)

# ----------------------------------------------------------------------------
# Panel — Error distribution & scatter (per‑model)
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("Per‑model diagnostics")
if chosen:
    model_pick = st.selectbox("Choose a model", options=chosen, index=0)
    ya, pa = DF["y"].align(DF[model_pick], join="inner")
    err = (ya - pa)

    colA, colB = st.columns(2)
    with colA:
        st.caption("Error histogram")
        hist = px.histogram(err.dropna(), nbins=max(10, min(40, int(np.sqrt(len(err.dropna()))))), title=None)
        hist.update_layout(template="plotly_white", height=340, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(hist, use_container_width=True)
        st.write(f"Bias (mean error): **{err.mean():.3f}**")
    with colB:
        st.caption("Actual vs forecast (R²)")
        sc = px.scatter(x=pa, y=ya, trendline="ols")
        sc.update_layout(template="plotly_white", height=340, xaxis_title="Forecast", yaxis_title="Actual")
        st.plotly_chart(sc, use_container_width=True)
        st.write(f"R²: **{_r2_scatter(ya, pa):.3f}**")

# ----------------------------------------------------------------------------
# Diebold–Mariano test (pairwise)
# ----------------------------------------------------------------------------

if dm_run and dm_m1 in DF and dm_m2 in DF and dm_m1 != dm_m2:
    power = 1 if dm_loss.startswith("AE") else 2
    t_stat, p_val, T_eff = dm_test(DF["y"], DF[dm_m1], DF[dm_m2], h=1, power=power)
    if np.isfinite(t_stat):
        decision = "Reject H0 (different accuracy)" if p_val < 0.05 else "Fail to reject H0"
        st.success(f"DM test ({dm_m1} vs {dm_m2}, loss={dm_loss}): t={t_stat:.3f}, p={p_val:.3f}, T={T_eff}. **{decision}**")
    else:
        st.info("DM test not feasible (insufficient overlapping data).")

# ----------------------------------------------------------------------------
# Export block
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("Export")

# Combined predictions & residuals
P = DF.copy()
for c in [c for c in P.columns if c != "y"]:
    P[f"resid__{c}"] = (P["y"] - P[c])

csv_bytes = P.to_csv().encode("utf-8")
st.download_button("predictions_and_residuals.csv", data=csv_bytes, file_name="preds_residuals.csv")

# Metrics CSV (filtered)
if not M.empty:
    st.download_button("metrics_filtered.csv", data=M.to_csv(index=False).encode("utf-8"), file_name="metrics_filtered.csv")

# Plot JSON (for reproducibility)
try:
    fig_bundle = {
        "actual_vs": fig.to_plotly_json(),
        "rolling_mae": figr.to_plotly_json(),
        "cum_abs": figc.to_plotly_json(),
    }
    st.download_button("figures.json", data=json.dumps(fig_bundle).encode("utf-8"), file_name="figures.json")
except Exception:
    pass

st.caption("This page summarizes performance and supports statistical testing. Use SHAP & Events (page 7) for interpretability.")
