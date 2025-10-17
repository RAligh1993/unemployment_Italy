# pages/7_SHAP_Events.py — SHAP (Interpretability) & Events Timeline (EN)
# ============================================================================
# Role
#   • Fit a linear (Ridge) model on the monthly panel for interpretability
#   • Compute SHAP values (if `shap` is installed); otherwise fall back to
#     coefficient-based contributions (|β|·σ for global; β·x for local)
#   • Global importance, local explanation for a chosen month, contribution table
#   • Events timeline: add/import/export events, overlay on charts, and
#     pre/post performance comparisons for any selected backtest model
#
# Notes
#   • This page does *not* depend on the walk-forward training from page 4.
#     It fits a fresh (windowed) linear model purely for interpretability.
#   • For SHAP on linear models we use `shap.Explainer` when available and
#     fall back gracefully if the API variant is different.
#   • All data are read from AppState (y_monthly, panel_monthly, bt_results).
# ============================================================================

from __future__ import annotations

import io
import json
import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

# Optional deps
try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # type: ignore

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    RidgeCV = None  # type: ignore
    class StandardScaler:  # no-op fallback
        def fit(self, X): return self
        def transform(self, X): return X

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

st.title("🧭 SHAP & Events")
st.caption("Interpret your model with SHAP (or coefficient contributions) and analyze performance around timeline events.")

if state.y_monthly is None or state.y_monthly.empty or state.panel_monthly is None or state.panel_monthly.empty:
    st.warning("Target or monthly panel missing. Build them in pages 1–3 first.")
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("Model window & features")
    years = sorted({d.year for d in state.y_monthly.index})
    s_year = st.select_slider("Start year", options=years, value=years[max(0, len(years) - 6)])
    e_year = st.select_slider("End year", options=years, value=years[-1])

    num_cols = state.panel_monthly.select_dtypes(include=[np.number]).columns.tolist()
    default_cols = num_cols[: min(20, len(num_cols))]
    cols_sel = st.multiselect("Select features (monthly)", options=num_cols, default=default_cols)

    st.markdown("---")
    st.header("Ridge & SHAP")
    alpha_txt = st.text_input("Ridge alphas (comma)", value="0.1,0.3,1,3,10,30,100")
    standardize = st.checkbox("Standardize X", value=True)
    use_shap = st.checkbox("Use SHAP if available", value=True)

    st.markdown("---")
    st.header("Events")
    model_for_events = st.selectbox("Backtest model for pre/post metrics", options=sorted(list(state.bt_results.keys())) if state.bt_results else ["(none)"])
    win_pre = st.slider("Pre window (months)", 1, 24, 6)
    win_post = st.slider("Post window (months)", 1, 24, 6)

# ----------------------------------------------------------------------------
# Data slicing
# ----------------------------------------------------------------------------

mask = (state.y_monthly.index.year >= s_year) & (state.y_monthly.index.year <= e_year)
y_win = state.y_monthly.loc[mask].copy()
X_win_full = state.panel_monthly.loc[y_win.index].copy()
X_win = X_win_full[cols_sel].select_dtypes(include=[np.number]) if cols_sel else pd.DataFrame(index=y_win.index)

if X_win.empty or y_win.isna().all():
    st.info("Select at least one numeric feature and a valid date range.")
    st.stop()

# Drop rows with any NA across y and X
Z = pd.concat([y_win.rename("y"), X_win], axis=1).dropna()
if Z.empty or Z.shape[0] < 12:
    st.info("Too few complete observations in the selected window (need ≥ 12).")
    st.stop()

y_al = Z["y"].astype(float)
X_al = Z.drop(columns=["y"]).astype(float)

# ----------------------------------------------------------------------------
# Fit ridge & compute SHAP (or contributions)
# ----------------------------------------------------------------------------

# Parse alphas
try:
    alphas = [float(a.strip()) for a in alpha_txt.split(",") if a.strip()]
    if not alphas:
        alphas = [0.1, 1.0, 10.0]
except Exception:
    alphas = [0.1, 1.0, 10.0]

scaler = StandardScaler() if standardize else None
if scaler is not None:
    X_fit = scaler.fit_transform(X_al.values)
else:
    X_fit = X_al.values

# Train ridge
if RidgeCV is None:
    st.error("scikit-learn not installed. Install scikit-learn to enable SHAP/coeff contributions.")
    st.stop()

ridge = RidgeCV(alphas=np.array(alphas), store_cv_values=False)
ridge.fit(X_fit, y_al.values)

# Predictions and residuals
pred_in = ridge.predict(X_fit)
resid_in = y_al.values - pred_in

# SHAP values (if available & requested)
shap_vals = None
baseline_value = float(np.mean(y_al.values))
if use_shap and shap is not None:
    try:
        # Explainer API differs by version; try new then legacy
        try:
            expl = shap.Explainer(ridge, X_fit)
            S = expl(X_fit)
            shap_vals = np.array(S.values)
            baseline_value = float(np.array(S.base_values).mean())
        except Exception:
            expl = shap.LinearExplainer(ridge, X_fit)
            shap_vals = np.array(expl.shap_values(X_fit))
            baseline_value = float(np.mean(expl.expected_value))
    except Exception:
        shap_vals = None

# Fallback contributions using coefficients
coef = ridge.coef_.ravel()
if scaler is not None:
    # contributions in original scale: β·z, where z is standardized X
    X_std = X_fit
else:
    # approx standardize by z-scoring once to make |β| comparable
    mu = X_al.values.mean(axis=0)
    sd = X_al.values.std(axis=0) + 1e-9
    X_std = (X_al.values - mu) / sd

contrib_matrix = (X_std * coef.reshape(1, -1))  # shape T × p

# Global importance (SHAP if available, else |β|·σ)
if shap_vals is not None:
    global_imp = np.nanmean(np.abs(shap_vals), axis=0)
else:
    # |β| * std(X) on the (standardized) scale
    if scaler is not None:
        # standard scaler makes std=1 → use |β|
        global_imp = np.abs(coef)
    else:
        global_imp = np.abs(coef) * (X_al.values.std(axis=0) + 1e-9)

imp_df = pd.DataFrame({"feature": X_al.columns, "importance": global_imp}).sort_values("importance", ascending=False)

# ----------------------------------------------------------------------------
# Display — KPIs & Global importance
# ----------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)
col1.metric("R² (in-sample)", f"{1.0 - np.var(resid_in)/ (np.var(y_al.values)+1e-12):.3f}")
col2.metric("MAE (in-sample)", f"{np.mean(np.abs(resid_in)):.3f}")
col3.metric("α (selected)", f"{ridge.alpha_ if hasattr(ridge,'alpha_') else alphas[0]:.3f}")

st.subheader("Global feature importance")
fig_imp = px.bar(imp_df.head(25)[::-1], x="importance", y="feature", orientation="h", title="Top features by importance")
fig_imp.update_layout(template="plotly_white", height=560, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_imp, use_container_width=True)

# ----------------------------------------------------------------------------
# Local explanation — pick a month
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("Local explanation (single month)")

idx_al = Z.index
pick = st.select_slider("Pick a month", options=list(idx_al), value=idx_al[-1])
row_idx = list(idx_al).index(pick)

if shap_vals is not None:
    contrib = shap_vals[row_idx, :]
    base = baseline_value
    method = "SHAP"
else:
    contrib = contrib_matrix[row_idx, :]
    base = float(np.mean(y_al.values))
    method = "β·x contributions (fallback)"

row = pd.DataFrame({
    "feature": X_al.columns,
    "contribution": contrib,
    "value": X_al.iloc[row_idx].values,
}).sort_values("contribution", ascending=False)

# Pred decomposition
pred_val = float(pred_in[row_idx])
actual_val = float(y_al.values[row_idx])

cA, cB = st.columns([1.2, 1])
with cA:
    st.caption(f"Method: {method}")
    # Waterfall-like bar: positives vs negatives
    pos = row[row["contribution"] > 0].head(12)
    neg = row[row["contribution"] < 0].tail(12)
    wf = pd.concat([pos, neg])
    fig_wf = px.bar(wf, x="contribution", y="feature", orientation="h", color=(wf["contribution"] > 0),
                    title=f"Contributions at {pd.to_datetime(pick).date()} (pred≈{pred_val:.3f}, actual={actual_val:.3f})",
                    labels={"color": "+ / -"})
    fig_wf.update_layout(template="plotly_white", height=640, margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    st.plotly_chart(fig_wf, use_container_width=True)
with cB:
    st.caption("Top contributors (table)")
    st.dataframe(row.head(20).style.format({"contribution": "{:.4f}", "value": "{:.4f}"}), use_container_width=True)

# ----------------------------------------------------------------------------
# Events Timeline — storage & UI
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("Events timeline & impact")

# Initialize events in session_state
st.session_state.setdefault("events", [])  # list of {date:str, label:str}

colE1, colE2 = st.columns([1.2, 1])
with colE1:
    st.caption("Add event")
    e_date = st.date_input("Event date (EOM aligned recommended)")
    e_label = st.text_input("Label", value="Policy / Shock")
    add_btn = st.button("Add event")
    if add_btn and e_date:
        st.session_state["events"].append({"date": str(pd.to_datetime(e_date) + pd.offsets.MonthEnd(0)).split(" ")[0],
                                            "label": e_label})
        st.success("Event added.")
with colE2:
    st.caption("Import / Export")
    up = st.file_uploader("Import events.json", type=["json"], key="ev_imp")
    if up is not None:
        try:
            data = json.load(up)
            if isinstance(data, list):
                st.session_state["events"] = data
                st.success(f"Imported {len(data)} events.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
    if st.session_state["events"]:
        st.download_button("Export events.json", data=json.dumps(st.session_state["events"], indent=2).encode("utf-8"), file_name="events.json")

# Show events table
if st.session_state["events"]:
    ev_df = pd.DataFrame(st.session_state["events"]).assign(date=lambda d: pd.to_datetime(d["date"]))
    ev_df = ev_df.sort_values("date")
    st.dataframe(ev_df, use_container_width=True)

# Overlay on target and selected model (if any)
if st.session_state.get("events"):
    st.markdown("**Timeline overlay**")
    fig_t = go.Figure(); fig_t.add_trace(go.Scatter(x=state.y_monthly.index, y=state.y_monthly.values, name="Target", mode="lines", line=dict(width=3)))
    if model_for_events and model_for_events in state.bt_results:
        s = state.bt_results[model_for_events]
        fig_t.add_trace(go.Scatter(x=s.index, y=s.values, name=model_for_events, mode="lines"))
    for _, r in ev_df.iterrows():
        ts = r["date"]
        fig_t.add_vline(x=ts, line_dash="dot", line_color="#EF4444")
        fig_t.add_annotation(x=ts, y=float(state.y_monthly.loc[state.y_monthly.index.get_loc(ts, method='nearest')]),
                             text=str(r["label"]), showarrow=True, arrowhead=1, yshift=20)
    fig_t.update_layout(template="plotly_white", height=460, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_t, use_container_width=True)

# ----------------------------------------------------------------------------
# Pre/Post event performance (chosen model)
# ----------------------------------------------------------------------------

if st.session_state.get("events") and model_for_events and model_for_events in state.bt_results:
    st.markdown("**Pre/Post event metrics**")
    model_s = state.bt_results[model_for_events]
    rows = []
    for _, r in ev_df.iterrows():
        t0 = r["date"]
        pre_mask = (state.y_monthly.index <= t0) & (state.y_monthly.index > (t0 - pd.offsets.MonthEnd(win_pre)))
        post_mask = (state.y_monthly.index > t0) & (state.y_monthly.index <= (t0 + pd.offsets.MonthEnd(win_post)))
        # align
        y_pre, p_pre = state.y_monthly.loc[pre_mask].align(model_s, join="inner")
        y_post, p_post = state.y_monthly.loc[post_mask].align(model_s, join="inner")
        def _mae(a, b):
            a, b = a.align(b, join="inner"); return float((a - b).abs().mean()) if len(a) else float("nan")
        rows.append({
            "event": r["label"],
            "date": str(t0.date()),
            f"MAE_pre_{win_pre}m": _mae(y_pre, p_pre),
            f"MAE_post_{win_post}m": _mae(y_post, p_post),
            "obs_pre": int(len(y_pre)),
            "obs_post": int(len(y_post)),
        })
    ev_metrics = pd.DataFrame(rows)
    if not ev_metrics.empty:
        # delta column
        pre_col = [c for c in ev_metrics.columns if c.startswith("MAE_pre_")][0]
        post_col = [c for c in ev_metrics.columns if c.startswith("MAE_post_")][0]
        ev_metrics["ΔMAE (post-pre)"] = ev_metrics[post_col] - ev_metrics[pre_col]
        st.dataframe(ev_metrics.style.format({pre_col:"{:.3f}", post_col:"{:.3f}", "ΔMAE (post-pre)":"{:.3f}"}), use_container_width=True)

# ----------------------------------------------------------------------------
# (Optional) SHAP before/after an event — mean contribution shift
# ----------------------------------------------------------------------------

if shap_vals is not None and st.session_state.get("events"):
    st.markdown("---")
    st.subheader("Contribution shift around an event (experimental)")
    pick_event = st.selectbox("Event", options=[f"{r['date']:%Y-%m-%d} — {r['label']}" for _, r in ev_df.iterrows()]) if st.session_state["events"] else None
    if pick_event:
        # parse date
        idx = [i for i, r in ev_df.iterrows()]
        # find the chosen row by label string
        sel_row = None
        for _, r in ev_df.iterrows():
            tag = f"{r['date']:%Y-%m-%d} — {r['label']}"
            if tag == pick_event:
                sel_row = r
                break
        if sel_row is not None:
            t0 = sel_row["date"]
            pre_mask = (Z.index <= t0) & (Z.index > (t0 - pd.offsets.MonthEnd(win_pre)))
            post_mask = (Z.index > t0) & (Z.index <= (t0 + pd.offsets.MonthEnd(win_post)))
            # map back into shap_vals rows by matching index positions
            idx_all = list(Z.index)
            pre_idx = [i for i, ts in enumerate(idx_all) if pre_mask.loc[ts]]
            post_idx = [i for i, ts in enumerate(idx_all) if post_mask.loc[ts]]
            if pre_idx and post_idx:
                pre_mean = np.nanmean(shap_vals[pre_idx, :], axis=0)
                post_mean = np.nanmean(shap_vals[post_idx, :], axis=0)
                diff = post_mean - pre_mean
                df_shift = pd.DataFrame({"feature": X_al.columns, "Δcontrib": diff}).sort_values("Δcontrib", ascending=False)
                fig_shift = px.bar(df_shift.head(20)[::-1], x="Δcontrib", y="feature", orientation="h", title="Mean contribution shift (post − pre)")
                fig_shift.update_layout(template="plotly_white", height=520, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_shift, use_container_width=True)
            else:
                st.info("Insufficient pre/post samples within the selected windows for this event.")

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------

st.caption("SHAP requires the `shap` package. When unavailable, a consistent \n"
           "coefficient-based fallback is used for interpretability. Events \n"
           "analysis uses your backtest predictions (page 4).")
