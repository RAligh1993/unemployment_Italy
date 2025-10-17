# pages/3_Feature_Engineering.py — Feature transforms, lags & rolling (EN)
# ============================================================================
# Role
#   • Transform the monthly panel: diff/pct/log/winsorize/z-score
#   • Generate lags and rolling statistics (mean/std/min/max)
#   • Handle missing values (ffill/bfill) and preview quality
#   • Maintain a JSON "recipe" for reproducibility; reset to original
#
# Notes
#   • Page config is centralized in app.py
#   • Uses utils/* if available; otherwise includes local fallbacks
# ============================================================================

from __future__ import annotations

import json
import math
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

# ----------------------------------------------------------------------------
# State (robust import with fallback)
# ----------------------------------------------------------------------------
try:
    from utils.state import AppState  # type: ignore
except Exception:
    class _State:
        def __init__(self) -> None:
            self.panel_monthly: pd.DataFrame | None = None
            self.y_monthly: pd.Series | None = None
            self._panel_monthly_orig: pd.DataFrame | None = None
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
# Local helpers (safe EOM & naming) — fall back to utils if present
# ----------------------------------------------------------------------------
try:
    from utils.time_ops import end_of_month as _eom  # type: ignore
except Exception:
    def _eom(s: pd.Series) -> pd.Series:
        return (pd.to_datetime(s, errors="coerce").dt.tz_localize(None) + pd.offsets.MonthEnd(0)).dt.normalize()


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)


def _make_unique(name: str, existing: set[str]) -> str:
    base = name
    k = 1
    while name in existing:
        name = f"{base}__{k}"
        k += 1
    existing.add(name)
    return name


def _slug(s: str) -> str:
    s = str(s).strip().lower().replace(" ", "_")
    for ch in ["/", "(", ")", "-", "%", ":", ",", "."]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

# ----------------------------------------------------------------------------
# Header & guards
# ----------------------------------------------------------------------------

st.title("🧪 Feature Engineering")
st.caption("Transform, lag, roll, and standardize the **monthly** panel. All changes are kept in session state.")

if state.panel_monthly is None or state.panel_monthly.empty:
    st.warning("Monthly panel is empty. Build it in **Data & Aggregation** first.")
    st.stop()

# Create an original snapshot once
if getattr(state, "_panel_monthly_orig", None) is None:
    state._panel_monthly_orig = state.panel_monthly.copy()

panel = state.panel_monthly.copy()
all_cols = [c for c in panel.columns if _is_numeric(panel[c])]

# ----------------------------------------------------------------------------
# Sidebar — global options
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Global options")
    cap_preview = st.checkbox("Limit preview to last 120 months", value=True)
    show_memory = st.checkbox("Show memory usage", value=False)

    st.markdown("---")
    if st.button("Reset panel to original", help="Restore the very first monthly panel snapshot."):
        state.panel_monthly = state._panel_monthly_orig.copy() if state._panel_monthly_orig is not None else state.panel_monthly
        st.experimental_rerun()

# ----------------------------------------------------------------------------
# Tabs: Transforms | Lags & Rolling | Missing | Standardize | Recipe
# ----------------------------------------------------------------------------

T1, T2, T3, T4, T5 = st.tabs([
    "Transforms",
    "Lags & Rolling",
    "Missing handling",
    "Standardize",
    "Recipe",
])

# =============
# TAB 1 — Transforms
# =============
with T1:
    st.subheader("Deterministic transforms")
    cols_sel = st.multiselect("Select numeric features", options=all_cols, default=all_cols[: min(10, len(all_cols))])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        do_diff = st.checkbox("diff(1)", value=False)
    with c2:
        do_pct = st.checkbox("pct_change(1)", value=False)
    with c3:
        do_log = st.checkbox("log(x)", value=False, help="Safe log: log(x) after shifting by a small epsilon if needed")
    with c4:
        do_winsor = st.checkbox("winsorize", value=False)

    c5, c6 = st.columns(2)
    with c5:
        w_low = st.slider("Winsor lower pctl", 0.0, 10.0, 1.0, step=0.5)
    with c6:
        w_high = st.slider("Winsor upper pctl", 90.0, 100.0, 99.0, step=0.5)

    st.markdown("— New columns will be appended with prefixes: `diff1__`, `pct1__`, `log__`, `win__`. Naming is collision‑safe.")

    def _safe_log(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        eps = max(1e-9, np.nanmin(s[s > 0]) * 1e-6) if np.any(s > 0) else 1e-6
        return np.log(s + eps)

    def _winsor(s: pd.Series, lo: float, hi: float) -> pd.Series:
        lo_v = np.nanpercentile(s, lo)
        hi_v = np.nanpercentile(s, hi)
        return s.clip(lo_v, hi_v)

    if st.button("Apply transforms", type="primary"):
        if not cols_sel:
            st.warning("Select at least one feature.")
        else:
            df = state.panel_monthly.copy()
            existing = set(df.columns)
            for c in cols_sel:
                s = pd.to_numeric(df[c], errors="coerce")
                if do_diff:
                    name = _make_unique(f"diff1__{_slug(c)}", existing)
                    df[name] = s.diff(1)
                if do_pct:
                    name = _make_unique(f"pct1__{_slug(c)}", existing)
                    df[name] = s.pct_change(1)
                if do_log:
                    name = _make_unique(f"log__{_slug(c)}", existing)
                    df[name] = _safe_log(s)
                if do_winsor:
                    name = _make_unique(f"win__{_slug(c)}_{int(w_low)}_{int(w_high)}", existing)
                    df[name] = _winsor(s, w_low, w_high)
            state.panel_monthly = df
            # update recipe
            rec = st.session_state.get("fe_recipe", [])
            rec.append({
                "op": "transform",
                "columns": cols_sel,
                "params": {"diff": do_diff, "pct1": do_pct, "log": do_log, "winsor": do_winsor, "w_low": w_low, "w_high": w_high},
            })
            st.session_state["fe_recipe"] = rec
            st.success("Transforms applied and recipe updated.")

# =============
# TAB 2 — Lags & Rolling
# =============
with T2:
    st.subheader("Generate lags and rolling features")
    cols_sel2 = st.multiselect("Select features for lags/rolling", options=all_cols, default=all_cols[: min(8, len(all_cols))], key="lag_cols")
    c1, c2 = st.columns(2)
    with c1:
        lag_list_str = st.text_input("Lags (comma‑sep)", value="1,3,6,12")
    with c2:
        roll_list_str = st.text_input("Rolling windows (comma‑sep)", value="3,6,12")

    r_stats = st.multiselect("Rolling stats", options=["mean", "std", "min", "max"], default=["mean", "std"]) 

    def _parse_int_list(s: str) -> list[int]:
        nums = []
        for tok in s.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = int(tok)
                if v > 0:
                    nums.append(v)
            except Exception:
                pass
        return sorted(set(nums))

    if st.button("Apply lags/rolling", type="primary"):
        L = _parse_int_list(lag_list_str)
        W = _parse_int_list(roll_list_str)
        if not cols_sel2 or (not L and not W):
            st.warning("Provide at least one column and one lag/window.")
        else:
            df = state.panel_monthly.copy()
            existing = set(df.columns)
            for c in cols_sel2:
                s = pd.to_numeric(df[c], errors="coerce")
                for l in L:
                    name = _make_unique(f"lag{l}__{_slug(c)}", existing)
                    df[name] = s.shift(l)
                for w in W:
                    r = s.rolling(w, min_periods=max(1, w // 2))
                    if "mean" in r_stats:
                        name = _make_unique(f"roll{w}_mean__{_slug(c)}", existing)
                        df[name] = r.mean()
                    if "std" in r_stats:
                        name = _make_unique(f"roll{w}_std__{_slug(c)}", existing)
                        df[name] = r.std(ddof=0)
                    if "min" in r_stats:
                        name = _make_unique(f"roll{w}_min__{_slug(c)}", existing)
                        df[name] = r.min()
                    if "max" in r_stats:
                        name = _make_unique(f"roll{w}_max__{_slug(c)}", existing)
                        df[name] = r.max()
            state.panel_monthly = df
            rec = st.session_state.get("fe_recipe", [])
            rec.append({"op": "lags_rolling", "columns": cols_sel2, "params": {"lags": L, "windows": W, "stats": r_stats}})
            st.session_state["fe_recipe"] = rec
            st.success("Lags/Rolling features generated.")

# =============
# TAB 3 — Missing handling
# =============
with T3:
    st.subheader("Missing values")
    method = st.selectbox("Fill method", ["none", "ffill", "bfill", "ffill_then_bfill"], index=0)
    drop_leading = st.checkbox("Drop leading all‑NA rows after fill", value=True)

    if st.button("Apply fill", type="primary"):
        df = state.panel_monthly.copy()
        if method == "ffill":
            df = df.ffill()
        elif method == "bfill":
            df = df.bfill()
        elif method == "ffill_then_bfill":
            df = df.ffill().bfill()
        # optional drop leading NaN rows
        if drop_leading:
            df = df.loc[~df.isna().all(axis=1)]
        state.panel_monthly = df
        rec = st.session_state.get("fe_recipe", [])
        rec.append({"op": "fill", "params": {"method": method, "drop_leading": drop_leading}})
        st.session_state["fe_recipe"] = rec
        st.success("Missing value handling applied.")

# =============
# TAB 4 — Standardize
# =============
with T4:
    st.subheader("Z-score standardization")
    cols_std = st.multiselect("Select features to z‑score (create new __z columns)", options=all_cols, default=[])
    expanding = st.checkbox("Use expanding mean/std (time‑aware)", value=False, help="If checked, uses expanding moments to avoid look‑ahead bias.")

    if st.button("Apply z‑score", type="primary"):
        df = state.panel_monthly.copy()
        existing = set(df.columns)
        for c in cols_std:
            s = pd.to_numeric(df[c], errors="coerce")
            if expanding:
                mu = s.expanding().mean()
                sd = s.expanding().std(ddof=0)
            else:
                mu = s.mean(); sd = s.std(ddof=0)
            z = (s - mu) / (sd + 1e-9)
            name = _make_unique(f"{_slug(c)}__z", existing)
            df[name] = z
        state.panel_monthly = df
        rec = st.session_state.get("fe_recipe", [])
        rec.append({"op": "zscore", "columns": cols_std, "params": {"expanding": expanding}})
        st.session_state["fe_recipe"] = rec
        st.success("Z-score features appended.")

# =============
# TAB 5 — Recipe (export/import/apply)
# =============
with T5:
    st.subheader("Recipe (JSON)")
    recipe = st.session_state.get("fe_recipe", [])
    st.code(json.dumps(recipe, indent=2), language="json")

    st.markdown("**Import recipe**")
    rec_txt = st.text_area("Paste JSON recipe here", value="", height=120)
    c1, c2, c3 = st.columns(3)

    def _apply_recipe(df: pd.DataFrame, rec: list[dict]) -> pd.DataFrame:
        existing = set(df.columns)
        def safe_log(s: pd.Series) -> pd.Series:
            s = s.astype(float); eps = max(1e-9, np.nanmin(s[s > 0]) * 1e-6) if np.any(s > 0) else 1e-6
            return np.log(s + eps)
        def winsor(s: pd.Series, lo: float, hi: float) -> pd.Series:
            return s.clip(np.nanpercentile(s, lo), np.nanpercentile(s, hi))
        def slug(x: str) -> str:
            return _slug(x)
        for step in rec:
            op = step.get("op")
            if op == "transform":
                cols = step.get("columns", [])
                P = step.get("params", {})
                for c in cols:
                    s = pd.to_numeric(df[c], errors="coerce") if c in df.columns else None
                    if s is None: continue
                    if P.get("diff"):
                        df[_make_unique(f"diff1__{slug(c)}", existing)] = s.diff(1)
                    if P.get("pct1"):
                        df[_make_unique(f"pct1__{slug(c)}", existing)] = s.pct_change(1)
                    if P.get("log"):
                        df[_make_unique(f"log__{slug(c)}", existing)] = safe_log(s)
                    if P.get("winsor"):
                        df[_make_unique(f"win__{slug(c)}_{int(P.get('w_low',1))}_{int(P.get('w_high',99))}", existing)] = winsor(s, float(P.get('w_low',1.0)), float(P.get('w_high',99.0)))
            elif op == "lags_rolling":
                cols = step.get("columns", [])
                P = step.get("params", {})
                L = P.get("lags", []) or []
                W = P.get("windows", []) or []
                stats = P.get("stats", ["mean"]) or []
                for c in cols:
                    if c not in df.columns: continue
                    s = pd.to_numeric(df[c], errors="coerce")
                    for l in L:
                        df[_make_unique(f"lag{int(l)}__{slug(c)}", existing)] = s.shift(int(l))
                    for w in W:
                        r = s.rolling(int(w), min_periods=max(1, int(w)//2))
                        if "mean" in stats:
                            df[_make_unique(f"roll{int(w)}_mean__{slug(c)}", existing)] = r.mean()
                        if "std" in stats:
                            df[_make_unique(f"roll{int(w)}_std__{slug(c)}", existing)] = r.std(ddof=0)
                        if "min" in stats:
                            df[_make_unique(f"roll{int(w)}_min__{slug(c)}", existing)] = r.min()
                        if "max" in stats:
                            df[_make_unique(f"roll{int(w)}_max__{slug(c)}", existing)] = r.max()
            elif op == "fill":
                P = step.get("params", {})
                m = P.get("method", "none")
                if m == "ffill": df = df.ffill()
                elif m == "bfill": df = df.bfill()
                elif m == "ffill_then_bfill": df = df.ffill().bfill()
                if P.get("drop_leading", True):
                    df = df.loc[~df.isna().all(axis=1)]
            elif op == "zscore":
                cols = step.get("columns", [])
                P = step.get("params", {})
                expanding = bool(P.get("expanding", False))
                for c in cols:
                    if c not in df.columns: continue
                    s = pd.to_numeric(df[c], errors="coerce")
                    if expanding:
                        mu = s.expanding().mean(); sd = s.expanding().std(ddof=0)
                    else:
                        mu = s.mean(); sd = s.std(ddof=0)
                    df[_make_unique(f"{slug(c)}__z", existing)] = (s - mu) / (sd + 1e-9)
        return df

    with c1:
        if st.button("Apply imported recipe", use_container_width=True):
            try:
                rec_in = json.loads(rec_txt) if rec_txt.strip() else []
                df = _apply_recipe(state.panel_monthly.copy(), rec_in)
                state.panel_monthly = df
                st.success("Imported recipe applied to panel.")
            except Exception as e:
                st.error(f"Invalid recipe JSON: {e}")
    with c2:
        if st.button("Clear recipe", use_container_width=True):
            st.session_state["fe_recipe"] = []
            st.info("Recipe cleared. Existing panel untouched.")
    with c3:
        if st.button("Download recipe.json", use_container_width=True):
            st.download_button("Click to download", data=json.dumps(recipe, indent=2).encode("utf-8"), file_name="feature_recipe.json")

# ----------------------------------------------------------------------------
# Preview & diagnostics
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("Preview & diagnostics")

pm = state.panel_monthly
if cap_preview and len(pm) > 120:
    pm_prev = pm.tail(120)
else:
    pm_prev = pm

st.markdown(f"**Panel shape:** {pm.shape[0]} rows × {pm.shape[1]} cols")
if show_memory:
    mem_mb = pm.memory_usage(deep=True).sum() / (1024 ** 2)
    st.caption(f"Approx. memory: {mem_mb:.2f} MB")

st.dataframe(pm_prev.tail(24), use_container_width=True)

# Presence heatmap
try:
    presence = pm_prev.notna().astype(int)
    hm = px.imshow(
        presence.T,
        aspect="auto",
        color_continuous_scale=[[0, "#F3F4F6"], [1, "#0EA5E9"]],
        title="Presence heatmap (preview)",
    )
    hm.update_layout(height=520, template="plotly_white", coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(hm, use_container_width=True)
except Exception:
    pass

# Optional overlay with target
if state.y_monthly is not None and not state.y_monthly.empty:
    st.markdown("**Overlay with target (pick a feature):**")
    num_cols = [c for c in pm.columns if _is_numeric(pm[c])]
    if num_cols:
        f = st.selectbox("Feature", options=num_cols, index=0)
        y, x = state.y_monthly.align(pm[f], join="inner")
        xz = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines", name="Target", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=xz.index, y=xz.values, mode="lines", name=f"std({f})"))
        fig.update_layout(template="plotly_white", height=360, title="Target vs standardized feature")
        st.plotly_chart(fig, use_container_width=True)

st.caption("Feature‑engineered panel is now ready for Backtesting.")
