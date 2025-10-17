# pages/2_Data_Aggregation.py — Data intake & monthly/quarterly panel builder (EN)
# ============================================================================
# Role
#   • Upload & normalize: Monthly target, Daily CSVs, Google Trends (weekly/monthly)
#   • Robust date handling (tz‑naive, EOM/EOQ), duplicate/NaN controls
#   • Daily → Monthly aggregation (mean/sum/last, business‑days toggle)
#   • Build clean MONTHLY and QUARTERLY panels and persist to AppState
#   • Coverage diagnostics, heatmap, and CSV/Excel export
#
# Notes
#   • Does NOT set page_config (centralized in app.py)
#   • Uses utils/* if available; otherwise falls back to local helpers
# ============================================================================

from __future__ import annotations

import json
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
            self.y_monthly: pd.Series | None = None
            self.panel_monthly: pd.DataFrame | None = None
            self.panel_quarterly: pd.DataFrame | None = None
            self.raw_daily: list[pd.DataFrame] = []
            self.google_trends: pd.DataFrame | None = None
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
# Try to import utils helpers, but keep local fallbacks
# ----------------------------------------------------------------------------
try:
    from utils.time_ops import to_datetime_naive as _to_dt, end_of_month as _eom  # type: ignore
except Exception:
    def _to_dt(v) -> pd.Series:
        return pd.to_datetime(pd.Series(v), errors="coerce").dt.tz_localize(None).dt.normalize()

    def _eom(s: pd.Series) -> pd.Series:
        return (_to_dt(s) + pd.offsets.MonthEnd(0)).dt.normalize()

try:
    from utils.io_ops import read_csv_safe as _read_csv, excel_export_bytes  # type: ignore
except Exception:
    def _read_csv(file):
        return pd.read_csv(file)

    def excel_export_bytes(panel_m, panel_q, preds, metrics, cfg_json) -> bytes:
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            if panel_m is not None:
                panel_m.reset_index().to_excel(writer, sheet_name="monthly_panel", index=False)
            if panel_q is not None:
                panel_q.reset_index().to_excel(writer, sheet_name="quarterly_panel", index=False)
            if metrics is not None:
                metrics.to_excel(writer, sheet_name="metrics", index=False)
            pd.DataFrame({"config_json": [cfg_json]}).to_excel(writer, sheet_name="config", index=False)
        return output.getvalue()

# ----------------------------------------------------------------------------
# Page header
# ----------------------------------------------------------------------------

st.title("🧱 Data & Aggregation")
st.caption("Upload daily / Google Trends / monthly target and build clean monthly/quarterly panels.")

# ----------------------------------------------------------------------------
# Sidebar — controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Aggregation settings")
    agg_method = st.selectbox("Daily → Monthly aggregation", ["mean", "sum", "last"], index=0)
    use_business_days = st.checkbox("Business days only (Mon–Fri)", value=False, help="If checked, weekends are ignored before monthly aggregation.")
    min_days_per_month = st.slider("Min. valid days per month", 1, 28, 10, help="If a month has fewer valid daily observations than this, set that month to NaN.")

    st.markdown("---")
    st.header("Panel cleaning")
    drop_const = st.checkbox("Drop constant / near-constant columns", value=True)
    corr_prune = st.checkbox("Prune highly correlated duplicates", value=False)
    corr_thr = st.slider("Correlation threshold", 0.80, 0.99, 0.95) if corr_prune else 0.95

    st.markdown("---")
    exp_to = st.radio("Export format", ["CSV", "Excel"], index=0)

# ----------------------------------------------------------------------------
# Step 1 — Upload (target + daily + GT)
# ----------------------------------------------------------------------------

st.subheader("1) Upload data")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    tgt_file = st.file_uploader("Monthly target CSV (date,value)", type=["csv"], key="tgt_agg")
with c2:
    daily_files = st.file_uploader("Daily CSV(s)", type=["csv"], accept_multiple_files=True, key="daily_agg")
with c3:
    gt_files = st.file_uploader("Google Trends CSV(s)", type=["csv"], accept_multiple_files=True, key="gt_agg")

# Local helpers ---------------------------------------------------------------

def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ("date", "Date", "ds", "time", "Time", "period", "Week", "Month"):
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return df.columns[0]


def load_target(file) -> pd.Series | None:
    try:
        df = _read_csv(file)
        dcol = _detect_date_col(df)
        vcol = [c for c in df.columns if c != dcol][0]
        df = df[[dcol, vcol]].copy()
        df.columns = ["date", "y"]
        df["date"] = _eom(df["date"])  # enforce EOM
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        s = df.dropna().set_index("date")["y"].sort_index()
        return s
    except Exception as e:
        st.error(f"Failed to load target: {e}")
        return None


def slugify(name: str) -> str:
    s = str(name).strip().lower().replace(" ", "_")
    s = (s.replace("/", "_").replace("(", "_").replace(")", "_").replace("-", "_").replace("__", "_")
           .replace("%", "pct"))
    return s


def load_daily(file) -> pd.DataFrame | None:
    try:
        df = _read_csv(file)
        dcol = _detect_date_col(df)
        df.rename(columns={dcol: "date"}, inplace=True)
        df["date"] = _to_dt(df["date"])  # daily
        num_cols = []
        for c in df.columns:
            if c == "date":
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
            num_cols.append(c)
        df = df[["date"] + num_cols].dropna(subset=["date"]).sort_values("date")
        # slugify and prefix by filename
        stem = Path(getattr(file, 'name', 'daily')).stem
        prefix = slugify(stem)
        rename_map = {c: f"{prefix}__{slugify(c)}" for c in num_cols}
        df.rename(columns=rename_map, inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to load daily CSV: {e}")
        return None


def load_gt(files: list) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = _read_csv(f)
            date_col = "Week" if "Week" in df.columns else ("Month" if "Month" in df.columns else _detect_date_col(df))
            series_cols = [c for c in df.columns if c != date_col]
            df = df[[date_col] + series_cols].copy()
            df.rename(columns={date_col: "date"}, inplace=True)
            df["date"] = _to_dt(df["date"])  # weekly or monthly
            new_cols = [f"gt__{slugify(c)}" for c in series_cols]
            df.columns = ["date"] + new_cols
            for c in new_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            frames.append(df)
        except Exception as e:
            st.warning(f"GT load failed for one file: {e}")
    if not frames:
        return pd.DataFrame(columns=["date"])  # empty
    g = frames[0]
    for k in frames[1:]:
        g = pd.merge(g, k, on="date", how="outer")
    return g.sort_values("date").reset_index(drop=True)


# Persist uploaded assets to state (if provided)
if tgt_file is not None:
    state.y_monthly = load_target(tgt_file)
    if state.y_monthly is not None:
        st.success(f"Target loaded: {len(state.y_monthly)} months")

if daily_files:
    state.raw_daily = []
    for f in daily_files:
        d = load_daily(f)
        if d is not None:
            state.raw_daily.append(d)
    st.success(f"Loaded {len(state.raw_daily)} daily file(s)")

if gt_files:
    state.google_trends = load_gt(gt_files)
    st.success(f"Google Trends: {0 if state.google_trends is None else state.google_trends.shape[1]-1} series")

# Preview target
if state.y_monthly is not None:
    st.line_chart(state.y_monthly.rename("target"))

st.markdown("---")

# ----------------------------------------------------------------------------
# Step 2 — Aggregation functions
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def aggregate_daily_to_monthly(
    df: pd.DataFrame,
    method: str = "mean",
    business_days_only: bool = False,
    min_days: int = 10,
) -> pd.DataFrame:
    """Aggregate a daily frame into monthly EOM index.
    Returns: DataFrame with columns preserved and 'date' aligned to EOM.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])  # empty
    d = df.copy()
    d["date"] = _to_dt(d["date"])  # normalize
    if business_days_only:
        d = d[d["date"].dt.dayofweek < 5]
    d = d.set_index("date")
    if method == "sum":
        m = d.resample("M").sum(min_count=1)
    elif method == "last":
        m = d.resample("M").last()
    else:
        m = d.resample("M").mean()
    # drop months with too few valid rows
    counts = d.resample("M").count()
    m[counts < min_days] = np.nan
    m = m.reset_index()
    m["date"] = _eom(m["date"])  # enforce EOM
    return m


@st.cache_data(show_spinner=False)
def build_monthly_panel(
    daily_frames: list[pd.DataFrame],
    gt_frame: pd.DataFrame | None,
    method: str,
    business_days_only: bool,
    min_days: int,
) -> pd.DataFrame:
    panel = None
    for df in (daily_frames or []):
        if df is None or df.empty:
            continue
        m = aggregate_daily_to_monthly(df, method=method, business_days_only=business_days_only, min_days=min_days)
        panel = m if panel is None else panel.merge(m, on="date", how="outer")
    if gt_frame is not None and not gt_frame.empty:
        # GT weekly/monthly → monthly mean by default
        gtm = gt_frame.set_index("date").resample("M").mean().reset_index()
        gtm["date"] = _eom(gtm["date"])  # EOM
        panel = gtm if panel is None else panel.merge(gtm, on="date", how="outer")
    if panel is None:
        return pd.DataFrame()
    panel = panel.sort_values("date").set_index("date")
    # coerce numeric
    for c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    return panel


@st.cache_data(show_spinner=False)
def build_quarterly_panel(panel_m: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if panel_m is None or panel_m.empty:
        return pd.DataFrame()
    if how == "last":
        q = panel_m.resample("Q").last()
    elif how == "sum":
        q = panel_m.resample("Q").sum(min_count=1)
    else:
        q = panel_m.resample("Q").mean()
    q.index = pd.to_datetime(q.index).tz_localize(None)
    return q


def drop_constant_cols(df: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    keep = []
    for c in df.columns:
        s = df[c].dropna().values
        if len(s) == 0:
            continue
        if np.nanmax(s) - np.nanmin(s) > tol:
            keep.append(c)
    return df[keep]


def prune_correlated(df: pd.DataFrame, thr: float = 0.95) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    corr = df.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column].abs() > thr)]
    return df.drop(columns=drop_cols, errors="ignore")

# ----------------------------------------------------------------------------
# Step 3 — Build panel
# ----------------------------------------------------------------------------

st.subheader("2) Build monthly/quarterly panels")

build_btn = st.button("Build panel", type="primary")

if build_btn:
    with st.spinner("Aggregating and merging…"):
        panel_m = build_monthly_panel(state.raw_daily, state.google_trends, agg_method, use_business_days, min_days_per_month)
        if panel_m is None or panel_m.empty:
            st.error("No input to build from. Upload at least one daily file (and optionally GT).")
            st.stop()
        # cleaning
        if drop_const:
            panel_m = drop_constant_cols(panel_m)
        if corr_prune:
            panel_m = prune_correlated(panel_m, thr=corr_thr)
        # align with target
        if state.y_monthly is not None:
            panel_m = panel_m.loc[(panel_m.index >= state.y_monthly.index.min()) & (panel_m.index <= state.y_monthly.index.max())]
        state.panel_monthly = panel_m
        state.panel_quarterly = build_quarterly_panel(panel_m, how="mean")
    st.success(f"Built monthly panel: {state.panel_monthly.shape[0]} rows × {state.panel_monthly.shape[1]} features")

# Preview tables
if state.panel_monthly is not None and not state.panel_monthly.empty:
    st.markdown("### Monthly panel (tail)")
    st.dataframe(state.panel_monthly.tail(12), use_container_width=True)

    # Coverage diagnostics
    st.markdown("---")
    st.subheader("Coverage diagnostics")
    cov = state.panel_monthly.notna().mean().sort_values(ascending=False)
    st.dataframe(cov.to_frame("coverage_ratio").style.format({"coverage_ratio": "{:.2%}"}), use_container_width=True)

    # Heatmap of presence
    try:
        presence = state.panel_monthly.notna().astype(int)
        hm = px.imshow(
            presence.tail(60).T,
            aspect="auto",
            color_continuous_scale=[[0, "#F3F4F6"], [1, "#0EA5E9"]],
            title="Presence heatmap (last 60 months)"
        )
        hm.update_layout(height=520, template="plotly_white", coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(hm, use_container_width=True)
    except Exception:
        pass

    # Simple correlation with target (optional)
    if state.y_monthly is not None:
        y_al, X_al = state.y_monthly.align(state.panel_monthly, join="inner")
        if not X_al.empty:
            corr = X_al.corrwith(y_al).sort_values(ascending=False).head(15)
            st.markdown("**Top correlations with target (whole sample)**")
            st.dataframe(corr.to_frame("corr").style.format({"corr": "{:.3f}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("Quarterly panel (tail)")
    st.dataframe(state.panel_quarterly.tail(12) if state.panel_quarterly is not None else pd.DataFrame(), use_container_width=True)

# ----------------------------------------------------------------------------
# Step 4 — Export & config snapshot
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("3) Export & configuration snapshot")

cfg = {
    "agg_method": agg_method,
    "business_days_only": use_business_days,
    "min_days_per_month": int(min_days_per_month),
    "drop_const": bool(drop_const),
    "corr_prune": bool(corr_prune),
    "corr_thr": float(corr_thr),
}

cfg_json = json.dumps(cfg, indent=2)
st.code(cfg_json, language="json")

if state.panel_monthly is not None and not state.panel_monthly.empty:
    if exp_to == "CSV":
        st.download_button("Download monthly_panel.csv", data=state.panel_monthly.to_csv().encode("utf-8"), file_name="monthly_panel.csv")
        if state.panel_quarterly is not None:
            st.download_button("Download quarterly_panel.csv", data=state.panel_quarterly.to_csv().encode("utf-8"), file_name="quarterly_panel.csv")
    else:
        try:
            xlsb = excel_export_bytes(state.panel_monthly, state.panel_quarterly, None, None, cfg_json)
            st.download_button("Download panels.xlsx", data=xlsb, file_name="panels.xlsx")
        except Exception as e:
            st.info(f"Excel export failed: {e}")

st.caption("Panels are persisted in session state and ready for Feature Engineering & Backtesting.")
