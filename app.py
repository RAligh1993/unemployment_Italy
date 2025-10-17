# app.py — ISTAT Unemployment Nowcasting Lab (Core Shell, EN)
# =============================================================================
# Purpose
#   • Lightweight home page (navigation + theme)
#   • Central session state bootstrap (via utils/state.py, with safe fallback)
#   • Quick data intake (monthly target + daily CSVs + Google Trends)
#   • Instant sanity checks & mini correlation preview
#   • Hands off heavy lifting to pages/* and utils/* modules
# =============================================================================

from __future__ import annotations

import os
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

APP_NAME = "ISTAT Nowcasting Lab"
APP_VERSION = "1.0.0"

# =============================================================================
# 0) PAGE CONFIG & THEME
# =============================================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS loader ---------------------------------------------------------------
DEFAULT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root { --pri:#0F766E; --acc:#7C3AED; --ink:#0F172A; --mut:#6B7280; }
*{font-family:Inter,system-ui,-apple-system,'Segoe UI',Roboto,Arial}
.block-container{padding-top:1.2rem}
h1,h2,h3{color:var(--pri);letter-spacing:.2px}
.badge{display:inline-flex;align-items:center;gap:.5rem;padding:.2rem .6rem;border-radius:999px;background:#EEF2FF;color:var(--acc);font-weight:600;border:1px solid #E5E7EB}
.card{background:#fff;border:1px solid #E5E7EB;border-radius:14px;padding:16px;box-shadow:0 2px 12px rgba(0,0,0,.04)}
.kpi{background:#F8FAFC;border-radius:14px;padding:10px 12px;border:1px solid #EDF2F7}
.small{color:var(--mut);font-size:.85rem}
.stButton>button{background:linear-gradient(135deg,var(--pri),var(--acc));color:#fff;border:0;border-radius:12px;padding:.55rem .9rem;box-shadow:0 6px 18px rgba(15,118,110,.25)}
.stButton>button:hover{filter:brightness(1.05);transform:translateY(-1px)}
hr{border:none;height:1px;background:#E5E7EB}
"""

def load_css() -> None:
    css_path = Path("assets/style.css")
    css = DEFAULT_CSS
    if css_path.exists():
        try:
            css = css_path.read_text(encoding="utf-8")
        except Exception:
            pass
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# =============================================================================
# 1) CENTRAL STATE (robust import with fallback)
# =============================================================================
try:
    from utils.state import AppState  # type: ignore
except Exception:  # minimal fallback to keep Home usable before utils exists
    class _State:
        def __init__(self) -> None:
            self.y_monthly: pd.Series | None = None
            self.panel_monthly: pd.DataFrame | None = None
            self.panel_quarterly: pd.DataFrame | None = None
            self.bt_results: dict[str, pd.Series] = {}
            self.bt_metrics: pd.DataFrame | None = None
            self.raw_daily: list[pd.DataFrame] = []
            self.google_trends: pd.DataFrame | None = None

    class AppState:  # type: ignore
        @staticmethod
        def init() -> _State:
            if "_app" not in st.session_state:
                st.session_state["_app"] = _State()
            return t.cast(_State, st.session_state["_app"])  # noqa: E999
        @staticmethod
        def get() -> _State:
            return AppState.init()

state = AppState.init()

# =============================================================================
# 2) SIDEBAR – Navigation & environment status
# =============================================================================
with st.sidebar:
    if Path("assets/logo.svg").exists():
        st.image("assets/logo.svg", width=120)
    st.title("Navigation")

    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/1_Dashboard.py", label="📊 Dashboard")
    st.page_link("pages/2_Data_Aggregation.py", label="🧱 Data & Aggregation")
    st.page_link("pages/3_Feature_Engineering.py", label="🧪 Feature Engineering")
    st.page_link("pages/4_Backtesting.py", label="🧮 Backtesting")
    st.page_link("pages/5_Results.py", label="📈 Results")
    st.page_link("pages/6_AI_Assistant.py", label="🤖 AI Assistant")
    st.page_link("pages/7_SHAP_Events.py", label="🧭 SHAP & Events")
    st.page_link("pages/8_News_Impact.py", label="📰 News Impact")
    st.page_link("pages/9_Report.py", label="📋 Final Report")

    st.markdown("---")
    st.markdown(f"<span class='badge'>v{APP_VERSION}</span>", unsafe_allow_html=True)

    def env_badge(name: str) -> None:
        ok = bool(os.getenv(name))
        color = "#10B981" if ok else "#F59E0B"
        st.markdown(f"<div class='small'>🔑 {name}: <b style='color:{color}'>{'set' if ok else 'missing'}</b></div>", unsafe_allow_html=True)

    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "NEWSAPI_KEY"):
        env_badge(key)

# =============================================================================
# 3) HERO
# =============================================================================

st.title("📈 ISTAT Nowcasting Lab — Home")
st.caption("Orchestrator page: theme, central state, quick intake, and sanity checks.")

if state.bt_metrics is not None and not state.bt_metrics.empty:
    best = state.bt_metrics.sort_values("MAE").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", f"{best['model']}")
    c2.metric("MAE", f"{best['MAE']:.3f}")
    c3.metric("RMSE", f"{best['RMSE']:.3f}")
    c4.metric("SMAPE", f"{best.get('SMAPE', np.nan):.3f}")
else:
    st.info("No backtest yet. Use the left menu to build data and run models.")

st.markdown("---")

# =============================================================================
# 4) QUICK START — Optional one‑screen intake
# =============================================================================

st.subheader("🚀 Quick Start (optional)")
st.write(
    "Upload a **monthly target** (CSV with columns: `date,value`), any number of **daily CSVs**, and optional **Google Trends** CSVs (weekly or monthly).\n"
    "This creates a preview and basic correlations. For full control, prefer Pages 2–3."
)

c1, c2, c3 = st.columns([1.3, 1, 1])
with c1:
    tgt_file = st.file_uploader("Monthly target CSV", type=["csv"], key="tgt_home")
with c2:
    daily_files = st.file_uploader("Daily CSV(s)", type=["csv"], accept_multiple_files=True, key="daily_home")
with c3:
    gt_files = st.file_uploader("Google Trends CSV(s)", type=["csv"], accept_multiple_files=True, key="gt_home")

# --- helper functions (local to Home) ----------------------------------------

def detect_date_col(df: pd.DataFrame) -> str:
    for c in ("date", "Date", "ds", "time", "Time", "period", "Week", "Month"):
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return df.columns[0]


def to_dt(s: t.Iterable) -> pd.Series:
    return pd.to_datetime(pd.Series(s), errors="coerce").dt.tz_localize(None).dt.normalize()


def end_of_month(s: pd.Series) -> pd.Series:
    return (to_dt(s) + pd.offsets.MonthEnd(0)).dt.normalize()


def read_csv_safe(file) -> pd.DataFrame:
    try:
        from utils.io_ops import read_csv_safe as _read  # type: ignore
        return _read(file)
    except Exception:
        return pd.read_csv(file)


def load_target(file) -> pd.Series | None:
    try:
        df = read_csv_safe(file)
        dcol = detect_date_col(df)
        vcol = [c for c in df.columns if c != dcol][0]
        df = df[[dcol, vcol]].copy()
        df.columns = ["date", "y"]
        df["date"] = end_of_month(df["date"])  # enforce monthly EOM
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        s = df.dropna().set_index("date")["y"].sort_index()
        return s
    except Exception as e:
        st.error(f"Failed to load target: {e}")
        return None


def load_daily(file) -> pd.DataFrame | None:
    try:
        df = read_csv_safe(file)
        dcol = detect_date_col(df)
        df.rename(columns={dcol: "date"}, inplace=True)
        df["date"] = to_dt(df["date"])  # daily expected
        for c in df.columns:
            if c == "date":
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.columns = ["date"] + [str(c).strip().replace(" ", "_") for c in df.columns[1:]]
        return df.dropna(subset=["date"]).sort_values("date")
    except Exception as e:
        st.error(f"Failed to load daily CSV: {e}")
        return None


def load_gt(files: list) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = read_csv_safe(f)
            date_col = "Week" if "Week" in df.columns else ("Month" if "Month" in df.columns else detect_date_col(df))
            series_cols = [c for c in df.columns if c != date_col]
            df = df[[date_col] + series_cols].copy()
            df.rename(columns={date_col: "date"}, inplace=True)
            df["date"] = to_dt(df["date"])  # weekly or monthly
            new_cols = [f"gt_{str(c).lower().strip().replace(' ', '_')}" for c in series_cols]
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

# --- wire Quick Start ---------------------------------------------------------

if tgt_file is not None:
    state.y_monthly = load_target(tgt_file)
    if state.y_monthly is not None:
        st.success(
            f"Target loaded: {len(state.y_monthly)} months · "
            f"[{state.y_monthly.index.min().date()} → {state.y_monthly.index.max().date()}]"
        )
        st.line_chart(state.y_monthly)

if daily_files:
    state.raw_daily = []
    for f in daily_files:
        d = load_daily(f)
        if d is not None:
            state.raw_daily.append(d)
    st.success(f"Loaded {len(state.raw_daily)} daily file(s)")

if gt_files:
    state.google_trends = load_gt(gt_files)
    st.success(
        f"Google Trends: {0 if state.google_trends is None else state.google_trends.shape[1]-1} series, "
        f"{0 if state.google_trends is None else len(state.google_trends)} rows"
    )

# =============================================================================
# 5) MINI PREVIEW — quick monthly resample + top correlations
# =============================================================================
if state.y_monthly is not None and state.raw_daily:
    st.markdown("### 🔎 Instant monthly preview & correlations")
    panel = None
    for df in state.raw_daily:
        m = df.set_index("date").resample("M").mean().reset_index()
        m["date"] = end_of_month(m["date"])  # EOM alignment avoids merge errors
        panel = m if panel is None else panel.merge(m, on="date", how="outer")

    # add Google Trends monthly mean if present
    if state.google_trends is not None and not state.google_trends.empty:
        gtm = state.google_trends.set_index("date").resample("M").mean().reset_index()
        gtm["date"] = end_of_month(gtm["date"]) 
        panel = gtm if panel is None else panel.merge(gtm, on="date", how="outer")

    if panel is not None:
        panel = panel.sort_values("date").set_index("date")
        y = state.y_monthly
        X = panel.select_dtypes(include=[np.number])
        y_al, X_al = y.align(X, join="inner")
        if not X_al.empty and len(X_al.columns) > 0:
            corr = X_al.corrwith(y_al).sort_values(ascending=False)
            top = corr.head(10).dropna()
            st.dataframe(top.to_frame("corr_with_target").style.format({"corr_with_target": "{:.3f}"}), use_container_width=True)

            show_cols = list(top.index[:3])
            if show_cols:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_al.index, y=y_al.values, name="Target", mode="lines", line=dict(width=3)))
                for c in show_cols:
                    s = X_al[c]
                    s = (s - s.mean()) / (s.std() + 1e-9)
                    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=c, mode="lines"))
                fig.update_layout(template="plotly_white", height=360, title="Target vs top standardized features (preview)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload at least one daily file with numeric columns to get a preview.")
else:
    st.info("For an instant preview, upload a monthly target and at least one daily CSV.")

# =============================================================================
# 6) NEXT STEPS
# =============================================================================

st.markdown("---")
st.subheader("Next steps")
col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/2_Data_Aggregation.py", label="➡️ Build clean monthly/quarterly panels", help="Resample & merge with safe date handling")
with col2:
    st.page_link("pages/4_Backtesting.py", label="➡️ Run walk‑forward backtests", help="Choose models and evaluate")
with col3:
    st.page_link("pages/7_SHAP_Events.py", label="➡️ Explain & analyze events", help="SHAP + Timeline")

st.caption("© Nowcasting Lab · Experimental research tool. Use responsibly.")
