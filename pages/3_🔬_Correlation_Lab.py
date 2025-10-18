"""
🇮🇹 ITALIAN UNEMPLOYMENT DATA – HARDENED, FULLY AUTOMATED BUILD
Version: 3.1
Date: 2025-10-18

Run locally:
  streamlit run italian_auto_fetch_app_hardened.py

Key changes vs v3.0:
- Robust Eurostat fetch with *dynamic* dimension resolution + multi-step fallbacks
- Reliable Yahoo Finance with multi-ticker + start/period fallback + retry/backoff
- Google Trends with chunking, exponential backoff, and automatic column merge
- Concurrent fetching for speed (ThreadPoolExecutor)
- Better status/logging, clearer errors, deterministic caching
- Trends tab shows multi-series selector

Dependencies (requirements.txt):
streamlit
pandas
numpy
plotly
scipy
eurostat
yfinance
pytrends
requests
xlsxwriter
"""

import os
import time
import math
import random
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Italian Data Auto‑Fetch (Hardened)",
    page_icon="🇮🇹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero{background:linear-gradient(90deg,#009246 0%,#fff 33%,#fff 66%,#CE2B37 100%);padding:36px;border-radius:16px;text-align:center;margin-bottom:24px}
    .hero h1{margin:0;font-weight:900}
    .kpi{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:18px;border-radius:14px;text-align:center}
    .card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin-bottom:12px}
    code{white-space:pre-wrap}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------
ITALIAN_KEYWORDS = [
    "offerte di lavoro",
    "disoccupazione",
    "naspi",
    "indeed lavoro",
    "ricerca lavoro",
    "cerco lavoro",
    "centro per l'impiego",
    "cassa integrazione",
    "reddito di cittadinanza",
    "curriculum vitae",
]

DATA_SOURCES = {
    "unemp": {
        "name": "Italian Unemployment Rate",
        "provider": "Eurostat",
        "dataset": "une_rt_m",
        "mandatory": True,
    },
    "cci": {
        "name": "Consumer Confidence Index",
        "provider": "Eurostat",
        "dataset": "ei_bsco_m",
        "mandatory": False,
    },
    "hicp": {
        "name": "HICP (All items)",
        "provider": "Eurostat",
        "dataset": "prc_hicp_midx",
        "mandatory": False,
    },
    "iip": {
        "name": "Industrial Production Index",
        "provider": "Eurostat",
        "dataset": "sts_inpr_m",
        "mandatory": False,
    },
    "mib": {
        "name": "FTSE MIB Index",
        "provider": "Yahoo Finance",
        "dataset": "^FTSEMIB",
        "mandatory": False,
    },
    "vix": {
        "name": "V2TX/VIX Volatility",
        "provider": "Yahoo Finance",
        "dataset": "^V2TX",
        "mandatory": False,
    },
    "trends": {
        "name": "Google Trends (job keywords)",
        "provider": "Google Trends",
        "dataset": "keywords",
        "mandatory": False,
    },
}

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def _backoff_sleep(base=1.5, attempt=1, max_seconds=30):
    # exponential backoff with jitter
    delay = min(max_seconds, base ** attempt + random.uniform(0, 0.5))
    time.sleep(delay)


def _internet_ok(url="https://www.google.com", timeout=5) -> bool:
    try:
        import requests
        requests.get(url, timeout=timeout)
        return True
    except Exception:
        return False


@st.cache_data(ttl=1800, show_spinner=False)
def _eurostat_pars(dataset: str) -> Dict[str, List[str]]:
    # Cache parameter spaces to avoid repeated metadata calls
    import eurostat
    return eurostat.get_par_values(dataset)


def _eurostat_resolve(dataset: str, prefs: Dict[str, List[str]]) -> Dict[str, str]:
    """Pick the first available value for each dimension based on preferences."""
    pars = _eurostat_pars(dataset)
    resolved = {}
    # Only choose for dims that actually exist
    for dim, pref_list in prefs.items():
        options = pars.get(dim, [])
        for v in pref_list:
            if v in options:
                resolved[dim] = v
                break
    return resolved


@st.cache_data(ttl=3600)
def fetch_eurostat(dataset: str, preferred_filters: Dict[str, List[str]], start_year: int) -> Tuple[Optional[pd.DataFrame], str]:
    """Robust Eurostat fetch: resolve dims, try filtered pulls, fallbacks, tidy output."""
    try:
        import eurostat
        # 1) try preferred resolution
        resolved = _eurostat_resolve(dataset, preferred_filters)
        attempts: List[Dict[str, str]] = []
        if resolved:
            attempts.append(resolved)
        # 2) fallbacks (drop the strict dims one by one)
        # common dimension priorities to relax
        relax_order = ["s_adj", "unit", "sex", "age", "indic", "indic_bt", "nace_r2"]
        for dim in relax_order:
            if dim in resolved:
                att = {k: v for k, v in resolved.items() if k != dim}
                if att not in attempts:
                    attempts.append(att)
        last_err = None
        for filt in attempts:
            try:
                df = eurostat.get_data_df(dataset, filter_pars=filt, flags=False)
                if df is None or df.empty:
                    continue
                # tidy
                # time cols look like '2000M01' etc.
                time_cols = [c for c in df.columns if isinstance(c, str) and "M" in c and c.replace("M", "").isdigit()]
                if not time_cols:
                    # some tables come already tidy; attempt to find 'time' column
                    if "time" in df.columns and "values" in df.columns:
                        tidy = df[["time", "values"]].rename(columns={"time": "date", "values": "value"})
                        tidy["date"] = pd.to_datetime(tidy["date"]) + pd.offsets.MonthEnd(0)
                        tidy = tidy.sort_values("date")
                        tidy = tidy[tidy["date"].dt.year >= start_year]
                        if tidy.empty:
                            continue
                        return tidy.reset_index(drop=True), f"✅ {dataset} fetched with filters {filt}"
                    continue
                id_cols = [c for c in df.columns if c not in time_cols]
                melted = df.melt(id_vars=id_cols, value_vars=time_cols, var_name="period", value_name="value")
                melted["period"] = melted["period"].astype(str).str.replace("M", "-")
                melted["date"] = pd.to_datetime(melted["period"], format="%Y-%m", errors="coerce") + pd.offsets.MonthEnd(0)
                melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
                out = melted[["date", "value"]].dropna().sort_values("date")
                out = out[out["date"].dt.year >= start_year]
                if out.empty:
                    continue
                return out.reset_index(drop=True), f"✅ {dataset} fetched with filters {filt}"
            except Exception as e:
                last_err = e
                continue
        if last_err:
            return None, f"❌ Eurostat error: {last_err}"
        return None, "❌ Eurostat: no data with given filters/fallbacks"
    except ImportError:
        return None, "❌ eurostat not installed"


@st.cache_data(ttl=3600)
def fetch_yahoo(tickers: List[str], start_year: int) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        import yfinance as yf
    except ImportError:
        return None, "❌ yfinance not installed"
    start = f"{start_year}-01-01"
    errors = []
    for t in tickers:
        for attempt in range(1, 4):
            try:
                # Try history via Ticker first
                df = yf.Ticker(t).history(start=start, auto_adjust=True)
                if df is None or df.empty:
                    # fallback: period=max
                    df = yf.download(t, period="max", auto_adjust=True, progress=False)
                if df is not None and not df.empty:
                    res = pd.DataFrame({
                        "date": pd.to_datetime(df.index).tz_localize(None),
                        "close": df["Close"].values,
                        "volume": df.get("Volume", pd.Series([np.nan]*len(df))).values,
                    })
                    res = res[res["date"].dt.year >= start_year].reset_index(drop=True)
                    if not res.empty:
                        return res, f"✅ {t} {len(res)} rows"
            except Exception as e:
                errors.append(str(e))
                _backoff_sleep(attempt=attempt)
                continue
    return None, f"❌ Yahoo: all tickers failed ({'; '.join(errors[-2:])})"


@st.cache_data(ttl=1800)
def fetch_trends(keywords: List[str], geo: str, start_year: int) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return None, "❌ pytrends not installed"
    if not keywords:
        return None, "❌ no keywords"
    end_date = datetime.now().strftime("%Y-%m-%d")
    py = TrendReq(hl="it-IT", tz=60, timeout=(10, 30), retries=0)
    chunks = [keywords[i:i+5] for i in range(0, len(keywords), 5)]
    merged = None
    for chunk in chunks:
        ok = False
        for attempt in range(1, 4):
            try:
                py.build_payload(chunk, cat=0, timeframe=f"{start_year}-01-01 {end_date}", geo=geo)
                df = py.interest_over_time()
                if df is None or df.empty:
                    raise RuntimeError("empty trends")
                df = df.drop(columns=[c for c in df.columns if c.lower()=="ispartial"], errors="ignore")
                df = df.reset_index().rename(columns={"date": "date"})
                df["date"] = pd.to_datetime(df["date"]) + pd.offsets.Week(weekday=6)  # align to week end
                # rename columns to safe names
                rename = {c: f"gt_{c.replace(' ', '_')}" for c in df.columns if c != "date"}
                df = df.rename(columns=rename)
                merged = df if merged is None else pd.merge(merged, df, on="date", how="outer")
                ok = True
                break
            except Exception:
                _backoff_sleep(attempt=attempt)
        if not ok:
            return None, "❌ Google Trends rate-limited or unavailable"
    merged = merged.sort_values("date").reset_index(drop=True)
    # drop all-NaN cols (rare)
    merged = merged.dropna(axis=1, how="all")
    return merged, f"✅ Trends {len(merged)} weeks, {len(keywords)} keywords"


# -----------------------------------------------------------------------------
# VISUALS
# -----------------------------------------------------------------------------

def line_fig(df: pd.DataFrame, y_cols: List[str], title: str):
    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col], mode="lines", name=col,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title=title, template="plotly_white", height=480,
        hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        xaxis_title="Date", yaxis_title="Value"
    )
    return fig


def describe_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    s = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame({"Metric": ["Observations"], "Value": ["0"]})
    stats = {
        "Metric": ["Latest", "Mean", "Median", "Std", "Min", "Max", "Range", "Observations"],
        "Value": [f"{s.iloc[-1]:.2f}", f"{s.mean():.2f}", f"{s.median():.2f}", f"{s.std():.2f}", f"{s.min():.2f}", f"{s.max():.2f}", f"{(s.max()-s.min()):.2f}", f"{len(s)}"],
    }
    return pd.DataFrame(stats)


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/03/Flag_of_Italy.svg/320px-Flag_of_Italy.svg.png", width=96)
    st.markdown("### ⚙️ Settings")
    start_year = st.slider("Start Year", min_value=2000, max_value=datetime.now().year, value=2010)

    st.markdown("---")
    st.markdown("### 📊 Data Sources")
    selected = {}
    for key, info in DATA_SOURCES.items():
        if info.get("mandatory", False):
            selected[key] = st.checkbox(f"{info['name']} (mandatory)", value=True, disabled=True)
        else:
            selected[key] = st.checkbox(info["name"], value=(key in ["cci", "hicp"]))

    if selected.get("trends", False):
        n_kw = st.slider("Keywords count", 1, 10, 5)
        chosen_kw = ITALIAN_KEYWORDS[:n_kw]
    else:
        chosen_kw = []

    st.markdown("---")
    auto_resolve = st.toggle("Auto‑resolve Eurostat dimensions", value=True, help="Pick best available codes (unit/s_adj/etc) dynamically with fallbacks.")
    concurrent = st.toggle("Concurrent fetching", value=True)

    st.markdown("---")
    fetch = st.button("🚀 Fetch Data", type="primary", use_container_width=True)
    clr = st.button("🔄 Clear Cache", use_container_width=True)
    if clr:
        st.cache_data.clear()
        st.success("Cache cleared")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
st.markdown('<div class="hero"><h1>Italian Economic Data – Auto Fetch</h1><p>Eurostat • Yahoo Finance • Google Trends</p></div>', unsafe_allow_html=True)

if not fetch:
    st.info("Select sources in the sidebar and hit **Fetch Data**. No manual downloads needed.")
    st.stop()

# Health check
if not _internet_ok():
    st.error("No internet connectivity. Check connection and retry.")
    st.stop()

# Preferences for Eurostat dimensions (priority-ordered lists)
EUROSTAT_PREFS = {
    # Unemployment rate
    "une_rt_m": {
        "geo": ["IT"],
        "s_adj": ["SA", "SCA", "NSA"],
        "sex": ["T"],
        "age": ["TOTAL", "Y15-74", "Y15-64"],
        "unit": ["PC_ACT", "PC_POP", "PC_Y15-74"],
    },
    # Consumer Confidence
    "ei_bsco_m": {
        "geo": ["IT"],
        "indic": ["BS-CSMCI", "BS-CSMCI-BAL"],
        "s_adj": ["NSA", "SA"],
        "unit": ["BAL"],
    },
    # HICP
    "prc_hicp_midx": {
        "geo": ["IT"],
        "coicop": ["CP00"],
        "unit": ["I15", "I21", "I2015=100", "I2015"],
    },
    # Industrial Production
    "sts_inpr_m": {
        "geo": ["IT"],
        "s_adj": ["SCA", "SA", "NSA"],
        "nace_r2": ["B-E", "B-D"],
        "indic_bt": ["PRD"],
        "unit": ["I15", "I21", "I2015=100", "I2015"],
    },
}

# -----------------------------------------------------------------------------
# FETCH PIPELINE
# -----------------------------------------------------------------------------

jobs = []
results: Dict[str, Tuple[Optional[pd.DataFrame], str]] = {}
log_box = st.empty()
progress = st.progress(0.0)
sel_keys = [k for k, v in selected.items() if v]


def submit_job(key: str):
    info = DATA_SOURCES[key]
    if info["provider"] == "Eurostat":
        prefs = EUROSTAT_PREFS.get(info["dataset"], {}) if auto_resolve else {}
        return fetch_eurostat(info["dataset"], prefs, start_year)
    elif info["provider"] == "Yahoo Finance":
        if key == "mib":
            tickers = ["^FTSEMIB", "FTSEMIB.MI", "EWI"]
        else:
            tickers = ["^V2TX", "^VIX"]
        return fetch_yahoo(tickers, start_year)
    elif info["provider"] == "Google Trends":
        return fetch_trends(chosen_kw, "IT", max(start_year, 2015))
    else:
        return None, "❌ unknown provider"


if concurrent and len(sel_keys) > 1:
    with ThreadPoolExecutor(max_workers=min(6, len(sel_keys))) as ex:
        future_map = {ex.submit(submit_job, k): k for k in sel_keys}
        done = 0
        for fut in as_completed(future_map):
            k = future_map[fut]
            try:
                results[k] = fut.result()
            except Exception as e:
                results[k] = (None, f"❌ {e}")
            done += 1
            progress.progress(done / len(sel_keys))
else:
    for i, k in enumerate(sel_keys, 1):
        results[k] = submit_job(k)
        progress.progress(i / len(sel_keys))

progress.progress(1.0)

# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------

succ = {k: v for k, v in results.items() if v[0] is not None}
fail = {k: v for k, v in results.items() if v[0] is None}

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='kpi'><h3>Total Sources</h3><h2>{len(sel_keys)}</h2></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='kpi' style='background:linear-gradient(135deg,#10B981,#059669)'><h3>Successful</h3><h2>{len(succ)}</h2></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='kpi' style='background:linear-gradient(135deg,#EF4444,#DC2626)'><h3>Failed</h3><h2>{len(fail)}</h2></div>", unsafe_allow_html=True)

st.markdown("### Status")
for k in sel_keys:
    name = DATA_SOURCES[k]["name"]
    msg = results[k][1]
    (st.success if (k in succ) else st.error)(f"**{name}:** {msg}")

if not succ:
    st.stop()

# Tabs per dataset
tabs = st.tabs([DATA_SOURCES[k]["name"] for k in succ.keys()])
for (tab, (k, (df, _))) in zip(tabs, succ.items()):
    with tab:
        name = DATA_SOURCES[k]["name"]
        st.markdown(f"#### {name}")
        if k in ["unemp", "cci", "hicp", "iip"]:
            # single value column named 'value'
            fig = line_fig(df, ["value"], name)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.tail(10), use_container_width=True)
            st.dataframe(describe_series(df, "value"), use_container_width=True, hide_index=True)
        elif k in ["mib", "vix"]:
            fig = line_fig(df, ["close"], name)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.tail(10), use_container_width=True)
            st.dataframe(describe_series(df, "close"), use_container_width=True, hide_index=True)
        elif k == "trends":
            # multi-series: let user pick
            value_cols = [c for c in df.columns if c != "date"]
            pick = st.multiselect("Series", value_cols, default=value_cols[: min(4,len(value_cols))])
            if pick:
                st.plotly_chart(line_fig(df, pick, name), use_container_width=True)
            st.dataframe(df.tail(10), use_container_width=True)

        # downloads
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"📥 Download {name} (CSV)",
            csv,
            f"{k}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True,
        )

# Bulk export
st.markdown("### 💾 Bulk Export")
if st.button("Download all as Excel", type="primary"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for k, (df, _) in succ.items():
            sheet = DATA_SOURCES[k]["name"][:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
    st.download_button(
        "Save Excel",
        output.getvalue(),
        f"italian_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.caption("This app auto-resolves Eurostat dimensions and applies sane fallbacks. No manual files needed.")
