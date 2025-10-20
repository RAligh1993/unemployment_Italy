"""
Unemployment Auto‑Fetcher (ISTAT + Eurostat)
===========================================
A robust, single‑file Streamlit app that automatically fetches **monthly unemployment**
(time series) from **ISTAT** (primary) with **Eurostat** fallback, plus optional
monthly/quarterly auxiliary indicators. Designed to be resilient on Streamlit Cloud.

Key features
------------
- Primary source: ISTAT SDMX REST (new endpoint). Fallback: Eurostat via `pandasdmx` if available.
- Dynamic **edition** discovery for ISTAT (latest release auto‑detection, manual override).
- Dimension‑aware key building for ISTAT (FREQ, GEO, DATA_TYPE, ADJUSTMENT, SEX, AGE, EDITION).
- Clean SDMX‑JSON parsing (no XML required). Optional `pandasdmx` for Eurostat; if absent, app stays functional.
- Caching, retry & timeout, graceful error messages, never crashes—falls back to sample data.
- Target series: Unemployment Rate; optional helpers (Eurostat STS/CI) for nowcasting.

Notes
-----
- ISTAT codes: SEX {"1": male, "2": female, "9": total}; ADJUSTMENT {"N": NSA, "Y": SA}.
- Eurostat codes: `une_rt_m` with s_adj {NSA, SA, TC}, sex {M, F, T}, age {TOTAL, Y_LT25, Y25-74}, unit {PC_ACT}.
- For auxiliary monthly indicators (Eurostat): IPI (sts_inpr_m), Retail Volume (sts_trtu_m), Services Turnover (sts_setu_m), Hours worked (sts_inlb_m), Consumer Confidence (ei_bsco_m). All are optional.

This file intentionally contains **no requirements.txt**. If Eurostat access via `pandasdmx` is desired, add it
in your deployment (Streamlit Cloud → App packages) together with `lxml`. The app still works without them.
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st

# Try using pandasdmx for Eurostat (optional). If missing, we keep running without it.
try:
    import pandasdmx as sdmx  # type: ignore
    HAS_PANDASDMX = True
except Exception:
    HAS_PANDASDMX = False

# =============================================================================
# Streamlit page config & styles
# =============================================================================
st.set_page_config(page_title="Unemployment Auto‑Fetcher — ISTAT + Eurostat", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .title {font-size:2rem;font-weight:800;text-align:center;margin:0.3rem 0 0.6rem;}
      .badge {display:inline-block;padding:.15rem .4rem;border-radius:.4rem;border:1px solid #e5e7eb;background:#f9fafb;font-size:.75rem}
      .card {background:#fff;border:1px solid #e5e7eb;border-radius:.6rem;padding:1rem}
      .ok {border-left:4px solid #10b981}
      .warn {border-left:4px solid #f59e0b}
      .bad {border-left:4px solid #ef4444}
      .muted{color:#64748b}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">📈 Unemployment Auto‑Fetcher (ISTAT → Eurostat fallback)</div>', unsafe_allow_html=True)

# =============================================================================
# HTTP session with retries
# =============================================================================

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "ISTAT-Eurostat-Unemployment/1.0 (+streamlit)",
        "Accept": "application/vnd.sdmx.data+json, application/json; q=0.9, */*; q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    retry = Retry(total=4, connect=4, read=4, backoff_factor=0.7,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "HEAD"],
                  raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

HTTP = make_session()

# =============================================================================
# SDMX‑JSON parsing (generic)
# =============================================================================

def parse_sdmx_json(j: Dict[str, Any]) -> pd.DataFrame:
    """Convert SDMX‑JSON to a tidy DataFrame with columns [date, value].
    Returns empty DataFrame on any issue (never raises)."""
    try:
        root = j.get("data", j)
        datasets = root.get("dataSets", [])
        if not datasets:
            return pd.DataFrame()
        ds = datasets[0]
        structure = root.get("structure", {})
        obs_dims = (structure.get("dimensions", {}) or {}).get("observation", [])
        time_values: Optional[List[str]] = None
        for d in obs_dims:
            if d.get("id") == "TIME_PERIOD":
                time_values = [v.get("id") for v in d.get("values", [])]
                break
        if not time_values:
            return pd.DataFrame()
        rec: List[Dict[str, Any]] = []
        series = ds.get("series", {}) or {}
        if series:
            for s in series.values():
                for idx, raw in (s.get("observations", {}) or {}).items():
                    t = time_values[int(idx)]
                    v = raw[0] if isinstance(raw, list) else raw
                    rec.append({"time": t, "value": float(v)})
        else:
            for idx, raw in (ds.get("observations", {}) or {}).items():
                t = time_values[int(idx)]
                v = raw[0] if isinstance(raw, list) else raw
                rec.append({"time": t, "value": float(v)})
        if not rec:
            return pd.DataFrame()
        df = pd.DataFrame(rec)
        # map period to timestamp (monthly preferred)
        def to_date(p: str) -> Optional[pd.Timestamp]:
            p = str(p)
            if re.match(r"^\d{4}-\d{2}$", p):
                return pd.to_datetime(p) + pd.offsets.MonthEnd(0)
            if "-Q" in p:
                y, q = p.split("-Q"); y = int(y); m = int(q) * 3
                return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
            if re.match(r"^\d{4}$", p):
                return pd.Timestamp(int(p), 12, 31)
            return pd.to_datetime(p, errors="coerce")
        df["date"] = df["time"].map(to_date)
        return df.dropna(subset=["date"]).sort_values("date")[ ["date", "value"] ]
    except Exception:
        return pd.DataFrame()

# =============================================================================
# ISTAT helpers (dataflow 151_874: monthly unemployment rate)
# =============================================================================
ISTAT_BASE = "https://esploradati.istat.it/SDMXWS/rest"
ISTAT_FLOW_UNEM = "151_874"          # Unemployment rate — monthly
ISTAT_INDICATOR = "UNEM_R"

# ISTAT codes (fixed)
ISTAT_SEX_MAP = {"Total": "9", "Male": "1", "Female": "2"}
ISTAT_ADJ_MAP = {"NSA": "N", "SA": "Y"}  # Trend not available in ISTAT monthly

# Expose common age bands used by ISTAT (Eurostat will be mapped separately)
AGE_CHOICES = {
    "15–74 (Total)": "Y15-74",
    "15–24": "Y15-24",
    "25–74": "Y25-74",
    "15–34": "Y15-34",
    "35–49": "Y35-49",
    "50–64": "Y50-64",
    "50–74": "Y50-74",
    "15–64": "Y15-64",
}

@st.cache_data(show_spinner=False)
def istat_list_editions(sex_code: str = "9", age_code: str = "Y15-74", adj_code: str = "N") -> List[str]:
    """Return all available EDITION codes for the given slice using `serieskeysonly`.
    We query a standard slice and collect the EDITION codes from series keys.
    """
    key = f"M.IT.{ISTAT_INDICATOR}.{adj_code}.{sex_code}.{age_code}"
    url = f"{ISTAT_BASE}/data/ISTAT/{ISTAT_FLOW_UNEM}/{key}"
    params = {"detail": "serieskeysonly", "format": "sdmx-json"}
    r = HTTP.get(url, params=params, timeout=(10, 45))
    if r.status_code != 200:
        return []
    j = r.json()
    root = j.get("data", j)
    structure = root.get("structure", {})
    # series dimension metadata
    s_dims = (structure.get("dimensions", {}) or {}).get("series", [])
    idx_by_id = {d.get("id"): i for i, d in enumerate(s_dims)}
    if "EDITION" not in idx_by_id:
        return []
    ed_idx = idx_by_id["EDITION"]
    # list of code values for each series dimension
    value_lists = [ [v.get("id") for v in (d.get("values") or [])] for d in s_dims ]
    out: List[str] = []
    for key_str in (root.get("dataSets", [{}])[0].get("series") or {}).keys():
        # series key is colon‑separated indices
        try:
            parts = [int(x) for x in key_str.split(":")]
            code = value_lists[ed_idx][parts[ed_idx]]
            out.append(code)
        except Exception:
            continue
    return sorted(list(dict.fromkeys(out)))  # unique & ordered

def istat_pick_latest_edition(editions: List[str]) -> Optional[str]:
    if not editions:
        return None
    def parse(code: str) -> Tuple[int, int, int]:
        # e.g., 2025M10G2 → (2025, 10, 2)
        try:
            y, rest = code.split("M", 1)
            m, g = rest.split("G", 1)
            return (int(y), int(m), int(g))
        except Exception:
            return (0, 0, 0)
    return sorted(editions, key=parse)[-1]

@st.cache_data(show_spinner=False)
def istat_fetch_unemployment(sex: str, age: str, adj: str, start: str, end: str, edition: Optional[str]) -> pd.DataFrame:
    """Fetch ISTAT monthly unemployment rate for Italy with given dimensions.
    Dimensions:
      - sex in {"1","2","9"}
      - age like "Y15-74"
      - adj in {"N","Y"}
      - edition required (latest auto‑picked if None)
    """
    ed = edition or istat_pick_latest_edition(istat_list_editions(sex, age, adj))
    if not ed:
        return pd.DataFrame()
    key = f"M.IT.{ISTAT_INDICATOR}.{adj}.{sex}.{age}.{ed}"
    url = f"{ISTAT_BASE}/data/ISTAT/{ISTAT_FLOW_UNEM}/{key}"
    params = {"startPeriod": start, "endPeriod": end, "format": "sdmx-json"}
    r = HTTP.get(url, params=params, timeout=(10, 90))
    if r.status_code != 200:
        return pd.DataFrame()
    return parse_sdmx_json(r.json())

# =============================================================================
# Eurostat helpers (optional via pandasdmx)
# =============================================================================
EUROSTAT_DATASET = "une_rt_m"
EUROSTAT_AGE_ALLOWED = {"TOTAL", "Y_LT25", "Y25-74"}

# Map ISTAT‑style age to Eurostat set (fallback to TOTAL if not available)

def map_age_to_eurostat(age_code: str) -> str:
    if age_code in ("Y15-24", "Y15-34"):
        return "Y_LT25"
    if age_code == "Y25-74":
        return "Y25-74"
    # default bucket
    return "TOTAL"

@st.cache_data(show_spinner=False)
def eurostat_fetch_unemployment(geo: str, sex: str, age: str, s_adj: str, start: str, end: str) -> pd.DataFrame:
    """Fetch Eurostat monthly unemployment (une_rt_m) using pandasdmx if available.
    Returns empty DataFrame if pandasdmx is not installed or request fails.
    """
    if not HAS_PANDASDMX:
        return pd.DataFrame()
    try:
        req = sdmx.Request("ESTAT")
        params = {"startPeriod": start, "endPeriod": end}
        # unit=PC_ACT (rate, % of active population)
        key = {"unit": "PC_ACT", "s_adj": s_adj, "sex": sex, "age": age, "geo": geo}
        resp = req.data(EUROSTAT_DATASET, key=key, params=params)
        # Convert to pandas Series/DataFrame
        try:
            ser = resp.to_pandas()
        except Exception:
            # fallback via SDMX‑JSON representation if available
            j = resp.msg.to_json()
            df = parse_sdmx_json(j if isinstance(j, dict) else {})
            return df.rename(columns={"value": "value"})
        if isinstance(ser, pd.Series):
            df = ser.to_frame(name="value").reset_index()
            # detect time column
            tcol = next((c for c in df.columns if str(c).upper() in ("TIME_PERIOD", "TIME", "index")), None)
            if tcol is None:
                return pd.DataFrame()
            df.rename(columns={tcol: "date"}, inplace=True)
            # unify timestamps
            df["date"] = pd.to_datetime(df["date"], errors="coerce").map(lambda x: x + pd.offsets.MonthEnd(0) if pd.notna(x) else x)
            return df.dropna(subset=["date"])[["date", "value"]].sort_values("date")
        # If DataFrame with explicit column
        if "TIME_PERIOD" in ser.columns:
            df = ser.rename(columns={"TIME_PERIOD": "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce").map(lambda x: x + pd.offsets.MonthEnd(0) if pd.notna(x) else x)
            # find numeric column
            vcol = next((c for c in df.columns if c not in {"date"} and pd.api.types.is_numeric_dtype(df[c])), None)
            if vcol is None:
                return pd.DataFrame()
            df = df[["date", vcol]].rename(columns={vcol: "value"}).dropna()
            return df.sort_values("date")
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Optional Eurostat auxiliary fetchers (monthly):
EUROSTAT_AUX_DATASETS = {
    "IPI (industry production index)": ("sts_inpr_m", "I19: Volume index (2015=100)", {}),
    "Retail trade volume": ("sts_trtu_m", "I19: Volume index (2015=100)", {}),
    "Services turnover": ("sts_setu_m", "I19: Working day adj index", {}),
    "Hours worked (industry)": ("sts_inlb_m", "I19: Hours worked index", {}),
    "Consumer confidence": ("ei_bsco_m", "CI: balance", {}),
}

@st.cache_data(show_spinner=False)
def eurostat_fetch_aux(dataset: str, geo: str, start: str, end: str) -> pd.DataFrame:
    if not HAS_PANDASDMX:
        return pd.DataFrame()
    try:
        req = sdmx.Request("ESTAT")
        # We keep filters minimal; many STS datasets accept s_adj and unit; here we rely on defaults
        resp = req.data(dataset, params={"startPeriod": start, "endPeriod": end})
        ser = resp.to_pandas()
        # try to collapse to single series by selecting geo
        if isinstance(ser, pd.Series):
            df = ser.reset_index()
        else:
            df = ser.reset_index()
        # try to filter chosen geo column
        if "geo" in df.columns:
            df = df[df["geo"] == geo]
        # standardize columns
        tcol = next((c for c in df.columns if str(c).upper() in ("TIME_PERIOD", "TIME", "index")), None)
        vcol = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None)
        if tcol is None or vcol is None:
            return pd.DataFrame()
        out = df[[tcol, vcol]].rename(columns={tcol: "date", vcol: "value"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce").map(lambda x: x + pd.offsets.MonthEnd(0) if pd.notna(x) else x)
        return out.dropna().sort_values("date")
    except Exception:
        return pd.DataFrame()

# =============================================================================
# Demo data fallback
# =============================================================================

def demo_monthly(start: str, end: str, base: float = 9.5) -> pd.DataFrame:
    idx = pd.date_range(start + "-01", end + "-01", freq="MS") + pd.offsets.MonthEnd(0)
    n = len(idx)
    np.random.seed(42)
    seasonal = np.sin(np.arange(n) * 2 * np.pi / 12) * 0.25
    drift = np.linspace(0, -0.8, n)
    noise = np.random.randn(n) * 0.2
    val = base + seasonal + drift + noise
    return pd.DataFrame({"date": idx, "value": val}).sort_values("date")

# =============================================================================
# Business objects
# =============================================================================

@dataclass
class FetchOptions:
    geo: str = "IT"
    sex_label: str = "Total"    # Total/Male/Female
    age_code: str = "Y15-74"
    s_adj_label: str = "SA"      # NSA/SA/TC (TC for Eurostat only)
    start_year: int = 2010
    start_month: int = 1
    end_year: int = datetime.utcnow().year
    end_month: int = datetime.utcnow().month
    istat_edition: Optional[str] = None
    source_priority: List[str] = field(default_factory=lambda: ["ISTAT", "EUROSTAT"])  # order

    @property
    def start(self) -> str:
        return f"{self.start_year}-{self.start_month:02d}"

    @property
    def end(self) -> str:
        return f"{self.end_year}-{self.end_month:02d}"

@dataclass
class SeriesResult:
    df: pd.DataFrame
    source: str
    meta: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Fetcher orchestrator
# =============================================================================

class UnemploymentFetcher:
    def __init__(self):
        pass

    def fetch(self, opt: FetchOptions) -> SeriesResult:
        # Prepare dimension codes for both providers
        istat_sex = ISTAT_SEX_MAP.get(opt.sex_label, "9")
        istat_adj = ISTAT_ADJ_MAP.get(opt.s_adj_label, "Y") if opt.s_adj_label in ("NSA", "SA") else ISTAT_ADJ_MAP["SA"]
        euro_sex = {"Total": "T", "Male": "M", "Female": "F"}[opt.sex_label]
        euro_age = map_age_to_eurostat(opt.age_code)
        euro_adj = opt.s_adj_label if opt.s_adj_label in ("NSA", "SA", "TC") else "SA"

        # Decide priority
        sources = [s.upper() for s in opt.source_priority if s]
        if not sources:
            sources = ["ISTAT", "EUROSTAT"]

        for s in sources:
            if s == "ISTAT":
                if opt.geo != "IT":
                    continue  # ISTAT monthly unemployment is national (IT) only
                if opt.s_adj_label == "TC":
                    continue  # trend not in ISTAT dataset
                df = istat_fetch_unemployment(istat_sex, opt.age_code, istat_adj, opt.start, opt.end, opt.istat_edition)
                if not df.empty:
                    return SeriesResult(df=df.rename(columns={"value": "unemployment"}), source="ISTAT", meta={"edition": opt.istat_edition or "(latest)"})
            elif s == "EUROSTAT":
                df = eurostat_fetch_unemployment(opt.geo, euro_sex, euro_age, euro_adj, opt.start, opt.end)
                if not df.empty:
                    return SeriesResult(df=df.rename(columns={"value": "unemployment"}), source="EUROSTAT", meta={})
        # Final fallback
        df = demo_monthly(opt.start, opt.end)
        return SeriesResult(df=df.rename(columns={"value": "unemployment"}), source="DEMO")

# =============================================================================
# UI — Controls
# =============================================================================
with st.sidebar:
    st.subheader("Settings")
    st.caption("Primary = ISTAT (IT only), fallback = Eurostat. The app won’t crash; it will show demo data if needed.")

    geo = st.selectbox("Geo (country)", options=["IT", "DE", "FR", "ES", "EU27_2020"], index=0, help="Eurostat supports many geos; ISTAT is Italy only.")
    sex_label = st.selectbox("Sex", options=["Total", "Male", "Female"], index=0)
    age_label = st.selectbox("Age band (ISTAT style)", options=list(AGE_CHOICES.keys()), index=0)
    age_code = AGE_CHOICES[age_label]

    s_adj_label = st.selectbox("Seasonal adjustment", options=["SA", "NSA", "TC"], index=0, help="ISTAT supports SA/NSA; Eurostat supports SA/NSA/TC.")

    c1, c2 = st.columns(2)
    with c1:
        start_year = st.number_input("Start year", min_value=1990, max_value=2100, value=2015, step=1)
        start_month = st.number_input("Start month", min_value=1, max_value=12, value=1, step=1)
    with c2:
        end_year = st.number_input("End year", min_value=1990, max_value=2100, value=datetime.utcnow().year, step=1)
        end_month = st.number_input("End month", min_value=1, max_value=12, value=min(datetime.utcnow().month, 12), step=1)

    st.markdown("---")
    st.markdown("**Source priority**")
    srcs = st.multiselect("Try in order", options=["ISTAT", "EUROSTAT"], default=["ISTAT", "EUROSTAT"])
    if not srcs:
        srcs = ["ISTAT", "EUROSTAT"]

    st.markdown("---")
    st.markdown("**ISTAT edition**")
    # Only meaningful for Italy + SA/NSA
    istat_editions = istat_list_editions(ISTAT_SEX_MAP.get(sex_label, "9"), age_code, ISTAT_ADJ_MAP.get(s_adj_label, "Y")) if geo == "IT" and s_adj_label in ("SA", "NSA") else []
    ed_choice = None
    if istat_editions:
        ed_choice = st.selectbox("Edition (release)", options=["<latest>"] + istat_editions, index=0)
        if ed_choice == "<latest>":
            ed_choice = None

    st.markdown("---")
    st.markdown("**Auxiliary indicators (Eurostat)**")
    aux_selected = st.multiselect("Optional monthly helpers", options=list(EUROSTAT_AUX_DATASETS.keys()), default=["IPI (industry production index)", "Consumer confidence"])    

    st.markdown("---")
    show_debug = st.checkbox("Show debug info", value=False)

fetcher = UnemploymentFetcher()

if st.button("Fetch data", type="primary", use_container_width=True):
    opt = FetchOptions(
        geo=geo, sex_label=sex_label, age_code=age_code, s_adj_label=s_adj_label,
        start_year=int(start_year), start_month=int(start_month), end_year=int(end_year), end_month=int(end_month),
        istat_edition=ed_choice, source_priority=srcs,
    )

    with st.spinner("Fetching unemployment series…"):
        res = fetcher.fetch(opt)

    # status card
    if res.source == "ISTAT":
        st.markdown("<div class='card ok'><b>Success:</b> Source = ISTAT (official, Italy)</div>", unsafe_allow_html=True)
    elif res.source == "EUROSTAT":
        st.markdown("<div class='card ok'><b>Success:</b> Source = Eurostat</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card warn'><b>Using demo data</b> — all primary sources failed.</div>", unsafe_allow_html=True)

    df = res.df.copy().dropna()
    if df.empty:
        st.markdown("<div class='card bad'>No observations returned. Try different dimensions or a wider time window.</div>", unsafe_allow_html=True)
    else:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", len(df))
        c2.metric("Start", df["date"].min().strftime("%Y-%m"))
        c3.metric("End", df["date"].max().strftime("%Y-%m"))
        c4.metric("Latest", f"{df['unemployment'].iloc[-1]:.2f}%")

        st.markdown("### Target series")
        st.line_chart(df.set_index("date")["unemployment"])
        st.dataframe(df, use_container_width=True)

        # Derived metrics
        m = df.set_index("date").sort_index()
        m["mom"] = m["unemployment"].pct_change() * 100
        m["yoy"] = m["unemployment"].pct_change(12) * 100
        with st.expander("Derived rates (m/m %, y/y %)"):
            st.dataframe(m[["mom", "yoy"]].reset_index())

        # Aux series
        aux_frames: List[Tuple[str, pd.DataFrame]] = []
        if aux_selected:
            with st.spinner("Fetching auxiliary indicators…"):
                for label in aux_selected:
                    ds, _desc, _kw = EUROSTAT_AUX_DATASETS[label]
                    aux = eurostat_fetch_aux(ds, opt.geo, opt.start, opt.end)
                    if not aux.empty:
                        aux_frames.append((label, aux))
                    elif show_debug:
                        st.info(f"Aux dataset '{label}' returned empty or pandasdmx missing.")
        if aux_frames:
            st.markdown("### Auxiliary indicators")
            # Plot each and compute correlation with unemployment
            for label, adf in aux_frames:
                merged = pd.merge(df[["date", "unemployment"]], adf, on="date", how="inner")
                merged.rename(columns={"value": label}, inplace=True)
                st.line_chart(merged.set_index("date")[[label]])
                if len(merged) >= 3:
                    corr = merged["unemployment"].corr(merged[label])
                    st.caption(f"Correlation with unemployment: **{corr:.2f}**")

        # Download
        st.download_button(
            "Download CSV (target)", data=df.to_csv(index=False), mime="text/csv",
            file_name=f"unemployment_{res.source.lower()}_{opt.geo}_{opt.start}_{opt.end}.csv",
            use_container_width=True,
        )

    if show_debug:
        st.markdown("---")
        st.subheader("Debug info")
        st.write({
            "HAS_PANDASDMX": HAS_PANDASDMX,
            "options": opt.__dict__,
            "source": res.source,
        })

# Helpful footer
st.markdown(
    """
    <div class='muted' style='margin-top:1rem'>
      <span class='badge'>Tips</span>
      If Eurostat access is desired, install <code>pandasdmx</code> (and <code>lxml</code>) in your environment.
      ISTAT endpoint is rate‑limited; caching is enabled to reduce calls.
    </div>
    """,
    unsafe_allow_html=True,
)
