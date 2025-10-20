"""
Streamlit App — Unemployment Auto‑Fetcher (ISTAT primary, Eurostat fallback)
- Zero hard dependency on pandasdmx (optional). App never crashes: ISTAT → Eurostat → Demo data.
- Nice UI, KPI cards, chart, CSV download, caching & retries.

Deploy notes (Streamlit Cloud):
- Minimum packages: streamlit, pandas, numpy, requests, urllib3
- Optional for Eurostat live fallback: pandasdmx, lxml
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st

# -----------------------------------------------------------------------------
# Optional: use pandasdmx if available (Eurostat convenience). Keep app alive if absent.
# -----------------------------------------------------------------------------
try:  # why: avoid ModuleNotFoundError on Streamlit Cloud
    import pandasdmx as sdmx  # type: ignore
    HAS_PANDASDMX = True
except Exception:
    HAS_PANDASDMX = False

# -----------------------------------------------------------------------------
# Streamlit page config & styles
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Unemployment Auto‑Fetcher — ISTAT + Eurostat", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      :root { --card-b: #ffffff; --card-s: #e5e7eb; --ok: #10b981; --warn:#f59e0b; --bad:#ef4444; --muted:#64748b; }
      .title {font-size:2rem;font-weight:800;text-align:center;margin:.25rem 0 .75rem}
      .badge {display:inline-block;padding:.2rem .5rem;border-radius:.5rem;border:1px solid var(--card-s);background:#f8fafc;font-size:.75rem}
      .card {background:var(--card-b);border:1px solid var(--card-s);border-radius:.75rem;padding:1rem}
      .ok {border-left:6px solid var(--ok)}
      .warn {border-left:6px solid var(--warn)}
      .bad {border-left:6px solid var(--bad)}
      .muted{color:var(--muted)}
      .kpi {text-align:center;border:1px solid var(--card-s);border-radius:.75rem;padding:.6rem}
      .kpi .label{color:var(--muted);font-size:.8rem}
      .kpi .val{font-size:1.3rem;font-weight:700}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">📈 Unemployment Auto‑Fetcher (ISTAT → Eurostat → Demo)</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HTTP session with retries
# -----------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "ISTAT-Eurostat-Unemployment/1.0 (+streamlit)",
        "Accept": "application/vnd.sdmx.data+json, application/json; q=0.9, */*; q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    retry = Retry(
        total=4, connect=4, read=4, backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

HTTP = make_session()

# -----------------------------------------------------------------------------
# SDMX‑JSON parsing (generic)
# -----------------------------------------------------------------------------

def parse_sdmx_json(j: Dict[str, Any]) -> pd.DataFrame:
    """Convert SDMX‑JSON to tidy DataFrame [date,value]. Returns empty on any issue."""
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

        def to_date(p: str) -> Optional[pd.Timestamp]:  # why: support M/Q/Y periods
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

# -----------------------------------------------------------------------------
# ISTAT helpers (dataflow 151_874: monthly unemployment rate)
# -----------------------------------------------------------------------------
ISTAT_BASE = "https://esploradati.istat.it/SDMXWS/rest"
ISTAT_FLOW_UNEM = "151_874"          # Unemployment rate — monthly
ISTAT_INDICATOR = "UNEM_R"

ISTAT_SEX_MAP = {"Total": "9", "Male": "1", "Female": "2"}
ISTAT_ADJ_MAP = {"NSA": "N", "SA": "Y"}  # Trend not available in ISTAT monthly

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
    key = f"M.IT.{ISTAT_INDICATOR}.{adj_code}.{sex_code}.{age_code}"
    url = f"{ISTAT_BASE}/data/ISTAT/{ISTAT_FLOW_UNEM}/{key}"
    params = {"detail": "serieskeysonly", "format": "sdmx-json"}
    try:
        r = HTTP.get(url, params=params, timeout=(10, 45))
        if r.status_code != 200:
            return []
        j = r.json()
        root = j.get("data", j)
        structure = root.get("structure", {})
        s_dims = (structure.get("dimensions", {}) or {}).get("series", [])
        idx_by_id = {d.get("id"): i for i, d in enumerate(s_dims)}
        if "EDITION" not in idx_by_id:
            return []
        ed_idx = idx_by_id["EDITION"]
        value_lists = [[v.get("id") for v in (d.get("values") or [])] for d in s_dims]
        out: List[str] = []
        for key_str in (root.get("dataSets", [{}])[0].get("series") or {}).keys():
            try:
                parts = [int(x) for x in key_str.split(":")]
                code = value_lists[ed_idx][parts[ed_idx]]
                out.append(code)
            except Exception:
                continue
        return sorted(list(dict.fromkeys(out)))
    except Exception:
        return []


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
    ed = edition or istat_pick_latest_edition(istat_list_editions(sex, age, adj))
    if not ed:
        return pd.DataFrame()
    key = f"M.IT.{ISTAT_INDICATOR}.{adj}.{sex}.{age}.{ed}"
    url = f"{ISTAT_BASE}/data/ISTAT/{ISTAT_FLOW_UNEM}/{key}"
    params = {"startPeriod": start, "endPeriod": end, "format": "sdmx-json"}
    try:
        r = HTTP.get(url, params=params, timeout=(10, 90))
        if r.status_code != 200:
            return pd.DataFrame()
        return parse_sdmx_json(r.json())
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# Eurostat helpers (fallback).
# If pandasdmx is unavailable, we return empty and let orchestrator fallback to demo.
# -----------------------------------------------------------------------------
EUROSTAT_DATASET = "une_rt_m"


def map_age_to_eurostat(age_code: str) -> str:
    if age_code in ("Y15-24", "Y15-34"):
        return "Y_LT25"
    if age_code == "Y25-74":
        return "Y25-74"
    return "TOTAL"


@st.cache_data(show_spinner=False)
def eurostat_fetch_unemployment(geo: str, sex: str, age: str, s_adj: str, start: str, end: str) -> pd.DataFrame:
    if not HAS_PANDASDMX:
        return pd.DataFrame()
    try:
        req = sdmx.Request("ESTAT")
        params = {"startPeriod": start, "endPeriod": end}
        key = {"unit": "PC_ACT", "s_adj": s_adj, "sex": sex, "age": age, "geo": geo}
        resp = req.data(EUROSTAT_DATASET, key=key, params=params)
        try:
            ser = resp.to_pandas()
        except Exception:
            j = resp.msg.to_json()
            if isinstance(j, dict):
                return parse_sdmx_json(j)
            return pd.DataFrame()
        if isinstance(ser, pd.Series):
            df = ser.to_frame(name="value").reset_index()
            tcol = next((c for c in df.columns if str(c).upper() in ("TIME_PERIOD", "TIME", "index")), None)
            if tcol is None:
                return pd.DataFrame()
            df.rename(columns={tcol: "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce").map(lambda x: x + pd.offsets.MonthEnd(0) if pd.notna(x) else x)
            return df.dropna(subset=["date"])[["date", "value"]].sort_values("date")
        if isinstance(ser, pd.DataFrame) and "TIME_PERIOD" in ser.columns:
            df = ser.rename(columns={"TIME_PERIOD": "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce").map(lambda x: x + pd.offsets.MonthEnd(0) if pd.notna(x) else x)
            vcol = next((c for c in df.columns if c not in {"date"} and pd.api.types.is_numeric_dtype(df[c])), None)
            if vcol is None:
                return pd.DataFrame()
            return df[["date", vcol]].rename(columns={vcol: "value"}).dropna().sort_values("date")
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# Demo data fallback
# -----------------------------------------------------------------------------

def demo_monthly(start: str, end: str, base: float = 9.5) -> pd.DataFrame:
    idx = pd.date_range(start + "-01", end + "-01", freq="MS") + pd.offsets.MonthEnd(0)
    n = len(idx)
    np.random.seed(42)
    seasonal = np.sin(np.arange(n) * 2 * np.pi / 12) * 0.25
    drift = np.linspace(0, -0.8, n)
    noise = np.random.randn(n) * 0.2
    val = base + seasonal + drift + noise
    return pd.DataFrame({"date": idx, "value": val}).sort_values("date")

# -----------------------------------------------------------------------------
# Business objects
# -----------------------------------------------------------------------------

@dataclass
class FetchOptions:
    geo: str = "IT"
    sex_label: str = "Total"    # Total/Male/Female
    age_code: str = "Y15-74"
    s_adj_label: str = "SA"      # NSA/SA/TC (TC Eurostat only)
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


class UnemploymentFetcher:
    def fetch(self, opt: FetchOptions) -> SeriesResult:
        istat_sex = ISTAT_SEX_MAP.get(opt.sex_label, "9")
        istat_adj = ISTAT_ADJ_MAP.get(opt.s_adj_label, "Y") if opt.s_adj_label in ("NSA", "SA") else ISTAT_ADJ_MAP["SA"]
        euro_sex = {"Total": "T", "Male": "M", "Female": "F"}[opt.sex_label]
        euro_age = map_age_to_eurostat(opt.age_code)
        euro_adj = opt.s_adj_label if opt.s_adj_label in ("NSA", "SA", "TC") else "SA"

        sources = [s.upper() for s in opt.source_priority if s] or ["ISTAT", "EUROSTAT"]

        for s in sources:
            if s == "ISTAT":
                if opt.geo != "IT":
                    continue
                if opt.s_adj_label == "TC":
                    continue
                df = istat_fetch_unemployment(istat_sex, opt.age_code, istat_adj, opt.start, opt.end, opt.istat_edition)
                if not df.empty:
                    return SeriesResult(df=df.rename(columns={"value": "unemployment"}), source="ISTAT", meta={"edition": opt.istat_edition or "(latest)"})
            elif s == "EUROSTAT":
                df = eurostat_fetch_unemployment(opt.geo, euro_sex, euro_age, euro_adj, opt.start, opt.end)
                if not df.empty:
                    return SeriesResult(df=df.rename(columns={"value": "unemployment"}), source="EUROSTAT", meta={"via": "pandasdmx" if HAS_PANDASDMX else "requests"})

        df = demo_monthly(opt.start, opt.end)
        return SeriesResult(df=df.rename(columns={"value": "unemployment"}), source="DEMO")


# -----------------------------------------------------------------------------
# UI — Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Settings")
    st.caption("Primary = ISTAT (IT only), fallback = Eurostat. The app won’t crash; demo data used if needed.")

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
    istat_editions = istat_list_editions(ISTAT_SEX_MAP.get(sex_label, "9"), age_code, ISTAT_ADJ_MAP.get(s_adj_label, "Y")) if geo == "IT" and s_adj_label in ("SA", "NSA") else []
    ed_choice: Optional[str] = None
    if istat_editions:
        ed_choice = st.selectbox("Edition (release)", options=["<latest>"] + istat_editions, index=0)
        if ed_choice == "<latest>":
            ed_choice = None

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
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"<div class='kpi'><div class='label'>Observations</div><div class='val'>{len(df)}</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'><div class='label'>Start</div><div class='val'>{df['date'].min().strftime('%Y-%m')}</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'><div class='label'>End</div><div class='val'>{df['date'].max().strftime('%Y-%m')}</div></div>", unsafe_allow_html=True)
        k4.markdown(f"<div class='kpi'><div class='label'>Latest</div><div class='val'>{df['unemployment'].iloc[-1]:.2f}%</div></div>", unsafe_allow_html=True)

        st.markdown("### Target series")
        st.line_chart(df.set_index("date")["unemployment"])
        st.dataframe(df, use_container_width=True)

        m = df.set_index("date").sort_index()
        m["mom"] = m["unemployment"].pct_change() * 100
        m["yoy"] = m["unemployment"].pct_change(12) * 100
        with st.expander("Derived rates (m/m %, y/y %)"):
            st.dataframe(m[["mom", "yoy"]].reset_index())

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

st.markdown(
    """
    <div class='muted' style='margin-top:1rem'>
      <span class='badge'>Tips</span>
      For Eurostat live fallback install <code>pandasdmx</code> and <code>lxml</code> in your environment (optional).
      ISTAT endpoint is rate‑limited; caching reduces calls.
    </div>
    """,
    unsafe_allow_html=True,
)
