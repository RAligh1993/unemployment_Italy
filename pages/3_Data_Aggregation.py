"""
Streamlit App — Unemployment Auto‑Fetcher (ISTAT primary, Eurostat fallback w/o pandasdmx)
- Robust on Streamlit Cloud. No hard dependency except: streamlit, pandas, numpy, requests, urllib3.
- Order: ISTAT → Eurostat (Statistics API JSON‑stat) → Demo. No crashes.
- Clean UI, KPI, chart, CSV, caching & retries.

Deploy notes (Streamlit Cloud):
- requirements.txt minimal:
    streamlit
    pandas
    numpy
    requests
    urllib3
- (Optional) Add `pandasdmx` and `lxml` if you also want the classic SDMX client; this app does not require them.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st

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
        "User-Agent": "ISTAT-Eurostat-Unemployment/1.1 (+streamlit)",
        "Accept": "application/json, application/vnd.sdmx.data+json; q=0.9, */*; q=0.5",
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
# Parsers: SDMX‑JSON and JSON‑stat 2.0
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
                    if v is None:
                        continue
                    rec.append({"time": t, "value": float(v)})
        else:
            for idx, raw in (ds.get("observations", {}) or {}).items():
                t = time_values[int(idx)]
                v = raw[0] if isinstance(raw, list) else raw
                if v is None:
                    continue
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


def parse_jsonstat_dataset(obj: Dict[str, Any]) -> pd.DataFrame:
    """Parse Eurostat Statistics API JSON‑stat dataset to DataFrame[date,value]."""
    try:
        if obj.get("class") != "dataset":
            datasets = obj.get("datasets") or obj.get("data")
            if isinstance(datasets, list) and datasets:
                obj = datasets[0]
        dim = obj["dimension"]
        ids: List[str] = obj.get("id") or dim.get("id")
        size: List[int] = obj.get("size") or dim.get("size")
        role = (dim.get("role") or {})
        time_id = (role.get("time") or [None])[0] or next((i for i in ids if i.lower() == "time"), None)
        if not time_id:
            return pd.DataFrame()
        def codes_for(did: str) -> List[str]:
            cat = dim[did]["category"]
            idx = cat.get("index")
            if isinstance(idx, dict):
                return [k for k, _ in sorted(idx.items(), key=lambda kv: kv[1])]
            if isinstance(idx, list):
                return idx
            if "label" in cat and isinstance(cat["label"], dict):
                return list(cat["label"].keys())
            return []
        codes: Dict[str, List[str]] = {did: codes_for(did) for did in ids}
        values = obj.get("value")
        if isinstance(values, dict):
            full = [None] * int(np.prod(size))
            for k, v in values.items():
                try:
                    full[int(k)] = v
                except Exception:
                    pass
            values = full
        if not isinstance(values, list):
            return pd.DataFrame()
        if len(ids) == 1 and ids[0] == time_id:
            t_codes = codes[time_id]
            rec = [
                {"time": t_codes[i], "value": float(values[i])}
                for i in range(min(len(values), len(t_codes)))
                if values[i] is not None
            ]
            return _jsonstat_time_to_df(rec)
        strides = _compute_strides(size)
        ti = ids.index(time_id)
        t_codes = codes[time_id]
        selector = [0] * len(ids)
        out: List[Dict[str, Any]] = []
        for tpos, t in enumerate(t_codes):
            selector[ti] = tpos
            lin = sum(selector[i] * strides[i] for i in range(len(ids)))
            if lin < len(values) and values[lin] is not None:
                out.append({"time": t, "value": float(values[lin])})
        return _jsonstat_time_to_df(out)
    except Exception:
        return pd.DataFrame()


def _compute_strides(size: List[int]) -> List[int]:
    stride = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        stride[i] = stride[i + 1] * size[i + 1]
    return stride


def _jsonstat_time_to_df(rec: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rec:
        return pd.DataFrame()
    df = pd.DataFrame(rec)
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

# -----------------------------------------------------------------------------
# ISTAT helpers — 151_874 Unemployment rate (monthly). No EDITION needed.
# -----------------------------------------------------------------------------
ISTAT_BASE = "https://esploradati.istat.it/SDMXWS/rest"
ISTAT_FLOW_ID = "151_874"
ISTAT_AGENCY = "IT1"
ISTAT_VERSIONS_TRY = ["1.2", "1.1", "1.0"]
ISTAT_INDICATOR = "UNEM_R"

ISTAT_SEX_MAP = {"Total": "9", "Male": "1", "Female": "2"}
ISTAT_ADJ_MAP = {"NSA": "N", "SA": "Y"}

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
def istat_fetch_unemployment(sex: str, age: str, adj: str, start: str, end: str) -> pd.DataFrame:
    """Query ISTAT SDMX REST. Build key: FREQ.GEO.INDICATOR.ADJ.SEX.AGE"""
    key = f"M.IT.{ISTAT_INDICATOR}.{adj}.{sex}.{age}"
    params = {"startPeriod": start, "endPeriod": end, "format": "sdmx-json"}
    for ver in ISTAT_VERSIONS_TRY:
        flowref = f"{ISTAT_AGENCY},{ISTAT_FLOW_ID},{ver}"
        url = f"{ISTAT_BASE}/data/{flowref}/{key}"
        try:
            r = HTTP.get(url, params=params, timeout=(10, 90))
            if r.status_code == 200:
                df = parse_sdmx_json(r.json())
                if not df.empty:
                    return df
        except Exception:
            continue
    # try wildcard extra slot (harmless if 404)
    for ver in ISTAT_VERSIONS_TRY:
        flowref = f"{ISTAT_AGENCY},{ISTAT_FLOW_ID},{ver}"
        url = f"{ISTAT_BASE}/data/{flowref}/{key}.*"
        try:
            r = HTTP.get(url, params=params, timeout=(10, 90))
            if r.status_code == 200:
                df = parse_sdmx_json(r.json())
                if not df.empty:
                    return df
        except Exception:
            continue
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# Eurostat — Statistics API (JSON‑stat). No external libs.
# -----------------------------------------------------------------------------
EUROSTAT_STATS_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
EUROSTAT_DATASET = "UNE_RT_M"


def map_age_to_eurostat(age_code: str) -> str:
    if age_code in ("Y15-24", "Y15-34"):
        return "Y_LT25"
    if age_code == "Y25-74":
        return "Y25-74"
    return "TOTAL"


@st.cache_data(show_spinner=False)
def eurostat_fetch_unemployment(geo: str, sex: str, age: str, s_adj: str, start: str, end: str) -> pd.DataFrame:
    url = f"{EUROSTAT_STATS_BASE}/{EUROSTAT_DATASET}"
    params = {
        "lang": "en",
        "unit": "PC_ACT",
        "s_adj": s_adj,
        "sex": sex,
        "age": age,
        "geo": geo,
        "time": f"{start[:4]}-{end[:4]}",
    }
    try:
        r = HTTP.get(url, params=params, timeout=(10, 60))
        if r.status_code != 200:
            return pd.DataFrame()
        df = parse_jsonstat_dataset(r.json())
        if df.empty:
            return pd.DataFrame()
        sdt = pd.to_datetime(f"{start}-01") + pd.offsets.MonthEnd(0)
        edt = pd.to_datetime(f"{end}-01") + pd.offsets.MonthEnd(0)
        df = df[(df["date"] >= sdt) & (df["date"] <= edt)]
        return df
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
    s_adj_label: str = "SA"      # NSA/SA/TC (Eurostat supports TC)
    start_year: int = 2010
    start_month: int = 1
    end_year: int = datetime.utcnow().year
    end_month: int = datetime.utcnow().month
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
                if opt.geo == "IT" and opt.s_adj_label != "TC":
                    df = istat_fetch_unemployment(istat_sex, opt.age_code, istat_adj, opt.start, opt.end)
                    if not df.empty:
                        return SeriesResult(df=df.rename(columns={"value": "unemployment"}), source="ISTAT")
            elif s == "EUROSTAT":
                df = eurostat_fetch_unemployment(opt.geo, euro_sex, euro_age, euro_adj, opt.start, opt.end)
                if not df.empty:
                    return SeriesResult(df=df.rename(columns={"value": "unemployment"}), source="EUROSTAT")

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
    show_debug = st.checkbox("Show debug info", value=False)

fetcher = UnemploymentFetcher()

if st.button("Fetch data", type="primary", use_container_width=True):
    opt = FetchOptions(
        geo=geo, sex_label=sex_label, age_code=age_code, s_adj_label=s_adj_label,
        start_year=int(start_year), start_month=int(start_month), end_year=int(end_year), end_month=int(end_month),
        source_priority=srcs,
    )

    with st.spinner("Fetching unemployment series…"):
        res = fetcher.fetch(opt)

    if res.source == "ISTAT":
        st.markdown("<div class='card ok'><b>Success:</b> Source = ISTAT (official, Italy)</div>", unsafe_allow_html=True)
    elif res.source == "EUROSTAT":
        st.markdown("<div class='card ok'><b>Success:</b> Source = Eurostat (Statistics API)</div>", unsafe_allow_html=True)
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
            "options": opt.__dict__,
            "source": res.source,
        })

st.markdown(
    """
    <div class='muted' style='margin-top:1rem'>
      <span class='badge'>Tips</span>
      ISTAT via SDMX REST (IT1, 151_874). Eurostat via Statistics API (JSON‑stat). No extra libs required.
    </div>
    """,
    unsafe_allow_html=True,
)
