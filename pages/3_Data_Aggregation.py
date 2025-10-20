# app.py
from __future__ import annotations
import re, json, textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st

# =============================================================================
# Page setup
# =============================================================================
st.set_page_config(page_title="Unemployment Auto-Fetcher (ISTAT + Eurostat)", page_icon="📈", layout="wide")
st.markdown("<h2 style='text-align:center'>📈 Unemployment Auto-Fetcher (ISTAT → Eurostat → Demo)</h2>", unsafe_allow_html=True)

# =============================================================================
# HTTP with retries
# =============================================================================
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "ISTAT-Eurostat-Unemployment/1.3 (+streamlit)",
        "Accept": "application/vnd.sdmx.data+json, application/json;q=0.9, */*;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    retry = Retry(
        total=4, connect=4, read=4, backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
    s.mount("http://", adapter); s.mount("https://", adapter)
    return s

HTTP = make_session()

# =============================================================================
# Helpers: parse SDMX-JSON & JSON-stat
# =============================================================================
def _period_to_date(p: str) -> Optional[pd.Timestamp]:
    p = str(p)
    if re.match(r"^\d{4}-\d{2}$", p):
        return pd.to_datetime(p) + pd.offsets.MonthEnd(0)
    if "-Q" in p:
        y, q = p.split("-Q"); y = int(y); m = int(q) * 3
        return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    if re.match(r"^\d{4}$", p):
        return pd.Timestamp(int(p), 12, 31)
    return pd.to_datetime(p, errors="coerce")

def parse_sdmx_json(j: Dict[str, Any]) -> pd.DataFrame:
    try:
        root = j.get("data", j)
        datasets = root.get("dataSets") or []
        if not datasets: return pd.DataFrame()
        ds = datasets[0]
        structure = root.get("structure", {})
        obs_dims = (structure.get("dimensions", {}) or {}).get("observation", [])
        time_values: Optional[List[str]] = None
        for d in obs_dims:
            if d.get("id") in ("TIME_PERIOD", "TIME"):
                time_values = [v.get("id") for v in d.get("values", [])]
                break
        if not time_values: return pd.DataFrame()

        rec: List[Dict[str, Any]] = []
        series = ds.get("series") or {}
        if series:
            for s in series.values():
                for idx, raw in (s.get("observations") or {}).items():
                    t = time_values[int(idx)]
                    v = raw[0] if isinstance(raw, list) else raw
                    if v is None: continue
                    rec.append({"time": t, "value": float(v)})
        else:
            for idx, raw in (ds.get("observations") or {}).items():
                t = time_values[int(idx)]
                v = raw[0] if isinstance(raw, list) else raw
                if v is None: continue
                rec.append({"time": t, "value": float(v)})

        if not rec: return pd.DataFrame()
        df = pd.DataFrame(rec)
        df["date"] = df["time"].map(_period_to_date)
        return df.dropna(subset=["date"]).sort_values("date")[["date", "value"]]
    except Exception:
        return pd.DataFrame()

def _compute_strides(size: List[int]) -> List[int]:
    stride = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        stride[i] = stride[i + 1] * size[i + 1]
    return stride

def parse_jsonstat_dataset(obj: Dict[str, Any]) -> pd.DataFrame:
    try:
        if obj.get("class") != "dataset":
            datasets = obj.get("datasets") or obj.get("data")
            if isinstance(datasets, list) and datasets:
                obj = datasets[0]
        dim = obj["dimension"]
        ids: List[str] = obj.get("id") or dim.get("id")
        size: List[int] = obj.get("size") or dim.get("size")
        role = (dim.get("role") or {})
        time_id = (role.get("time") or [None])[0] or next((i for i in ids if i.lower()=="time"), None)
        if not time_id: return pd.DataFrame()

        def codes_for(did: str) -> List[str]:
            cat = dim[did]["category"]
            idx = cat.get("index")
            if isinstance(idx, dict):  # mapping code->pos
                return [k for k, _ in sorted(idx.items(), key=lambda kv: kv[1])]
            if isinstance(idx, list):
                return idx
            if "label" in cat and isinstance(cat["label"], dict):
                return list(cat["label"].keys())
            return []

        codes: Dict[str, List[str]] = {did: codes_for(did) for did in ids}
        values = obj.get("value")

        if isinstance(values, dict):  # sparse
            full = [None] * int(np.prod(size))
            for k, v in values.items():
                try: full[int(k)] = v
                except Exception: pass
            values = full
        if not isinstance(values, list): return pd.DataFrame()

        if len(ids)==1 and ids[0]==time_id:
            t_codes = codes[time_id]
            rec = [
                {"time": t_codes[i], "value": float(values[i])}
                for i in range(min(len(values), len(t_codes))) if values[i] is not None
            ]
            df = pd.DataFrame(rec); df["date"] = df["time"].map(_period_to_date)
            return df.dropna(subset=["date"]).sort_values("date")[["date","value"]]

        strides = _compute_strides(size)
        ti = ids.index(time_id); t_codes = codes[time_id]
        selector = [0]*len(ids); out=[]
        for tpos, t in enumerate(t_codes):
            selector[ti] = tpos
            lin = sum(selector[i]*strides[i] for i in range(len(ids)))
            if lin < len(values) and values[lin] is not None:
                out.append({"time": t, "value": float(values[lin])})
        df = pd.DataFrame(out); df["date"] = df["time"].map(_period_to_date)
        return df.dropna(subset=["date"]).sort_values("date")[["date","value"]]
    except Exception:
        return pd.DataFrame()

# =============================================================================
# ISTAT auto-try
# =============================================================================
ISTAT_BASES = [
    "https://sdmx.istat.it/SDMXWS/rest",        # newer
    "https://esploradati.istat.it/SDMXWS/rest", # legacy
]
ISTAT_AGENCY = "IT1"
ISTAT_FLOW_ID = "151_874"
ISTAT_VERSIONS_TRY = ["1.2", "1.1", "1.0"]

ISTAT_SEX_MAP = {"Total": "9", "Male": "1", "Female": "2"}
ISTAT_ADJ_MAP = {"NSA": "N", "SA": "Y"}  # adjustment codes

# چند الگوی کلید که در عمل برای این فلو دیده می‌شود (برخی آیینه‌ها UNEM_R را می‌خواهند)
def istat_key_candidates(sex: str, age: str, adj: str) -> List[str]:
    return [
        f"M.IT.UNEM_R.{adj}.{sex}.{age}",  # با DATA_TYPE=UNEM_R
        f"M.IT.{sex}.{age}.{adj}",         # ساده بدون DATA_TYPE
        f"M.IT.{adj}.{sex}.{age}",         # ترتیب جایگزین
    ]

@st.cache_data(show_spinner=False)
def istat_fetch_auto(age_code: str, sex_label: str, s_adj_label: str,
                     start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Try multiple ISTAT endpoints/versions/keys. Returns (df, debug)."""
    sex = ISTAT_SEX_MAP.get(sex_label, "9")
    adj = ISTAT_ADJ_MAP.get(s_adj_label, "Y")
    params = {"startPeriod": start, "endPeriod": end, "format": "sdmx-json"}

    attempts: List[Dict[str, Any]] = []
    for base in ISTAT_BASES:
        for ver in ISTAT_VERSIONS_TRY:
            flowref = f"{ISTAT_AGENCY},{ISTAT_FLOW_ID},{ver}"
            for key in istat_key_candidates(sex, age_code, adj):
                url = f"{base}/data/{flowref}/{key}"
                try:
                    r = HTTP.get(url, params=params, timeout=(12, 90))
                    info = {
                        "url": r.url, "status": r.status_code,
                        "elapsed_sec": getattr(r, "elapsed", None).total_seconds() if hasattr(r,"elapsed") else None,
                    }
                    if r.status_code == 200:
                        try:
                            df = parse_sdmx_json(r.json())
                        except Exception:
                            info["error"] = "JSON parse failed"
                            attempts.append(info); continue
                        if not df.empty:
                            attempts.append({**info, "ok": True})
                            return df, {"source":"ISTAT", "ok": True, "attempts": attempts}
                        else:
                            info["note"] = "Empty after parse"
                    else:
                        txt = r.text[:300] if isinstance(r.text, str) else ""
                        info["body_head"] = txt
                    attempts.append(info)
                except Exception as ex:
                    attempts.append({"url": url, "status": None, "error": str(ex)})
                # یک تلاش اضافه با وایلدکارد
                wc_url = url + ".*"
                try:
                    r = HTTP.get(wc_url, params=params, timeout=(12, 90))
                    info = {"url": r.url, "status": r.status_code,
                            "elapsed_sec": getattr(r,"elapsed",None).total_seconds() if hasattr(r,"elapsed") else None}
                    if r.status_code == 200:
                        try:
                            df = parse_sdmx_json(r.json())
                        except Exception:
                            info["error"] = "JSON parse failed (wc)"
                            attempts.append(info); continue
                        if not df.empty:
                            attempts.append({**info, "ok": True, "wildcard": True})
                            return df, {"source":"ISTAT", "ok": True, "attempts": attempts}
                        else:
                            info["note"] = "Empty after parse (wc)"
                    else:
                        info["body_head"] = (r.text[:300] if isinstance(r.text,str) else "")
                    attempts.append(info)
                except Exception as ex:
                    attempts.append({"url": wc_url, "status": None, "error": str(ex)})

    return pd.DataFrame(), {"source":"ISTAT", "ok": False, "attempts": attempts}

# =============================================================================
# Eurostat
# =============================================================================
EUROSTAT_STATS_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
EUROSTAT_DATASET = "une_rt_m"  # MUST be lowercase

def map_age_to_eurostat(age_code: str) -> str:
    # une_rt_m supports: TOTAL, Y15-24, Y25-74, (and sometimes Y15-74, Y_LT25)
    if age_code in ("Y15-24","Y25-74","Y15-74"):
        return age_code
    if age_code in ("Y15-34","Y35-49","Y50-64","Y50-74","Y15-64"):
        return "TOTAL"
    return "TOTAL"

@st.cache_data(show_spinner=False)
def eurostat_fetch(geo: str, sex_label: str, age_code: str, s_adj_label: str,
                   start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sex_map = {"Total":"T","Male":"M","Female":"F"}
    sex = sex_map.get(sex_label,"T")
    age = map_age_to_eurostat(age_code)
    s_adj = s_adj_label if s_adj_label in ("SA","NSA","TC") else "SA"

    params = {
        "lang": "en",
        "unit": "PC_ACT",
        "s_adj": s_adj,
        "sex": sex,
        "age": age,
        "geo": geo,
        "time": f"{start[:4]}-{end[:4]}",
    }
    url = f"{EUROSTAT_STATS_BASE}/{EUROSTAT_DATASET}"
    attempts = []
    try:
        r = HTTP.get(url, params=params, timeout=(12, 60))
        info = {
            "url": r.url, "status": r.status_code,
            "elapsed_sec": getattr(r,"elapsed",None).total_seconds() if hasattr(r,"elapsed") else None
        }
        if r.status_code == 200:
            try:
                df = parse_jsonstat_dataset(r.json())
            except Exception as ex:
                info["error"] = f"JSON-stat parse failed: {ex}"
                attempts.append(info)
                return pd.DataFrame(), {"source":"EUROSTAT", "ok": False, "attempts": attempts}
            if df.empty:
                info["note"] = "Empty after parse"
                attempts.append(info)
                return pd.DataFrame(), {"source":"EUROSTAT", "ok": False, "attempts": attempts}
            # clip to exact months
            sdt = pd.to_datetime(f"{start}-01") + pd.offsets.MonthEnd(0)
            edt = pd.to_datetime(f"{end}-01") + pd.offsets.MonthEnd(0)
            df = df[(df["date"] >= sdt) & (df["date"] <= edt)]
            attempts.append({**info, "ok": True})
            return df, {"source":"EUROSTAT", "ok": True, "attempts": attempts}
        else:
            info["body_head"] = r.text[:300] if isinstance(r.text,str) else ""
            attempts.append(info)
            return pd.DataFrame(), {"source":"EUROSTAT", "ok": False, "attempts": attempts}
    except Exception as ex:
        attempts.append({"url": url, "status": None, "error": str(ex)})
        return pd.DataFrame(), {"source":"EUROSTAT", "ok": False, "attempts": attempts}

# =============================================================================
# Demo fallback
# =============================================================================
def demo_monthly(start: str, end: str, base: float = 9.5) -> pd.DataFrame:
    idx = pd.date_range(start + "-01", end + "-01", freq="MS") + pd.offsets.MonthEnd(0)
    n = len(idx); np.random.seed(42)
    seasonal = np.sin(np.arange(n)*2*np.pi/12) * 0.25
    drift = np.linspace(0, -0.8, n)
    noise = np.random.randn(n) * 0.2
    val = base + seasonal + drift + noise
    return pd.DataFrame({"date": idx, "unemployment": val}).sort_values("date")

# =============================================================================
# Business types
# =============================================================================
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

@dataclass
class FetchOptions:
    geo: str = "IT"
    sex_label: str = "Total"
    age_code: str = "Y15-74"
    s_adj_label: str = "SA"
    start_year: int = 2015
    start_month: int = 1
    end_year: int = datetime.utcnow().year
    end_month: int = datetime.utcnow().month
    source_priority: List[str] = field(default_factory=lambda: ["ISTAT","EUROSTAT"])

    @property
    def start(self) -> str: return f"{self.start_year}-{self.start_month:02d}"
    @property
    def end(self) -> str:   return f"{self.end_year}-{self.end_month:02d}"

@dataclass
class SeriesResult:
    df: pd.DataFrame
    source: str
    debug: Dict[str, Any]

class UnemploymentFetcher:
    def fetch(self, opt: FetchOptions) -> SeriesResult:
        debug_all = {"tries": []}
        for src in [s.upper() for s in (opt.source_priority or [])]:
            if src == "ISTAT" and opt.geo == "IT" and opt.s_adj_label in ("SA","NSA"):
                df, dbg = istat_fetch_auto(opt.age_code, opt.sex_label, opt.s_adj_label, opt.start, opt.end)
                debug_all["tries"].append(dbg)
                if not df.empty:
                    return SeriesResult(df=df.rename(columns={"value":"unemployment"}), source="ISTAT", debug=debug_all)
            if src == "EUROSTAT":
                df, dbg = eurostat_fetch(opt.geo, opt.sex_label, opt.age_code, opt.s_adj_label, opt.start, opt.end)
                debug_all["tries"].append(dbg)
                if not df.empty:
                    return SeriesResult(df=df.rename(columns={"value":"unemployment"}), source="EUROSTAT", debug=debug_all)
        # demo
        ddf = demo_monthly(opt.start, opt.end)
        debug_all["demo"] = True
        return SeriesResult(df=ddf, source="DEMO", debug=debug_all)

# =============================================================================
# UI
# =============================================================================
with st.sidebar:
    st.subheader("Settings")
    geo = st.selectbox("Geo (country)", options=["IT","DE","FR","ES","EU27_2020"], index=0)
    sex_label = st.selectbox("Sex", options=["Total","Male","Female"], index=0)
    age_label = st.selectbox("Age band (ISTAT style)", options=list(AGE_CHOICES.keys()), index=0)
    age_code = AGE_CHOICES[age_label]
    s_adj_label = st.selectbox("Seasonal adjustment", options=["SA","NSA","TC"], index=0,
                               help="ISTAT supports SA/NSA; Eurostat supports SA/NSA/TC.")
    c1, c2 = st.columns(2)
    with c1:
        start_year = st.number_input("Start year", 1990, 2100, 2015, 1)
        start_month = st.number_input("Start month", 1, 12, 1, 1)
    with c2:
        end_year = st.number_input("End year", 1990, 2100, datetime.utcnow().year, 1)
        end_month = st.number_input("End month", 1, 12, min(datetime.utcnow().month,12), 1)

    st.markdown("---")
    st.markdown("**Source priority**")
    srcs = st.multiselect("Try in order", options=["ISTAT","EUROSTAT"], default=["ISTAT","EUROSTAT"])
    if not srcs: srcs = ["ISTAT","EUROSTAT"]
    st.markdown("---")
    show_debug = st.checkbox("Show detailed debug attempts", value=True)

fetcher = UnemploymentFetcher()

colA, colB = st.columns([2,1])
with colA:
    fetch_btn = st.button("🔎 Fetch data", type="primary")
with colB:
    test_btn = st.button("🧪 Self-test (known-good queries)")

def render_series(res: SeriesResult, opt: FetchOptions):
    if res.source == "ISTAT":
        st.success("Source = **ISTAT** (official Italy)")
    elif res.source == "EUROSTAT":
        st.success("Source = **Eurostat** (Statistics API v1.0)")
    else:
        st.warning("Using **Demo data** — all primary sources failed.")

    df = res.df.dropna()
    if df.empty:
        st.error("No observations returned. Try different dimensions or a wider time window.")
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Observations", f"{len(df)}")
    k2.metric("Start", df["date"].min().strftime("%Y-%m"))
    k3.metric("End", df["date"].max().strftime("%Y-%m"))
    k4.metric("Latest", f"{df['unemployment'].iloc[-1]:.2f}%")

    st.markdown("### Target series")
    st.line_chart(df.set_index("date")["unemployment"])
    st.dataframe(df, use_container_width=True)

    # Derived
    m = df.set_index("date").sort_index()
    m["mom_%"] = m["unemployment"].pct_change()*100
    m["yoy_%"] = m["unemployment"].pct_change(12)*100
    with st.expander("Derived changes (m/m %, y/y %)"):
        st.dataframe(m[["mom_%","yoy_%"]].reset_index())

    st.download_button("Download CSV", data=df.to_csv(index=False), mime="text/csv",
                       file_name=f"unemployment_{res.source.lower()}_{opt.geo}_{opt.start}_{opt.end}.csv")

    if show_debug:
        st.markdown("---")
        st.subheader("Debug attempts")
        st.caption("Every HTTP attempt (URL, status, first bytes). Use this to see invalid keys or 404/400 causes.")
        for i, t in enumerate(res.debug.get("tries", []), start=1):
            st.markdown(f"**Attempt group {i}: {t.get('source')}**")
            st.json(t, expanded=False)

# Run
if fetch_btn:
    opt = FetchOptions(
        geo=geo, sex_label=sex_label, age_code=age_code, s_adj_label=s_adj_label,
        start_year=int(start_year), start_month=int(start_month),
        end_year=int(end_year), end_month=int(end_month),
        source_priority=srcs,
    )
    with st.spinner("Fetching unemployment series…"):
        res = fetcher.fetch(opt)
    render_series(res, opt)

if test_btn:
    st.info("Running self-tests…")
    tests = [
        # Known-good Eurostat
        FetchOptions(geo="IT", sex_label="Total", age_code="Y15-74", s_adj_label="SA",
                     start_year=2015, start_month=1, end_year=datetime.utcnow().year, end_month=datetime.utcnow().month,
                     source_priority=["EUROSTAT"]),
        # Try ISTAT (Italy only, SA)
        FetchOptions(geo="IT", sex_label="Total", age_code="Y15-74", s_adj_label="SA",
                     start_year=2015, start_month=1, end_year=datetime.utcnow().year, end_month=datetime.utcnow().month,
                     source_priority=["ISTAT"]),
    ]
    for idx, opt in enumerate(tests, start=1):
        with st.spinner(f"Test {idx}: {opt.source_priority[0]} {opt.geo} {opt.sex_label}/{opt.age_code}/{opt.s_adj_label} …"):
            res = fetcher.fetch(opt)
        st.markdown(f"### Test {idx} → Source: **{res.source}**")
        render_series(res, opt)
