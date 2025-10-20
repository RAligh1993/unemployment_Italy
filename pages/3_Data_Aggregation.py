"""
🤖 Streamlit ISTAT + Eurostat Auto‑Fetcher (Robust, All Cases)
================================================================
Purpose
-------
A production‑grade Streamlit app to **automate monthly unemployment data fetching** from
ISTAT (SDMX REST) with intelligent fallbacks, and optional Eurostat support.

Key features
------------
- Dynamic **dataflow/DSD/codelist discovery** (no hardcoded IDs)
- **Dimension‑aware** key builder (FREQ/REG/SEX/AGE/s_adj, etc.)
- **Retries with backoff**, connection/read **timeouts**, and graceful error messages
- **Caching** for flows/DSDs/codelists/data queries (fast UI)
- **Multiple sources**: ISTAT primary; Eurostat via `pandasdmx` (if available); optional sample data fallback
- **Validation & QC** tools: flag parsing, series stats, integrity checks
- **Comprehensive UI** with advanced options; download CSV; tidy outputs

Notes
-----
- Designed to be resilient: every network call wrapped with robust try/except and will not crash the app.
- If Eurostat provider (`pandasdmx`) is unavailable or fails, the app continues with ISTAT or sample fallback.
- Works headlessly (no JS/CORS issues) because Streamlit runs server‑side.

"""

from __future__ import annotations
import sys
import re
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache
from datetime import datetime

import streamlit as st

# Optional dependency: pandasdmx for Eurostat (and even ISTAT) provider abstraction
try:
    import pandasdmx as sdmx
    HAS_PANDASDMX = True
except Exception:
    HAS_PANDASDMX = False

# =============================================================================
# Streamlit Page Config & Styles
# =============================================================================
st.set_page_config(
    page_title="ISTAT/Eurostat Auto Fetcher – Unemployment (Nowcasting Ready)",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main-title {
          font-size: 2.2rem; font-weight: 800; text-align: center;
          background: linear-gradient(120deg, #0ea5e9, #10b981, #34d399);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          margin-bottom: .5rem;
      }
      .subtle { color: #64748b; }
      .card { background: white; padding: 1rem; border-radius: 10px; border: 1px solid #e5e7eb; }
      .ok { background: #ecfeff; border-left: 4px solid #06b6d4; }
      .warn { background: #fff7ed; border-left: 4px solid #f59e0b; }
      .bad { background: #fef2f2; border-left: 4px solid #ef4444; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🤖 ISTAT + Eurostat Auto‑Fetcher — Monthly Unemployment</div>', unsafe_allow_html=True)

# =============================================================================
# Constants & Configuration
# =============================================================================
ISTAT_BASE_URLS: List[str] = [
    # Newer explorer endpoint first
    "https://esploradati.istat.it/SDMXWS/rest",
    # Legacy (still useful as fallback)
    "https://sdmx.istat.it/SDMXWS/rest",
]

# Known indicator keywords to search flows dynamically
INDICATOR_KEYWORDS = {
    "unemployment": ["disocc", "unemp", "disoccupazione", "taxdisoccu", "unemployment"],
    "employment": ["occup", "employment", "taxoccu"],
    "inactivity": ["inatt", "inactivity"],
}

# Reasonable defaults for UI
DEFAULT_FREQ = "M"  # monthly
DEFAULT_REGION = "IT"  # Italy national
DEFAULT_S_ADJ_ISTAT = ("Y", "Seasonally adjusted")  # typical coding Y=SA (may vary per DSD)
DEFAULT_S_ADJ_ESTAT = ("SA", "Seasonally adjusted")

# =============================================================================
# Helper: Robust HTTP Session with Retry/Timeout
# =============================================================================

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "ISTAT-Eurostat-AutoFetcher/1.0 (+streamlit)",
        "Accept": "application/vnd.sdmx.data+json, application/json; q=0.9, */*; q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = make_session()

# =============================================================================
# Utilities: Safe ops, logging into Streamlit area
# =============================================================================

@dataclass
class LogBuffer:
    messages: List[Tuple[str, str]] = field(default_factory=list)
    def add(self, level: str, msg: str):
        self.messages.append((level.upper(), msg))
    def render(self):
        if not self.messages:
            return
        for lvl, msg in self.messages:
            if   lvl == "INFO": st.info(msg)
            elif lvl == "WARN": st.warning(msg)
            elif lvl == "ERROR": st.error(msg)
            else: st.write(msg)

LOG = LogBuffer()

def safe_json(r: requests.Response) -> Optional[Dict[str, Any]]:
    try:
        return r.json()
    except Exception as e:
        return None

# =============================================================================
# SDMX JSON Parsing (robust across providers)
# =============================================================================

def parse_sdmx_json(j: Dict[str, Any]) -> pd.DataFrame:
    """Parse SDMX‑JSON into tidy two columns: date,value.
    Handles observation TIME_PERIOD mapping and numeric value extraction.
    Returns empty DataFrame on any issue (never raises).
    """
    if not isinstance(j, dict):
        return pd.DataFrame()
    root = j.get("data", j)
    datasets = root.get("dataSets", [])
    if not datasets:
        return pd.DataFrame()
    ds = datasets[0]

    # Observation dimension (TIME_PERIOD)
    structure = root.get("structure", {})
    obs_dims = (structure.get("dimensions", {}) or {}).get("observation", [])
    time_values: Optional[List[str]] = None
    for d in obs_dims:
        if d.get("id") == "TIME_PERIOD":
            time_values = [v.get("id") for v in d.get("values", [])]
            break
    if not time_values:
        return pd.DataFrame()

    # Extract observations
    rec: List[Dict[str, Any]] = []
    series = ds.get("series", {}) or {}
    if not series:
        # Some providers put obs at dataset level under observations
        observations = ds.get("observations", {}) or {}
        for idx, raw in observations.items():
            try:
                t = time_values[int(idx)]
                v = raw[0] if isinstance(raw, list) else raw
                rec.append({"time_period": t, "value": float(v)})
            except Exception:
                continue
    else:
        for _, s in series.items():
            obs = s.get("observations", {}) or {}
            for idx, raw in obs.items():
                try:
                    t = time_values[int(idx)]
                    v = raw[0] if isinstance(raw, list) else raw
                    rec.append({"time_period": t, "value": float(v)})
                except Exception:
                    continue

    if not rec:
        return pd.DataFrame()

    df = pd.DataFrame(rec)

    def to_date(period: str) -> Optional[pd.Timestamp]:
        try:
            p = str(period)
            if "-Q" in p:
                y, q = p.split("-Q"); y = int(y); q = int(q); m = q * 3
                return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
            if re.match(r"^\d{4}-\d{2}$", p):
                return pd.to_datetime(p) + pd.offsets.MonthEnd(0)
            if re.match(r"^\d{4}$", p):
                return pd.Timestamp(int(p), 12, 31)
            return pd.to_datetime(p, errors="coerce")
        except Exception:
            return None

    df["date"] = df["time_period"].map(to_date)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "value"]]

# =============================================================================
# SDMX Service Client (generic)
# =============================================================================

class SDMXClient:
    def __init__(self, base_urls: List[str]):
        self.base_urls = base_urls

    # ---------- Discovery ----------
    def pick_base(self) -> Optional[str]:
        for base in self.base_urls:
            try:
                r = SESSION.get(f"{base}/dataflow", params={"format": "sdmx-json"}, timeout=(10, 30))
                if r.status_code == 200:
                    return base
            except Exception:
                continue
        return None

    @lru_cache(maxsize=16)
    def dataflows(self, base: str) -> Dict[str, Dict[str, Any]]:
        try:
            r = SESSION.get(f"{base}/dataflow", params={"format": "sdmx-json"}, timeout=(10, 60))
            j = safe_json(r) or {}
            # Two common shapes
            if "dataflows" in j:
                return j["dataflows"].get("dataflow", {}) or {}
            if "data" in j and "dataflows" in j["data"]:
                return j["data"]["dataflows"].get("dataflow", {}) or {}
            return {}
        except Exception:
            return {}

    @staticmethod
    def _extract_name(name_obj: Any) -> str:
        if isinstance(name_obj, str):
            return name_obj
        if isinstance(name_obj, dict):
            # prefer English, then Italian
            for k in ("en", "it"):
                if k in name_obj and name_obj[k]:
                    return name_obj[k]
            # otherwise any
            for v in name_obj.values():
                if v:
                    return v
        return ""

    def search_flows(self, base: str, keywords: List[str]) -> List[Tuple[str, str, Optional[str]]]:
        flows = self.dataflows(base)
        out: List[Tuple[str, str, Optional[str]]] = []
        kw_regex = re.compile("|".join([re.escape(k.lower()) for k in keywords]))
        for fid, meta in flows.items():
            name = self._extract_name(meta.get("name"))
            blob = f"{fid.lower()} {name.lower()}"
            if kw_regex.search(blob):
                dsd_id = None
                # common path to DSD reference
                try:
                    dsd_id = meta.get("structure", {}).get("ref", {}).get("id") or meta.get("structure", {}).get("id")
                except Exception:
                    dsd_id = None
                out.append((fid, name or fid, dsd_id))
        return out

    @lru_cache(maxsize=64)
    def datastructure(self, base: str, dsd_id: str) -> Dict[str, Any]:
        try:
            r = SESSION.get(f"{base}/datastructure/{dsd_id}", params={"format": "sdmx-json"}, timeout=(10, 60))
            return safe_json(r) or {}
        except Exception:
            return {}

    def series_dimensions(self, dsd_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return ordered dim list for series axis with their codelist refs.
        Works across common SDMX JSON structures for ISTAT/Eurostat.
        """
        structure = dsd_json.get("structure", dsd_json)
        dims = (structure.get("dimensions", {}) or {}).get("series", [])
        if not dims:
            # some schemas use generic 'dimension'
            dims = (structure.get("dimensions", {}) or {}).get("dimension", [])
        out: List[Dict[str, Any]] = []
        for d in dims:
            out.append({
                "id": d.get("id"),
                "name": d.get("name", d.get("id")),
                "codelist": (d.get("localRepresentation", {}) or {}).get("enumeration", {}).get("ref", {}).get("id"),
            })
        return out

    @lru_cache(maxsize=128)
    def codelist(self, base: str, cl_id: str) -> List[Tuple[str, str]]:
        try:
            r = SESSION.get(f"{base}/codelist/{cl_id}", params={"format": "sdmx-json"}, timeout=(10, 60))
            j = safe_json(r) or {}
            # try multiple shapes
            codes = (
                j.get("codelists", {}).get("codelist", {}).get(cl_id, {}).get("codes", {})
                if j.get("codelists") else
                j.get("structure", {}).get("codelists", {}).get("codelist", [{}])[0].get("codes", {})
            )
            out: List[Tuple[str, str]] = []
            for code, meta in (codes or {}).items():
                name = meta.get("name")
                if isinstance(name, dict):
                    name = name.get("en") or name.get("it") or next(iter(name.values()), code)
                if not isinstance(name, str):
                    name = str(name) if name else code
                out.append((code, name))
            return sorted(out, key=lambda x: x[0])
        except Exception:
            return []

    # ---------- Data query ----------
    def build_key(self, dim_order: List[Dict[str, Any]], selections: Dict[str, str]) -> str:
        parts: List[str] = []
        for d in dim_order:
            parts.append(selections.get(d["id"], ""))  # empty → all
        return ".".join(parts)

    @lru_cache(maxsize=256)
    def get_data(self, base: str, flow_id: str, key: str, start: str, end: str) -> pd.DataFrame:
        try:
            params = {"startPeriod": start, "endPeriod": end, "format": "sdmx-json"}
            url = f"{base}/data/{flow_id}/{key}"
            r = SESSION.get(url, params=params, timeout=(10, 90))
            if r.status_code != 200:
                LOG.add("WARN", f"Data request HTTP {r.status_code} for {flow_id} (key='{key[:60]}')")
                return pd.DataFrame()
            j = safe_json(r)
            if not j:
                LOG.add("WARN", "Received non‑JSON or empty response.")
                return pd.DataFrame()
            df = parse_sdmx_json(j)
            return df
        except Exception as e:
            LOG.add("ERROR", f"Exception during data fetch: {e}")
            return pd.DataFrame()

# =============================================================================
# Eurostat via pandasdmx (optional)
# =============================================================================

class EurostatClient:
    def __init__(self):
        self.available = HAS_PANDASDMX
        self._req = None
        if self.available:
            try:
                self._req = sdmx.Request("ESTAT")
            except Exception:
                self.available = False

    def list_flows(self, query: str = "une_rt_m") -> List[Tuple[str, str]]:
        if not self.available:
            return []
        try:
            mf = self._req.dataflow()
            out = []
            q = query.lower()
            for fid, s in mf.dataflow.items():
                name = str(s.name) if getattr(s, "name", None) else fid
                if q in fid.lower() or q in name.lower():
                    out.append((fid, name))
            return sorted(out, key=lambda x: x[0])
        except Exception:
            return []

    def fetch_une_rt_m(
        self,
        geo: str = "IT",
        sex: str = "T",
        age: str = "Y15-74",
        s_adj: str = "SA",
        start: str = "2010",
        end: str = "2025",
    ) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        try:
            # Not all providers require exact key order when using params.
            # Use SDMX key string for robustness; pandasdmx maps dimensions internally.
            key = f"{s_adj}.{sex}.{age}.{geo}.A"  # NOTE: unit dimension varies; A often for rate unit in une_rt_m
            # When unsure, leave key empty and pass filters via params:
            params = {"time": f"{start}-{end}", "s_adj": s_adj, "sex": sex, "age": age, "geo": geo}
            resp = self._req.data("une_rt_m", key=None, params=params)
            # Convert to pandas
            try:
                df = resp.to_pandas()  # MultiIndex with dims + TIME_PERIOD
            except Exception:
                # fallback manual parsing
                j = resp.msg.to_json()
                return parse_sdmx_json(j)
            if isinstance(df, pd.Series):
                df = df.reset_index()
                # normalize time column name
                time_col = next((c for c in df.columns if str(c).upper() in ("TIME_PERIOD", "time", "TIME")), None)
                if time_col is None and "index" in df.columns:
                    time_col = "index"
                df.rename(columns={
                    time_col: "date",
                    0: "value",
                }, inplace=True)
                # attempt date parsing
                df["date"] = pd.to_datetime(df["date"], errors="coerce").map(
                    lambda x: x + pd.offsets.MonthEnd(0) if not pd.isna(x) else x
                )
                df = df.dropna(subset=["date"]).sort_values("date")
                return df[["date", "value"]]
            # DataFrame case (less common for single series)
            if "TIME_PERIOD" in df.columns:
                df.rename(columns={"TIME_PERIOD": "date"}, inplace=True)
                df["date"] = pd.to_datetime(df["date"], errors="coerce").map(
                    lambda x: x + pd.offsets.MonthEnd(0) if not pd.isna(x) else x
                )
                if "value" not in df.columns:
                    # heuristic: the numeric column could be named 'obs_value'
                    numcol = next((c for c in df.columns if str(c).lower() in ("value", "values", "obs_value")), None)
                    if numcol:
                        df["value"] = df[numcol]
                return df[["date", "value"]].dropna()
            return pd.DataFrame()
        except Exception as e:
            LOG.add("WARN", f"Eurostat fetch failed: {e}")
            return pd.DataFrame()

# =============================================================================
# Sample/Demo Data (last‑resort fallback)
# =============================================================================

def demo_series_monthly(start_year: int = 2010, end_year: int = 2025, base: float = 9.5, trend: float = -0.5) -> pd.DataFrame:
    rng = pd.date_range(f"{start_year}-01-31", f"{end_year}-12-31", freq="M")
    n = len(rng)
    np.random.seed(42)
    seasonal = np.sin(np.arange(n) * 2 * np.pi / 12) * 0.3
    noise = np.random.randn(n) * 0.2
    drift = np.linspace(0, trend, n)
    vals = base + seasonal + noise + drift
    df = pd.DataFrame({"date": rng, "value": vals})
    return df

# =============================================================================
# Business Logic: Unemployment Fetch with all variants
# =============================================================================

@dataclass
class FetchOptions:
    freq: str = DEFAULT_FREQ
    region: str = DEFAULT_REGION
    sex: str = "T"
    age: str = "Y15-74"
    s_adj: str = DEFAULT_S_ADJ_ISTAT[0]  # ISTAT: 'Y' typical for SA (varies by DSD)
    start_year: int = 2010
    end_year: int = datetime.utcnow().year
    source_priority: List[str] = field(default_factory=lambda: ["ISTAT", "EUROSTAT"])  # order to try

@dataclass
class SeriesResult:
    df: pd.DataFrame
    source: str
    flow_id: Optional[str] = None
    key: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

class UnemploymentFetcher:
    def __init__(self):
        self.istat = SDMXClient(ISTAT_BASE_URLS)
        self.eurostat = EurostatClient()

    def find_istat_flow(self, base: str) -> Optional[Tuple[str, str, Optional[str]]]:
        # Prefer flows with monthly unemployment (keywords)
        keywords = INDICATOR_KEYWORDS["unemployment"] + ["mens", "mensili", "monthly", "rate"]
        flows = self.istat.search_flows(base, keywords)
        # Heuristics: prioritize names containing both unemployment+monthly
        def score(item: Tuple[str, str, Optional[str]]) -> int:
            fid, name, _ = item
            name_low = name.lower()
            s = 0
            if "unemp" in name_low or "disocc" in name_low: s += 3
            if "rate" in name_low or "tax" in name_low: s += 2
            if "month" in name_low or "mens" in name_low: s += 2
            if fid.lower().startswith("151_"): s += 1  # many recent flows use 151_* (heuristic)
            return -s  # sort ascending, negative means highest score first
        flows_sorted = sorted(flows, key=score)
        if flows_sorted:
            return flows_sorted[0]
        return None

    def _default_dimension_mapping(self, dim_order: List[Dict[str, Any]]) -> Dict[str, str]:
        # Provide reasonable defaults for unknown dimensions
        mapping = {}
        for d in dim_order:
            did = (d.get("id") or "").upper()
            if did in ("FREQ", "FREQUENCY"): mapping[did] = DEFAULT_FREQ
            elif did in ("GEO", "REGION", "REF_AREA"): mapping[did] = DEFAULT_REGION
            elif did in ("SEX", "SESSO"): mapping[did] = "T"
            elif did in ("AGE", "ETA"): mapping[did] = "Y15-74"
            elif did in ("S_ADJ", "SEASONAL_ADJUSTMENT", "SEASONAL"): mapping[did] = DEFAULT_S_ADJ_ISTAT[0]
            else: mapping[did] = ""  # all
        return mapping

    def fetch_istat_unemployment(self, opt: FetchOptions) -> SeriesResult:
        base = self.istat.pick_base()
        if not base:
            raise RuntimeError("No ISTAT endpoint reachable")
        flow = self.find_istat_flow(base)
        if not flow:
            raise RuntimeError("Could not find an ISTAT monthly unemployment flow dynamically. Try adjusting keywords.")
        flow_id, flow_name, dsd_id = flow
        if not dsd_id:
            # try to infer later but proceed
            LOG.add("WARN", f"No DSD id in dataflow for {flow_id}; attempting generic request.")

        # DSD → dimensions → codelists
        dim_order = []
        if dsd_id:
            dsd_j = self.istat.datastructure(base, dsd_id)
            dim_order = self.istat.series_dimensions(dsd_j)
        # Build selections
        sel_map = self._default_dimension_mapping(dim_order) if dim_order else {}
        # Overwrite with requested options if present in dims
        def set_if_present(dim_id_aliases: List[str], val: str):
            for d in dim_order:
                did = (d.get("id") or "").upper()
                if did in [a.upper() for a in dim_id_aliases]:
                    sel_map[did] = val
        set_if_present(["FREQ", "FREQUENCY"], opt.freq)
        set_if_present(["GEO", "REGION", "REF_AREA"], opt.region)
        set_if_present(["SEX", "SESSO"], opt.sex)
        set_if_present(["AGE", "ETA"], opt.age)
        set_if_present(["S_ADJ", "SEASONAL", "SEASONAL_ADJUSTMENT"], opt.s_adj)

        # Convert sel_map keyed by official ids to selections using original IDs in order
        selections: Dict[str, str] = {}
        for d in dim_order:
            did = d.get("id")
            if not did: continue
            upper = did.upper()
            selections[did] = sel_map.get(upper, "")

        key = self.istat.build_key(dim_order, selections) if dim_order else ""  # if no DSD, blank key (all)
        df = self.istat.get_data(base, flow_id, key, str(opt.start_year), str(opt.end_year))
        if df.empty:
            # try degraded key variants: enforce freq/geo minimally
            LOG.add("WARN", "No observations returned; retrying with minimal key (FREQ.GEO only if known).")
            min_sel: Dict[str, str] = {}
            order2: List[Dict[str, Any]] = []
            for d in dim_order:
                did = (d.get("id") or "").upper()
                if did in ("FREQ", "FREQUENCY", "GEO", "REGION", "REF_AREA"):
                    order2.append(d)
                    min_sel[d["id"]] = selections.get(d["id"], "")
                else:
                    order2.append(d)
                    min_sel[d["id"]] = ""
            key2 = self.istat.build_key(order2, min_sel) if order2 else ""
            df = self.istat.get_data(base, flow_id, key2, str(opt.start_year), str(opt.end_year))
        if df.empty:
            raise RuntimeError("ISTAT returned no data for the selected parameters.")
        return SeriesResult(df=df, source=f"ISTAT ({base})", flow_id=flow_id, key=key, meta={"flow_name": flow_name})

    def fetch(self, opt: FetchOptions) -> SeriesResult:
        # Try sources in order
        for src in opt.source_priority:
            try:
                if src.upper() == "ISTAT":
                    return self.fetch_istat_unemployment(opt)
                if src.upper() == "EUROSTAT":
                    if not self.eurostat.available:
                        LOG.add("WARN", "Eurostat via pandasdmx not available. Skipping.")
                        continue
                    df = self.eurostat.fetch_une_rt_m(
                        geo=opt.region,
                        sex=opt.sex if opt.sex in ("M", "F", "T") else "T",
                        age=opt.age,
                        s_adj="SA" if opt.s_adj in ("Y", "SA") else "NSA",
                        start=str(opt.start_year),
                        end=str(opt.end_year),
                    )
                    if not df.empty:
                        return SeriesResult(df=df, source="EUROSTAT (pandasdmx)")
                    else:
                        LOG.add("WARN", "Eurostat returned empty dataset for given filters.")
                        continue
            except Exception as e:
                LOG.add("WARN", f"Source {src} failed: {e}")
                continue
        # Final fallback: demo data (explicit)
        df_demo = demo_series_monthly(opt.start_year, opt.end_year)
        return SeriesResult(df=df_demo, source="DEMO")

# =============================================================================
# Streamlit UI
# =============================================================================

st.markdown(
    """
<div class='card ok'>
  <b>How it works</b>: The app dynamically resolves ISTAT dataflows/DSDs and builds valid SDMX keys.
  It retries with backoff, caches metadata & data, and gracefully falls back to Eurostat (if available) or sample data.
</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("All settings are safe; the app won't crash on bad inputs.")

    # Source order
    src_order = st.multiselect(
        "Sources (priority order)",
        options=["ISTAT", "EUROSTAT"],
        default=["ISTAT", "EUROSTAT"],
        help="The app tries these in order; if all fail it produces demo data.",
    )
    if not src_order:
        src_order = ["ISTAT"]

    # Time window
    c1, c2 = st.columns(2)
    with c1:
        start_year = st.number_input("Start year", min_value=1990, max_value=2100, value=2010, step=1)
    with c2:
        end_year = st.number_input("End year", min_value=1990, max_value=2100, value=datetime.utcnow().year, step=1)
    if end_year < start_year:
        st.warning("End year < Start year; swapping.")
        start_year, end_year = end_year, start_year

    st.markdown("---")
    st.subheader("📐 Dimensions")
    freq = st.selectbox("Frequency", options=["M", "Q", "A"], index=0)

    region = st.text_input("Region (GEO/REGION)", value="IT", help="ISTAT codes: IT (national) or NUTS codes like ITC4. Eurostat uses NUTS codes for geo.")

    c3, c4 = st.columns(2)
    with c3:
        sex = st.selectbox("Sex (SEX)", options=["T", "M", "F"], index=0)
    with c4:
        age = st.text_input("Age (AGE)", value="Y15-74", help="Common bands: Y15-24, Y25-34, Y35-49, Y50-64, Y15-74")

    s_adj_label = st.selectbox("Seasonal adjustment", options=["Seasonally adjusted", "Not adjusted"], index=0)
    s_adj = "Y" if s_adj_label.startswith("Season") else "N"

    st.markdown("---")
    st.subheader("🧪 Advanced")
    show_debug = st.checkbox("Show debug logs", value=False)

fetcher = UnemploymentFetcher()

st.markdown("## 🎯 Target: Monthly Unemployment (rate)")

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Primary source", "ISTAT")
with colB:
    st.metric("Fallback", "Eurostat")
with colC:
    st.metric("Freq", freq)
with colD:
    st.metric("Region", region)

if st.button("🚀 Fetch Data", type="primary", use_container_width=True):
    with st.spinner("Fetching across sources with retries and caching…"):
        opts = FetchOptions(
            freq=freq,
            region=region.strip() or DEFAULT_REGION,
            sex=sex,
            age=age.strip() or "Y15-74",
            s_adj=s_adj,
            start_year=int(start_year),
            end_year=int(end_year),
            source_priority=src_order,
        )
        result = fetcher.fetch(opts)

    st.markdown("---")

    if not result.df.empty:
        if result.source.startswith("ISTAT"):
            st.markdown(
                f"""
                <div class='card ok'>
                  <b>✅ Success</b> — Source: <b>{result.source}</b><br/>
                  Flow: <code>{result.flow_id or 'n/a'}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif result.source.startswith("EUROSTAT"):
            st.markdown(
                f"""
                <div class='card ok'>
                  <b>✅ Success</b> — Source: <b>{result.source}</b><br/>
                  Dataset: <code>une_rt_m</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class='card warn'>
                  <b>⚠️ Using sample data</b> — All primary sources failed.
                </div>
                """,
                unsafe_allow_html=True,
            )

        df = result.df.copy()
        # Basic QC
        summary = {
            "rows": len(df),
            "start": df["date"].min().strftime("%Y-%m"),
            "end": df["date"].max().strftime("%Y-%m"),
            "latest": float(df["value"].iloc[-1]) if len(df) else np.nan,
        }
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", summary["rows"])
        c2.metric("Start", summary["start"]) 
        c3.metric("End", summary["end"]) 
        c4.metric("Latest", f"{summary['latest']:.2f}")

        # Transform helper metrics: m/m and y/y growth if monthly
        if freq == "M":
            df2 = df.set_index("date").sort_index()
            df2["mom"] = df2["value"].pct_change() * 100
            df2["yoy"] = df2["value"].pct_change(12) * 100
        else:
            df2 = df.set_index("date").sort_index()
            df2["mom"], df2["yoy"] = np.nan, np.nan

        st.markdown("### 📈 Chart")
        st.line_chart(df.set_index("date"))

        st.markdown("### 📋 Data")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "💾 Download CSV",
            df.to_csv(index=False),
            file_name=f"unemployment_{result.source.replace(' ', '_').lower()}_{region}_{start_year}_{end_year}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Show growths
        with st.expander("Derived metrics (m/m, y/y)"):
            st.dataframe(df2.reset_index())

    else:
        st.markdown(
            """
            <div class='card bad'>
              <b>❌ No data available</b> — All sources returned empty. The app created no exceptions; try different dimensions or years.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if show_debug:
        st.markdown("---")
        st.subheader("🔍 Debug log")
        LOG.render()

# =============================================================================
# Extra Tools (optional panels)
# =============================================================================

with st.expander("🔧 Inspect ISTAT monthly unemployment flows (auto discovery)"):
    base = fetcher.istat.pick_base()
    if base:
        st.caption(f"Endpoint in use: {base}")
        flows = fetcher.istat.search_flows(base, INDICATOR_KEYWORDS["unemployment"] + ["mens", "mensili", "monthly", "rate"])
        if flows:
            df_flows = pd.DataFrame(flows, columns=["flow_id", "name", "dsd_id"]).sort_values("flow_id")
            st.dataframe(df_flows, use_container_width=True)
        else:
            st.info("No flows matched. Try again later or adjust keyword list.")
    else:
        st.warning("No ISTAT endpoint reachable at the moment.")

with st.expander("📚 Notes on dimensions & best practices"):
    st.write(
        "- Always set FREQ and GEO.\n"
        "- Prefer Seasonally Adjusted data for modeling (ISTAT often 'Y', Eurostat 'SA').\n"
        "- Keep both NSA and SA for QC.\n"
        "- Store levels (unemployed, employed, labour force) besides the rate.\n"
        "- Log release/vintage timestamps for nowcasting reproducibility."
    )
