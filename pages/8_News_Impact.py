# pages/8_News_Impact.py — Pro v3.1 (fixed & hardened)
# =============================================================================
# Multi‑region news ingestion (RSS/NewsAPI/GDELT/CSV) → scoring (lexicon+VADER+LLM)
# → daily metrics → monthly signals per REGION (ITALY/EUROPE/INTERNATIONAL)
# → impact analysis vs. unemployment target. Works WITHOUT API via RSS.
# Key fixes vs v3:
#   • Robust RSS parser (multiple date fields) + dynamic TTL key for cache
#   • Auto‑collect RSS on page load (no clicks) with safe state merge
#   • Guards for missing libs/keys; never crashes → shows informative messages
#   • LLM scoring clamped to [-1,1] + safe JSON parsing
#   • Safer resampling, deduplication, and CSV upload handling
# =============================================================================
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

# Optional deps (graceful degradation)
try:
    import feedparser  # RSS
except Exception:  # pragma: no cover
    feedparser = None  # type: ignore

try:
    import requests  # HTTP for NewsAPI/GDELT/local LLM
except Exception:  # pragma: no cover
    requests = None  # type: ignore

# Sentiment (optional)
try:
    import nltk  # for VADER
    from nltk.sentiment import SentimentIntensityAnalyzer
    _HAVE_VADER = True
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore
    _HAVE_VADER = False

# LLM providers (all optional)
try:
    from openai import OpenAI  # >=1.30
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None  # type: ignore

# =============================================================================
# State (robust import with fallback)
# =============================================================================
try:
    from utils.state import AppState  # type: ignore
except Exception:
    class _State:
        def __init__(self) -> None:
            self.y_monthly: Optional[pd.Series] = None
            self.panel_monthly: Optional[pd.DataFrame] = None
            self.news_daily: Optional[pd.DataFrame] = None
            self.news_monthly: Optional[pd.DataFrame] = None
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

# =============================================================================
# Curated outlets by region
# =============================================================================
OUTLETS: Dict[str, List[Dict[str, List[str]]]] = {
    "ITALY": [
        {"name": "ANSA Economy", "rss": ["https://www.ansa.it/sito/notizie/economia/economia_rss.xml"], "domains": ["ansa.it"]},
        {"name": "Il Sole 24 Ore", "rss": ["https://www.ilsole24ore.com/rss/economia.xml"], "domains": ["ilsole24ore.com"]},
        {"name": "La Repubblica", "rss": ["https://www.repubblica.it/rss/economia/rss2.0.xml"], "domains": ["repubblica.it"]},
        {"name": "Corriere Economia", "rss": ["https://xml2.corriereobjects.it/rss/economia.xml"], "domains": ["corriere.it"]},
    ],
    "EUROPE": [
        {"name": "Euronews Economy", "rss": ["https://www.euronews.com/rss?level=theme&name=news&theme=economy"], "domains": ["euronews.com"]},
        {"name": "ECB Press", "rss": ["https://www.ecb.europa.eu/press/html/press.en.html"], "domains": ["ecb.europa.eu"]},
        {"name": "EU Commission Press", "rss": ["https://ec.europa.eu/commission/presscorner/home/en/rss"], "domains": ["ec.europa.eu"]},
    ],
    "INTERNATIONAL": [
        {"name": "BBC Business", "rss": ["https://feeds.bbci.co.uk/news/business/rss.xml"], "domains": ["bbc.co.uk","bbc.com"]},
        {"name": "Reuters Economy", "rss": ["https://www.reuters.com/markets/economicNews/rss"], "domains": ["reuters.com"]},
        {"name": "AP Business", "rss": ["https://apnews.com/hub/apf-business?utm_source=apnews.com&utm_medium=referral&utm_campaign=rss"], "domains": ["apnews.com"]},
    ],
}
REGIONS = list(OUTLETS.keys())

# =============================================================================
# Lexicon (transparent deterministic score)
# =============================================================================
ITALIAN_NEG = {"disoccupazione": -1.0, "licenziamenti": -1.0, "cassa integrazione": -0.8, "crisi": -0.6, "recessione": -0.8, "sciopero": -0.4}
ITALIAN_POS = {"assunzioni": +0.9, "occupazione": +0.7, "nuovi posti di lavoro": +1.0, "ripresa": +0.6}
EN_NEG = {"unemployment": -0.8, "layoffs": -1.0, "jobless": -0.7, "recession": -0.8, "strike": -0.4}
EN_POS = {"hiring": +0.9, "jobs added": +0.8, "job growth": +0.8, "recovery": +0.6}

# =============================================================================
# Helpers
# =============================================================================

def _to_dt(s):
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

@st.cache_data(show_spinner=False)
def _parse_rss_cached(url: str, bucket: str) -> dict:
    # bucket makes TTL configurable externally (e.g., rounded time string)
    return feedparser.parse(url) if feedparser is not None else {}

def _extract_dt(entry) -> Optional[datetime]:
    for k in ("published_parsed", "updated_parsed", "created_parsed"):
        val = getattr(entry, k, None)
        if val is not None:
            try: return datetime(*val[:6])
            except Exception: pass
    for k in ("published", "updated", "created", "pubDate", "dc_date"):
        val = getattr(entry, k, None)
        if val:
            try: return pd.to_datetime(val).to_pydatetime()
            except Exception: pass
    return None

# =============================================================================
# UI
# =============================================================================

st.title("📰 News → Impact on Unemployment (By Region)")
st.caption("RSS works out‑of‑the‑box (no API). Add NewsAPI/GDELT/CSV for more coverage. Score via lexicon/VADER/LLM and build monthly regional signals.")

with st.sidebar:
    st.header("Providers & Regions")
    use_rss = st.checkbox("Use RSS (no API)", value=True)
    use_newsapi = st.checkbox("Use NewsAPI (needs key)", value=False)
    use_gdelt = st.checkbox("Use GDELT (experimental)", value=False)

    st.markdown("---")
    st.header("Regions & Outlets")
    region = st.selectbox("Region", options=REGIONS, index=0)
    names = [o["name"] for o in OUTLETS[region]]
    selected = st.multiselect("Outlets", options=names, default=names[:min(3, len(names))])
    custom_rss = st.text_area("Custom RSS (one per line)", "", height=80)

    st.markdown("---")
    st.header("Date window")
    if state.y_monthly is not None and not state.y_monthly.empty:
        ymin, ymax = state.y_monthly.index.min().date(), state.y_monthly.index.max().date()
    else:
        ymin, ymax = datetime(2019,1,1).date(), datetime.today().date()
    dr = st.date_input("From / To", (ymin, ymax))

    st.markdown("---")
    st.header("Auto RSS")
    auto_collect = st.checkbox("Auto‑collect on load (RSS only)", value=True)
    rss_ttl_min = st.number_input("RSS cache TTL (minutes)", 5, 180, 30)

    st.markdown("---")
    st.header("Scoring")
    use_vader = st.checkbox("VADER sentiment (if available)", value=_HAVE_VADER)
    llm_enable = st.checkbox("LLM impact (econ & unemployment)", value=False)
    llm_provider = st.selectbox("Provider", ["openai","anthropic","gemini","local"], disabled=not llm_enable)
    llm_model = st.text_input("Model", value="gpt-4o-mini", disabled=not llm_enable)
    llm_temp = st.slider("Creativity", 0.0, 1.0, 0.2, 0.05, disabled=not llm_enable)
    llm_max = st.slider("Max LLM articles", 10, 300, 60, 10, disabled=not llm_enable)

    st.markdown("---")
    st.header("Aggregation")
    smooth_ma = st.slider("Monthly smoothing MA", 1, 12, 3)
    lead_lag = st.slider("Lead/Lag months", -6, 6, 0)
    append_panel = st.checkbox("Append to panel_monthly", value=True)

# Build selected sources
sel = [o for o in OUTLETS[region] if o["name"] in selected]
region_rss = [u for o in sel for u in o.get("rss", [])]
if custom_rss.strip():
    region_rss += [u.strip() for u in custom_rss.splitlines() if u.strip()]
region_domains = [d for o in sel for d in o.get("domains", [])]

# CSV upload
up = st.file_uploader("Upload CSV (date,title,source,content,url; UTF‑8)", type=["csv"])
upload_df = pd.read_csv(up) if up is not None else pd.DataFrame(columns=["date","title","source","content","url"]) 
if not upload_df.empty:
    for c in ["date","title","source","content","url"]:
        if c not in upload_df.columns: upload_df[c] = ""
    upload_df["date"] = _to_dt(upload_df["date"]).dt.normalize()
    upload_df["provider"], upload_df["region"] = "upload", region

# =============================================================================
# Providers
# =============================================================================

def _rss_fetch(urls: List[str], d_from: datetime, d_to: datetime, region_tag: str, ttl_bucket: str) -> pd.DataFrame:
    if not use_rss or feedparser is None or not urls:
        return pd.DataFrame(columns=["date","title","source","content","url","provider","region"]) 
    rows = []
    for u in urls:
        try:
            d = _parse_rss_cached(u, ttl_bucket) or {}
            feed_name = (d.get("feed") or {}).get("title", "RSS")
            for e in d.get("entries", []):
                dt = _extract_dt(e) or datetime.now(tz=timezone.utc).replace(tzinfo=None)
                if not (d_from <= dt.date() <= d_to):
                    continue
                rows.append({
                    "date": pd.to_datetime(dt).tz_localize(None).normalize(),
                    "title": getattr(e, "title", "") or "",
                    "source": feed_name,
                    "content": getattr(e, "summary", "") or "",
                    "url": getattr(e, "link", "") or "",
                    "provider": "rss",
                    "region": region_tag,
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


def _newsapi_fetch(domains: List[str], query: str, d_from: datetime, d_to: datetime, region_tag: str) -> pd.DataFrame:
    if not use_newsapi or requests is None or not domains:
        return pd.DataFrame(columns=["date","title","source","content","url","provider","region"]) 
    key = os.getenv("NEWSAPI_KEY") or (st.secrets.get("NEWSAPI_KEY") if hasattr(st, "secrets") else None)
    if not key:
        st.info("NEWSAPI_KEY not set; skipping NewsAPI.")
        return pd.DataFrame()
    base = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "domains": ",".join(domains),
        "from": str(d_from),
        "to": str(d_to),
        "language": "en,it",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "page": 1,
        "apiKey": key,
    }
    rows = []
    try:
        r = requests.get(base, params=params, timeout=30)
        if r.status_code != 200:
            st.warning(f"NewsAPI error {r.status_code}: {r.text[:120]}")
            return pd.DataFrame()
        for a in r.json().get("articles", []):
            dt = pd.to_datetime(a.get("publishedAt")).tz_localize(None).normalize()
            rows.append({
                "date": dt,
                "title": a.get("title", "") or "",
                "source": (a.get("source") or {}).get("name", "NewsAPI"),
                "content": a.get("description", "") or "",
                "url": a.get("url", ""),
                "provider": "newsapi",
                "region": region_tag,
            })
    except Exception as e:
        st.warning(f"NewsAPI request failed: {e}")
    return pd.DataFrame(rows)

# (GDELT kept optional/minimal; can be expanded later)

# =============================================================================
# Auto RSS on load (no API)
# =============================================================================

d_from, d_to = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
# TTL bucket string to drive cache invalidation (e.g., '2025-10-17 12:30')
ttl_bucket = pd.Timestamp.utcnow().floor(f"{int(rss_ttl_min)}min").strftime("%Y-%m-%d %H:%M")

if auto_collect and use_rss and region_rss:
    key = f"auto::{region}::{hash(tuple(region_rss))}::{d_from.date()}::{d_to.date()}::{ttl_bucket}"
    if st.session_state.get("_auto_key") != key:
        auto_df = _rss_fetch(region_rss, d_from, d_to, region, ttl_bucket)
        if not auto_df.empty:
            if state.news_daily is None or state.news_daily.empty:
                state.news_daily = auto_df
            else:
                state.news_daily = pd.concat([state.news_daily, auto_df], ignore_index=True)
                state.news_daily = state.news_daily.drop_duplicates(subset=["date","title","region"], keep="first")
            st.success(f"[Auto‑RSS] Collected {len(auto_df):,} items for {region}.")
        else:
            st.info("[Auto‑RSS] No items in current window.")
        st.session_state["_auto_key"] = key

# Manual collection
if st.button("Collect news", type="primary"):
    frames = [
        _rss_fetch(region_rss, d_from, d_to, region, ttl_bucket),
        _newsapi_fetch(region_domains, "(unemployment OR layoffs OR jobs OR lavoro OR disoccupazione)", d_from, d_to, region),
    ]
    if not upload_df.empty:
        frames.append(upload_df)
    all_news = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if all_news.empty:
        st.warning("No news collected. Check providers/keys or widen the date range.")
    else:
        all_news = all_news.dropna(subset=["date"]).copy()
        all_news["title"] = all_news["title"].fillna("").astype(str).str.strip()
        all_news = all_news.sort_values("date").drop_duplicates(subset=["date","title","region"], keep="first")
        if state.news_daily is None or state.news_daily.empty:
            state.news_daily = all_news
        else:
            state.news_daily = pd.concat([state.news_daily, all_news], ignore_index=True)
            state.news_daily = state.news_daily.drop_duplicates(subset=["date","title","region"], keep="first")
        st.success(f"Added {len(all_news):,} items. Total: {len(state.news_daily):,}.")

# Preview
if state.news_daily is not None and not state.news_daily.empty:
    st.markdown("### Latest 30 articles (all regions)")
    st.dataframe(state.news_daily.sort_values("date").tail(30), use_container_width=True)
else:
    st.info("No news in session yet. Enable Auto‑RSS or click Collect news.")

# =============================================================================
# Scoring — lexicon + optional VADER + optional LLM
# =============================================================================

st.markdown("---")
st.subheader("2) Score articles → daily metrics by region")

if state.news_daily is None or state.news_daily.empty:
    st.stop()

news = state.news_daily.copy()

# Lexicon score

def _kw_score(text: str) -> float:
    t = (text or "").lower()
    s = 0.0
    for w, v in ITALIAN_NEG.items():
        if w in t: s += v
    for w, v in ITALIAN_POS.items():
        if w in t: s += v
    for w, v in EN_NEG.items():
        if w in t: s += v
    for w, v in EN_POS.items():
        if w in t: s += v
    return float(s)

news["kw_score"] = [
    _kw_score(f"{t}\n{c}") for t, c in zip(news.get("title",""), news.get("content",""))
]

# VADER (optional)
if use_vader and _HAVE_VADER:
    try:
        try: nltk.data.find("sentiment/vader_lexicon.zip")
        except Exception: nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()
        news["vader_compound"] = [sia.polarity_scores((f"{t}. {c}"))['compound'] for t, c in zip(news.get("title",""), news.get("content",""))]
    except Exception:
        st.info("VADER unavailable; skipping.")
        news["vader_compound"] = np.nan
else:
    if "vader_compound" not in news.columns:
        news["vader_compound"] = np.nan

# LLM scoring (econ_impact & unemp_impact in [-1,1])
LLM_PROMPT = """
You are an economic analyst. Read the headline and the short summary and return
a STRICT JSON object with this exact schema:

{"econ_impact": 0.0, "unemp_impact": 0.0, "rationale": ""}

Constraints:
- econ_impact ∈ [-1, 1]
- unemp_impact ∈ [-1, 1]
- rationale ≤ 60 words
Guidance:
- econ_impact = macro/financial direction.
- unemp_impact = direction on unemployment (higher => more unemployment).
- If unsure, return zeros.
Output ONLY the JSON object, no extra text.
"""



def _llm_client(provider: str, model: str):
    if not llm_enable: return None
    if provider == "openai" and OpenAI is not None:
        key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
        return OpenAI(api_key=key) if key else None
    if provider == "anthropic" and anthropic is not None:
        key = os.getenv("ANTHROPIC_API_KEY") or (st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None)
        return anthropic.Anthropic(api_key=key) if key else None
    if provider == "gemini" and genai is not None:
        key = os.getenv("GOOGLE_API_KEY") or (st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None)
        if key:
            genai.configure(api_key=key)
            return genai.GenerativeModel(model)
    if provider == "local" and requests is not None:
        return os.getenv("OLLAMA_ENDPOINT") or (st.secrets.get("OLLAMA_ENDPOINT") if hasattr(st, "secrets") else None)
    return None


def _clamp01(x: float) -> float:
    try: return float(max(-1.0, min(1.0, x)))
    except Exception: return 0.0


def _llm_score_subset(df: pd.DataFrame) -> pd.DataFrame:
    if not llm_enable or df.empty:
        return pd.DataFrame(columns=["econ_impact","unemp_impact","rationale"]) 
    client = _llm_client(llm_provider, llm_model)
    if client is None:
        st.info("LLM provider not configured; skipping.")
        return pd.DataFrame(columns=["econ_impact","unemp_impact","rationale"]) 
    n = min(llm_max, len(df))
    out = []
    prog = st.progress(0.0, text=f"LLM scoring ({llm_provider})…")
    for i, (_, r) in enumerate(df.head(n).iterrows(), 1):
        txt = f"Title: {r['title']}\nSource: {r['source']}\nRegion: {r['region']}\nSummary: {r['content']}"
        try:
            if llm_provider == "openai":
                resp = client.chat.completions.create(
                    model=llm_model, temperature=float(llm_temp),
                    messages=[{"role":"system","content": LLM_PROMPT},{"role":"user","content": txt}],
                    response_format={"type":"json_object"},
                )
                js = json.loads(resp.choices[0].message.content)
            elif llm_provider == "anthropic":
                m = client.messages.create(
                    model=llm_model, temperature=float(llm_temp), max_tokens=256,
                    system=LLM_PROMPT, messages=[{"role":"user","content": txt}],
                )
                js = json.loads(m.content[0].text)
            elif llm_provider == "gemini":
                rp = client.generate_content([{"role":"user","parts":[LLM_PROMPT + "\n\n" + txt]}])
                js = json.loads(rp.text)
            else:  # local HTTP endpoint
                rr = requests.post(client, json={"model": llm_model, "messages":[{"role":"system","content": LLM_PROMPT},{"role":"user","content": txt}]}, timeout=60)
                data = rr.json()
                content = data.get("message",{}).get("content") or data.get("choices", [{}])[0].get("message",{}).get("content")
                js = json.loads(content)
            out.append({
                "econ_impact": _clamp01(js.get("econ_impact", 0.0)),
                "unemp_impact": _clamp01(js.get("unemp_impact", 0.0)),
                "rationale": str(js.get("rationale", ""))[:240],
            })
        except Exception:
            out.append({"econ_impact": 0.0, "unemp_impact": 0.0, "rationale": "n/a"})
        prog.progress(i/n)
    prog.empty()
    return pd.DataFrame(out)

# ensure LLM columns exist
for c in ["econ_impact","unemp_impact","rationale"]:
    if c not in news.columns:
        news[c] = np.nan if c != "rationale" else ""

if llm_enable:
    to_score_idx = news[news["econ_impact"].isna()].index
    if len(to_score_idx) > 0:
        scored = _llm_score_subset(news.loc[to_score_idx])
        if not scored.empty:
            news.loc[to_score_idx[:len(scored)], ["econ_impact","unemp_impact","rationale"]] = scored.values

# persist
state.news_daily = news

# Daily metrics
daily = (
    news.groupby([news["date"].dt.normalize(), "region"]).agg(
        n=("title","count"),
        kw_mean=("kw_score","mean"),
        vader=("vader_compound","mean"),
        llm_econ=("econ_impact","mean"),
        llm_unemp=("unemp_impact","mean"),
    ).reset_index().rename(columns={"date":"day"})
)

st.markdown("**Daily metrics (tail)**")
st.dataframe(daily.tail(30), use_container_width=True)

figd = px.bar(daily, x="day", y="n", color="region", title="#articles per day (stacked)")
figd.update_layout(template="plotly_white", height=360)
st.plotly_chart(figd, use_container_width=True)

# =============================================================================
# Monthly signals (per region) + append to panel
# =============================================================================

st.markdown("---")
st.subheader("3) Build monthly signals by region")

monthly_parts = []
for reg in REGIONS:
    g = daily[daily["region"] == reg].set_index("day")
    if g.empty: continue
    mm = pd.DataFrame({
        f"news_{reg.lower()}__count": g["n"].resample("M").sum(min_count=1),
        f"news_{reg.lower()}__kw": g["kw_mean"].resample("M").mean(),
        f"news_{reg.lower()}__vader": g["vader"].resample("M").mean(),
        f"news_{reg.lower()}__llm_econ": g["llm_econ"].resample("M").mean(),
        f"news_{reg.lower()}__llm_unemp": g["llm_unemp"].resample("M").mean(),
    })
    if smooth_ma and smooth_ma > 1:
        for c in list(mm.columns):
            mm[c + f"_ma{smooth_ma}"] = mm[c].rolling(smooth_ma, min_periods=1).mean()
    if lead_lag != 0:
        mm = mm.shift(int(lead_lag))
    monthly_parts.append(mm)

monthly = pd.concat(monthly_parts, axis=1).sort_index() if monthly_parts else pd.DataFrame()
state.news_monthly = monthly

st.markdown("**Monthly signals (tail)**")
st.dataframe(monthly.tail(24), use_container_width=True)

if append_panel and state.panel_monthly is not None:
    state.panel_monthly = state.panel_monthly.join(monthly, how="outer")
    st.success("Signals appended to panel_monthly.")

# =============================================================================
# Impact analysis vs target
# =============================================================================

st.markdown("---")
st.subheader("4) Impact analysis vs target")

if state.y_monthly is None or state.y_monthly.empty or monthly.empty:
    st.info("Need target and monthly signals.")
else:
    y = state.y_monthly
    figm = go.Figure(); figm.add_trace(go.Scatter(x=y.index, y=(y-y.mean())/(y.std(ddof=0)+1e-9), name="Target (z)", mode="lines", line=dict(width=3)))
    for reg in REGIONS:
        col = f"news_{reg.lower()}__llm_unemp"
        if col in monthly.columns:
            z = (monthly[col] - monthly[col].mean())/(monthly[col].std(ddof=0)+1e-9)
            figm.add_trace(go.Scatter(x=monthly.index, y=z, name=f"{reg}: llm_unemp (z)", mode="lines"))
    figm.update_layout(template="plotly_white", height=420, title="Target vs LLM unemployment impact (z‑scores)")
    st.plotly_chart(figm, use_container_width=True)

    rows = []
    for reg in REGIONS:
        for sig in [f"news_{reg.lower()}__kw", f"news_{reg.lower()}__llm_unemp", f"news_{reg.lower()}__vader"]:
            if sig not in monthly.columns: continue
            for L in range(-6, 7):
                if L >= 0:
                    ya, xa = y.align(monthly[sig].shift(L), join="inner")
                else:
                    ya, xa = y.shift(-L).align(monthly[sig], join="inner")
                if len(ya) >= 6 and len(xa) == len(ya):
                    c = float(np.corrcoef(ya.values, xa.values)[0,1])
                    rows.append({"region": reg, "feature": sig, "lag": L, "corr": c})
    CC = pd.DataFrame(rows)
    if not CC.empty:
        best = (CC.sort_values("corr", ascending=False).groupby(["region","feature"]).head(1))
        st.markdown("**Best cross‑correlations (per region/feature)**")
        st.dataframe(best.sort_values(["corr"], ascending=False), use_container_width=True)
        figcc = px.bar(CC, x="lag", y="corr", color="region", facet_col="feature", facet_col_wrap=2, title="Cross‑correlation by region")
        figcc.update_layout(template="plotly_white", height=520)
        st.plotly_chart(figcc, use_container_width=True)

# =============================================================================
# Export + housekeeping
# =============================================================================

st.markdown("---")
st.subheader("5) Export")
if state.news_daily is not None and not state.news_daily.empty:
    st.download_button("news_daily_scored.csv", state.news_daily.to_csv(index=False).encode("utf-8"), file_name="news_daily_scored.csv")
if state.news_monthly is not None and not state.news_monthly.empty:
    st.download_button("news_monthly_signals.csv", state.news_monthly.to_csv().encode("utf-8"), file_name="news_monthly_signals.csv")

with st.expander("⚙️ Debug & maintenance", expanded=False):
    st.write("Providers:", {"rss": bool(use_rss and feedparser), "newsapi": use_newsapi, "gdelt": use_gdelt})
    st.write("Selected RSS count:", len(region_rss))
    if state.news_daily is not None:
        st.write("Daily rows:", len(state.news_daily))
    if state.news_monthly is not None:
        st.write("Monthly rows:", len(state.news_monthly))
    if st.button("Reset news state"):
        state.news_daily = None; state.news_monthly = None
        st.success("Cleared.")

st.caption("RSS requires no API and updates automatically with cache TTL. For historical archives, use CSV or APIs with keys. LLM impact is optional and supports OpenAI/Claude/Gemini or a local HTTP endpoint.")
