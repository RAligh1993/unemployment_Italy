# pages/8_News_Impact.py — Multi‑source News (Italy/EU/International) + LLM Impact Scoring
# ============================================================================
# What this page does (v2, pro):
#   • Curated outlets grouped by Region: ITALY / EUROPE / INTERNATIONAL
#   • User chooses outlets; for each outlet we pull via NewsAPI (if key & toggle)
#     otherwise we fall back to RSS (or user‑provided custom RSS). CSV upload too.
#   • Two‑stage scoring per article: keyword_index (fast) + optional LLM scoring
#     (econ_impact, unemployment_impact in [-1, +1] with rationale). Providers:
#     OpenAI / Claude / Gemini / Local HTTP (e.g., Ollama). Fully optional-safe.
#   • Daily → Monthly signals per REGION (counts, kw_score, llm_econ, llm_unemp)
#     with smoothing and lead/lag; append to panel_monthly with region prefix.
#   • Impact analysis vs target: overlays, cross‑corr, rolling corr per region.
#   • Export CSVs; persistent state: news_daily, news_monthly.
# ============================================================================

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

# Optional deps
try:
    import feedparser  # RSS
except Exception:  # pragma: no cover
    feedparser = None  # type: ignore

try:
    import requests  # NewsAPI / Local HTTP LLM
except Exception:  # pragma: no cover
    requests = None  # type: ignore

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

# ----------------------------------------------------------------------------
# State (robust import with fallback)
# ----------------------------------------------------------------------------
try:
    from utils.state import AppState  # type: ignore
except Exception:
    class _State:
        def __init__(self) -> None:
            self.y_monthly: Optional[pd.Series] = None
            self.panel_monthly: Optional[pd.DataFrame] = None
            self.bt_results: Dict[str, pd.Series] = {}
            self.bt_metrics: Optional[pd.DataFrame] = None
            self.news_daily: Optional[pd.DataFrame] = None
            self.news_monthly: Optional[pd.DataFrame] = None
    class AppState:  # type: ignore
        @staticmethod
        def init() -> _State:
            if "_app" not in st.session_state:
                st.session_state["_app"] = _State()
            return st.session_state["_app"]  # type: ignore
        @staticmethod
        def get() -> _State:  # noqa
            return AppState.init()

state = AppState.init()

# ----------------------------------------------------------------------------
# Curated outlets by region (editable). Each item: name, rss (list), newsapi_domains (list)
# ----------------------------------------------------------------------------
OUTLETS = {
    "ITALY": [
        {"name": "ANSA Economy", "rss": ["https://www.ansa.it/sito/notizie/economia/economia_rss.xml"], "domains": ["ansa.it"]},
        {"name": "Il Sole 24 Ore", "rss": ["https://www.ilsole24ore.com/rss/economia.xml"], "domains": ["ilsole24ore.com"]},
        {"name": "La Repubblica", "rss": ["https://www.repubblica.it/rss/economia/rss2.0.xml"], "domains": ["repubblica.it"]},
        {"name": "Corriere Economia", "rss": ["https://xml2.corriereobjects.it/rss/economia.xml"], "domains": ["corriere.it"]},
    ],
    "EUROPE": [
        {"name": "Euronews Economy", "rss": ["https://www.euronews.com/rss?level=theme&name=news&theme=economy"], "domains": ["euronews.com"]},
        {"name": "ECB Press", "rss": ["https://www.ecb.europa.eu/press/html/press.en.html"], "domains": ["ecb.europa.eu"]},
        {"name": "EU Commission Economy", "rss": ["https://ec.europa.eu/commission/presscorner/home/en/rss"], "domains": ["ec.europa.eu"]},
    ],
    "INTERNATIONAL": [
        {"name": "BBC Business", "rss": ["https://feeds.bbci.co.uk/news/business/rss.xml"], "domains": ["bbc.co.uk","bbc.com"]},
        {"name": "Reuters Economy", "rss": ["https://www.reuters.com/markets/economicNews/rss"], "domains": ["reuters.com"]},
        {"name": "AP Business", "rss": ["https://apnews.com/hub/apf-business?utm_source=apnews.com&utm_medium=referral&utm_campaign=rss"], "domains": ["apnews.com"]},
    ],
}

REGIONS = list(OUTLETS.keys())

# ----------------------------------------------------------------------------
# Simple lexicon for keyword_index (IT/EN)
# ----------------------------------------------------------------------------
ITALIAN_NEG = {"disoccupazione": -1.0, "licenziamenti": -1.0, "cassa integrazione": -0.8, "crisi": -0.6, "recessione": -0.8}
ITALIAN_POS = {"assunzioni": +0.9, "occupazione": +0.8, "nuovi posti di lavoro": +1.0, "ripresa": +0.6}
EN_NEG = {"unemployment": -0.8, "layoffs": -1.0, "jobless": -0.7, "recession": -0.8, "strike": -0.4}
EN_POS = {"hiring": +0.9, "jobs added": +0.8, "job growth": +0.8, "recovery": +0.6}

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def _to_dt(s):
    return pd.to_datetime(s, errors="coerce").tz_localize(None)

def _normalize_date(s):
    return _to_dt(s).dt.normalize()

@st.cache_data(show_spinner=False)
def _hash_id(*parts) -> str:
    m = hashlib.sha1()
    for p in parts:
        m.update(str(p).encode("utf-8"))
    return m.hexdigest()[:16]

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------

st.title("📰 News → Impact on Unemployment (by Region)")
st.caption("Select outlets by region, collect news via RSS/NewsAPI, score with keywords and optional LLM, build monthly signals, and analyze impact vs target.")

with st.sidebar:
    st.header("Providers & Regions")
    use_rss = st.checkbox("Use RSS", value=True)
    use_newsapi = st.checkbox("Use NewsAPI (needs key)", value=False)
    st.markdown("— You can also upload a CSV (date,title,source,content,url) in the page.")

    st.markdown("---")
    st.header("Outlets by region")
    region = st.selectbox("Region", REGIONS, index=0)
    names = [o["name"] for o in OUTLETS[region]]
    selected_outlets = st.multiselect("Choose outlets", options=names, default=names[: min(3, len(names))])
    add_custom_rss = st.text_area("Custom RSS (one per line)", value="", height=80)

    st.markdown("---")
    st.header("Date window")
    if state.y_monthly is not None and not state.y_monthly.empty:
        ymin, ymax = state.y_monthly.index.min().date(), state.y_monthly.index.max().date()
    else:
        ymin, ymax = datetime(2019, 1, 1).date(), datetime.today().date()
    dr = st.date_input("From / To", value=(ymin, ymax))

    st.markdown("---")
    st.header("LLM Scoring (optional)")
    llm_enable = st.checkbox("LLM economic/unemployment impact scoring", value=False)
    llm_provider = st.selectbox("Provider", ["openai","anthropic","gemini","local"], index=0, disabled=not llm_enable)
    llm_model = st.text_input("Model", value="gpt-4o-mini", disabled=not llm_enable)
    llm_temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.05, disabled=not llm_enable)
    llm_max = st.slider("Max articles to LLM‑score per run", 10, 300, 60, 10, disabled=not llm_enable)

    st.markdown("---")
    st.header("Aggregation")
    smooth_ma = st.slider("Monthly smoothing MA", 1, 12, 3)
    lead_lag = st.slider("Lead/Lag months (signal shift)", -6, 6, 0)
    append_panel = st.checkbox("Append monthly signals to panel_monthly", value=True)

# ----------------------------------------------------------------------------
# Ingest controls (center)
# ----------------------------------------------------------------------------

st.subheader("1) Ingest news")

col1, col2, col3 = st.columns([1.2, 1.2, 1.0])

with col1:
    st.markdown("**Selected outlets (current region)**")
    st.write(", ".join(selected_outlets) if selected_outlets else "—")
with col2:
    st.markdown("**Providers enabled**")
    st.write(("RSS" if use_rss else "") + (" + NewsAPI" if use_newsapi else ""))
with col3:
    up = st.file_uploader("Upload CSV (date,title,source,content,url)", type=["csv"])

# ----------------------------------------------------------------------------
# Build outlet list / domains / rss for selected region
# ----------------------------------------------------------------------------

sel_objs = [o for o in OUTLETS[region] if o["name"] in selected_outlets]
region_rss = [u for o in sel_objs for u in o.get("rss", [])]
region_domains = [d for o in sel_objs for d in o.get("domains", [])]
if add_custom_rss.strip():
    region_rss.extend([u.strip() for u in add_custom_rss.splitlines() if u.strip()])

# CSV upload parse
csv_df = pd.read_csv(up) if up is not None else pd.DataFrame(columns=["date","title","source","content","url"]) 
if not csv_df.empty:
    for c in ["date","title","source","content","url"]:
        if c not in csv_df.columns:
            csv_df[c] = ""
    csv_df["date"] = _normalize_date(csv_df["date"])  # normalize tz
    csv_df["provider"] = "upload"; csv_df["region"] = region

# ----------------------------------------------------------------------------
# Provider helpers
# ----------------------------------------------------------------------------

def _rss_fetch(urls: List[str], d_from: datetime, d_to: datetime, region_tag: str) -> pd.DataFrame:
    if not use_rss or feedparser is None or not urls:
        return pd.DataFrame(columns=["date","title","source","content","url","provider","region"]) 
    rows = []
    for u in urls:
        try:
            d = feedparser.parse(u)
            feed_name = d.feed.get("title", "RSS") if hasattr(d, "feed") else "RSS"
            for e in d.entries:
                dt = None
                for k in ("published_parsed","updated_parsed"):
                    if hasattr(e, k) and getattr(e, k) is not None:
                        try:
                            dt = datetime(*getattr(e, k)[:6])
                            break
                        except Exception:
                            pass
                if dt is None:
                    continue
                dt = pd.to_datetime(dt).tz_localize(None)
                if not (d_from <= dt.date() <= d_to):
                    continue
                rows.append({
                    "date": dt.normalize(),
                    "title": getattr(e, "title", "").strip(),
                    "source": feed_name,
                    "content": getattr(e, "summary", ""),
                    "url": getattr(e, "link", ""),
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
        st.info("NEWSAPI_KEY not set; skipping NewsAPI calls.")
        return pd.DataFrame(columns=["date","title","source","content","url","provider","region"]) 
    base = "https://newsapi.org/v2/everything"
    rows = []
    # NewsAPI limits: pass domains (comma‑sep) and restrict languages
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
    try:
        r = requests.get(base, params=params, timeout=30)
        if r.status_code != 200:
            st.warning(f"NewsAPI error {r.status_code}: {r.text[:120]}")
            return pd.DataFrame()
        data = r.json()
        for a in data.get("articles", []):
            dt = pd.to_datetime(a.get("publishedAt")).tz_localize(None)
            rows.append({
                "date": dt.normalize(),
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

# ----------------------------------------------------------------------------
# Run ingestion
# ----------------------------------------------------------------------------

if st.button("Collect news", type="primary"):
    d_from, d_to = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    frames = []
    # Selected region only in this run; user can switch region and press again
    frames.append(_rss_fetch(region_rss, d_from, d_to, region))
    query = "(unemployment OR layoffs OR jobs OR lavoro OR disoccupazione)"
    frames.append(_newsapi_fetch(region_domains, query, d_from, d_to, region))
    if not csv_df.empty:
        frames.append(csv_df)

    all_news = (
        pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
        if frames else pd.DataFrame(columns=["date","title","source","content","url","provider","region"]) 
    )
    if all_news.empty:
        st.warning("No news collected for this run. Check providers/keys or widen date range.")
    else:
        # de‑dup by (date,title) within region
        all_news = all_news.dropna(subset=["date"]).copy()
        all_news["title"] = all_news["title"].fillna("").astype(str).str.strip()
        all_news["region"] = all_news["region"].fillna(region)
        all_news = all_news.sort_values("date").drop_duplicates(subset=["date","title","region"], keep="first")
        # merge to existing state (append)
        if state.news_daily is None or state.news_daily.empty:
            state.news_daily = all_news
        else:
            state.news_daily = pd.concat([state.news_daily, all_news], ignore_index=True).drop_duplicates(subset=["date","title","region"], keep="first")
        st.success(f"Collected {len(all_news):,} articles (region={region}). Total in session: {len(state.news_daily):,}")

# Preview
if state.news_daily is not None and not state.news_daily.empty:
    st.markdown("### Preview (latest 30 across regions)")
    st.dataframe(state.news_daily.sort_values("date").tail(30), use_container_width=True)

# ----------------------------------------------------------------------------
# 2) Scoring — keyword index and (optional) LLM impact
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("2) Score articles → daily metrics by region")

if state.news_daily is None or state.news_daily.empty:
    st.info("Collect some news first.")
    st.stop()

# Keyword score

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

news = state.news_daily.copy()
news["kw_score"] = [
    _kw_score(f"{t}\n{c}") for t, c in zip(news.get("title",""), news.get("content",""))
]

# Optional LLM scoring (econ_impact, unemp_impact in [-1,1])

def _llm_clients():
    cl = {
        "openai": None,
        "anthropic": None,
        "gemini": None,
    }
    if llm_enable:
        if llm_provider == "openai" and OpenAI is not None:
            key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
            if key:
                cl["openai"] = OpenAI(api_key=key)
        if llm_provider == "anthropic" and anthropic is not None:
            key = os.getenv("ANTHROPIC_API_KEY") or (st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None)
            if key:
                cl["anthropic"] = anthropic.Anthropic(api_key=key)
        if llm_provider == "gemini" and genai is not None:
            key = os.getenv("GOOGLE_API_KEY") or (st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None)
            if key:
                genai.configure(api_key=key)
                cl["gemini"] = genai.GenerativeModel(llm_model)
    return cl

LLM_PROMPT = (
    "You are an economic analyst. Read the headline and short summary. "
    "Return a strict JSON with fields: econ_impact [-1..1], unemp_impact [-1..1], rationale (<=60 words). "
    "econ_impact captures overall macro/financial impact; unemp_impact captures likely direction on unemployment (higher=more unemployment). "
    "Be conservative; if unsure, output 0.0. Example: {\"econ_impact\": -0.2, \"unemp_impact\": 0.1, \"rationale\": \"mild slowdown\"}."
)

@st.cache_data(show_spinner=False)
def _llm_cache_key(txt: str, prov: str, model: str) -> str:
    return _hash_id("llm", prov, model, txt[:500])


def _llm_score_rows(df_in: pd.DataFrame) -> pd.DataFrame:
    if not llm_enable:
        return pd.DataFrame(columns=["econ_impact","unemp_impact","rationale"]) 
    clients = _llm_clients()
    prov = llm_provider
    if prov == "local":
        endpoint = os.getenv("OLLAMA_ENDPOINT") or (st.secrets.get("OLLAMA_ENDPOINT") if hasattr(st, "secrets") else None)
    n = min(llm_max, len(df_in))
    out_rows = []
    bar = st.progress(0.0, text=f"LLM scoring ({prov})…")
    for i, (_, r) in enumerate(df_in.head(n).iterrows(), 1):
        txt = f"Title: {r['title']}\nSource: {r['source']}\nRegion: {r['region']}\nSummary: {r['content']}"
        ck = _llm_cache_key(txt, prov, llm_model)
        cached = st.session_state.get(ck)
        if cached:
            out_rows.append(cached)
            bar.progress(i / n)
            continue
        res = {"econ_impact": 0.0, "unemp_impact": 0.0, "rationale": "n/a"}
        try:
            if prov == "openai" and clients["openai"] is not None:
                c = clients["openai"]
                resp = c.chat.completions.create(
                    model=llm_model,
                    temperature=float(llm_temperature),
                    messages=[
                        {"role":"system","content": LLM_PROMPT},
                        {"role":"user","content": txt},
                    ],
                    response_format={"type":"json_object"},
                )
                res = json.loads(resp.choices[0].message.content)
            elif prov == "anthropic" and clients["anthropic"] is not None:
                c = clients["anthropic"]
                m = c.messages.create(
                    model=llm_model,
                    temperature=float(llm_temperature),
                    max_tokens=256,
                    system=LLM_PROMPT,
                    messages=[{"role":"user","content": txt}],
                )
                res = json.loads(m.content[0].text)
            elif prov == "gemini" and clients["gemini"] is not None:
                model = clients["gemini"]
                rp = model.generate_content([
                    {"role":"user","parts":[LLM_PROMPT + "\n\n" + txt]}  # simple prompt
                ])
                res = json.loads(rp.text)
            elif prov == "local" and requests is not None and (endpoint := (os.getenv("OLLAMA_ENDPOINT") or (st.secrets.get("OLLAMA_ENDPOINT") if hasattr(st,"secrets") else None))):
                payload = {"model": llm_model, "messages": [
                    {"role":"system","content": LLM_PROMPT},
                    {"role":"user","content": txt},
                ]}
                rr = requests.post(endpoint, json=payload, timeout=60)
                try:
                    data = rr.json()
                    content = data.get("message",{}).get("content") or data.get("choices", [{}])[0].get("message",{}).get("content")
                    res = json.loads(content)
                except Exception:
                    pass
        except Exception:
            pass
        # sanitize
        try:
            res = {
                "econ_impact": float(res.get("econ_impact", 0.0)),
                "unemp_impact": float(res.get("unemp_impact", 0.0)),
                "rationale": str(res.get("rationale", ""))[:240],
            }
        except Exception:
            res = {"econ_impact": 0.0, "unemp_impact": 0.0, "rationale": "n/a"}
        st.session_state[ck] = res
        out_rows.append(res)
        bar.progress(i / n)
    bar.empty()
    return pd.DataFrame(out_rows)

# Run LLM scoring on *new* rows without econ/unemp columns
need_llm = llm_enable and ("econ_impact" not in news.columns or news["econ_impact"].isna().all())
if need_llm:
    scored = _llm_score_rows(news)
    if not scored.empty:
        # pad if we scored subset only
        for col in ["econ_impact","unemp_impact","rationale"]:
            news[col] = news.get(col)
        news.loc[news.index[:len(scored)], ["econ_impact","unemp_impact","rationale"]] = scored.values
else:
    # ensure columns exist
    for col in ["econ_impact","unemp_impact","rationale"]:
        if col not in news.columns:
            news[col] = np.nan if col != "rationale" else ""

state.news_daily = news

# Daily metrics by region
metrics_daily = (
    news.groupby([news["date"].dt.normalize(), "region"]).agg(
        n=("title","count"),
        kw_mean=("kw_score","mean"),
        econ_llm=("econ_impact","mean"),
        unemp_llm=("unemp_impact","mean"),
    ).reset_index().rename(columns={"date":"day"})
)

st.markdown("**Daily metrics (tail, all regions)**")
st.dataframe(metrics_daily.tail(30), use_container_width=True)

# Plot daily counts (stacked by region)
figd = px.bar(metrics_daily, x="day", y="n", color="region", title="#articles per day (stacked)")
figd.update_layout(template="plotly_white", height=380)
st.plotly_chart(figd, use_container_width=True)

# ----------------------------------------------------------------------------
# 3) Monthly signals by region
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("3) Build monthly signals (by region)")

m = metrics_daily.set_index("day")
M_list = []
for reg in REGIONS:
    g = m[m["region"] == reg]
    if g.empty:
        continue
    mm = pd.DataFrame({
        f"news_{reg.lower()}__count": g["n"].resample("M").sum(min_count=1),
        f"news_{reg.lower()}__kw": g["kw_mean"].resample("M").mean(),
        f"news_{reg.lower()}__llm_econ": g["econ_llm"].resample("M").mean(),
        f"news_{reg.lower()}__llm_unemp": g["unemp_llm"].resample("M").mean(),
    })
    if smooth_ma and smooth_ma > 1:
        for c in list(mm.columns):
            mm[c + f"_ma{smooth_ma}"] = mm[c].rolling(smooth_ma, min_periods=1).mean()
    if lead_lag != 0:
        mm = mm.shift(int(lead_lag))
    M_list.append(mm)

monthly = pd.concat(M_list, axis=1).sort_index() if M_list else pd.DataFrame()
state.news_monthly = monthly

st.markdown("**Monthly signals (tail)**")
st.dataframe(monthly.tail(24), use_container_width=True)

if append_panel and state.panel_monthly is not None:
    P = state.panel_monthly.copy()
    P = P.join(monthly, how="outer")
    state.panel_monthly = P
    st.success("Appended regional news signals to panel_monthly.")

# ----------------------------------------------------------------------------
# 4) Impact analysis vs target (per region)
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("4) Impact analysis vs target (per region)")

if state.y_monthly is None or state.y_monthly.empty or monthly.empty:
    st.info("Need target and monthly signals.")
else:
    y = state.y_monthly
    # Overlay standardized
    figm = go.Figure(); figm.add_trace(go.Scatter(x=y.index, y=(y-y.mean())/(y.std(ddof=0)+1e-9), name="Target (z)", mode="lines", line=dict(width=3)))
    for reg in REGIONS:
        col = f"news_{reg.lower()}__llm_unemp"
        if col in monthly.columns:
            z = (monthly[col] - monthly[col].mean()) / (monthly[col].std(ddof=0)+1e-9)
            figm.add_trace(go.Scatter(x=monthly.index, y=z, name=f"{reg}: llm_unemp (z)", mode="lines"))
    figm.update_layout(template="plotly_white", height=420, title="Target vs LLM unemployment impact (z‑scores)")
    st.plotly_chart(figm, use_container_width=True)

    # Cross‑correlation table (±6)
    rows = []
    for reg in REGIONS:
        for sig in [f"news_{reg.lower()}__kw", f"news_{reg.lower()}__llm_unemp"]:
            if sig not in monthly.columns:
                continue
            for L in range(-6, 7):
                if L >= 0:
                    ya, xa = y.align(monthly[sig].shift(L), join="inner")
                else:
                    ya, xa = y.shift(-L).align(monthly[sig], join="inner")
                if len(ya) >= 6:
                    corr = float(np.corrcoef(ya.values, xa.values)[0,1])
                    rows.append({"region": reg, "feature": sig, "lag": L, "corr": corr})
    CC = pd.DataFrame(rows)
    if not CC.empty:
        best = CC.loc[CC.groupby(["region","feature"])]["corr"].idxmax()
        st.dataframe(CC.loc[best].sort_values("corr", ascending=False), use_container_width=True)
        figcc = px.bar(CC, x="lag", y="corr", color="region", facet_col="feature", facet_col_wrap=2, title="Cross‑correlation by region")
        figcc.update_layout(template="plotly_white", height=520)
        st.plotly_chart(figcc, use_container_width=True)

# ----------------------------------------------------------------------------
# 5) Export
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("5) Export")

if state.news_daily is not None and not state.news_daily.empty:
    st.download_button("news_daily_scored.csv", state.news_daily.to_csv(index=False).encode("utf-8"), file_name="news_daily_scored.csv")
if state.news_monthly is not None and not state.news_monthly.empty:
    st.download_button("news_monthly_signals.csv", state.news_monthly.to_csv().encode("utf-8"), file_name="news_monthly_signals.csv")

st.caption("Tip: Toggle regions and run multiple collection passes; the page merges results across runs. LLM scoring is optional and rate‑limited via the slider. If APIs are unavailable, RSS + CSV is sufficient for research workflows.")
