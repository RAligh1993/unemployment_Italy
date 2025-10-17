# pages/8_News_Impact.py — Real‑time News → Monthly Signal & Impact (EN)
# ============================================================================
# Role
#   • Pull news from RSS / NewsAPI / (optionally) GDELT* or accept user CSV
#   • Score articles (keyword‑based; optional VADER if available) → daily metrics
#   • Aggregate daily → monthly signals (counts, sentiment, spikes) with EOM alignment
#   • Analyze impact: overlay with target, cross‑correlation (±6), rolling corr,
#     (optional) Granger causality if statsmodels is installed
#   • Persist signals to AppState; optionally append to monthly panel (feature prefix)
#   • Export CSV; diagnostics: top sources, spikes, coverage
#
# *GDELT REST is provided as a convenience; if HTTP blocked, fallback to upload.
# ============================================================================

from __future__ import annotations

import io
import os
import json
import typing as t
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

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
    import requests  # NewsAPI / GDELT
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore

try:
    from statsmodels.tsa.stattools import grangercausalitytests
except Exception:  # pragma: no cover
    grangercausalitytests = None  # type: ignore

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
            self.bt_results: dict[str, pd.Series] = {}
            self.bt_metrics: pd.DataFrame | None = None
            self.news_daily: pd.DataFrame | None = None
            self.news_monthly: pd.DataFrame | None = None
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
# Safe time helpers (fallback to utils.time_ops if present)
# ----------------------------------------------------------------------------
try:
    from utils.time_ops import to_datetime_naive as _to_dt, end_of_month as _eom  # type: ignore
except Exception:
    def _to_dt(v) -> pd.Series:
        return pd.to_datetime(pd.Series(v), errors="coerce").dt.tz_localize(None).dt.normalize()
    def _eom(s: pd.Series) -> pd.Series:
        return (pd.to_datetime(s, errors="coerce").dt.tz_localize(None) + pd.offsets.MonthEnd(0)).dt.normalize()

# ----------------------------------------------------------------------------
# UI — header
# ----------------------------------------------------------------------------

st.title("📰 News → Impact on Unemployment")
st.caption("Fetch/score news, build monthly signals, and analyze correlations with the target. Internet calls are optional — you can always upload your own CSV.")

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("Providers & Ingest")
    mode_rss = st.checkbox("Use RSS", value=False)
    mode_newsapi = st.checkbox("Use NewsAPI", value=False)
    mode_gdelt = st.checkbox("Use GDELT (experimental)", value=False)
    st.markdown("— Or upload a CSV with columns: date,title,source,content (UTF‑8)")

    st.markdown("---")
    st.header("Date window")
    if state.y_monthly is not None and not state.y_monthly.empty:
        ymin, ymax = state.y_monthly.index.min().date(), state.y_monthly.index.max().date()
    else:
        ymin, ymax = datetime(2016, 1, 31).date(), datetime.today().date()
    dr = st.date_input("From / To", value=(ymin, ymax))

    st.markdown("---")
    st.header("Scoring")
    score_method = st.selectbox("Method", ["keyword_index", "vader_if_available"], index=0)
    st.caption("keyword_index uses a curated IT/EN lexicon; VADER works best for English and may be noisy on Italian.")

    st.markdown("---")
    st.header("Monthly aggregation")
    daily_agg = st.selectbox("Daily → Monthly for counts", ["sum", "mean", "last"], index=0)
    sent_agg = st.selectbox("Daily → Monthly for sentiment", ["mean", "median"], index=0)
    smooth_ma = st.slider("Smoothing MA (months)", 1, 12, 3)
    lead_lag = st.slider("Lead/Lag months (signal → shift)", -6, 6, 0)
    add_to_panel = st.checkbox("Append monthly signal(s) to panel_monthly", value=True)
    prefix = st.text_input("Feature prefix", value="news")

# ----------------------------------------------------------------------------
# Inputs area (center) — RSS / NewsAPI / GDELT / Upload
# ----------------------------------------------------------------------------

st.subheader("1) Ingest news")

col_rss, col_newsapi, col_gdelt, col_up = st.columns([1.3, 1.2, 1.2, 1.0])

rss_df = pd.DataFrame()
newsapi_df = pd.DataFrame()
gdelt_df = pd.DataFrame()

with col_rss:
    st.markdown("**RSS**")
    rss_urls = st.text_area("Feed URLs (one per line)", value="https://www.ansa.it/sito/notizie/economia/economia_rss.xml\nhttps://feeds.bbci.co.uk/news/business/rss.xml", height=120)
    fetch_rss = st.button("Fetch RSS", use_container_width=True)

with col_newsapi:
    st.markdown("**NewsAPI**")
    st.caption("Set NEWSAPI_KEY env/secret; limited free tier. Language: it,en.")
    newsapi_q = st.text_input("Query", value="(unemployment OR layoffs OR jobs) AND (Italy OR Italia)")
    newsapi_lang = st.multiselect("Languages", options=["it", "en"], default=["it", "en"])
    newsapi_page = st.slider("Pages (×100)", 1, 5, 1)
    fetch_newsapi = st.button("Fetch NewsAPI", use_container_width=True)

with col_gdelt:
    st.markdown("**GDELT (events/gkg)**")
    st.caption("Keyword search on GDELT 2.1 GKG. If HTTP disabled, skip.")
    gdelt_q = st.text_input("Keywords (comma)", value="disoccupazione,lavoro,occupazione,licenziamenti")
    gdelt_fetch = st.button("Fetch GDELT", use_container_width=True)

with col_up:
    st.markdown("**Upload CSV**")
    up = st.file_uploader("CSV: date,title,source,content", type=["csv"]) 

# ----------------------------------------------------------------------------
# Scoring lexicon (IT + EN) — simple directional weights
# ----------------------------------------------------------------------------

ITALIAN_NEG = {
    "disoccupazione": -1.0, "licenziamenti": -1.0, "cassa integrazione": -0.8,
    "crisi": -0.6, "recessione": -0.8, "sciopero": -0.4, "chiusura": -0.7,
}
ITALIAN_POS = {
    "assunzioni": +0.9, "occupazione": +0.8, "nuovi posti di lavoro": +1.0,
    "ripresa": +0.6, "crescita": +0.5, "incentivi": +0.4,
}
EN_NEG = {
    "unemployment": -0.8, "layoffs": -1.0, "jobless": -0.7, "recession": -0.8, "strike": -0.4,
}
EN_POS = {
    "hiring": +0.9, "jobs added": +0.8, "job growth": +0.8, "recovery": +0.6,
}

# ----------------------------------------------------------------------------
# Parsers — RSS / NewsAPI / GDELT / CSV upload
# ----------------------------------------------------------------------------

def _parse_rss(urls: list[str], d_from: datetime, d_to: datetime) -> pd.DataFrame:
    if not mode_rss or feedparser is None:
        return pd.DataFrame(columns=["date","title","source","content","provider","url"])
    rows = []
    for u in urls:
        try:
            d = feedparser.parse(u)
            for e in d.entries:
                # best effort to parse date
                dt = None
                for key in ("published_parsed", "updated_parsed"):
                    if hasattr(e, key) and getattr(e, key) is not None:
                        try:
                            dt = datetime(*getattr(e, key)[:6])
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
                    "source": d.feed.get("title", "RSS"),
                    "content": getattr(e, "summary", ""),
                    "provider": "rss",
                    "url": getattr(e, "link", ""),
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


def _parse_newsapi(query: str, langs: list[str], pages: int, d_from: datetime, d_to: datetime) -> pd.DataFrame:
    if not mode_newsapi or requests is None:
        return pd.DataFrame(columns=["date","title","source","content","provider","url"])
    key = os.getenv("NEWSAPI_KEY") or st.secrets.get("NEWSAPI_KEY", None) if hasattr(st, "secrets") else None
    if not key:
        st.info("NEWSAPI_KEY not set; skipping NewsAPI.")
        return pd.DataFrame(columns=["date","title","source","content","provider","url"])
    base = "https://newsapi.org/v2/everything"
    rows = []
    for page in range(1, pages + 1):
        params = {
            "q": query,
            "language": ",".join(langs) if langs else None,
            "from": str(d_from), "to": str(d_to),
            "pageSize": 100, "page": page,
            "sortBy": "publishedAt", "apiKey": key,
        }
        try:
            r = requests.get(base, params={k:v for k,v in params.items() if v is not None}, timeout=30)
            if r.status_code != 200:
                break
            data = r.json()
            for a in data.get("articles", []):
                dt = pd.to_datetime(a.get("publishedAt")).tz_localize(None)
                rows.append({
                    "date": dt.normalize(),
                    "title": a.get("title", "") or "",
                    "source": (a.get("source") or {}).get("name", "NewsAPI"),
                    "content": a.get("description", "") or "",
                    "provider": "newsapi",
                    "url": a.get("url", ""),
                })
        except Exception:
            break
    return pd.DataFrame(rows)


def _parse_gdelt(keywords: list[str], d_from: datetime, d_to: datetime) -> pd.DataFrame:
    if not mode_gdelt or requests is None:
        return pd.DataFrame(columns=["date","title","source","content","provider","url"])
    # Use GDELT GKG 2.1 CSV through a simple search API (experimental); we query per day
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    rows = []
    days = pd.date_range(d_from, d_to, freq="D")
    for d in days:
        try:
            q = " OR ".join([f'"{k.strip()}"' for k in keywords if k.strip()])
            params = {"query": q, "startdatetime": d.strftime("%Y%m%d000000"), "enddatetime": d.strftime("%Y%m%d235959"), "format": "json"}
            r = requests.get(base, params=params, timeout=30)
            if r.status_code != 200:
                continue
            data = r.json()
            for a in data.get("articles", []):
                dt = pd.to_datetime(a.get("seendate")).tz_localize(None)
                rows.append({
                    "date": dt.normalize(),
                    "title": a.get("title", ""),
                    "source": a.get("sourceCommonName", "GDELT"),
                    "content": a.get("socialimage", ""),
                    "provider": "gdelt",
                    "url": a.get("url", ""),
                })
        except Exception:
            continue
    return pd.DataFrame(rows)

# Upload CSV
up_df = pd.read_csv(up) if up is not None else pd.DataFrame(columns=["date","title","source","content"]) 
if not up_df.empty:
    up_df["date"] = _to_dt(up_df["date"])  # normalize
    up_df["provider"] = "upload"
    up_df["url"] = ""

# Fetch
if st.button("Run ingestion", type="primary"):
    d_from, d_to = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    frames = []
    if mode_rss and fetch_rss:
        frames.append(_parse_rss([u.strip() for u in rss_urls.splitlines() if u.strip()], d_from, d_to))
    if mode_newsapi and fetch_newsapi:
        frames.append(_parse_newsapi(newsapi_q, newsapi_lang, newsapi_page, d_from, d_to))
    if mode_gdelt and gdelt_fetch:
        frames.append(_parse_gdelt([x.strip() for x in gdelt_q.split(",")], d_from, d_to))
    if not up_df.empty:
        frames.append(up_df)
    all_news = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if frames else pd.DataFrame(columns=["date","title","source","content","provider","url"]) 
    if all_news.empty:
        st.warning("No news collected. Check providers, keys, or upload a CSV.")
    else:
        # De‑dup (title+source per day)
        all_news = all_news.dropna(subset=["date"]).copy()
        all_news["title"] = all_news["title"].fillna("").astype(str).str.strip()
        all_news["source"] = all_news["source"].fillna("").astype(str).str.strip()
        all_news["dedup_key"] = all_news["title"].str.lower().str.replace(r"\s+", " ", regex=True) + "|" + all_news["source"].str.lower()
        all_news = all_news.sort_values("date").drop_duplicates(subset=["date","dedup_key"], keep="first")
        state.news_daily = all_news
        st.success(f"Collected {len(all_news):,} unique articles.")

# Preview
if state.news_daily is not None and not state.news_daily.empty:
    st.markdown("### Preview (latest 25)")
    st.dataframe(state.news_daily.tail(25), use_container_width=True)

# ----------------------------------------------------------------------------
# 2) Scoring → daily metrics
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("2) Score articles → daily metrics")

if state.news_daily is None or state.news_daily.empty:
    st.info("Ingest news first (or upload CSV).")
    st.stop()

# Build scorer
sia = None
if score_method == "vader_if_available" and SentimentIntensityAnalyzer is not None:
    try:
        # NLTK VADER requires lexicon; let it try — if missing, it may raise
        sia = SentimentIntensityAnalyzer()
    except Exception:
        sia = None


def _score_row(title: str, content: str) -> float:
    text = (title or "") + "\n" + (content or "")
    text_l = text.lower()
    # Keyword index
    score = 0.0
    for w, v in ITALIAN_NEG.items():
        if w in text_l: score += v
    for w, v in ITALIAN_POS.items():
        if w in text_l: score += v
    for w, v in EN_NEG.items():
        if w in text_l: score += v
    for w, v in EN_POS.items():
        if w in text_l: score += v
    # VADER (if exists) — blend
    if sia is not None:
        try:
            vs = sia.polarity_scores(text)
            score = 0.5 * score + 0.5 * float(vs.get("compound", 0.0))
        except Exception:
            pass
    return float(score)

# Score
df = state.news_daily.copy()
df["score"] = [
    _score_row(str(t), str(c)) for t, c in zip(df.get("title",""), df.get("content",""))
]

# Daily metrics
daily = (
    df.groupby("date").agg(
        n=("title", "count"),
        score_mean=("score", "mean"),
        score_median=("score", "median"),
        pos_ratio=("score", lambda s: float(np.mean(np.array(s) > 0))),
    ).reset_index()
)

state.news_daily = df  # persist with scores

st.markdown("**Daily metrics (tail)**")
st.dataframe(daily.tail(20), use_container_width=True)

# Daily plot
figd = go.Figure()
figd.add_trace(go.Bar(x=daily["date"], y=daily["n"], name="# articles", opacity=0.6))
figd.add_trace(go.Scatter(x=daily["date"], y=daily["score_mean"].rolling(7, min_periods=3).mean(), name="score_mean (7d MA)", mode="lines"))
figd.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(figd, use_container_width=True)

# ----------------------------------------------------------------------------
# 3) Monthly signals (EOM alignment) + smoothing/shift
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("3) Build monthly signals")

# EOM index
daily["date"] = _to_dt(daily["date"])  # ensure tz‑naive normalized
m = daily.set_index("date")

# choose aggregators
if daily_agg == "sum":
    m_n = m["n"].resample("M").sum(min_count=1)
elif daily_agg == "last":
    m_n = m["n"].resample("M").last()
else:
    m_n = m["n"].resample("M").mean()

if sent_agg == "median":
    m_s = m["score_mean"].resample("M").median()
else:
    m_s = m["score_mean"].resample("M").mean()

monthly = pd.DataFrame({
    f"{prefix}__news_count": m_n,
    f"{prefix}__news_sent": m_s,
})
monthly.index = _eom(monthly.index)

# smoothing MA
if smooth_ma and smooth_ma > 1:
    monthly[f"{prefix}__news_count_ma{smooth_ma}"] = monthly[f"{prefix}__news_count"].rolling(smooth_ma, min_periods=1).mean()
    monthly[f"{prefix}__news_sent_ma{smooth_ma}"] = monthly[f"{prefix}__news_sent"].rolling(smooth_ma, min_periods=1).mean()

# lead/lag shift
if lead_lag != 0:
    monthly = monthly.shift(int(lead_lag))

state.news_monthly = monthly

st.markdown("**Monthly signals (tail)**")
st.dataframe(monthly.tail(24), use_container_width=True)

# Optionally append to panel
if add_to_panel and state.panel_monthly is not None:
    # align keys and merge on EOM
    P = state.panel_monthly.copy()
    P.index = _eom(P.index)
    P = P.join(monthly, how="outer")
    state.panel_monthly = P
    st.success("Appended monthly news signals to panel_monthly.")

# ----------------------------------------------------------------------------
# 4) Impact analysis vs target
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("4) Impact analysis vs target")

if state.y_monthly is None or state.y_monthly.empty:
    st.info("No target in AppState; load on page 2.")
else:
    y, X = state.y_monthly.align(monthly.iloc[:, :2], join="inner")  # align with core two cols
    if len(y) >= 6:
        # Overlay
        figm = go.Figure()
        figm.add_trace(go.Scatter(x=y.index, y=(y - y.mean())/(y.std(ddof=0)+1e-9), name="Target (z)", mode="lines", line=dict(width=3)))
        for c in X.columns:
            z = (X[c] - X[c].mean()) / (X[c].std(ddof=0)+1e-9)
            figm.add_trace(go.Scatter(x=X.index, y=z, name=c, mode="lines"))
        figm.update_layout(template="plotly_white", height=420, title="Target vs news signals (standardized)")
        st.plotly_chart(figm, use_container_width=True)

        # Cross‑correlation (±6)
        st.markdown("**Cross‑correlation (lags in months; positive lag ⇒ news leads)**")
        lags = range(-6, 7)
        rows = []
        for c in X.columns:
            for L in lags:
                if L >= 0:
                    yc, xc = y.align(X[c].shift(L), join="inner")
                else:
                    yc, xc = y.shift(-L).align(X[c], join="inner")
                if len(yc) >= 6:
                    rows.append({"feature": c, "lag": L, "corr": float(np.corrcoef(yc.values, xc.values)[0,1])})
        C = pd.DataFrame(rows)
        if not C.empty:
            best = C.loc[C.groupby("feature")["corr"].idxmax()].sort_values("corr", ascending=False)
            st.dataframe(best, use_container_width=True)
            figcc = px.bar(C, x="lag", y="corr", color="feature", barmode="group")
            figcc.update_layout(template="plotly_white", height=420)
            st.plotly_chart(figcc, use_container_width=True)

        # Rolling correlation (24m)
        st.markdown("**Rolling correlation (24m window, with news_sent)**")
        if f"{prefix}__news_sent" in X.columns:
            r = (
                pd.concat([y.rename("y"), X[f"{prefix}__news_sent"].rename("x")], axis=1)
                .dropna()
                .rolling(24, min_periods=12)
                .apply(lambda a: np.corrcoef(a[:,0], a[:,1])[0,1], raw=False)
            )
            figroll = go.Figure(); figroll.add_trace(go.Scatter(x=r.index, y=r.values, name="rolling corr", mode="lines"))
            figroll.update_layout(template="plotly_white", height=300)
            st.plotly_chart(figroll, use_container_width=True)

        # Granger causality (optional)
        if grangercausalitytests is not None:
            st.markdown("**Granger causality (y ~ news_sent)**")
            YY = pd.concat([y.rename("y"), X[f"{prefix}__news_sent"].rename("x")], axis=1).dropna()
            if len(YY) >= 36:
                maxlag = 6
                try:
                    res = grangercausalitytests(YY[["y","x"]], maxlag=maxlag, verbose=False)
                    rows = []
                    for L in range(1, maxlag+1):
                        p = res[L][0]["ssr_ftest"][1]
                        rows.append({"lag": L, "p_value": float(p)})
                    GC = pd.DataFrame(rows)
                    st.dataframe(GC, use_container_width=True)
                except Exception:
                    st.info("Granger test failed on this slice (singular matrix or not enough variation).")
            else:
                st.info("Need ≥36 months for a reliable Granger test.")

# ----------------------------------------------------------------------------
# 5) Diagnostics: sources and spikes
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("5) Diagnostics")

if state.news_daily is not None and not state.news_daily.empty:
    # Top sources
    topS = (
        state.news_daily.groupby("source").size().sort_values(ascending=False).head(15)
    )
    figsrc = px.bar(topS[::-1], x=topS.values[::-1], y=topS.index[::-1], orientation="h", title="Top sources by count")
    figsrc.update_layout(template="plotly_white", height=420)
    st.plotly_chart(figsrc, use_container_width=True)

    # Spike months (z of counts)
    m_counts = monthly[f"{prefix}__news_count"].dropna()
    if not m_counts.empty:
        z = (m_counts - m_counts.mean()) / (m_counts.std(ddof=0) + 1e-9)
        spikes = z[z > 2.0].sort_values(ascending=False).to_frame("z_score")
        if not spikes.empty:
            st.markdown("**Spike months (z>2)**")
            st.dataframe(spikes, use_container_width=True)

# ----------------------------------------------------------------------------
# 6) Export
# ----------------------------------------------------------------------------

st.markdown("---")
st.subheader("6) Export")

if state.news_daily is not None and not state.news_daily.empty:
    st.download_button("news_daily_scored.csv", data=state.news_daily.to_csv(index=False).encode("utf-8"), file_name="news_daily_scored.csv")
if state.news_monthly is not None and not state.news_monthly.empty:
    st.download_button("news_monthly_signals.csv", data=state.news_monthly.to_csv().encode("utf-8"), file_name="news_monthly_signals.csv")

st.caption("If internet access is restricted, use the Upload CSV path. All merges are EOM‑aligned to avoid date‑type mismatches.")
