# app_nowcast_it.py
# 🇮🇹 Fully-automatic fetch + baseline nowcast for Italian unemployment
import io, itertools, json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="IT Unemployment Nowcast", page_icon="🇮🇹", layout="wide")

# ========= Eurostat JSON-stat 2.0 (generic) =========
BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

def _get_jsonstat(dataset: str, params: Dict[str,str]) -> dict:
    url = f"{BASE}{dataset}?lang=EN&" + "&".join([f"{k}={v}" for k,v in params.items()])
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return {}
    return r.json()

def _get_dims(dataset: str) -> Tuple[List[str], Dict[str, List[str]]]:
    js = _get_jsonstat(dataset, {"lastTimePeriod":"1"})
    if not js or js.get("class") != "dataset":
        raise RuntimeError("Eurostat metadata fetch failed")
    ids = js["id"]
    dim = js["dimension"]
    avail = {}
    for d in ids:
        cats = dim[d]["category"]["index"]  # dict code->ordinal
        avail[d] = [c for c,_ in sorted(cats.items(), key=lambda x: x[1])]
    return ids, avail

def _parse_jsonstat_timeseries(js: dict, fixed: Dict[str,str]) -> pd.DataFrame:
    ids = js["id"]; sizes = js["size"]; dim = js["dimension"]
    idx = {d:i for i,d in enumerate(ids)}
    if "time" not in idx: raise ValueError("No time dim")
    t_i = idx["time"]
    t_sorted = sorted(dim["time"]["category"]["index"].items(), key=lambda x: x[1])
    labels = [dim["time"]["category"]["label"].get(code, code) for code,_ in t_sorted]
    pos = [0]*len(ids)
    for d in ids:
        if d=="time": continue
        cat_index = dim[d]["category"]["index"]
        if d in fixed and fixed[d] in cat_index:
            pos[idx[d]] = cat_index[fixed[d]]
        elif len(cat_index)==1:
            pos[idx[d]] = next(iter(cat_index.values()))
        else:
            pos[idx[d]] = 0
    strides = [1]*len(ids)
    for i in range(len(ids)-2, -1, -1):
        strides[i] = strides[i+1]*sizes[i+1]
    values = js["value"]
    out = []
    for code, tpos in t_sorted:
        pos[t_i] = tpos
        lin = sum(pos[k]*strides[k] for k in range(len(ids)))
        if isinstance(values, list):
            v = values[lin] if lin < len(values) else None
        else:
            v = values.get(str(lin), None)
        out.append((code, v))
    df = pd.DataFrame(out, columns=["period","value"])
    df["date"] = pd.to_datetime(df["period"], errors="coerce")
    if df["date"].isna().any():
        df["date"] = pd.to_datetime(df["period"].astype(str).str.replace("M","-"), format="%Y-%m", errors="coerce")
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().sort_values("date")[["date","value"]].reset_index(drop=True)

def eurostat_fetch(dataset: str, prefs: Dict[str, List[str]], start_year: int) -> Tuple[Optional[pd.DataFrame], str, Dict[str,str]]:
    ids, avail = _get_dims(dataset)
    # build candidates by intersecting prefs with availability
    cand = {}
    for d in ids:
        if d == "time": continue
        want = prefs.get(d, [])
        have = avail.get(d, [])
        if want:
            inter = [x for x in want if x in have]
            cand[d] = inter if inter else (have[:1] if have else [])
        else:
            cand[d] = have[:1] if have else []
    order = [d for d in ["freq","geo","sex","age","s_adj","unit","indic","coicop","nace_r2","indic_bt"] if d in cand]
    tries = [dict(zip(order, combo)) for combo in itertools.product(*[cand[d] for d in order])]
    last = None
    for i, filt in enumerate(tries, 1):
        js = _get_jsonstat(dataset, filt)
        if not js or js.get("class") != "dataset":
            last = f"bad response for {filt}"
            continue
        df = _parse_jsonstat_timeseries(js, filt)
        df = df[df["date"].dt.year >= start_year]
        if not df.empty:
            return df, f"✅ {dataset} attempt#{i} {len(df)} rows", filt
        last = f"empty for {filt}"
    return None, f"❌ {dataset} failed. Last: {last}", {}

EUROSTAT_PREFS = {
    "une_rt_m": {"freq":["M"], "geo":["IT"], "sex":["T"], "age":["TOTAL","Y15-74"], "s_adj":["SA","SCA","NSA"], "unit":["PC_ACT","PC_POP"]},
    "ei_bsco_m": {"freq":["M"], "geo":["IT"], "indic":["BS-CSMCI","BS-CSMCI-BAL"], "s_adj":["NSA","SA"], "unit":["BAL"]},
    "prc_hicp_midx": {"freq":["M"], "geo":["IT"], "coicop":["CP00"], "unit":["I21","I15"]},
    "sts_inpr_m": {"freq":["M"], "geo":["IT"], "s_adj":["SCA","SA","NSA"], "nace_r2":["B-E","B-D"], "indic_bt":["PRD"], "unit":["I21","I15"]},
}

# ========= Finance (Stooq/FRED/CBOE) =========
def stooq_csv(symbol: str) -> Optional[pd.DataFrame]:
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        r = requests.get(url, timeout=12)
        if r.status_code != 200 or not r.text.strip(): return None
        df = pd.read_csv(io.StringIO(r.text))
        if {"Date","Close"}.issubset(df.columns):
            out = df.rename(columns={"Date":"date","Close":"close"})
            out["date"] = pd.to_datetime(out["date"])
            return out[["date","close"]].sort_values("date").reset_index(drop=True)
    except Exception:
        return None
    return None

def fred_vix_csv() -> Optional[pd.DataFrame]:
    try:
        url = "https://fred.stlouisfed.org/series/VIXCLS/downloaddata/VIXCLS.csv"
        r = requests.get(url, timeout=12, allow_redirects=True)
        df = pd.read_csv(io.StringIO(r.text))
        if {"DATE","VIXCLS"}.issubset(df.columns):
            df = df.rename(columns={"DATE":"date","VIXCLS":"close"})
            df["date"] = pd.to_datetime(df["date"]); df["close"] = pd.to_numeric(df["close"], errors="coerce")
            return df.dropna().sort_values("date").reset_index(drop=True)
    except Exception:
        return None
    return None

def cboe_vix_csv() -> Optional[pd.DataFrame]:
    try:
        url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
        r = requests.get(url, timeout=12)
        df = pd.read_csv(io.StringIO(r.text))
        cols = {c.lower(): c for c in df.columns}
        if not cols.get("date") or not cols.get("close"): return None
        df = df.rename(columns={cols["date"]:"date", cols["close"]:"close"})[["date","close"]]
        df["date"] = pd.to_datetime(df["date"]); df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df.dropna().sort_values("date").reset_index(drop=True)
    except Exception:
        return None
    return None

def finance_fetch(start_year: int):
    mib = stooq_csv("^fmib")
    vix = fred_vix_csv() or cboe_vix_csv() or stooq_csv("vi.f")
    out = {}
    if mib is not None and not mib.empty:
        out["mib"] = mib[mib["date"].dt.year >= start_year].copy()
    if vix is not None and not vix.empty:
        out["vix"] = vix[vix["date"].dt.year >= start_year].copy()
    return out

# ========= Utility =========
def line_fig(df: pd.DataFrame, y_col: str, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df[y_col], mode="lines", name=title))
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified", height=420)
    return fig

def month_end(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(0)
    return df

def to_monthly_finance(df: pd.DataFrame) -> pd.DataFrame:
    # EOM close + monthly return
    m = df.set_index("date").resample("M").last().rename_axis("date").reset_index()
    m["ret"] = m["close"].pct_change()
    return m

# ========= Sidebar =========
with st.sidebar:
    st.header("⚙️ Settings")
    start_year = st.slider("Start year", 2000, datetime.now().year, 2010)
    fetch_btn = st.button("🚀 Fetch data", type="primary", use_container_width=True)
    st.caption("Sources: Eurostat API • Stooq • FRED/CBOE")

st.title("🇮🇹 Italian Unemployment — Auto Fetch & Nowcast")

if not fetch_btn:
    st.info("از سایدبار **Fetch data** را بزن. هیچ دانلود دستی لازم نیست.")
    st.stop()

# ========= Fetch block =========
status = []
data = {}

# Eurostat: Unemployment (mandatory)
df_u, msg_u, filt_u = eurostat_fetch("une_rt_m", EUROSTAT_PREFS["une_rt_m"], start_year)
status.append(("Unemployment (Eurostat)", msg_u))
if df_u is not None:
    data["unemp"] = df_u

# Optional Eurostat: CCI/HICP/IIP
for code, label in [("ei_bsco_m","CCI"), ("prc_hicp_midx","HICP (All)"), ("sts_inpr_m","IIP")]:
    df, msg, filt = eurostat_fetch(code, EUROSTAT_PREFS[code], start_year)
    status.append((label, msg))
    if df is not None:
        data[code] = df

# Finance: MIB + VIX
fin = finance_fetch(start_year)
if "mib" in fin:
    status.append(("FTSE MIB (Stooq)", f"✅ {len(fin['mib'])} rows"))
    data["mib_m"] = to_monthly_finance(month_end(fin["mib"]))
else:
    status.append(("FTSE MIB (Stooq)", "❌ fail"))
if "vix" in fin:
    status.append(("VIX (FRED/CBOE/Stooq)", f"✅ {len(fin['vix'])} rows"))
    data["vix_m"] = to_monthly_finance(month_end(fin["vix"]))
else:
    status.append(("VIX (FRED/CBOE/Stooq)", "❌ fail"))

# Show status
cols = st.columns(2)
for i,(name,msg) in enumerate(status):
    (cols[i%2].success if "✅" in msg else cols[i%2].warning)(f"**{name}:** {msg}")

if "unemp" not in data:
    st.error("نرخ بیکاری نگرفتیم؛ بدون آن الآن کاری نمی‌کنیم.")
    st.stop()

# ========= Charts =========
st.subheader("📈 Series")
st.plotly_chart(line_fig(data["unemp"], "value", "Unemployment rate (%)"), use_container_width=True)
with st.expander("Preview tables"):
    for k,v in data.items():
        st.write(k, v.tail(6))

# ========= Baseline Nowcast (Ridge, expanding window) =========
st.subheader("🤖 Baseline Nowcast (expanding Ridge)")
# Build monthly features
feat = []
# Eurostat predictors (same-month rhs):
if "ei_bsco_m" in data: feat.append(data["ei_bsco_m"].rename(columns={"value":"cci"}))
if "prc_hicp_midx" in data: feat.append(data["prc_hicp_midx"].rename(columns={"value":"hicp"}))
if "sts_inpr_m" in data: feat.append(data["sts_inpr_m"].rename(columns={"value":"iip"}))
# Finance monthly (returns / levels)
if "mib_m" in data: feat.append(data["mib_m"][["date","ret"]].rename(columns={"ret":"mib_ret"}))
if "vix_m" in data: 
    v = data["vix_m"].rename(columns={"close":"vix"}); v = v[["date","vix"]]
    feat.append(v)

X = None
if feat:
    X = feat[0]
    for z in feat[1:]:
        X = pd.merge(X, z, on="date", how="outer")
    # Add first differences & lags (simple)
    for c in [c for c in X.columns if c not in ["date"]]:
        X[f"d_{c}"] = X[c].diff()
        X[f"lag1_{c}"] = X[c].shift(1)
else:
    X = pd.DataFrame({"date": data["unemp"]["date"]})

# target
y = data["unemp"].rename(columns={"value":"u"}).copy()

# align
dfm = pd.merge(y, X, on="date", how="left").sort_values("date").reset_index(drop=True)
# we model u(t) with rhs available by end-of-month t (approximation)
dfm = dfm.dropna(subset=["u"]).copy()

# minimal feature selection
feature_cols = [c for c in dfm.columns if c not in ["date","u"]]
df_train = dfm.dropna(subset=feature_cols).copy()
if len(df_train) > 60 and len(feature_cols) > 0:
    # expanding-window CV for alpha
    alphas = [0.1, 0.3, 1.0, 3.0, 10.0]
    best_alpha, best_mae = None, 1e9
    for a in alphas:
        maes = []
        for split in range(48, len(df_train)-1):  # start after 4y data
            tr = df_train.iloc[:split]
            va = df_train.iloc[split:split+1]
            model = Pipeline([("sc", StandardScaler(with_mean=True, with_std=True)),
                              ("rg", Ridge(alpha=a))])
            model.fit(tr[feature_cols], tr["u"])
            pred = model.predict(va[feature_cols])[0]
            maes.append(abs(pred - va["u"].values[0]))
        m = float(np.mean(maes)) if maes else 1e9
        if m < best_mae: best_mae, best_alpha = m, a

    # final fit on all data until last-1 (to nowcast last)
    model = Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=best_alpha))])
    model.fit(df_train[feature_cols], df_train["u"])

    # last available month in dfm (target known) + next month (nowcast) using latest features
    last_month = dfm["date"].max()
    # construct feature row for current month end
    cur_date = (pd.to_datetime(datetime.now().date()) + pd.offsets.MonthEnd(0)).normalize()
    # forward-fill features to current EOM
    fx = dfm.set_index("date")[feature_cols].sort_index().copy()
    fx = fx.reindex(pd.date_range(fx.index.min(), cur_date, freq="M")).ffill().iloc[[-1]]
    nowcast = float(model.predict(fx[feature_cols])[0])
    resid_sd = float(np.std(model.named_steps["rg"].predict(df_train[feature_cols]) - df_train["u"]))
    lo, hi = nowcast - 1.0*resid_sd, nowcast + 1.0*resid_sd

    st.success(f"Nowcast for {cur_date.date()}: **{nowcast:.2f}%**  (±{resid_sd:.2f}, 68% CI ≈ [{lo:.2f}, {hi:.2f}])")
    st.caption(f"Best alpha={best_alpha}, expanding-window MAE≈{best_mae:.2f}. Features: {len(feature_cols)}")
else:
    st.warning("برای nowcast دادهٔ پیش‌بین کافی/پاک در دسترس نیست؛ یا فیچرها خیلی کم‌اند.")
