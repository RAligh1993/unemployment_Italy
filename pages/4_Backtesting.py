# pages/4_Backtesting.py — Walk‑forward backtesting incl. U‑MIDAS, ARIMAX, SARIMAX (EN)
# ============================================================================
# Role
#   • One‑step walk‑forward backtesting on monthly data (nowcast h=0 / forecast h=1)
#   • Baselines (Naive, Seasonal Naive, Moving Average, optional ETS)
#   • Ridge‑ARX (lagged target + exogenous monthly features)
#   • NEW: U‑MIDAS (unrestricted MIDAS on a chosen daily regressor)
#   • NEW: ARIMAX & SARIMAX (from statsmodels) with selected exogenous set
#   • Ensembles (uniform / inverse‑MAE)
#   • Rolling vs Expanding windows; per‑step standardization
#   • Metrics (MAE/RMSE/SMAPE/MASE) + plots + export; persisted to AppState
#
# Notes
#   • Page config handled in app.py
#   • Uses utils/* if present; includes safe fallbacks otherwise
#   • Horizon supported: 0 (nowcast) and 1 (one month ahead). h>1 is disabled
#     for ARIMAX/SARIMAX and U‑MIDAS to avoid exog leakage unless you provide
#     a future exog provider.
# ============================================================================

from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

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
            self.raw_daily: list[pd.DataFrame] = []
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

# ----------------------------------------------------------------------------n# Optional utils imports
# ----------------------------------------------------------------------------
try:
    from utils.models.metrics import mae, rmse, smape, mase  # type: ignore
except Exception:
    def mae(y, p): y, p = y.align(p, join="inner"); return float((y - p).abs().mean())
    def rmse(y, p): y, p = y.align(p, join="inner"); return float(np.sqrt(((y - p) ** 2).mean()))
    def smape(y, p): y, p = y.align(p, join="inner"); return float((200 * (y - p).abs() / (y.abs() + p.abs() + 1e-12)).mean())
    def mase(y, p): y, p = y.align(p, join="inner"); den=float(y.diff().abs().mean()); return float(mae(y,p)/(den+1e-12)) if np.isfinite(den) else float("nan")

try:
    from utils.time_ops import end_of_month as _eom  # type: ignore
except Exception:
    def _eom(s: pd.Series) -> pd.Series:
        return (pd.to_datetime(s, errors="coerce").dt.tz_localize(None) + pd.offsets.MonthEnd(0)).dt.normalize()

# ETS (optional)
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ETS  # type: ignore
except Exception:
    _ETS = None

# Sklearn Ridge
try:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
except Exception:
    RidgeCV = None  # type: ignore
    class StandardScaler:  # fallback no‑op
        def fit(self, X): return self
        def transform(self, X): return X
    class SimpleImputer:
        def __init__(self, strategy="median"): pass
        def fit_transform(self, X): return np.asarray(X)
        def transform(self, X): return np.asarray(X)

# ARIMAX / SARIMAX
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
except Exception:
    SARIMAX = None

# ----------------------------------------------------------------------------
# Config dataclass
# ----------------------------------------------------------------------------
@dataclass
class BTConfig:
    start_year: int
    end_year: int
    horizon: int  # 0 or 1
    window_mode: str  # "expanding" or "rolling"
    min_train: int
    rolling_window: int
    y_lags: int
    x_lags: int
    standardize: bool
    seasonal_period: int
    # model toggles
    use_naive: bool
    use_snaive: bool
    use_ma: bool
    ma_k: int
    use_ets: bool
    use_ridge: bool
    use_umidas: bool
    use_arimax: bool
    use_sarimax: bool
    # ridge
    alpha_grid: list[float]
    # U‑MIDAS
    umidas_daily_col: str | None
    umidas_n_lags: int
    umidas_n_bins: int
    umidas_agg: str  # sum/mean
    vintage_cutoff_day: int
    # ARIMAX/SARIMAX
    arimax_order: tuple[int, int, int]
    sarimax_order: tuple[int, int, int]
    sarimax_seasonal: tuple[int, int, int, int]
    arimax_exog_cols: list[str]

# ----------------------------------------------------------------------------
# UI — controls
# ----------------------------------------------------------------------------

st.title("🧮 Backtesting")
st.caption("Walk‑forward evaluation on the monthly target with baselines, Ridge‑ARX, **U‑MIDAS**, **ARIMAX/SARIMAX**.")

if state.y_monthly is None or state.y_monthly.empty:
    st.warning("Monthly target not found. Build/Load it in previous pages.")
    st.stop()

if state.panel_monthly is None or state.panel_monthly.empty:
    st.warning("Monthly panel is empty. Build it in **Data & Aggregation** and optionally engineer in page 3.")
    st.stop()

with st.sidebar:
    st.header("Evaluation window")
    years = sorted({d.year for d in state.y_monthly.index})
    s_year = st.select_slider("Start year", options=years, value=years[max(0, len(years) - 5)])
    e_year = st.select_slider("End year", options=years, value=years[-1])

    st.header("Forecast setup")
    horizon = st.selectbox("Horizon (months ahead)", options=[0, 1], index=1, help="0=nowcast, 1=one‑month ahead")
    window_mode = st.radio("Training window", ["expanding", "rolling"], horizontal=True, index=0)
    min_train = st.slider("Min train size (months)", 24, 120, 48, step=6)
    rolling_window = st.slider("Rolling window (months)", 12, 120, 60, step=6)

    st.header("Baselines & Ridge")
    use_naive = st.checkbox("NAIVE (t‑1)", value=True)
    use_snaive = st.checkbox("SNAIVE (seasonal)", value=True)
    use_ma = st.checkbox("Moving Average", value=False)
    ma_k = st.slider("MA window (k)", 2, 24, 3)
    use_ets = st.checkbox("ETS (if available)", value=False)
    seas = st.slider("Seasonal period (months)", 3, 24, 12)
    use_ridge = st.checkbox("Ridge‑ARX", value=True)
    alpha_grid_txt = st.text_input("Ridge alphas (comma)", value="0.1,0.3,1,3,10,30,100")

    st.header("U‑MIDAS (daily regressor)")
    use_umidas = st.checkbox("Enable U‑MIDAS", value=False)
    vintage_cut = st.slider("Vintage cutoff day", 1, 28, 15, help="Include only daily data available up to this day each month.")
    umidas_nlags = st.slider("Daily lags (D)", 10, 90, 44, help="Number of last daily observations before the anchor date.")
    umidas_bins = st.slider("Bins (B)", 2, 15, 6, help="Group D daily lags into B bins → unrestricted coefficients.")
    umidas_agg = st.radio("Bin aggregator", ["sum", "mean"], horizontal=True)

    # Gather daily candidates from uploaded daily frames
    daily_candidates: list[str] = []
    if state.raw_daily:
        for df in state.raw_daily:
            for c in df.columns:
                if c != "date" and pd.api.types.is_numeric_dtype(df[c]):
                    daily_candidates.append(f"{c}")
    daily_candidates = sorted(list(set(daily_candidates)))
    umidas_col = st.selectbox("Daily column for U‑MIDAS", options=["(none)"] + daily_candidates, index=0)

    st.header("ARIMAX / SARIMAX")
    use_arimax = st.checkbox("Enable ARIMAX", value=False)
    use_sarimax = st.checkbox("Enable SARIMAX", value=False)
    st.caption("Tip: choose a small, stable exog set to avoid singular fits.")

    # Exog selection from monthly panel
    num_cols = state.panel_monthly.select_dtypes(include=[np.number]).columns.tolist()
    default_exog = [c for c in num_cols if c.endswith("_lag1")][:5]
    exog_cols = st.multiselect("Exogenous columns (monthly)", options=num_cols, default=default_exog)

    c1, c2 = st.columns(2)
    with c1:
        p = st.number_input("p (AR)", 0, 12, 1)
        d = st.number_input("d (diff)", 0, 2, 0)
        q = st.number_input("q (MA)", 0, 12, 1)
    with c2:
        P = st.number_input("P (seasonal AR)", 0, 2, 0)
        D = st.number_input("D (seasonal diff)", 0, 2, 0)
        Q = st.number_input("Q (seasonal MA)", 0, 2, 0)
    m_season = st.number_input("Season length m", 0, 24, 12)

    run_btn = st.button("Run backtest", type="primary")

# ----------------------------------------------------------------------------
# Helper: training window slicing
# ----------------------------------------------------------------------------

def _train_window(idx: pd.DatetimeIndex, t: pd.Timestamp, mode: str, rolling_window: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = t - pd.offsets.MonthEnd(1)  # strictly prior month end
    if mode == "rolling":
        start = end - pd.DateOffset(months=rolling_window - 1)
        start = max(idx.min(), start)
    else:
        start = idx.min()
    return start, end

# ----------------------------------------------------------------------------
# Baseline predictors
# ----------------------------------------------------------------------------

def _pred_naive(y_hist: pd.Series) -> float:
    return float(y_hist.iloc[-1])

def _pred_snaive(y_hist: pd.Series, season: int) -> float:
    return float(y_hist.iloc[-season]) if len(y_hist) > season else float("nan")

def _pred_ma(y_hist: pd.Series, k: int) -> float:
    return float(y_hist.iloc[-k:].mean()) if len(y_hist) >= k else float("nan")

def _pred_ets(y_hist: pd.Series, season: int) -> float:
    if _ETS is None or len(y_hist) < max(12, season + 2):
        return float("nan")
    try:
        mdl = _ETS(y_hist, trend="add", seasonal="add", seasonal_periods=season, initialization_method="estimated")
        fit = mdl.fit(optimizable=True, disp=False)
        return float(fit.forecast(1).iloc[0])
    except Exception:
        return float("nan")

# ----------------------------------------------------------------------------
# Ridge‑ARX
# ----------------------------------------------------------------------------

def _prepare_design(y: pd.Series, X: pd.DataFrame, y_lags: int, x_lags: int, horizon: int) -> pd.DataFrame:
    df = pd.DataFrame(index=y.index)
    for L in range(1, y_lags + 1):
        df[f"y_lag{L}"] = y.shift(L)
    for c in X.columns:
        for L in range(1, x_lags + 1):
            df[f"{c}__lag{L}"] = X[c].shift(L)
    target = y.shift(-horizon).rename("target")
    return pd.concat([target, df], axis=1)


def _fit_predict_ridge(y_tr: pd.Series, X_tr: pd.DataFrame, x_pred: pd.Series, alphas: list[float]) -> float:
    if RidgeCV is None or X_tr.shape[1] == 0:
        return float("nan")
    model = RidgeCV(alphas=np.array(alphas, dtype=float), store_cv_values=False)
    model.fit(X_tr.values, y_tr.values)
    return float(model.predict(x_pred.values.reshape(1, -1))[0])

# ----------------------------------------------------------------------------
# U‑MIDAS (unrestricted, daily → B bins)
# ----------------------------------------------------------------------------

def _collect_daily_series(colname: str) -> pd.DataFrame:
    """Find the first daily frame that contains `colname` and return date + col."""
    if not state.raw_daily:
        return pd.DataFrame()
    for df in state.raw_daily:
        if colname in df.columns:
            z = df[["date", colname]].dropna()
            z = z.copy(); z["date"] = pd.to_datetime(z["date"]).dt.tz_localize(None).dt.normalize()
            z = z.sort_values("date").reset_index(drop=True)
            return z
    return pd.DataFrame()


def _daily_lags_bins_for_anchor(daily: pd.DataFrame, anchor_date: pd.Timestamp, D: int, B: int, agg: str, cutoff_day: int) -> np.ndarray:
    """Build B‑bin features from the last D daily observations available as of anchor month (vintage cutoff).
    anchor_date is the target month end (EOM) we want to nowcast/forecast.
    For horizon h=0, anchor is month t; for h=1, anchor is month t (use info up to month t to forecast t+1).
    """
    if daily.empty:
        return np.full(B, np.nan)
    # compute vintage cutoff date for the anchor month
    month_start = pd.Timestamp(anchor_date.year, anchor_date.month, 1)
    last_dom = (month_start + pd.offsets.MonthEnd(0)).day
    cut_day = min(cutoff_day, last_dom)
    cutoff = month_start + pd.Timedelta(days=cut_day - 1)
    # use all daily strictly ≤ cutoff
    d = daily[daily["date"] <= cutoff]
    if d.empty:
        return np.full(B, np.nan)
    # take last D observations
    vals = d.iloc[-D:][d.columns[-1]].astype(float).values
    if len(vals) < 3:
        return np.full(B, np.nan)
    # split into B bins (almost equal sizes)
    idx = np.linspace(0, len(vals), B + 1).astype(int)
    feats = []
    for i in range(B):
        seg = vals[idx[i]: idx[i+1]]
        if seg.size == 0:
            feats.append(np.nan)
        else:
            feats.append(np.nanmean(seg) if agg == "mean" else np.nansum(seg))
    return np.array(feats, dtype=float)


def _walk_forward_umidas(y: pd.Series, horizon: int, mode: str, min_train: int, rolling_window: int,
                         daily_col: str, D: int, B: int, agg: str, cutoff_day: int) -> pd.Series:
    """Unrestricted MIDAS via binning last D daily lags into B features and OLS/Ridge.
    Uses only information available as‑of the anchor month (vintage cutoff).
    """
    if not daily_col or daily_col == "(none)":
        return pd.Series(dtype=float)
    daily = _collect_daily_series(daily_col)
    if daily.empty:
        return pd.Series(dtype=float)

    # Timeline to evaluate
    idx = y.index
    eval_idx = idx  # y itself will be aligned to horizon below when evaluating

    preds = []
    dates = []

    alphas = [0.1, 1.0, 10.0] if RidgeCV is None else [0.1, 1.0, 10.0]

    for t in eval_idx:
        # Define training window bounds relative to t
        start, end = _train_window(idx, t, mode, rolling_window)
        y_hist = y.loc[(y.index >= start) & (y.index <= end)].dropna()
        if len(y_hist) < min_train:
            continue

        # Build training design by iterating months in y_hist.index
        X_rows = []
        y_rows = []
        for m_end in y_hist.index:
            # anchor month is m_end for horizon=0, still m_end for horizon=1 (forecast t+1 using info up to t)
            anchor = m_end
            feats = _daily_lags_bins_for_anchor(daily, anchor, D, B, agg, cutoff_day)
            if np.all(np.isnan(feats)):
                continue
            X_rows.append(feats)
            # target at m_end + h
            if horizon == 0:
                y_rows.append(y.loc[m_end])
            else:
                # y at next month end if available
                nxt = m_end + pd.offsets.MonthEnd(1)
                if nxt in y.index:
                    y_rows.append(y.loc[nxt])
                else:
                    X_rows.pop();  # drop row if target missing
        if len(y_rows) < max(24, min_train // 2):
            continue
        X_tr = np.asarray(X_rows, dtype=float)
        y_tr = np.asarray(y_rows, dtype=float)

        # Test row at time t
        anchor_test = t  # use info up to t to predict y_t (h=0) or y_{t+1} (h=1)
        x_pred = _daily_lags_bins_for_anchor(daily, anchor_test, D, B, agg, cutoff_day)
        if np.any(np.isnan(x_pred)):
            continue

        # Fit ridge (if available) else OLS
        if RidgeCV is not None:
            model = RidgeCV(alphas=np.array(alphas), store_cv_values=False)
            try:
                model.fit(X_tr, y_tr)
                yhat = float(model.predict(x_pred.reshape(1, -1))[0])
            except Exception:
                continue
        else:
            # OLS closed‑form with Tikhonov 1e-6
            try:
                XtX = X_tr.T @ X_tr + 1e-6 * np.eye(X_tr.shape[1])
                beta = np.linalg.solve(XtX, X_tr.T @ y_tr)
                yhat = float(x_pred @ beta)
            except Exception:
                continue

        # Output date is t (+h months)
        out_date = t if horizon == 0 else (t + pd.offsets.MonthEnd(1))
        preds.append(yhat)
        dates.append(out_date)

    if not preds:
        return pd.Series(dtype=float)
    s = pd.Series(preds, index=pd.to_datetime(dates)).sort_index()
    return s

# ----------------------------------------------------------------------------
# ARIMAX / SARIMAX walk‑forward
# ----------------------------------------------------------------------------

def _walk_forward_sarimax(y: pd.Series, X: pd.DataFrame, horizon: int, mode: str, min_train: int, rolling_window: int,
                          order: tuple[int,int,int], seasonal_order: tuple[int,int,int,int], exog_cols: list[str], label: str) -> pd.Series:
    if SARIMAX is None:
        return pd.Series(dtype=float)
    idx = y.index
    preds = []
    dates = []

    # build exog matrix once
    Xn = X[exog_cols].copy() if exog_cols else pd.DataFrame(index=X.index)
    Xn = Xn.astype(float)

    for t in idx:
        start, end = _train_window(idx, t, mode, rolling_window)
        y_tr = y.loc[(y.index >= start) & (y.index <= end)].dropna()
        if len(y_tr) < min_train:
            continue
        ex_tr = Xn.loc[y_tr.index] if not Xn.empty else None

        try:
            mdl = SARIMAX(endog=y_tr, exog=ex_tr, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
            fit = mdl.fit(disp=False)
        except Exception:
            continue

        # Forecast step
        if horizon == 0:
            # need exog for next step = t (current month end)
            ex_f = Xn.loc[[t]] if not Xn.empty and t in Xn.index else None
            try:
                f = fit.forecast(steps=1, exog=ex_f)
                yhat = float(f.iloc[0])
                out_date = t
            except Exception:
                continue
        else:  # horizon=1
            # need exog for t+1 — we approximate by using exog at t (naive hold) if available
            t1 = t + pd.offsets.MonthEnd(1)
            ex_f = Xn.loc[[t1]] if not Xn.empty and t1 in Xn.index else (Xn.loc[[t]] if not Xn.empty and t in Xn.index else None)
            try:
                f = fit.forecast(steps=1, exog=ex_f)
                yhat = float(f.iloc[0])
                out_date = t1
            except Exception:
                continue

        preds.append(yhat)
        dates.append(out_date)

    if not preds:
        return pd.Series(dtype=float)
    return pd.Series(preds, index=pd.to_datetime(dates)).sort_index().rename(label)

# ----------------------------------------------------------------------------
# Main backtest run
# ----------------------------------------------------------------------------

def run_backtest(cfg: BTConfig, y: pd.Series, X: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame, dict]:
    # Slice evaluation window
    mask = (y.index.year >= cfg.start_year) & (y.index.year <= cfg.end_year)
    y_eval = y.loc[mask].copy()
    X_eval = X.loc[y_eval.index].copy()

    # Predictions holder
    pred_series: dict[str, pd.Series] = {}

    # Baselines
    idx = y_eval.index
    y_full = y
    for t in idx:
        pass  # just to keep idx in scope for helper

    # Build once for Ridge‑ARX
    full_design = _prepare_design(y_full, X, cfg.y_lags, cfg.x_lags, cfg.horizon)

    # Walk and produce for each model
    # Baselines: produce by single pass and convert to series
    def _baseline_series(name: str, func):
        preds = []
        dates = []
        for t in idx:
            start, end = _train_window(y_full.index, t, cfg.window_mode, cfg.rolling_window)
            y_hist = y_full.loc[(y_full.index >= start) & (y_full.index <= end)].dropna()
            if len(y_hist) < cfg.min_train:
                continue
            try:
                yhat = func(y_hist)
                out_date = t if cfg.horizon == 0 else (t + pd.offsets.MonthEnd(1))
                preds.append(yhat); dates.append(out_date)
            except Exception:
                continue
        if preds:
            pred_series[name] = pd.Series(preds, index=pd.to_datetime(dates)).sort_index()

    if cfg.use_naive: _baseline_series("NAIVE", _pred_naive)
    if cfg.use_snaive: _baseline_series(f"SNAIVE_s{cfg.seasonal_period}", lambda s: _pred_snaive(s, cfg.seasonal_period))
    if cfg.use_ma: _baseline_series(f"MA_{cfg.ma_k}", lambda s: _pred_ma(s, cfg.ma_k))
    if cfg.use_ets: _baseline_series("ETS", lambda s: _pred_ets(s, cfg.seasonal_period))

    # Ridge‑ARX
    if cfg.use_ridge and RidgeCV is not None:
        alphas = cfg.alpha_grid if cfg.alpha_grid else [0.1, 1.0, 10.0]
        preds, dates = [], []
        for t in idx:
            start, end = _train_window(y_full.index, t, cfg.window_mode, cfg.rolling_window)
            train = full_design.loc[(full_design.index >= start) & (full_design.index <= end)].dropna()
            if len(train) < max(12, cfg.min_train // 2):
                continue
            row = full_design.loc[[t]] if t in full_design.index else None
            if row is None or row.empty or row.isna().any(axis=1).iloc[0]:
                continue
            X_tr = train.drop(columns=["target"]).astype(float)
            y_tr = train["target"].astype(float)
            x_pred = row.drop(columns=["target"]).iloc[0].astype(float)
            if cfg.standardize and len(X_tr.columns) > 0:
                scl = StandardScaler(); X_tr[:] = scl.fit_transform(X_tr.values); x_pred[:] = scl.transform(x_pred.values.reshape(1,-1)).ravel()
            try:
                yhat = _fit_predict_ridge(y_tr, X_tr, x_pred, alphas)
            except Exception:
                continue
            out_date = t if cfg.horizon == 0 else (t + pd.offsets.MonthEnd(1))
            preds.append(yhat); dates.append(out_date)
        if preds:
            pred_series["RidgeARX"] = pd.Series(preds, index=pd.to_datetime(dates)).sort_index()

    # U‑MIDAS
    if cfg.use_umidas and cfg.umidas_daily_col and cfg.umidas_daily_col != "(none)":
        s = _walk_forward_umidas(y_eval, cfg.horizon, cfg.window_mode, cfg.min_train, cfg.rolling_window,
                                 cfg.umidas_daily_col, cfg.umidas_n_lags, cfg.umidas_n_bins, cfg.umidas_agg, cfg.vintage_cutoff_day)
        if not s.empty:
            pred_series[f"UMIDAS_{cfg.umidas_daily_col}_D{cfg.umidas_n_lags}_B{cfg.umidas_n_bins}"] = s

    # ARIMAX / SARIMAX
    if (cfg.use_arimax or cfg.use_sarimax) and SARIMAX is not None:
        Xnum = X_eval.select_dtypes(include=[np.number])
        excols = [c for c in cfg.arimax_exog_cols if c in Xnum.columns]
        if cfg.use_arimax:
            s = _walk_forward_sarimax(y_eval, Xnum, cfg.horizon, cfg.window_mode, cfg.min_train, cfg.rolling_window,
                                      order=cfg.arimax_order, seasonal_order=(0,0,0,0), exog_cols=excols, label="ARIMAX")
            if not s.empty:
                pred_series["ARIMAX"] = s
        if cfg.use_sarimax:
            s = _walk_forward_sarimax(y_eval, Xnum, cfg.horizon, cfg.window_mode, cfg.min_train, cfg.rolling_window,
                                      order=cfg.sarimax_order, seasonal_order=cfg.sarimax_seasonal, exog_cols=excols, label="SARIMAX")
            if not s.empty:
                pred_series["SARIMAX"] = s

    # Metrics
    metrics_rows = []
    for name, s in pred_series.items():
        y_al, p_al = state.y_monthly.align(s, join="inner")
        metrics_rows.append({"model": name, "MAE": mae(y_al, p_al), "RMSE": rmse(y_al, p_al), "SMAPE": smape(y_al, p_al), "MASE": mase(y_al, p_al), "obs": int(len(y_al))})
    metrics = pd.DataFrame(metrics_rows).sort_values("MAE") if metrics_rows else pd.DataFrame(columns=["model","MAE","RMSE","SMAPE","MASE","obs"])

    # Ensembles (uniform & inverse‑MAE)
    if len(pred_series) >= 2 and not metrics.empty:
        # align
        all_df = pd.DataFrame({"y": state.y_monthly}).join(pd.DataFrame(pred_series)).dropna()
        if all_df.shape[1] > 2:
            # uniform
            ens_u = all_df.drop(columns=["y"]).mean(axis=1)
            pred_series["Ensemble_Uniform"] = ens_u
            # inverse‑MAE
            m = metrics.set_index("model")["MAE"]
            valid = [c for c in all_df.columns if c != "y" and c in m.index]
            w = (1.0 / (m.loc[valid] + 1e-9)).values
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            ens_w = (all_df[valid].values @ w)
            pred_series["Ensemble_InvMAE"] = pd.Series(ens_w, index=all_df.index)
            # recompute metrics including ensembles
            metrics_rows = []
            for name, s in pred_series.items():
                y_al, p_al = state.y_monthly.align(s, join="inner")
                metrics_rows.append({"model": name, "MAE": mae(y_al, p_al), "RMSE": rmse(y_al, p_al), "SMAPE": smape(y_al, p_al), "MASE": mase(y_al, p_al), "obs": int(len(y_al))})
            metrics = pd.DataFrame(metrics_rows).sort_values("MAE")

    cfg_dump = json.dumps(cfg.__dict__, indent=2)
    return pred_series, metrics, {"config": cfg_dump}

# ----------------------------------------------------------------------------
# Run & display
# ----------------------------------------------------------------------------

if run_btn:
    # Parse ridge alphas
    try:
        alpha_vals = [float(a.strip()) for a in alpha_grid_txt.split(",") if a.strip()]
    except Exception:
        alpha_vals = [0.1, 1.0, 10.0]

    # Build X from panel (numeric only)
    X_full = state.panel_monthly.select_dtypes(include=[np.number]).copy()

    cfg = BTConfig(
        start_year=int(s_year), end_year=int(e_year), horizon=int(horizon),
        window_mode=str(window_mode), min_train=int(min_train), rolling_window=int(rolling_window),
        y_lags=12, x_lags=6, standardize=True, seasonal_period=int(seas),
        use_naive=bool(use_naive), use_snaive=bool(use_snaive), use_ma=bool(use_ma), ma_k=int(ma_k), use_ets=bool(use_ets),
        use_ridge=bool(use_ridge), use_umidas=bool(use_umidas), use_arimax=bool(use_arimax), use_sarimax=bool(use_sarimax),
        alpha_grid=alpha_vals,
        umidas_daily_col=(None if umidas_col == "(none)" else umidas_col), umidas_n_lags=int(umidas_nlags), umidas_n_bins=int(umidas_bins), umidas_agg=str(umidas_agg), vintage_cutoff_day=int(vintage_cut),
        arimax_order=(int(p), int(d), int(q)), sarimax_order=(int(p), int(d), int(q)), sarimax_seasonal=(int(P), int(D), int(Q), int(m_season)),
        arimax_exog_cols=list(exog_cols),
    )

    with st.spinner("Running walk‑forward backtest…"):
        preds, metrics, meta = run_backtest(cfg, state.y_monthly, X_full)

    if not preds:
        st.warning("No predictions produced. Adjust settings (train size, horizon, exog selection).")
    else:
        state.bt_results = preds
        state.bt_metrics = metrics

        st.success("Backtest completed.")
        st.markdown("### Results table")
        st.dataframe(metrics.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","SMAPE":"{:.2f}","MASE":"{:.3f}"}), use_container_width=True)

        # Plot: target vs top‑k predictions
        topK = st.slider("Top‑k models to plot", 1, min(7, len(preds)), min(4, len(preds)))
        order = metrics.sort_values("MAE")["model"].tolist()[:topK]
        y = state.y_monthly
        fig = go.Figure(); fig.add_trace(go.Scatter(x=y.index, y=y.values, name="Target", mode="lines", line=dict(width=3)))
        for name in order:
            s = preds[name]
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines"))
        fig.update_layout(template="plotly_white", height=480, title="Target vs top‑k predictions")
        st.plotly_chart(fig, use_container_width=True)

        # Export
        st.markdown("---"); st.subheader("Export")
        out = pd.DataFrame({"date": state.y_monthly.index, "y": state.y_monthly.values}).set_index("date")
        for name, s in preds.items():
            out = out.join(s.rename(name), how="outer")
        out = out.sort_index()
        st.download_button("predictions.csv", data=out.to_csv().encode("utf-8"), file_name="predictions.csv")
        st.download_button("metrics.csv", data=metrics.to_csv(index=False).encode("utf-8"), file_name="metrics.csv")
        st.markdown("**Config JSON**"); st.code(meta["config"], language="json")
else:
    st.info("Configure the window and models in the sidebar, then click **Run backtest**. For ARIMAX/SARIMAX and U‑MIDAS, keep horizon ≤ 1.")
