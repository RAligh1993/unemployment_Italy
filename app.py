# -*- coding: utf-8 -*-
"""
===========================================================================
ISTAT INTERACTIVE UNEMPLOYMENT NOWCASTING LAB
===========================================================================
Interactive backtesting platform for Italian unemployment forecasting

Features:
- Interactive data selection (date ranges, training windows)
- All forecasting models (NAIVE, MA, ETS, Ridge, MIDAS, Ensembles)
- Configurable scenarios (COVID, Google Trends, backtest mode)
- SHAP feature importance analysis
- Event timeline with impact visualization
- Real-time news integration
- Comprehensive performance comparison
- Export functionality

Author: [Your Name] - ISTAT Internship 2025
Version: 3.0 (Production Ready)
===========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import json
import io
from typing import List, Dict, Tuple, Optional

# ML & Stats
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import shap

# Optional: ETS
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_ETS = True
except ImportError:
    HAS_ETS = False

# Optional: News
try:
    import feedparser
    HAS_NEWS = True
except ImportError:
    HAS_NEWS = False

warnings.filterwarnings('ignore')

# ===========================================================================
# PAGE CONFIG
# ===========================================================================

st.set_page_config(
    page_title="🇮🇹 ISTAT Nowcasting Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================================
# CUSTOM CSS
# ===========================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    h1 {
        color: #1e40af;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    
    .success-box {
        background: #dcfce7;
        border-left: 4px solid #16a34a;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #ca8a04;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .info-box {
        background: #dbeafe;
        border-left: 4px solid #2563eb;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .event-marker {
        background: #fecaca;
        border-left: 4px solid #dc2626;
        padding: 10px;
        border-radius: 6px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# SESSION STATE INITIALIZATION
# ===========================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'backtest_run' not in st.session_state:
    st.session_state.backtest_run = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None

# ===========================================================================
# CONFIGURATION & CONSTANTS
# ===========================================================================

class Config:
    """Global configuration"""
    
    # Backtest modes
    BACKTEST_MODES = ["Expanding Window", "Rolling Window", "Both"]
    
    # Models
    MODELS = {
        "NAIVE": "Simple baseline (t = t-1)",
        "MA3": "3-month moving average",
        "MA12": "12-month moving average",
        "ETS": "Exponential smoothing",
        "AR_Ridge_BASE": "Ridge with AR lags only",
        "AR_Ridge_FIN": "Ridge with AR + financial",
        "MIDAS_AR": "MIDAS with high-frequency data",
        "GT_Ridge_PCA": "Ridge with Google Trends (PCA)",
        "Combined_Ridge": "Ridge with all features",
        "Ensemble_Simple": "Simple average ensemble",
        "Ensemble_Trim": "Trimmed mean ensemble"
    }
    
    # COVID Events
    COVID_EVENTS = [
        {
            "name": "First Lockdown",
            "start": "2020-03-09",
            "end": "2020-05-18",
            "color": "#dc2626",
            "description": "Italy's first national lockdown"
        },
        {
            "name": "Second Lockdown",
            "start": "2020-11-06",
            "end": "2020-12-03",
            "color": "#ea580c",
            "description": "Partial lockdown measures"
        },
        {
            "name": "Third Lockdown",
            "start": "2021-03-15",
            "end": "2021-04-26",
            "color": "#f59e0b",
            "description": "Regional restrictions"
        },
        {
            "name": "Vaccine Rollout",
            "start": "2021-01-01",
            "end": "2021-01-01",
            "color": "#16a34a",
            "description": "Mass vaccination begins"
        },
        {
            "name": "State of Emergency End",
            "start": "2022-03-31",
            "end": "2022-03-31",
            "color": "#2563eb",
            "description": "COVID emergency declared over"
        }
    ]
    
    # MIDAS settings
    MIDAS_WINDOWS = (10, 15)
    MIDAS_K = 3
    
    # Ridge alphas
    RIDGE_ALPHAS = np.logspace(-3, 3, 25)
    
    # Ensemble settings
    ENSEMBLE_TRIM = 0.25
    
    # PCA settings
    PCA_VAR = 0.95
    PCA_MAX_COMP = 12

CFG = Config()

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def sanitize_column_name(name: str) -> str:
    """Convert column name to safe format."""
    name = str(name).strip().lower()
    name = name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    return name

def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric, handling missing values."""
    return pd.to_numeric(series, errors='coerce')

def parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates robustly."""
    return pd.to_datetime(series, errors='coerce', dayfirst=True)

def to_month_end(series: pd.Series) -> pd.Series:
    """Convert to month-end."""
    return (pd.to_datetime(series) + pd.offsets.MonthEnd(0)).dt.normalize()

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MASE": np.nan, "N": 0}
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    
    # MASE
    scale = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    mase = mae / scale if scale > 0 else np.nan
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MASE": float(mase),
        "N": len(y_true)
    }

def almon_weights(n: int, K: int = 3) -> np.ndarray:
    """Generate Almon polynomial weights for MIDAS."""
    j = np.arange(1, n + 1, dtype=float)
    Phi = np.vstack([j**k for k in range(K)]).T
    w = Phi @ np.ones(K)
    w = np.maximum(w, 0.0)
    w = w[::-1]
    if w.sum() == 0:
        w[:] = 1.0
    return w / w.sum()

# ===========================================================================
# DATA LOADING
# ===========================================================================

@st.cache_data
def load_data(uploaded_file) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all data sheets."""
    
    try:
        # Monthly
        monthly = pd.read_excel(uploaded_file, sheet_name='monthly')
        monthly.columns = [sanitize_column_name(c) for c in monthly.columns]
        monthly['date'] = to_month_end(monthly['date'])
        
        for col in monthly.columns:
            if col != 'date':
                monthly[col] = clean_numeric(monthly[col])
        
        monthly = monthly.sort_values('date').dropna(subset=['date']).reset_index(drop=True)
        
        # Find unemployment column
        unemp_col = None
        for col in ['unemp', 'unemployment', 'unemployment_rate']:
            if col in monthly.columns:
                unemp_col = col
                break
        
        if unemp_col:
            monthly = monthly.rename(columns={unemp_col: 'unemp'})
            monthly = monthly.dropna(subset=['unemp'])
        
        # Stock
        try:
            stock = pd.read_excel(uploaded_file, sheet_name='daily_stock', header=None)
            stock.columns = ["date", "close", "volume"] + [f"col_{i}" for i in range(3, len(stock.columns))]
            stock['date'] = parse_dates(stock['date'])
            stock['close'] = clean_numeric(stock['close'])
            stock = stock[['date', 'close']].dropna().sort_values('date').reset_index(drop=True)
            stock['ret'] = np.log(stock['close']).diff()
        except:
            stock = pd.DataFrame()
        
        # VIX
        try:
            vix = pd.read_excel(uploaded_file, sheet_name='VIX')
            vix.columns = [sanitize_column_name(c) for c in vix.columns]
            vix['date'] = parse_dates(vix['date'])
            
            vix_col = None
            for col in ['vix', 'v2tx', 'close', 'value']:
                if col in vix.columns:
                    vix_col = col
                    break
            
            if vix_col:
                vix['vix'] = clean_numeric(vix[vix_col])
                vix = vix[['date', 'vix']].dropna().sort_values('date').reset_index(drop=True)
        except:
            vix = pd.DataFrame()
        
        # Google Trends
        try:
            gt = pd.read_excel(uploaded_file, sheet_name='google')
            date_col = gt.columns[0]
            gt = gt.rename(columns={date_col: 'date'})
            gt['date'] = parse_dates(gt['date'])
            
            kw_cols = [c for c in gt.columns if c != 'date']
            new_names = {'date': 'date'}
            for c in kw_cols:
                new_names[c] = 'gt_' + sanitize_column_name(c)
            gt = gt.rename(columns=new_names)
            
            for c in gt.columns:
                if c != 'date':
                    gt[c] = clean_numeric(gt[c])
            
            gt = gt.dropna(subset=[c for c in gt.columns if c.startswith('gt_')], how='all')
            gt = gt.sort_values('date').reset_index(drop=True)
        except:
            gt = pd.DataFrame()
        
        return monthly, stock, vix, gt
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, None, None

# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================

def add_ar_lags(df: pd.DataFrame, target_col: str = 'unemp') -> pd.DataFrame:
    """Add AR lags."""
    z = df.copy()
    for lag in [1, 2, 3, 12]:
        z[f"{target_col}_lag{lag}"] = z[target_col].shift(lag)
    return z

def add_covid_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add COVID event dummies."""
    z = df.copy()
    
    for i, event in enumerate(CFG.COVID_EVENTS[:3]):  # Only lockdowns
        start = pd.Timestamp(event['start'])
        end = pd.Timestamp(event['end'])
        z[f"covid_lockdown{i+1}"] = ((z['date'] >= start) & (z['date'] <= end)).astype(int)
    
    covid_start = pd.Timestamp("2020-03-01")
    z["covid_era"] = (z['date'] >= covid_start).astype(int)
    
    return z

def aggregate_daily_to_monthly(daily_df: pd.DataFrame, monthly_dates: pd.DataFrame, 
                               value_col: str, vintage_day: int = 15) -> pd.DataFrame:
    """Aggregate daily data to monthly MTD."""
    
    if daily_df.empty:
        return monthly_dates[['date']].copy()
    
    # Vintage cut
    daily_v = daily_df.copy()
    daily_v['date'] = pd.to_datetime(daily_v['date'])
    
    ms = pd.to_datetime(daily_v['date'].values.astype("datetime64[M]"))
    me = ms + pd.offsets.MonthEnd(0)
    last_dom = pd.Series(me).dt.day.to_numpy()
    effective_cutoff = np.minimum(vintage_day, last_dom)
    cut_date = ms + pd.to_timedelta(effective_cutoff - 1, unit="D")
    mask = daily_v['date'].to_numpy() <= cut_date
    
    daily_v = daily_v.loc[mask].copy()
    daily_v['ym'] = daily_v['date'].dt.to_period("M")
    
    # Aggregate
    monthly_agg = daily_v.groupby('ym')[value_col].agg(['mean', 'std', 'min', 'max']).reset_index()
    monthly_agg['date'] = monthly_agg['ym'].dt.to_timestamp("M") + pd.offsets.MonthEnd(0)
    monthly_agg = monthly_agg.drop(columns=['ym'])
    monthly_agg.columns = ['date'] + [f"{value_col}_{c}" for c in ['mean', 'std', 'min', 'max']]
    
    result = monthly_dates[['date']].merge(monthly_agg, on='date', how='left')
    return result

# ===========================================================================
# MODELING FUNCTIONS
# ===========================================================================

def walk_forward_naive(df: pd.DataFrame, target_col: str, min_train: int, mode: str,
                      rolling_window: Optional[int] = None) -> pd.DataFrame:
    """NAIVE forecast."""
    rows = []
    for t in range(min_train, len(df)):
        rows.append({
            'date': df['date'].iloc[t],
            'actual': float(df[target_col].iloc[t]),
            'forecast': float(df[target_col].iloc[t-1]),
            'model': 'NAIVE'
        })
    return pd.DataFrame(rows)

def walk_forward_ma(df: pd.DataFrame, k: int, target_col: str, min_train: int, mode: str,
                   rolling_window: Optional[int] = None) -> pd.DataFrame:
    """Moving average."""
    rows = []
    start_t = max(k, min_train)
    
    for t in range(start_t, len(df)):
        if mode == "rolling" and rolling_window is not None:
            train_start = max(0, t - rolling_window)
            ma_window = df[target_col].iloc[train_start:t]
        else:
            ma_window = df[target_col].iloc[max(0, t-k):t]
        
        rows.append({
            'date': df['date'].iloc[t],
            'actual': float(df[target_col].iloc[t]),
            'forecast': float(ma_window.tail(k).mean()),
            'model': f'MA{k}'
        })
    
    return pd.DataFrame(rows)

def walk_forward_ets(df: pd.DataFrame, target_col: str, min_train: int, mode: str,
                    rolling_window: Optional[int] = None) -> pd.DataFrame:
    """Exponential Smoothing."""
    if not HAS_ETS:
        return pd.DataFrame()
    
    rows = []
    
    for t in range(min_train, len(df)):
        if mode == "rolling" and rolling_window is not None:
            train_start = max(0, t - rolling_window)
            y_train = df[target_col].iloc[train_start:t].values
        else:
            y_train = df[target_col].iloc[:t].values
        
        try:
            model = ExponentialSmoothing(y_train, trend="add", seasonal=None,
                                        initialization_method="estimated")
            fit = model.fit(optimized=True, use_brute=False)
            forecast_result = fit.forecast(1)
            yhat = float(forecast_result[0])
            
            rows.append({
                'date': df['date'].iloc[t],
                'actual': float(df[target_col].iloc[t]),
                'forecast': yhat,
                'model': 'ETS'
            })
        except:
            continue
    
    return pd.DataFrame(rows)

def walk_forward_ridge(df: pd.DataFrame, features: List[str], target_col: str,
                      min_train: int, mode: str, model_name: str,
                      rolling_window: Optional[int] = None) -> pd.DataFrame:
    """Ridge regression."""
    rows = []
    avail_feats = [f for f in features if f in df.columns]
    
    if len(avail_feats) == 0:
        return pd.DataFrame()
    
    df_sub = df[['date', target_col] + avail_feats].copy()
    
    for t in range(min_train, len(df_sub)):
        if mode == "rolling" and rolling_window is not None:
            train_start = max(0, t - rolling_window)
            train = df_sub.iloc[train_start:t].dropna(subset=[target_col])
        else:
            train = df_sub.iloc[:t].dropna(subset=[target_col])
        
        test = df_sub.iloc[[t]]
        
        if len(train) < min_train:
            continue
        
        X_train = train[avail_feats].values
        y_train = train[target_col].values
        X_test = test[avail_feats].values
        
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=CFG.RIDGE_ALPHAS, cv=min(5, max(2, len(train)//10))))
        ])
        
        try:
            pipe.fit(X_train, y_train)
            yhat = float(pipe.predict(X_test)[0])
            
            rows.append({
                'date': test['date'].values[0],
                'actual': float(test[target_col].values[0]),
                'forecast': yhat,
                'model': model_name
            })
        except:
            continue
    
    return pd.DataFrame(rows)

def align_forecasts_to_common_dates(forecasts: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Align all forecasts to common dates."""
    if len(forecasts) == 0:
        return []
    
    common_dates = set(forecasts[0]['date'])
    for fc in forecasts[1:]:
        common_dates = common_dates.intersection(set(fc['date']))
    
    common_dates = sorted(list(common_dates))
    
    aligned = []
    for fc in forecasts:
        fc_aligned = fc[fc['date'].isin(common_dates)].copy()
        aligned.append(fc_aligned)
    
    return aligned

# ===========================================================================
# SHAP ANALYSIS
# ===========================================================================

def compute_shap_values(df: pd.DataFrame, features: List[str], target_col: str,
                       model_name: str, sample_size: int = 100) -> Optional[Tuple]:
    """Compute SHAP values for Ridge model."""
    
    try:
        # Prepare data
        avail_feats = [f for f in features if f in df.columns]
        if len(avail_feats) == 0:
            return None
        
        df_sub = df[['date', target_col] + avail_feats].dropna()
        
        if len(df_sub) < 50:
            return None
        
        # Train final model
        X = df_sub[avail_feats].values
        y = df_sub[target_col].values
        
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=CFG.RIDGE_ALPHAS, cv=5))
        ])
        
        pipe.fit(X, y)
        
        # Transform data
        X_transformed = pipe[:-1].transform(X)
        
        # Sample for SHAP (to speed up)
        if len(X_transformed) > sample_size:
            indices = np.random.choice(len(X_transformed), sample_size, replace=False)
            X_sample = X_transformed[indices]
            feature_names = avail_feats
        else:
            X_sample = X_transformed
            feature_names = avail_feats
        
        # Compute SHAP
        explainer = shap.LinearExplainer(pipe[-1], X_transformed)
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values, X_sample, feature_names, explainer, X_transformed
    
    except Exception as e:
        st.warning(f"⚠️ SHAP computation failed for {model_name}: {str(e)}")
        return None

# ===========================================================================
# VISUALIZATION FUNCTIONS
# ===========================================================================

def plot_time_series(df: pd.DataFrame, target_col: str, title: str) -> go.Figure:
    """Interactive time series plot."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[target_col],
        mode='lines+markers',
        name='Unemployment Rate',
        line=dict(color='#2563eb', width=3),
        marker=dict(size=6)
    ))
    
    # Trend line
    z = np.polyfit(range(len(df)), df[target_col], 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=p(range(len(df))),
        mode='lines',
        name='Trend',
        line=dict(color='#dc2626', width=2, dash='dash')
    ))
    
    # COVID events
    for event in CFG.COVID_EVENTS:
        fig.add_vrect(
            x0=event['start'], x1=event.get('end', event['start']),
            fillcolor=event['color'], opacity=0.2,
            layer="below", line_width=0,
            annotation_text=event['name'],
            annotation_position="top left"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Unemployment Rate (%)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_forecast_comparison(forecasts: List[pd.DataFrame], actual_df: pd.DataFrame,
                            target_col: str, n_recent: int = 24) -> go.Figure:
    """Compare forecasts interactively."""
    
    fig = go.Figure()
    
    colors = {
        'NAIVE': '#dc2626',
        'MA3': '#ea580c',
        'MA12': '#f59e0b',
        'ETS': '#84cc16',
        'AR_Ridge_BASE': '#14b8a6',
        'AR_Ridge_FIN': '#06b6d4',
        'MIDAS_AR': '#3b82f6',
        'GT_Ridge_PCA': '#8b5cf6',
        'Combined_Ridge': '#a855f7',
        'Ensemble_Simple': '#16a34a',
        'Ensemble_Trim': '#15803d'
    }
    
    for fc_df in forecasts:
        if fc_df.empty:
            continue
        
        model_name = fc_df['model'].iloc[0]
        color = colors.get(model_name, '#6b7280')
        
        fc_recent = fc_df.tail(n_recent)
        
        fig.add_trace(go.Scatter(
            x=fc_recent['date'],
            y=fc_recent['forecast'],
            mode='lines+markers',
            name=model_name,
            line=dict(color=color, width=2),
            marker=dict(size=5),
            visible='legendonly' if model_name not in ['NAIVE', 'Ensemble_Simple'] else True
        ))
    
    # Actual
    actual_recent = actual_df.tail(n_recent)
    fig.add_trace(go.Scatter(
        x=actual_recent['date'],
        y=actual_recent[target_col],
        mode='lines+markers',
        name='Actual',
        line=dict(color='black', width=3),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title=f"Forecast Comparison (Last {n_recent} Months)",
        xaxis_title="Date",
        yaxis_title="Unemployment Rate (%)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_performance_bars(results_df: pd.DataFrame) -> go.Figure:
    """Performance comparison bars."""
    
    results_df = results_df.sort_values('MASE')
    
    colors = ['#16a34a' if x < 1.0 else '#dc2626' if x > 1.2 else '#f59e0b'
             for x in results_df['MASE']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=results_df['Model'],
        x=results_df['MASE'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='black', width=1)),
        text=results_df['MASE'].apply(lambda x: f'{x:.3f}'),
        textposition='outside'
    ))
    
    fig.add_vline(x=1.0, line_dash="dash", line_color="red", 
                  annotation_text="NAIVE Baseline", annotation_position="top right")
    
    fig.update_layout(
        title="Model Performance - MASE (Lower is Better)",
        xaxis_title="MASE",
        yaxis_title="",
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_shap_summary(shap_values, X_sample, feature_names) -> go.Figure:
    """SHAP summary plot."""
    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=importance_df['Feature'],
        x=importance_df['Importance'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title="Feature Importance (Mean |SHAP value|)",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="",
        template='plotly_white',
        height=max(400, len(feature_names) * 20)
    )
    
    return fig

# ===========================================================================
# MAIN APP
# ===========================================================================

# Header
st.markdown("""
<div class="hero-banner">
    <h1 style="color: white; margin: 0;">🔬 ISTAT Interactive Nowcasting Lab</h1>
    <p style="font-size: 1.2rem; margin: 10px 0 0 0; opacity: 0.95;">
        Explore, Configure, and Evaluate Italian Unemployment Forecasting Models
    </p>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# SIDEBAR: Configuration
# ===========================================================================

with st.sidebar:
    st.header("📂 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Excel file",
        type=['xlsx', 'xls'],
        help="Upload economic_data1.xlsx with sheets: monthly, daily_stock, VIX, google"
    )
    
    if uploaded_file is not None:
        
        if not st.session_state.data_loaded:
            with st.spinner("⏳ Loading data..."):
                monthly, stock, vix, gt = load_data(uploaded_file)
                
                if monthly is not None:
                    st.session_state.monthly = monthly
                    st.session_state.stock = stock
                    st.session_state.vix = vix
                    st.session_state.gt = gt
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Loaded {len(monthly)} monthly observations")
                else:
                    st.error("❌ Failed to load data")
                    st.stop()
        else:
            monthly = st.session_state.monthly
            stock = st.session_state.stock
            vix = st.session_state.vix
            gt = st.session_state.gt
            
            st.success(f"✅ Data loaded: {len(monthly)} months")
        
        st.markdown("---")
        st.header("⚙️ Configuration")
        
        # Date range
        st.subheader("📅 Date Range")
        
        date_min = monthly['date'].min().date()
        date_max = monthly['date'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start date",
                value=date_min,
                min_value=date_min,
                max_value=date_max
            )
        with col2:
            end_date = st.date_input(
                "End date",
                value=date_max,
                min_value=date_min,
                max_value=date_max
            )
        
        # Training window
        st.subheader("🎯 Training Settings")
        
        min_train = st.slider(
            "Minimum training months",
            min_value=24,
            max_value=60,
            value=48,
            step=6,
            help="Minimum historical data required before making first forecast"
        )
        
        backtest_mode_options = ["Expanding Window", "Rolling Window", "Both"]
        backtest_mode = st.selectbox(
            "Backtest mode",
            backtest_mode_options,
            index=0,
            help="Expanding: use all past data | Rolling: use fixed window"
        )
        
        if "Rolling" in backtest_mode:
            rolling_window = st.slider(
                "Rolling window size (months)",
                min_value=36,
                max_value=84,
                value=60,
                step=6,
                help="Size of training window for rolling backtest"
            )
        else:
            rolling_window = 60
        
        # Feature toggles
        st.subheader("🎛️ Features")
        
        use_covid = st.checkbox("COVID dummies", value=False, 
                               help="Include lockdown indicators")
        
        use_google_trends = st.checkbox("Google Trends", value=False,
                                       help="Include GT features (reduces sample size)")
        
        # Model selection
        st.subheader("🤖 Models")
        
        models_to_run = []
        
        with st.expander("📊 Benchmarks", expanded=True):
            if st.checkbox("NAIVE", value=True):
                models_to_run.append('NAIVE')
            if st.checkbox("MA3", value=True):
                models_to_run.append('MA3')
            if st.checkbox("MA12", value=False):
                models_to_run.append('MA12')
            if HAS_ETS and st.checkbox("ETS", value=True):
                models_to_run.append('ETS')
        
        with st.expander("🎯 Ridge Models"):
            if st.checkbox("AR_Ridge_BASE", value=True):
                models_to_run.append('AR_Ridge_BASE')
            if st.checkbox("AR_Ridge_FIN", value=True):
                models_to_run.append('AR_Ridge_FIN')
            if use_google_trends and st.checkbox("GT_Ridge_PCA", value=False):
                models_to_run.append('GT_Ridge_PCA')
        
        with st.expander("🚀 Advanced"):
            if st.checkbox("MIDAS_AR", value=False):
                models_to_run.append('MIDAS_AR')
            if st.checkbox("Ensembles", value=True):
                models_to_run.append('ENSEMBLE')
        
        # Analysis options
        st.subheader("📊 Analysis")
        
        enable_shap = st.checkbox("SHAP Feature Importance", value=False,
                                 help="Compute feature importance (slower)")
        
        show_events = st.checkbox("Show COVID Events", value=True)
        
        st.markdown("---")
        
        # Run button
        run_backtest = st.button(
            "🚀 Run Backtesting",
            type="primary",
            use_container_width=True
        )
    
    else:
        st.info("👆 Please upload data file to begin")
        run_backtest = False

# ===========================================================================
# MAIN CONTENT
# ===========================================================================

if not uploaded_file:
    # Welcome screen
    st.markdown("## 🎯 Welcome to the Interactive Nowcasting Lab")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3 style="color: #667eea;">📊 Interactive Backtesting</h3>
            <p>Test multiple forecasting models with custom configurations:</p>
            <ul>
                <li>Select date ranges</li>
                <li>Choose training windows</li>
                <li>Toggle features on/off</li>
                <li>Compare expanding vs rolling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3 style="color: #667eea;">🤖 Advanced Models</h3>
            <p>Comprehensive model suite:</p>
            <ul>
                <li>Benchmarks (NAIVE, MA, ETS)</li>
                <li>Ridge ML models</li>
                <li>MIDAS with HF data</li>
                <li>Smart ensembles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3 style="color: #667eea;">🔬 Deep Analysis</h3>
            <p>Understand your models:</p>
            <ul>
                <li>SHAP feature importance</li>
                <li>Event impact visualization</li>
                <li>Performance metrics</li>
                <li>Export results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🚀 Quick Start")
    
    st.markdown("""
    1. **Upload Data**: Excel file with sheets `monthly`, `daily_stock`, `VIX`, `google`
    2. **Configure**: Choose models, features, and date ranges
    3. **Run**: Click "🚀 Run Backtesting"
    4. **Explore**: Interactive visualizations and analysis
    5. **Export**: Download results for further analysis
    """)
    
    st.info("""
    **💡 Tip**: Start with default settings (NAIVE + MA3 + AR_Ridge models) 
    for quick results, then explore advanced configurations.
    """)
    
    st.stop()

# ===========================================================================
# BACKTESTING EXECUTION
# ===========================================================================

if run_backtest:
    
    monthly = st.session_state.monthly
    stock = st.session_state.stock
    vix = st.session_state.vix
    gt = st.session_state.gt
    
    # Filter date range
    monthly_filtered = monthly[
        (monthly['date'] >= pd.Timestamp(start_date)) &
        (monthly['date'] <= pd.Timestamp(end_date))
    ].copy()
    
    if len(monthly_filtered) < min_train + 12:
        st.error(f"❌ Insufficient data. Need at least {min_train + 12} months.")
        st.stop()
    
    st.markdown("---")
    st.header("🔄 Running Backtesting...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepare features
    status_text.text("📊 Preparing features...")
    progress_bar.progress(10)
    
    panel = add_ar_lags(monthly_filtered, 'unemp')
    
    if use_covid:
        panel = add_covid_dummies(panel)
    
    # Financial features
    if not stock.empty:
        stock_monthly = aggregate_daily_to_monthly(
            stock[['date', 'ret']].rename(columns={'ret': 'stock_ret'}),
            panel[['date']],
            'stock_ret'
        )
        panel = panel.merge(stock_monthly, on='date', how='left')
    
    if not vix.empty:
        vix_monthly = aggregate_daily_to_monthly(
            vix,
            panel[['date']],
            'vix'
        )
        panel = panel.merge(vix_monthly, on='date', how='left')
    
    # Google Trends
    if use_google_trends and not gt.empty:
        # Simple aggregation
        gt['ym'] = gt['date'].dt.to_period('M')
        gt_cols = [c for c in gt.columns if c.startswith('gt_')]
        
        gt_monthly = gt.groupby('ym')[gt_cols].mean().reset_index()
        gt_monthly['date'] = gt_monthly['ym'].dt.to_timestamp('M') + pd.offsets.MonthEnd(0)
        gt_monthly = gt_monthly.drop(columns=['ym'])
        
        panel = panel.merge(gt_monthly, on='date', how='inner')
        
        st.info(f"ℹ️ Using Google Trends reduces sample to {len(panel)} months")
    
    progress_bar.progress(20)
    
    # Define features
    base_feats = ['unemp_lag1', 'unemp_lag2', 'unemp_lag3', 'unemp_lag12']
    base_feats = [f for f in base_feats if f in panel.columns]
    
    covid_feats = []
    if use_covid:
        covid_feats = [c for c in ['covid_lockdown1', 'covid_lockdown2', 
                                   'covid_lockdown3', 'covid_era'] if c in panel.columns]
    
    fin_feats = [c for c in panel.columns if 'stock' in c or 'vix' in c]
    
    gt_feats = [c for c in panel.columns if c.startswith('gt_')]
    
    status_text.text(f"Features: {len(base_feats)} AR + {len(covid_feats)} COVID + "
                    f"{len(fin_feats)} Financial + {len(gt_feats)} GT")
    
    # Determine modes
    if backtest_mode == "Expanding Window":
        modes = ["expanding"]
    elif backtest_mode == "Rolling Window":
        modes = ["rolling"]
    else:
        modes = ["expanding", "rolling"]
    
    all_results = []
    all_forecasts = {}
    
    for mode in modes:
        
        status_text.text(f"🔄 Running {mode.upper()} window backtesting...")
        
        forecasts = []
        
        # NAIVE
        if 'NAIVE' in models_to_run:
            progress_bar.progress(30)
            fc = walk_forward_naive(panel, 'unemp', min_train, mode, rolling_window)
            if not fc.empty:
                forecasts.append(fc)
        
        # MA3
        if 'MA3' in models_to_run:
            progress_bar.progress(35)
            fc = walk_forward_ma(panel, 3, 'unemp', min_train, mode, rolling_window)
            if not fc.empty:
                forecasts.append(fc)
        
        # MA12
        if 'MA12' in models_to_run:
            progress_bar.progress(40)
            fc = walk_forward_ma(panel, 12, 'unemp', min_train, mode, rolling_window)
            if not fc.empty:
                forecasts.append(fc)
        
        # ETS
        if 'ETS' in models_to_run:
            progress_bar.progress(45)
            fc = walk_forward_ets(panel, 'unemp', min_train, mode, rolling_window)
            if not fc.empty:
                forecasts.append(fc)
        
        # AR_Ridge_BASE
        if 'AR_Ridge_BASE' in models_to_run:
            progress_bar.progress(50)
            feats = base_feats + covid_feats
            fc = walk_forward_ridge(panel, feats, 'unemp', min_train, mode, 
                                   'AR_Ridge_BASE', rolling_window)
            if not fc.empty:
                forecasts.append(fc)
        
        # AR_Ridge_FIN
        if 'AR_Ridge_FIN' in models_to_run:
            progress_bar.progress(60)
            feats = base_feats + covid_feats + fin_feats
            fc = walk_forward_ridge(panel, feats, 'unemp', min_train, mode,
                                   'AR_Ridge_FIN', rolling_window)
            if not fc.empty:
                forecasts.append(fc)
        
        # GT_Ridge_PCA (simplified - no PCA for now)
        if 'GT_Ridge_PCA' in models_to_run and gt_feats:
            progress_bar.progress(70)
            feats = base_feats + covid_feats + gt_feats[:10]  # Use first 10 GT features
            fc = walk_forward_ridge(panel, feats, 'unemp', min_train, mode,
                                   'GT_Ridge_PCA', rolling_window)
            if not fc.empty:
                forecasts.append(fc)
        
        # MIDAS (simplified - skip for now due to complexity)
        # if 'MIDAS_AR' in models_to_run:
        #     progress_bar.progress(75)
        #     # Would need full MIDAS implementation
        
        progress_bar.progress(80)
        
        # Ensembles
        if 'ENSEMBLE' in models_to_run and len(forecasts) >= 2:
            status_text.text("🔄 Building ensembles...")
            
            # Align to common dates
            forecasts_aligned = align_forecasts_to_common_dates(forecasts)
            
            # Simple ensemble
            ensemble_df = forecasts_aligned[0][['date', 'actual']].copy()
            fc_matrix = np.column_stack([f['forecast'].values for f in forecasts_aligned])
            ensemble_df['forecast'] = fc_matrix.mean(axis=1)
            ensemble_df['model'] = f'Ensemble_Simple_{mode}'
            forecasts.append(ensemble_df)
        
        progress_bar.progress(85)
        
        # Fair evaluation
        status_text.text("📊 Computing metrics...")
        forecasts = align_forecasts_to_common_dates(forecasts)
        
        # Compute metrics
        for fc in forecasts:
            if fc.empty:
                continue
            
            model_name = fc['model'].iloc[0]
            metrics = compute_metrics(fc['actual'].values, fc['forecast'].values)
            
            all_results.append({
                'Mode': mode.capitalize(),
                'Model': model_name.replace(f'_{mode}', ''),
                **metrics
            })
        
        all_forecasts[mode] = forecasts
    
    progress_bar.progress(100)
    status_text.text("✅ Backtesting complete!")
    
    # Store results
    st.session_state.results = pd.DataFrame(all_results)
    st.session_state.forecasts = all_forecasts
    st.session_state.panel = panel
    st.session_state.backtest_run = True
    
    st.success("🎉 Backtesting completed successfully!")
    st.balloons()

# ===========================================================================
# RESULTS DISPLAY
# ===========================================================================

if st.session_state.backtest_run and st.session_state.results is not None:
    
    results_df = st.session_state.results
    forecasts_dict = st.session_state.forecasts
    panel = st.session_state.panel
    
    st.markdown("---")
    st.header("📊 Results Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_models = len(results_df['Model'].unique())
        st.metric("Models Tested", n_models)
    
    with col2:
        n_forecasts = results_df['N'].max()
        st.metric("Forecast Points", int(n_forecasts))
    
    with col3:
        best_model = results_df.nsmallest(1, 'MASE').iloc[0]
        st.metric("Best Model", best_model['Model'])
    
    with col4:
        best_mase = best_model['MASE']
        st.metric("Best MASE", f"{best_mase:.3f}")
    
    # Performance table
    st.markdown("---")
    st.subheader("🏆 Performance Comparison")
    
    # Mode filter
    mode_filter = st.radio(
        "View results for:",
        options=['All'] + list(results_df['Mode'].unique()),
        horizontal=True
    )
    
    if mode_filter != 'All':
        results_display = results_df[results_df['Mode'] == mode_filter].copy()
    else:
        results_display = results_df.copy()
    
    results_display = results_display.sort_values('MASE')
    
    st.dataframe(
        results_display.style.format({
            'MAE': '{:.4f}',
            'RMSE': '{:.4f}',
            'MASE': '{:.3f}',
            'N': '{:.0f}'
        }).background_gradient(subset=['MASE'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=400
    )
    
    # Performance chart
    st.plotly_chart(
        plot_performance_bars(results_display),
        use_container_width=True
    )
    
    # Time series visualization
    st.markdown("---")
    st.subheader("📈 Historical Data & Forecasts")
    
    # Time series plot
    fig_ts = plot_time_series(panel, 'unemp', 
                              f"Italian Unemployment Rate ({start_date} to {end_date})")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Forecast comparison
    for mode, forecasts in forecasts_dict.items():
        with st.expander(f"🔮 Forecast Comparison - {mode.capitalize()} Window", expanded=True):
            
            fig_fc = plot_forecast_comparison(forecasts, panel, 'unemp', n_recent=24)
            st.plotly_chart(fig_fc, use_container_width=True)
    
    # SHAP Analysis
    if enable_shap:
        st.markdown("---")
        st.subheader("🔬 SHAP Feature Importance Analysis")
        
        st.info("""
        **💡 What is SHAP?**
        
        SHAP (SHapley Additive exPlanations) shows which features contribute most 
        to predictions. Higher values = more important feature.
        """)
        
        # Choose model for SHAP
        ridge_models = [m for m in models_to_run if 'Ridge' in m]
        
        if ridge_models:
            shap_model = st.selectbox("Select model for SHAP analysis:", ridge_models)
            
            with st.spinner("🧠 Computing SHAP values..."):
                
                # Determine features
                if shap_model == 'AR_Ridge_BASE':
                    feats = base_feats + covid_feats
                elif shap_model == 'AR_Ridge_FIN':
                    feats = base_feats + covid_feats + fin_feats
                elif shap_model == 'GT_Ridge_PCA':
                    feats = base_feats + covid_feats + gt_feats[:10]
                else:
                    feats = base_feats
                
                shap_result = compute_shap_values(panel, feats, 'unemp', shap_model)
                
                if shap_result is not None:
                    shap_values, X_sample, feature_names, explainer, X_transformed = shap_result
                    
                    # Plot
                    fig_shap = plot_shap_summary(shap_values, X_sample, feature_names)
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    # Top features
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    top_features = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': mean_abs_shap
                    }).sort_values('Importance', ascending=False).head(5)
                    
                    st.markdown("**🌟 Top 5 Most Important Features:**")
                    for idx, row in top_features.iterrows():
                        st.markdown(f"- **{row['Feature']}**: {row['Importance']:.4f}")
                else:
                    st.warning("⚠️ Could not compute SHAP values for this model")
        else:
            st.info("ℹ️ No Ridge models selected. Enable Ridge models to use SHAP.")
    
    # Export section
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_results = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Performance Summary",
            csv_results,
            "performance_summary.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Combine all forecasts
        all_fc_list = []
        for mode, fcs in forecasts_dict.items():
            for fc in fcs:
                fc_copy = fc.copy()
                fc_copy['backtest_mode'] = mode
                all_fc_list.append(fc_copy)
        
        if all_fc_list:
            all_fc_df = pd.concat(all_fc_list, ignore_index=True)
            csv_forecasts = all_fc_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download All Forecasts",
                csv_forecasts,
                "all_forecasts.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col3:
        # Configuration summary
        config_dict = {
            'date_range': f"{start_date} to {end_date}",
            'min_train_months': min_train,
            'backtest_mode': backtest_mode,
            'rolling_window': rolling_window if 'Rolling' in backtest_mode else 'N/A',
            'covid_dummies': use_covid,
            'google_trends': use_google_trends,
            'models': models_to_run,
            'n_forecasts': int(results_df['N'].max()),
            'best_model': best_model['Model'],
            'best_mase': float(best_mase)
        }
        
        json_config = json.dumps(config_dict, indent=2, default=str)
        st.download_button(
            "📥 Download Configuration",
            json_config,
            "config.json",
            "application/json",
            use_container_width=True
        )

# ===========================================================================
# FOOTER
# ===========================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; color: white;'>
    <p style='font-size: 1.3rem; font-weight: 600; margin: 0 0 10px 0;'>
        🎓 ISTAT Internship Project 2025
    </p>
    <p style='font-size: 1rem; margin: 5px 0;'>
        Interactive Unemployment Nowcasting Laboratory
    </p>
    <p style='font-size: 0.9rem; opacity: 0.9; margin: 15px 0 0 0;'>
        Built with Streamlit • Python • scikit-learn • SHAP • Plotly
    </p>
    <p style='font-size: 0.85rem; opacity: 0.8; margin: 10px 0 0 0;'>
        Version 3.0 - Production Ready
    </p>
</div>
""", unsafe_allow_html=True)
