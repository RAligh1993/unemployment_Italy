"""
🏠 ISTAT Nowcasting Lab - Executive Home v2.0
=================================================
Professional command center for Italian unemployment nowcasting.
Features: Quick start, system status, data quality, model performance.

Author: AI Assistant
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import os
from typing import Optional, List, Dict, Tuple

# =============================================================================
# APP CONFIGURATION
# =============================================================================

APP_NAME = "ISTAT Nowcasting Lab"
APP_VERSION = "2.0.0"
APP_SUBTITLE = "Italian Unemployment Nowcasting System"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #0F766E, #14B8A6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1.25rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    .section-header {
        background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);
        color: white;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);
        padding: 1.75rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .status-good {
        background: #D1FAE5;
        color: #065F46;
        border: 2px solid #10B981;
    }
    
    .status-warn {
        background: #FEF3C7;
        color: #92400E;
        border: 2px solid #F59E0B;
    }
    
    .status-error {
        background: #FEE2E2;
        color: #991B1B;
        border: 2px solid #EF4444;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .quick-action {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #E5E7EB;
        text-align: center;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    .quick-action:hover {
        border-color: #0F766E;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.15);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.25);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(15, 118, 110, 0.35);
    }
    
    .progress-container {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .step-complete {
        background: #D1FAE5;
        border-left: 4px solid #10B981;
    }
    
    .step-active {
        background: #DBEAFE;
        border-left: 4px solid #3B82F6;
    }
    
    .step-pending {
        background: #F3F4F6;
        border-left: 4px solid #9CA3AF;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

try:
    from utils.state import AppState
except Exception:
    class _State:
        def __init__(self):
            self.y_monthly: Optional[pd.Series] = None
            self.panel_monthly: Optional[pd.DataFrame] = None
            self.panel_quarterly: Optional[pd.DataFrame] = None
            self.bt_results: Dict[str, pd.Series] = {}
            self.bt_metrics: Optional[pd.DataFrame] = None
            self.raw_daily: List[pd.DataFrame] = []
            self.google_trends: Optional[pd.DataFrame] = None
    
    class AppState:
        @staticmethod
        def init():
            if "_app" not in st.session_state:
                st.session_state["_app"] = _State()
            return st.session_state["_app"]
        
        @staticmethod
        def get():
            return AppState.init()

state = AppState.init()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_system_status() -> Dict[str, str]:
    """Get overall system status"""
    status = {
        'target': 'none',
        'panel': 'none',
        'features': 'none',
        'models': 'none'
    }
    
    # Check target
    try:
        if state.y_monthly is not None and not state.y_monthly.empty:
            status['target'] = 'ready'
    except:
        pass
    
    # Check panel
    try:
        if state.panel_monthly is not None and not state.panel_monthly.empty:
            status['panel'] = 'ready'
            if state.panel_monthly.shape[1] > 10:
                status['features'] = 'ready'
    except:
        pass
    
    # Check models - FIX: Safe check for bt_results
    try:
        if hasattr(state, 'bt_results') and state.bt_results is not None:
            if isinstance(state.bt_results, dict) and len(state.bt_results) > 0:
                status['models'] = 'ready'
    except:
        pass
    
    return status

def calculate_progress() -> int:
    """Calculate overall progress percentage"""
    try:
        status = get_system_status()
        completed = sum(1 for v in status.values() if v == 'ready')
        return int((completed / len(status)) * 100)
    except:
        return 0

def get_api_status() -> Dict[str, bool]:
    """Check API key availability"""
    try:
        return {
            'OpenAI': bool(os.getenv('OPENAI_API_KEY')),
            'Anthropic': bool(os.getenv('ANTHROPIC_API_KEY')),
            'Google': bool(os.getenv('GOOGLE_API_KEY')),
            'NewsAPI': bool(os.getenv('NEWSAPI_KEY'))
        }
    except:
        return {
            'OpenAI': False,
            'Anthropic': False,
            'Google': False,
            'NewsAPI': False
        }

def detect_date_col(df: pd.DataFrame) -> str:
    """Smart date column detection"""
    for c in ['date', 'Date', 'ds', 'time', 'Time', 'period', 'Week', 'Month']:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return df.columns[0]

def to_datetime_safe(s) -> pd.Series:
    """Safe datetime conversion"""
    return pd.to_datetime(pd.Series(s), errors='coerce').dt.tz_localize(None).dt.normalize()

def end_of_month(s: pd.Series) -> pd.Series:
    """Convert to month-end"""
    return (to_datetime_safe(s) + pd.offsets.MonthEnd(0)).dt.normalize()

def load_target(file) -> Optional[pd.Series]:
    """Load monthly target"""
    try:
        df = pd.read_csv(file)
        dcol = detect_date_col(df)
        vcol = [c for c in df.columns if c != dcol][0]
        df = df[[dcol, vcol]].copy()
        df.columns = ['date', 'y']
        df['date'] = end_of_month(df['date'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        return df.dropna().set_index('date')['y'].sort_index()
    except Exception as e:
        st.error(f"Error loading target: {str(e)}")
        return None

def load_daily(file) -> Optional[pd.DataFrame]:
    """Load daily data"""
    try:
        df = pd.read_csv(file)
        dcol = detect_date_col(df)
        df = df.rename(columns={dcol: 'date'})
        df['date'] = to_datetime_safe(df['date'])
        for c in df.columns:
            if c != 'date':
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['date']).sort_values('date')
    except Exception as e:
        st.error(f"Error loading daily: {str(e)}")
        return None

def load_google_trends(files: List) -> pd.DataFrame:
    """Load Google Trends"""
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            date_col = 'Week' if 'Week' in df.columns else ('Month' if 'Month' in df.columns else detect_date_col(df))
            df = df.rename(columns={date_col: 'date'})
            df['date'] = to_datetime_safe(df['date'])
            frames.append(df)
        except:
            continue
    
    if not frames:
        return pd.DataFrame()
    
    result = frames[0]
    for df in frames[1:]:
        result = pd.merge(result, df, on='date', how='outer')
    
    return result.sort_values('date').reset_index(drop=True)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_progress_chart(progress: int) -> go.Figure:
    """Create progress donut chart"""
    fig = go.Figure(data=[go.Pie(
        values=[progress, 100-progress],
        hole=0.7,
        marker=dict(colors=['#0F766E', '#F3F4F6']),
        textinfo='none',
        hoverinfo='none'
    )])
    
    fig.update_layout(
        showlegend=False,
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(
            text=f'{progress}%',
            x=0.5, y=0.5,
            font=dict(size=32, color='#0F766E', family='Inter'),
            showarrow=False
        )]
    )
    
    return fig

def create_mini_sparkline(series: pd.Series, color: str = '#0F766E') -> go.Figure:
    """Create mini sparkline"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba(15, 118, 110, 0.1)'
    ))
    
    fig.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# =============================================================================
# SIDEBAR - Simple version (Streamlit handles navigation automatically)
# =============================================================================

with st.sidebar:
    # Version badge
    st.markdown(f"""
    <div style='text-align: center; padding: 0.75rem; background: linear-gradient(135deg, #0F766E, #14B8A6); 
                color: white; border-radius: 10px; font-weight: 700; font-size: 1rem; margin-bottom: 1.5rem;'>
        {APP_NAME}<br/>
        <span style='font-size: 0.875rem; opacity: 0.9;'>v{APP_VERSION}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # API Status
    st.markdown("### 🔑 API Status")
    api_status = get_api_status()
    
    for api, available in api_status.items():
        status_color = "#10B981" if available else "#EF4444"
        status_icon = "✓" if available else "✗"
        st.markdown(f"""
        <div style='display: flex; justify-content: space-between; align-items: center; 
                    padding: 0.5rem; margin: 0.25rem 0; background: #F9FAFB; border-radius: 6px;'>
            <span style='color: #475569; font-weight: 500;'>{api}</span>
            <span style='color: {status_color}; font-weight: 700; font-size: 1.1rem;'>{status_icon}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Status Summary
    try:
        system_status = get_system_status()
        progress = calculate_progress()
        
        st.markdown("### 📊 Quick Status")
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: #F9FAFB; border-radius: 8px;'>
            <div style='font-size: 2rem; font-weight: 700; color: #0F766E;'>{progress}%</div>
            <div style='color: #64748B; font-size: 0.875rem;'>Complete</div>
        </div>
        """, unsafe_allow_html=True)
    except:
        pass

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Hero Section
st.markdown(f'<h1 class="hero-title">{APP_NAME}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="hero-subtitle">{APP_SUBTITLE}</p>', unsafe_allow_html=True)

# =============================================================================
# SYSTEM STATUS OVERVIEW
# =============================================================================

st.markdown('<div class="section-header">🎯 System Status</div>', unsafe_allow_html=True)

system_status = get_system_status()
progress = calculate_progress()

col1, col2, col3, col4, col5 = st.columns(5)

# Progress Circle
with col1:
    progress_fig = create_progress_chart(progress)
    st.plotly_chart(progress_fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.875rem;'>Overall Progress</p>", unsafe_allow_html=True)

# Status Cards
status_mapping = {
    'target': ('Target Data', '🎯'),
    'panel': ('Panel Built', '🧱'),
    'features': ('Features Ready', '🧪'),
    'models': ('Models Trained', '🤖')
}

for col, (key, (label, emoji)) in zip([col2, col3, col4, col5], status_mapping.items()):
    with col:
        status = system_status[key]
        
        if status == 'ready':
            badge_class = 'status-good'
            status_text = 'Ready'
        else:
            badge_class = 'status-warn'
            status_text = 'Pending'
        
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem;'>
            <div style='font-size: 2.5rem;'>{emoji}</div>
            <div style='margin: 0.5rem 0; font-weight: 600; color: #1F2937;'>{label}</div>
            <div class='status-badge {badge_class}'>{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# WORKFLOW STEPS
# =============================================================================

st.markdown('<div class="section-header">📋 Workflow Status</div>', unsafe_allow_html=True)

steps = [
    {
        'name': '1. Upload Target Data',
        'status': 'complete' if system_status['target'] == 'ready' else 'pending',
        'page': 'pages/2_Data_Aggregation.py'
    },
    {
        'name': '2. Build Monthly Panel',
        'status': 'complete' if system_status['panel'] == 'ready' else 'active' if system_status['target'] == 'ready' else 'pending',
        'page': 'pages/2_Data_Aggregation.py'
    },
    {
        'name': '3. Engineer Features',
        'status': 'complete' if system_status['features'] == 'ready' else 'active' if system_status['panel'] == 'ready' else 'pending',
        'page': 'pages/3_Feature_Engineering.py'
    },
    {
        'name': '4. Train & Backtest Models',
        'status': 'complete' if system_status['models'] == 'ready' else 'active' if system_status['features'] == 'ready' else 'pending',
        'page': 'pages/4_Backtesting.py'
    }
]

for step in steps:
    status_class = f"step-{step['status']}"
    
    if step['status'] == 'complete':
        icon = '✅'
    elif step['status'] == 'active':
        icon = '🔄'
    else:
        icon = '⭕'
    
    st.markdown(f"""
    <div class='step-indicator {status_class}'>
        <span style='font-size: 1.5rem;'>{icon}</span>
        <span style='flex: 1; font-weight: 600;'>{step['name']}</span>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# QUICK START
# =============================================================================

st.markdown('<div class="section-header">🚀 Quick Start</div>', unsafe_allow_html=True)

st.markdown("""
Upload your data to get started with nowcasting Italian unemployment.
All files should be CSV format with proper date columns.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🎯 Monthly Target")
    target_file = st.file_uploader(
        "Upload monthly unemployment data",
        type=['csv'],
        key='target_home',
        help="CSV with columns: date, value"
    )

with col2:
    st.markdown("#### 📊 Daily Data")
    daily_files = st.file_uploader(
        "Upload daily financial data",
        type=['csv'],
        accept_multiple_files=True,
        key='daily_home',
        help="Stock prices, VIX, etc."
    )

with col3:
    st.markdown("#### 🔍 Google Trends")
    trends_files = st.file_uploader(
        "Upload Google Trends data",
        type=['csv'],
        accept_multiple_files=True,
        key='trends_home',
        help="Weekly/monthly trends data"
    )

# Process uploads
if target_file:
    with st.spinner("Loading target data..."):
        state.y_monthly = load_target(target_file)
        if state.y_monthly is not None:
            st.success(f"✅ Loaded {len(state.y_monthly)} months of data")

if daily_files:
    with st.spinner("Loading daily data..."):
        state.raw_daily = []
        for f in daily_files:
            df = load_daily(f)
            if df is not None:
                state.raw_daily.append(df)
        st.success(f"✅ Loaded {len(state.raw_daily)} daily files")

if trends_files:
    with st.spinner("Loading Google Trends..."):
        state.google_trends = load_google_trends(trends_files)
        if not state.google_trends.empty:
            st.success(f"✅ Loaded {state.google_trends.shape[1]-1} trend series")

# =============================================================================
# DATA PREVIEW
# =============================================================================

if hasattr(state, 'y_monthly') and state.y_monthly is not None and not state.y_monthly.empty:
    st.markdown('<div class="section-header">📈 Data Preview</div>', unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Target Time Series")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=state.y_monthly.index,
                y=state.y_monthly.values,
                mode='lines+markers',
                name='Unemployment Rate',
                line=dict(color='#0F766E', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title='Italian Unemployment Rate',
                xaxis_title='Date',
                yaxis_title='Rate (%)',
                template='plotly_white',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Statistics")
            
            stats = {
                'Observations': len(state.y_monthly),
                'Start Date': state.y_monthly.index.min().strftime('%Y-%m'),
                'End Date': state.y_monthly.index.max().strftime('%Y-%m'),
                'Mean': f"{state.y_monthly.mean():.2f}%",
                'Std Dev': f"{state.y_monthly.std():.2f}%",
                'Min': f"{state.y_monthly.min():.2f}%",
                'Max': f"{state.y_monthly.max():.2f}%"
            }
            
            for label, value in stats.items():
                st.markdown(f"""
                <div style='padding: 0.5rem; margin: 0.25rem 0; background: #F9FAFB; border-radius: 6px;'>
                    <div style='color: #64748B; font-size: 0.75rem; text-transform: uppercase;'>{label}</div>
                    <div style='color: #1F2937; font-size: 1.25rem; font-weight: 700;'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Could not display data preview: {str(e)}")

# =============================================================================
# MODEL PERFORMANCE SUMMARY
# =============================================================================

if hasattr(state, 'bt_metrics') and state.bt_metrics is not None and not state.bt_metrics.empty:
    st.markdown('<div class="section-header">🏆 Model Performance</div>', unsafe_allow_html=True)
    
    try:
        # Best models
        top_models = state.bt_metrics.nsmallest(3, 'MAE')
        
        col1, col2, col3 = st.columns(3)
        
        for i, (col, (_, model)) in enumerate(zip([col1, col2, col3], top_models.iterrows())):
            with col:
                rank_emoji = ['🥇', '🥈', '🥉'][i]
                
                st.markdown(f"""
                <div class='info-card'>
                    <div style='font-size: 2rem; text-align: center;'>{rank_emoji}</div>
                    <div style='text-align: center; font-weight: 700; font-size: 1.1rem; margin: 0.5rem 0;'>
                        {model.get('model', 'Unknown')}
                    </div>
                    <div style='display: flex; justify-content: space-between; margin-top: 1rem;'>
                        <div>
                            <div style='color: #64748B; font-size: 0.75rem;'>MAE</div>
                            <div style='font-weight: 700;'>{model.get('MAE', 0):.4f}</div>
                        </div>
                        <div>
                            <div style='color: #64748B; font-size: 0.75rem;'>RMSE</div>
                            <div style='font-weight: 700;'>{model.get('RMSE', 0):.4f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Full metrics table
        st.markdown("#### All Models")
        
        format_dict = {}
        if 'MAE' in state.bt_metrics.columns:
            format_dict['MAE'] = '{:.4f}'
        if 'RMSE' in state.bt_metrics.columns:
            format_dict['RMSE'] = '{:.4f}'
        if 'SMAPE' in state.bt_metrics.columns:
            format_dict['SMAPE'] = '{:.2f}'
        if 'MASE' in state.bt_metrics.columns:
            format_dict['MASE'] = '{:.4f}'
        
        styled_df = state.bt_metrics.style.format(format_dict)
        if 'MAE' in state.bt_metrics.columns:
            styled_df = styled_df.background_gradient(subset=['MAE'], cmap='RdYlGn_r')
        
        st.dataframe(styled_df, use_container_width=True)
    
    except Exception as e:
        st.info("Model metrics available but couldn't display. Check data format.")

# =============================================================================
# QUICK ACTIONS
# =============================================================================

st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 View Dashboard", use_container_width=True, key="btn_dashboard"):
        st.switch_page("pages/1_Dashboard.py")

with col2:
    if st.button("🧱 Build Panel", use_container_width=True, key="btn_panel"):
        st.switch_page("pages/2_Data_Aggregation.py")

with col3:
    if st.button("🧪 Engineer Features", use_container_width=True, key="btn_features"):
        st.switch_page("pages/3_Feature_Engineering.py")

with col4:
    if st.button("🧮 Run Backtest", use_container_width=True, key="btn_backtest"):
        st.switch_page("pages/4_Backtesting.py")

# =============================================================================
# HELP & DOCUMENTATION
# =============================================================================

with st.expander("📚 Quick Help & Documentation", expanded=False):
    st.markdown("""
    ### Getting Started
    
    1. **Upload Data**: Start by uploading your monthly target and daily financial data
    2. **Build Panel**: Aggregate daily data to monthly frequency
    3. **Engineer Features**: Create lags, rolling statistics, and transforms
    4. **Run Backtest**: Train models using walk-forward validation
    5. **Analyze Results**: View performance metrics and visualizations
    
    ### Data Requirements
    
    - **Monthly Target**: CSV with `date` and `value` columns
    - **Daily Data**: CSV with `date` column and numeric features
    - **Google Trends**: CSV with `Week`/`Month` and trend values
    
    ### Supported Models
    
    - NAIVE, MA3, MA12: Benchmark models
    - ETS: Exponential smoothing
    - Ridge: Regularized regression
    - MIDAS: Mixed-data sampling
    - Ensembles: Combined predictions
    
    ### API Keys (Optional)
    
    Set environment variables for AI features:
    - `OPENAI_API_KEY`: For GPT models
    - `ANTHROPIC_API_KEY`: For Claude
    - `GOOGLE_API_KEY`: For Gemini
    - `NEWSAPI_KEY`: For news analysis
    """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"📅 Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

with col2:
    st.caption("🇮🇹 ISTAT Unemployment Nowcasting")

with col3:
    st.caption("💻 Built with Streamlit Pro")

st.markdown("""
<div style='text-align: center; color: #94A3B8; font-size: 0.875rem; margin-top: 2rem;'>
    © 2025 Nowcasting Lab · Experimental Research Tool · Use Responsibly
</div>
""", unsafe_allow_html=True)
