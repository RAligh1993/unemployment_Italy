"""
🧱 Data Aggregation Pro v2.0
================================
Professional data intake & panel builder for time series forecasting.
Features: Multi-source upload, smart aggregation, automated cleaning, visual diagnostics.

Author: AI Assistant
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import json
from io import BytesIO
from typing import Optional, List, Dict, Tuple

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
# HELPER FUNCTIONS
# =============================================================================

def to_datetime_safe(series: pd.Series) -> pd.Series:
    """Convert to datetime and normalize to date only (tz-naive)"""
    return pd.to_datetime(series, errors='coerce').dt.tz_localize(None).dt.normalize()

def end_of_month(series: pd.Series) -> pd.Series:
    """Align dates to end of month"""
    dt = to_datetime_safe(series)
    return (dt + pd.offsets.MonthEnd(0)).dt.normalize()

def slugify(name: str) -> str:
    """Convert name to clean column identifier"""
    return (name.strip().lower()
            .replace(' ', '_')
            .replace('/', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('-', '_')
            .replace('%', 'pct')
            .replace('__', '_'))

def detect_date_column(df: pd.DataFrame) -> str:
    """Smart detection of date column"""
    # Common date column names
    for name in ['date', 'Date', 'ds', 'time', 'Time', 'period', 'Week', 'Month', 'Day']:
        if name in df.columns:
            return name
    
    # Check for datetime types
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    # Default to first column
    return df.columns[0]

def validate_dataframe(df: pd.DataFrame, name: str) -> Tuple[bool, str]:
    """Validate uploaded dataframe"""
    if df is None or df.empty:
        return False, f"{name} is empty"
    
    if len(df.columns) < 2:
        return False, f"{name} must have at least 2 columns (date + value)"
    
    return True, "Valid"

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_target_series(file) -> Optional[pd.Series]:
    """Load monthly target variable from CSV"""
    try:
        df = pd.read_csv(file)
        
        valid, msg = validate_dataframe(df, "Target file")
        if not valid:
            st.error(f"❌ {msg}")
            return None
        
        # Detect columns
        date_col = detect_date_column(df)
        value_cols = [c for c in df.columns if c != date_col]
        
        if not value_cols:
            st.error("❌ No value column found in target file")
            return None
        
        value_col = value_cols[0]
        
        # Process data
        df = df[[date_col, value_col]].copy()
        df.columns = ['date', 'value']
        df['date'] = end_of_month(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Remove duplicates and NaN
        df = df.dropna()
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        series = df.set_index('date')['value'].sort_index()
        
        return series
    
    except Exception as e:
        st.error(f"❌ Error loading target: {str(e)}")
        return None

def load_daily_data(file) -> Optional[pd.DataFrame]:
    """Load daily time series data from CSV"""
    try:
        df = pd.read_csv(file)
        
        valid, msg = validate_dataframe(df, "Daily file")
        if not valid:
            st.warning(f"⚠️ {msg}")
            return None
        
        # Detect date column
        date_col = detect_date_column(df)
        df = df.rename(columns={date_col: 'date'})
        
        # Convert date
        df['date'] = to_datetime_safe(df['date'])
        df = df.dropna(subset=['date'])
        
        # Process numeric columns
        numeric_cols = []
        for col in df.columns:
            if col == 'date':
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().sum() > 0:
                numeric_cols.append(col)
        
        if not numeric_cols:
            st.warning(f"⚠️ No valid numeric columns in {file.name}")
            return None
        
        # Keep only date and numeric columns
        df = df[['date'] + numeric_cols].sort_values('date')
        
        # Rename columns with file prefix
        file_prefix = slugify(Path(file.name).stem)
        rename_map = {col: f"{file_prefix}__{slugify(col)}" for col in numeric_cols}
        df = df.rename(columns=rename_map)
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading {file.name}: {str(e)}")
        return None

def load_google_trends(files: List) -> Optional[pd.DataFrame]:
    """Load and merge multiple Google Trends CSV files"""
    if not files:
        return None
    
    frames = []
    
    for file in files:
        try:
            df = pd.read_csv(file)
            
            # Detect date column (Week or Month for GT)
            date_col = 'Week' if 'Week' in df.columns else ('Month' if 'Month' in df.columns else detect_date_column(df))
            
            # Get series columns
            series_cols = [c for c in df.columns if c != date_col]
            
            if not series_cols:
                st.warning(f"⚠️ No data columns in {file.name}")
                continue
            
            # Process
            df = df[[date_col] + series_cols].copy()
            df = df.rename(columns={date_col: 'date'})
            df['date'] = to_datetime_safe(df['date'])
            
            # Rename with gt__ prefix
            new_cols = [f"gt__{slugify(col)}" for col in series_cols]
            df.columns = ['date'] + new_cols
            
            # Convert to numeric
            for col in new_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            frames.append(df)
        
        except Exception as e:
            st.warning(f"⚠️ Error loading {file.name}: {str(e)}")
            continue
    
    if not frames:
        return None
    
    # Merge all Google Trends files
    result = frames[0]
    for df in frames[1:]:
        result = pd.merge(result, df, on='date', how='outer')
    
    return result.sort_values('date').reset_index(drop=True)

# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_to_monthly(
    df: pd.DataFrame,
    method: str = 'mean',
    business_days_only: bool = False,
    min_days: int = 10
) -> pd.DataFrame:
    """Aggregate daily data to monthly with various methods"""
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df['date'] = to_datetime_safe(df['date'])
    
    # Filter business days if requested
    if business_days_only:
        df = df[df['date'].dt.dayofweek < 5]
    
    # Set index and resample
    df = df.set_index('date')
    
    if method == 'sum':
        monthly = df.resample('M').sum(min_count=1)
    elif method == 'last':
        monthly = df.resample('M').last()
    else:  # mean
        monthly = df.resample('M').mean()
    
    # Apply minimum days filter
    counts = df.resample('M').count()
    monthly[counts < min_days] = np.nan
    
    # Reset index and align to EOM
    monthly = monthly.reset_index()
    monthly['date'] = end_of_month(monthly['date'])
    
    return monthly

def build_panel(
    daily_frames: List[pd.DataFrame],
    trends_df: Optional[pd.DataFrame],
    method: str,
    business_days: bool,
    min_days: int
) -> pd.DataFrame:
    """Build unified monthly panel from all data sources"""
    
    panel = None
    
    # Process daily files
    for df in daily_frames:
        if df is None or df.empty:
            continue
        
        monthly = aggregate_to_monthly(df, method, business_days, min_days)
        
        if panel is None:
            panel = monthly
        else:
            panel = pd.merge(panel, monthly, on='date', how='outer')
    
    # Process Google Trends
    if trends_df is not None and not trends_df.empty:
        # GT data is already weekly/monthly, resample to monthly mean
        gt_monthly = trends_df.set_index('date').resample('M').mean().reset_index()
        gt_monthly['date'] = end_of_month(gt_monthly['date'])
        
        if panel is None:
            panel = gt_monthly
        else:
            panel = pd.merge(panel, gt_monthly, on='date', how='outer')
    
    if panel is None or panel.empty:
        return pd.DataFrame()
    
    # Finalize
    panel = panel.sort_values('date').set_index('date')
    
    # Ensure all columns are numeric
    for col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors='coerce')
    
    return panel

def create_quarterly_panel(monthly_panel: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    """Aggregate monthly panel to quarterly"""
    
    if monthly_panel is None or monthly_panel.empty:
        return pd.DataFrame()
    
    if method == 'last':
        quarterly = monthly_panel.resample('Q').last()
    elif method == 'sum':
        quarterly = monthly_panel.resample('Q').sum(min_count=1)
    else:
        quarterly = monthly_panel.resample('Q').mean()
    
    return quarterly

# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def remove_constant_columns(df: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    """Remove columns with zero or near-zero variance"""
    
    if df is None or df.empty:
        return df
    
    keep_cols = []
    removed_cols = []
    
    for col in df.columns:
        values = df[col].dropna().values
        
        if len(values) == 0:
            removed_cols.append(col)
            continue
        
        if np.ptp(values) > tolerance:  # peak-to-peak range
            keep_cols.append(col)
        else:
            removed_cols.append(col)
    
    if removed_cols:
        st.info(f"🧹 Removed {len(removed_cols)} constant columns")
    
    return df[keep_cols]

def remove_correlated_duplicates(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove highly correlated duplicate columns"""
    
    if df is None or df.empty:
        return df
    
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Get upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns to drop
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    if to_drop:
        st.info(f"🧹 Removed {len(to_drop)} highly correlated columns (>{threshold:.0%})")
    
    return df.drop(columns=to_drop, errors='ignore')

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_coverage_heatmap(df: pd.DataFrame, n_months: int = 60):
    """Create presence/absence heatmap"""
    
    presence = df.notna().astype(int).tail(n_months)
    
    fig = px.imshow(
        presence.T,
        aspect='auto',
        color_continuous_scale=['#EF4444', '#10B981'],
        labels={'color': 'Present'},
        title=f'Data Coverage (Last {n_months} Months)'
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        xaxis_title='Date',
        yaxis_title='Feature',
        coloraxis_showscale=True
    )
    
    return fig

def plot_target_series(series: pd.Series):
    """Plot target time series"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines+markers',
        name='Target',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Monthly Target Variable',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_correlation_with_target(panel: pd.DataFrame, target: pd.Series, top_n: int = 15):
    """Plot top correlations with target"""
    
    y_aligned, X_aligned = target.align(panel, join='inner')
    
    if X_aligned.empty:
        return None
    
    correlations = X_aligned.corrwith(y_aligned).sort_values(ascending=False)
    top_corr = pd.concat([correlations.head(top_n//2), correlations.tail(top_n//2)])
    
    fig = go.Figure()
    
    colors = ['#10B981' if x > 0 else '#EF4444' for x in top_corr.values]
    
    fig.add_trace(go.Bar(
        x=top_corr.values,
        y=top_corr.index,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{x:.3f}' for x in top_corr.values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Correlations with Target',
        xaxis_title='Correlation Coefficient',
        yaxis_title='Feature',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_excel(
    monthly_panel: pd.DataFrame,
    quarterly_panel: Optional[pd.DataFrame],
    target: Optional[pd.Series],
    config: dict
) -> bytes:
    """Export all data to Excel with multiple sheets"""
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Monthly panel
        if monthly_panel is not None and not monthly_panel.empty:
            monthly_panel.reset_index().to_excel(writer, sheet_name='Monthly_Panel', index=False)
        
        # Quarterly panel
        if quarterly_panel is not None and not quarterly_panel.empty:
            quarterly_panel.reset_index().to_excel(writer, sheet_name='Quarterly_Panel', index=False)
        
        # Target
        if target is not None and not target.empty:
            target.reset_index().to_excel(writer, sheet_name='Target', index=False)
        
        # Config
        config_df = pd.DataFrame([config])
        config_df.to_excel(writer, sheet_name='Configuration', index=False)
    
    return output.getvalue()

# =============================================================================
# UI CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Data Aggregation Pro",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(120deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .step-header {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Header
st.markdown('<h1 class="main-title">🧱 Data Aggregation Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload, aggregate, and build clean time series panels for forecasting</p>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📊 Aggregation")
    agg_method = st.selectbox(
        "Daily → Monthly method:",
        options=['mean', 'sum', 'last'],
        help="How to aggregate daily data to monthly"
    )
    
    use_business_days = st.checkbox(
        "Business days only",
        value=False,
        help="Exclude weekends before aggregation"
    )
    
    min_days = st.slider(
        "Min days per month",
        min_value=1,
        max_value=28,
        value=10,
        help="Months with fewer days will be set to NaN"
    )
    
    st.markdown("---")
    st.subheader("🧹 Cleaning")
    
    drop_constant = st.checkbox(
        "Remove constant columns",
        value=True,
        help="Drop columns with zero variance"
    )
    
    drop_correlated = st.checkbox(
        "Remove correlated duplicates",
        value=False,
        help="Drop highly correlated columns"
    )
    
    if drop_correlated:
        corr_threshold = st.slider(
            "Correlation threshold",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f"
        )
    else:
        corr_threshold = 0.95
    
    st.markdown("---")
    st.subheader("💾 Export")
    export_format = st.radio("Format:", ['CSV', 'Excel'])
    
    st.markdown("---")
    if st.button("🗑️ Clear All Data", use_container_width=True):
        state.y_monthly = None
        state.panel_monthly = None
        state.panel_quarterly = None
        state.raw_daily = []
        state.google_trends = None
        st.success("✅ Cleared!")
        st.rerun()

# =============================================================================
# STEP 1: UPLOAD DATA
# =============================================================================

st.markdown('<div class="step-header"><h2>📤 Step 1: Upload Data</h2></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 🎯 Target Variable")
    target_file = st.file_uploader(
        "Monthly target (CSV)",
        type=['csv'],
        help="CSV with date and target value columns",
        key='target'
    )
    
    if target_file:
        with st.spinner("Loading target..."):
            state.y_monthly = load_target_series(target_file)
            if state.y_monthly is not None:
                st.success(f"✅ {len(state.y_monthly)} months loaded")

with col2:
    st.markdown("##### 📊 Daily Data")
    daily_files = st.file_uploader(
        "Daily time series (CSV)",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more daily CSV files",
        key='daily'
    )
    
    if daily_files:
        with st.spinner("Loading daily files..."):
            state.raw_daily = []
            for file in daily_files:
                df = load_daily_data(file)
                if df is not None:
                    state.raw_daily.append(df)
            
            if state.raw_daily:
                total_cols = sum(len(df.columns) - 1 for df in state.raw_daily)
                st.success(f"✅ {len(state.raw_daily)} files, {total_cols} series")

with col3:
    st.markdown("##### 🔍 Google Trends")
    trends_files = st.file_uploader(
        "Google Trends (CSV)",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload Google Trends weekly/monthly data",
        key='trends'
    )
    
    if trends_files:
        with st.spinner("Loading trends..."):
            state.google_trends = load_google_trends(trends_files)
            if state.google_trends is not None:
                n_series = len(state.google_trends.columns) - 1
                st.success(f"✅ {n_series} trends loaded")

# Preview target
if state.y_monthly is not None and not state.y_monthly.empty:
    st.markdown("##### 📈 Target Preview")
    fig = plot_target_series(state.y_monthly)
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", len(state.y_monthly))
    col2.metric("Start", state.y_monthly.index.min().strftime('%Y-%m'))
    col3.metric("End", state.y_monthly.index.max().strftime('%Y-%m'))
    col4.metric("Mean", f"{state.y_monthly.mean():.2f}")

# =============================================================================
# STEP 2: BUILD PANEL
# =============================================================================

st.markdown('<div class="step-header"><h2>🔨 Step 2: Build Panel</h2></div>', unsafe_allow_html=True)

if not state.raw_daily and state.google_trends is None:
    st.info("📌 Upload daily data or Google Trends to begin")
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        build_button = st.button("🚀 Build Panel", use_container_width=True, type="primary")
    
    if build_button:
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # Step 1: Aggregate
            status.text("📊 Aggregating daily data to monthly...")
            progress_bar.progress(0.2)
            
            panel = build_panel(
                state.raw_daily,
                state.google_trends,
                agg_method,
                use_business_days,
                min_days
            )
            
            if panel.empty:
                st.error("❌ Failed to build panel. Check your data.")
                st.stop()
            
            progress_bar.progress(0.4)
            
            # Step 2: Clean
            status.text("🧹 Cleaning panel...")
            
            if drop_constant:
                panel = remove_constant_columns(panel)
            progress_bar.progress(0.6)
            
            if drop_correlated:
                panel = remove_correlated_duplicates(panel, corr_threshold)
            progress_bar.progress(0.7)
            
            # Step 3: Align with target
            if state.y_monthly is not None:
                status.text("🎯 Aligning with target period...")
                panel = panel.loc[
                    (panel.index >= state.y_monthly.index.min()) &
                    (panel.index <= state.y_monthly.index.max())
                ]
            progress_bar.progress(0.85)
            
            # Step 4: Create quarterly
            status.text("📅 Creating quarterly panel...")
            state.panel_monthly = panel
            state.panel_quarterly = create_quarterly_panel(panel, agg_method)
            progress_bar.progress(1.0)
            
            status.empty()
            progress_bar.empty()
            
            st.success(f"✅ Built panel: {panel.shape[0]} months × {panel.shape[1]} features")
        
        except Exception as e:
            st.error(f"❌ Error building panel: {str(e)}")

# =============================================================================
# STEP 3: ANALYZE & VISUALIZE
# =============================================================================

if state.panel_monthly is not None and not state.panel_monthly.empty:
    st.markdown('<div class="step-header"><h2>📊 Step 3: Analysis & Diagnostics</h2></div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Months", state.panel_monthly.shape[0])
    
    with col2:
        st.metric("📊 Features", state.panel_monthly.shape[1])
    
    with col3:
        coverage = state.panel_monthly.notna().mean().mean()
        st.metric("✅ Coverage", f"{coverage:.1%}")
    
    with col4:
        if state.panel_quarterly is not None:
            st.metric("📅 Quarters", state.panel_quarterly.shape[0])
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "🔍 Coverage Analysis", "📈 Correlations", "💾 Export"])
    
    with tab1:
        st.subheader("Monthly Panel (Last 24 Months)")
        st.dataframe(
            state.panel_monthly.tail(24).style.format("{:.4f}"),
            use_container_width=True,
            height=400
        )
        
        if state.panel_quarterly is not None and not state.panel_quarterly.empty:
            st.subheader("Quarterly Panel (Last 8 Quarters)")
            st.dataframe(
                state.panel_quarterly.tail(8).style.format("{:.4f}"),
                use_container_width=True,
                height=300
            )
    
    with tab2:
        st.subheader("Feature Coverage Analysis")
        
        # Coverage table
        coverage_df = pd.DataFrame({
            'Feature': state.panel_monthly.columns,
            'Coverage': state.panel_monthly.notna().mean().values,
            'Missing': state.panel_monthly.isna().sum().values
        }).sort_values('Coverage', ascending=False)
        
        st.dataframe(
            coverage_df.style.format({'Coverage': '{:.1%}'}),
            use_container_width=True,
            height=400
        )
        
        # Heatmap
        st.subheader("Presence Heatmap")
        heatmap_fig = plot_coverage_heatmap(state.panel_monthly)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with tab3:
        if state.y_monthly is not None:
            st.subheader("Correlation with Target")
            
            corr_fig = plot_correlation_with_target(
                state.panel_monthly,
                state.y_monthly,
                top_n=20
            )
            
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.info("No overlapping data with target")
        else:
            st.info("Upload target variable to see correlations")
    
    with tab4:
        st.subheader("Export Data")
        
        # Configuration summary
        config = {
            'aggregation_method': agg_method,
            'business_days_only': use_business_days,
            'min_days_per_month': min_days,
            'drop_constant': drop_constant,
            'drop_correlated': drop_correlated,
            'correlation_threshold': corr_threshold,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with st.expander("⚙️ Configuration", expanded=False):
            st.json(config)
        
        # Export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if export_format == 'CSV':
                csv_data = state.panel_monthly.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Download Monthly Panel (CSV)",
                    csv_data,
                    "monthly_panel.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                if state.panel_quarterly is not None:
                    csv_q = state.panel_quarterly.to_csv().encode('utf-8')
                    st.download_button(
                        "📥 Download Quarterly Panel (CSV)",
                        csv_q,
                        "quarterly_panel.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                excel_data = export_to_excel(
                    state.panel_monthly,
                    state.panel_quarterly,
                    state.y_monthly,
                    config
                )
                st.download_button(
                    "📥 Download All Data (Excel)",
                    excel_data,
                    "panel_data.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col2:
            st.info("💡 **Tip:** Excel export includes all panels, target, and configuration in separate sheets!")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if state.panel_monthly is not None:
        st.caption(f"📊 Panel ready with {state.panel_monthly.shape[1]} features")

with col2:
    st.caption("🔄 Data persisted in session state")

with col3:
    st.caption("💻 Built with Streamlit Pro")
