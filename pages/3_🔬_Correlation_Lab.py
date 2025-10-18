"""
═══════════════════════════════════════════════════════════════════════════════
📤 MANUAL DATA UPLOAD & PANEL BUILDER
═══════════════════════════════════════════════════════════════════════════════

Upload your own CSV files and build custom time series panels.

This is the COMPLETE Data Aggregation Pro from your thesis, 
integrated into the multi-page system.

═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from io import BytesIO
from typing import Optional, List, Dict, Tuple

st.set_page_config(
    page_title="Manual Upload",
    page_icon="📤",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

if 'manual_state' not in st.session_state:
    st.session_state.manual_state = {
        'y_monthly': None,
        'panel_monthly': None,
        'panel_quarterly': None,
        'raw_daily': [],
        'google_trends': None
    }

state = st.session_state.manual_state

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def to_datetime_safe(series: pd.Series) -> pd.Series:
    """Convert to datetime and normalize"""
    return pd.to_datetime(series, errors='coerce').dt.tz_localize(None).dt.normalize()

def end_of_month(series: pd.Series) -> pd.Series:
    """Align dates to end of month"""
    dt = to_datetime_safe(series)
    return (dt + pd.offsets.MonthEnd(0)).dt.normalize()

def slugify(name: str) -> str:
    """Convert name to clean identifier"""
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
    for name in ['date', 'Date', 'ds', 'time', 'Time', 'period', 'Week', 'Month', 'Day']:
        if name in df.columns:
            return name
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return df.columns[0]

def validate_dataframe(df: pd.DataFrame, name: str) -> Tuple[bool, str]:
    """Validate uploaded dataframe"""
    if df is None or df.empty:
        return False, f"{name} is empty"
    if len(df.columns) < 2:
        return False, f"{name} must have at least 2 columns"
    return True, "Valid"

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_target_series(file) -> Optional[pd.Series]:
    """Load monthly target variable"""
    try:
        df = pd.read_csv(file)
        valid, msg = validate_dataframe(df, "Target")
        if not valid:
            st.error(f"❌ {msg}")
            return None
        
        date_col = detect_date_column(df)
        value_cols = [c for c in df.columns if c != date_col]
        if not value_cols:
            st.error("❌ No value column found")
            return None
        
        df = df[[date_col, value_cols[0]]].copy()
        df.columns = ['date', 'value']
        df['date'] = end_of_month(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna().drop_duplicates(subset=['date'], keep='last')
        
        return df.set_index('date')['value'].sort_index()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def load_daily_data(file) -> Optional[pd.DataFrame]:
    """Load daily time series"""
    try:
        df = pd.read_csv(file)
        valid, msg = validate_dataframe(df, "Daily")
        if not valid:
            st.warning(f"⚠️ {msg}")
            return None
        
        date_col = detect_date_column(df)
        df = df.rename(columns={date_col: 'date'})
        df['date'] = to_datetime_safe(df['date'])
        df = df.dropna(subset=['date'])
        
        numeric_cols = []
        for col in df.columns:
            if col == 'date':
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().sum() > 0:
                numeric_cols.append(col)
        
        if not numeric_cols:
            st.warning(f"⚠️ No valid columns in {file.name}")
            return None
        
        df = df[['date'] + numeric_cols].sort_values('date')
        file_prefix = slugify(Path(file.name).stem)
        rename_map = {col: f"{file_prefix}__{slugify(col)}" for col in numeric_cols}
        df = df.rename(columns=rename_map)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading {file.name}: {str(e)}")
        return None

def load_google_trends(files: List) -> Optional[pd.DataFrame]:
    """Load and merge Google Trends files"""
    if not files:
        return None
    
    frames = []
    for file in files:
        try:
            df = pd.read_csv(file)
            date_col = 'Week' if 'Week' in df.columns else ('Month' if 'Month' in df.columns else detect_date_column(df))
            series_cols = [c for c in df.columns if c != date_col]
            
            if not series_cols:
                st.warning(f"⚠️ No data in {file.name}")
                continue
            
            df = df[[date_col] + series_cols].copy()
            df = df.rename(columns={date_col: 'date'})
            df['date'] = to_datetime_safe(df['date'])
            
            new_cols = [f"gt__{slugify(col)}" for col in series_cols]
            df.columns = ['date'] + new_cols
            
            for col in new_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            frames.append(df)
        except Exception as e:
            st.warning(f"⚠️ Error: {str(e)}")
            continue
    
    if not frames:
        return None
    
    result = frames[0]
    for df in frames[1:]:
        result = pd.merge(result, df, on='date', how='outer')
    
    return result.sort_values('date').reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_to_monthly(
    df: pd.DataFrame,
    method: str = 'mean',
    business_days_only: bool = False,
    min_days: int = 10
) -> pd.DataFrame:
    """Aggregate daily to monthly"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df['date'] = to_datetime_safe(df['date'])
    
    if business_days_only:
        df = df[df['date'].dt.dayofweek < 5]
    
    df = df.set_index('date')
    
    if method == 'sum':
        monthly = df.resample('M').sum(min_count=1)
    elif method == 'last':
        monthly = df.resample('M').last()
    else:
        monthly = df.resample('M').mean()
    
    counts = df.resample('M').count()
    monthly[counts < min_days] = np.nan
    
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
    """Build unified monthly panel"""
    panel = None
    
    for df in daily_frames:
        if df is None or df.empty:
            continue
        monthly = aggregate_to_monthly(df, method, business_days, min_days)
        if panel is None:
            panel = monthly
        else:
            panel = pd.merge(panel, monthly, on='date', how='outer')
    
    if trends_df is not None and not trends_df.empty:
        gt_monthly = trends_df.set_index('date').resample('M').mean().reset_index()
        gt_monthly['date'] = end_of_month(gt_monthly['date'])
        if panel is None:
            panel = gt_monthly
        else:
            panel = pd.merge(panel, gt_monthly, on='date', how='outer')
    
    if panel is None or panel.empty:
        return pd.DataFrame()
    
    panel = panel.sort_values('date').set_index('date')
    for col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors='coerce')
    
    return panel

def create_quarterly_panel(monthly_panel: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    """Aggregate to quarterly"""
    if monthly_panel is None or monthly_panel.empty:
        return pd.DataFrame()
    
    if method == 'last':
        return monthly_panel.resample('Q').last()
    elif method == 'sum':
        return monthly_panel.resample('Q').sum(min_count=1)
    else:
        return monthly_panel.resample('Q').mean()

# ═══════════════════════════════════════════════════════════════════════════
# CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def remove_constant_columns(df: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    """Remove zero variance columns"""
    if df is None or df.empty:
        return df
    
    keep_cols = []
    removed_cols = []
    
    for col in df.columns:
        values = df[col].dropna().values
        if len(values) == 0:
            removed_cols.append(col)
            continue
        if np.ptp(values) > tolerance:
            keep_cols.append(col)
        else:
            removed_cols.append(col)
    
    if removed_cols:
        st.info(f"🧹 Removed {len(removed_cols)} constant columns")
    
    return df[keep_cols]

def remove_correlated_duplicates(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove highly correlated columns"""
    if df is None or df.empty:
        return df
    
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    if to_drop:
        st.info(f"🧹 Removed {len(to_drop)} correlated columns (>{threshold:.0%})")
    
    return df.drop(columns=to_drop, errors='ignore')

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_target_series(series: pd.Series):
    """Plot target"""
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

def plot_coverage_heatmap(df: pd.DataFrame, n_months: int = 60):
    """Coverage heatmap"""
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
        yaxis_title='Feature'
    )
    return fig

def plot_correlation_with_target(panel: pd.DataFrame, target: pd.Series, top_n: int = 15):
    """Correlation plot"""
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
        xaxis_title='Correlation',
        yaxis_title='Feature',
        template='plotly_white',
        height=600
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_to_excel(
    monthly_panel: pd.DataFrame,
    quarterly_panel: Optional[pd.DataFrame],
    target: Optional[pd.Series],
    config: dict
) -> bytes:
    """Export to Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if monthly_panel is not None and not monthly_panel.empty:
            monthly_panel.reset_index().to_excel(writer, sheet_name='Monthly_Panel', index=False)
        if quarterly_panel is not None and not quarterly_panel.empty:
            quarterly_panel.reset_index().to_excel(writer, sheet_name='Quarterly_Panel', index=False)
        if target is not None and not target.empty:
            target.reset_index().to_excel(writer, sheet_name='Target', index=False)
        config_df = pd.DataFrame([config])
        config_df.to_excel(writer, sheet_name='Configuration', index=False)
    return output.getvalue()

# ═══════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 15px; margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0;'>📤 Manual Data Upload</h1>
    <p style='color: white; margin: 10px 0 0 0; font-size: 1.2rem;'>Upload CSV files and build custom panels</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📊 Aggregation")
    agg_method = st.selectbox("Method:", ['mean', 'sum', 'last'])
    use_business_days = st.checkbox("Business days only", False)
    min_days = st.slider("Min days/month", 1, 28, 10)
    
    st.markdown("---")
    st.subheader("🧹 Cleaning")
    drop_constant = st.checkbox("Remove constant", True)
    drop_correlated = st.checkbox("Remove correlated", False)
    if drop_correlated:
        corr_threshold = st.slider("Correlation threshold", 0.80, 0.99, 0.95, 0.01)
    else:
        corr_threshold = 0.95
    
    st.markdown("---")
    export_format = st.radio("Export:", ['CSV', 'Excel'])
    
    st.markdown("---")
    if st.button("🗑️ Clear All", use_container_width=True):
        state['y_monthly'] = None
        state['panel_monthly'] = None
        state['panel_quarterly'] = None
        state['raw_daily'] = []
        state['google_trends'] = None
        st.success("✅ Cleared!")
        st.rerun()
    
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")

# Upload section
st.markdown("### 📤 Step 1: Upload Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 🎯 Target Variable")
    target_file = st.file_uploader("Monthly target (CSV)", type=['csv'], key='target')
    if target_file:
        state['y_monthly'] = load_target_series(target_file)
        if state['y_monthly'] is not None:
            st.success(f"✅ {len(state['y_monthly'])} months")

with col2:
    st.markdown("##### 📊 Daily Data")
    daily_files = st.file_uploader("Daily series (CSV)", type=['csv'], accept_multiple_files=True, key='daily')
    if daily_files:
        state['raw_daily'] = []
        for file in daily_files:
            df = load_daily_data(file)
            if df is not None:
                state['raw_daily'].append(df)
        if state['raw_daily']:
            total_cols = sum(len(df.columns) - 1 for df in state['raw_daily'])
            st.success(f"✅ {len(state['raw_daily'])} files, {total_cols} series")

with col3:
    st.markdown("##### 🔍 Google Trends")
    trends_files = st.file_uploader("Trends (CSV)", type=['csv'], accept_multiple_files=True, key='trends')
    if trends_files:
        state['google_trends'] = load_google_trends(trends_files)
        if state['google_trends'] is not None:
            n_series = len(state['google_trends'].columns) - 1
            st.success(f"✅ {n_series} trends")

# Preview target
if state['y_monthly'] is not None:
    st.markdown("---")
    st.markdown("##### 📈 Target Preview")
    fig = plot_target_series(state['y_monthly'])
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", len(state['y_monthly']))
    col2.metric("Start", state['y_monthly'].index.min().strftime('%Y-%m'))
    col3.metric("End", state['y_monthly'].index.max().strftime('%Y-%m'))
    col4.metric("Mean", f"{state['y_monthly'].mean():.2f}")

# Build panel
st.markdown("---")
st.markdown("### 🔨 Step 2: Build Panel")

if not state['raw_daily'] and state['google_trends'] is None:
    st.info("📌 Upload data first")
else:
    if st.button("🚀 Build Panel", type="primary", use_container_width=True):
        with st.spinner("Building..."):
            panel = build_panel(
                state['raw_daily'],
                state['google_trends'],
                agg_method,
                use_business_days,
                min_days
            )
            
            if panel.empty:
                st.error("❌ Failed")
            else:
                if drop_constant:
                    panel = remove_constant_columns(panel)
                if drop_correlated:
                    panel = remove_correlated_duplicates(panel, corr_threshold)
                
                if state['y_monthly'] is not None:
                    panel = panel.loc[
                        (panel.index >= state['y_monthly'].index.min()) &
                        (panel.index <= state['y_monthly'].index.max())
                    ]
                
                state['panel_monthly'] = panel
                state['panel_quarterly'] = create_quarterly_panel(panel, agg_method)
                
                st.success(f"✅ Panel: {panel.shape[0]} months × {panel.shape[1]} features")

# Results
if state['panel_monthly'] is not None:
    st.markdown("---")
    st.markdown("### 📊 Step 3: Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Months", state['panel_monthly'].shape[0])
    col2.metric("📊 Features", state['panel_monthly'].shape[1])
    coverage = state['panel_monthly'].notna().mean().mean()
    col3.metric("✅ Coverage", f"{coverage:.1%}")
    if state['panel_quarterly'] is not None:
        col4.metric("📅 Quarters", state['panel_quarterly'].shape[0])
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "🔍 Coverage", "📈 Correlations", "💾 Export"])
    
    with tab1:
        st.subheader("Monthly Panel")
        st.dataframe(state['panel_monthly'].tail(24), use_container_width=True, height=400)
        if state['panel_quarterly'] is not None:
            st.subheader("Quarterly Panel")
            st.dataframe(state['panel_quarterly'].tail(8), use_container_width=True, height=300)
    
    with tab2:
        coverage_df = pd.DataFrame({
            'Feature': state['panel_monthly'].columns,
            'Coverage': state['panel_monthly'].notna().mean().values,
            'Missing': state['panel_monthly'].isna().sum().values
        }).sort_values('Coverage', ascending=False)
        st.dataframe(coverage_df, use_container_width=True)
        
        heatmap = plot_coverage_heatmap(state['panel_monthly'])
        st.plotly_chart(heatmap, use_container_width=True)
    
    with tab3:
        if state['y_monthly'] is not None:
            corr_fig = plot_correlation_with_target(state['panel_monthly'], state['y_monthly'])
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.info("No overlap")
        else:
            st.info("Upload target first")
    
    with tab4:
        config = {
            'aggregation_method': agg_method,
            'business_days_only': use_business_days,
            'min_days_per_month': min_days,
            'drop_constant': drop_constant,
            'drop_correlated': drop_correlated,
            'correlation_threshold': corr_threshold,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with st.expander("⚙️ Config"):
            st.json(config)
        
        col1, col2 = st.columns(2)
        with col1:
            if export_format == 'CSV':
                csv = state['panel_monthly'].to_csv().encode('utf-8')
                st.download_button("📥 Monthly CSV", csv, "monthly.csv", "text/csv", use_container_width=True)
                if state['panel_quarterly'] is not None:
                    csv_q = state['panel_quarterly'].to_csv().encode('utf-8')
                    st.download_button("📥 Quarterly CSV", csv_q, "quarterly.csv", "text/csv", use_container_width=True)
            else:
                excel = export_to_excel(state['panel_monthly'], state['panel_quarterly'], state['y_monthly'], config)
                st.download_button("📥 Excel", excel, "panel.xlsx", use_container_width=True)
