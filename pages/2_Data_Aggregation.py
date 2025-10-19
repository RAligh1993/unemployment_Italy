"""
🧱 Data Aggregation Pro v3.0 (ADVANCED)
================================
Professional multi-source data intake & panel builder for unemployment nowcasting.

Features:
- Multi-target support (total, male, female, youth unemployment, etc.)
- Direct monthly indicators (no aggregation needed)
- Quarterly data with interpolation/forward-fill
- Daily data aggregation
- Google Trends integration
- Smart merging and alignment

Author: AI Assistant
Date: October 2025
Version: 3.0 - Multi-target and quarterly data support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json
from io import BytesIO
from typing import Optional, List, Dict, Tuple
import traceback

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

try:
    from utils.state import AppState
except Exception:
    class _State:
        def __init__(self):
            # Targets (can have multiple)
            self.y_monthly: Optional[pd.Series] = None  # Primary target
            self.targets_monthly: Optional[pd.DataFrame] = None  # All targets
            
            # Panels
            self.panel_monthly: Optional[pd.DataFrame] = None
            self.panel_quarterly: Optional[pd.DataFrame] = None
            
            # Raw data sources
            self.raw_daily: List[pd.DataFrame] = []
            self.raw_monthly: List[pd.DataFrame] = []  # Direct monthly data
            self.raw_quarterly: List[pd.DataFrame] = []  # Quarterly data
            self.google_trends: Optional[pd.DataFrame] = None
            
            # Original backup
            self._panel_monthly_orig: Optional[pd.DataFrame] = None
    
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
# CONFIGURATION
# =============================================================================

# Validation thresholds
MIN_OBSERVATIONS = 12
RECOMMENDED_OBSERVATIONS = 24
MAX_FILE_SIZE_MB = 100

# Quarterly interpolation methods
QUARTERLY_METHODS = {
    'forward_fill': 'Forward Fill (repeat quarterly value for 3 months)',
    'linear': 'Linear Interpolation (smooth transition)',
    'cubic': 'Cubic Spline (smooth curve)',
}

# Common unemployment categories
UNEMPLOYMENT_CATEGORIES = {
    'total': 'Total Unemployment Rate',
    'male': 'Male Unemployment Rate',
    'female': 'Female Unemployment Rate',
    'youth': 'Youth Unemployment (15-24)',
    'adult': 'Adult Unemployment (25+)',
    'long_term': 'Long-term Unemployment (>12 months)',
    'educated': 'Unemployment - Tertiary Education',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_datetime_safe(series: pd.Series) -> pd.Series:
    """Convert to datetime and normalize"""
    return pd.to_datetime(series, errors='coerce').dt.tz_localize(None).dt.normalize()


def end_of_month(series: pd.Series) -> pd.Series:
    """Align dates to end of month"""
    dt = to_datetime_safe(series)
    return (dt + pd.offsets.MonthEnd(0)).dt.normalize()


def end_of_quarter(series: pd.Series) -> pd.Series:
    """Align dates to end of quarter"""
    dt = to_datetime_safe(series)
    return (dt + pd.offsets.QuarterEnd(0)).dt.normalize()


def slugify(name: str) -> str:
    """Convert name to clean identifier"""
    return (str(name).strip().lower()
            .replace(' ', '_')
            .replace('/', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('-', '_')
            .replace('%', 'pct')
            .replace(':', '_')
            .replace(',', '_')
            .replace('__', '_')
            .strip('_'))


def detect_date_column(df: pd.DataFrame) -> str:
    """Smart detection of date column"""
    common_names = ['date', 'Date', 'DATE', 'ds', 'time', 'Time', 
                    'period', 'Period', 'Week', 'Month', 'Day', 'Quarter',
                    'timestamp', 'Timestamp', 'year_month', 'year_quarter']
    
    for name in common_names:
        if name in df.columns:
            return name
    
    # Check for datetime types
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    # Try first column
    try:
        first_col = df.columns[0]
        pd.to_datetime(df[first_col].dropna().iloc[0])
        return first_col
    except:
        pass
    
    return df.columns[0]


def detect_frequency(dates: pd.DatetimeIndex) -> str:
    """
    Detect frequency of time series
    
    Returns: 'daily', 'weekly', 'monthly', 'quarterly', 'annual', or 'unknown'
    """
    if len(dates) < 2:
        return 'unknown'
    
    # Calculate differences
    diffs = dates.to_series().diff().dt.days.dropna()
    median_diff = diffs.median()
    
    if median_diff <= 1:
        return 'daily'
    elif 5 <= median_diff <= 9:
        return 'weekly'
    elif 28 <= median_diff <= 31:
        return 'monthly'
    elif 85 <= median_diff <= 95:
        return 'quarterly'
    elif 360 <= median_diff <= 370:
        return 'annual'
    else:
        return 'unknown'


def show_dataframe_info(df: pd.DataFrame, title: str = "Data Info"):
    """Display DataFrame info in expander"""
    with st.expander(f"🔍 {title}", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory", f"{memory_mb:.2f} MB")
        
        st.write("**Columns:**", list(df.columns))
        st.dataframe(df.head(), use_container_width=True)

# =============================================================================
# DATA LOADING - MULTI-TARGET
# =============================================================================

def load_multi_target(file) -> Optional[pd.DataFrame]:
    """
    Load multiple unemployment targets from CSV
    
    Expected format:
    - First column: date
    - Other columns: different unemployment rates (total, male, female, youth, etc.)
    
    Returns:
        DataFrame with date index and multiple target columns
    """
    file_name = file.name if hasattr(file, 'name') else 'uploaded file'
    
    st.write("---")
    st.markdown("### 🎯 Loading Multi-Target Data")
    
    try:
        # Check file size
        if hasattr(file, 'size'):
            size_mb = file.size / 1024**2
            if size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File too large: {size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)")
                return None
        
        # Try multiple encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=encoding)
                if encoding != 'utf-8':
                    st.info(f"ℹ️ Loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None or df.empty:
            st.error("❌ Could not read file")
            return None
        
        show_dataframe_info(df, f"Original: {file_name}")
        
        # Detect date column
        date_col = detect_date_column(df)
        value_cols = [c for c in df.columns if c != date_col]
        
        if not value_cols:
            st.error(f"❌ No value columns found (date column: '{date_col}')")
            return None
        
        st.success(f"✅ Detected: date='{date_col}', {len(value_cols)} target columns")
        
        # Process
        df = df[[date_col] + value_cols].copy()
        df = df.rename(columns={date_col: 'date'})
        
        # Convert date
        st.write("**🔄 Processing dates...**")
        df['date'] = end_of_month(df['date'])
        
        nat_count = df['date'].isna().sum()
        if nat_count > 0:
            st.error(f"❌ {nat_count} invalid dates")
            return None
        
        # Convert all value columns to numeric
        st.write("**🔄 Processing values...**")
        
        original_cols = value_cols.copy()
        valid_cols = []
        
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            valid_count = df[col].notna().sum()
            total_count = len(df)
            
            if valid_count > 0:
                valid_cols.append(col)
                st.write(f"   ✅ `{col}`: {valid_count}/{total_count} valid values")
            else:
                st.warning(f"   ⚠️ `{col}`: No valid values (skipped)")
        
        if not valid_cols:
            st.error("❌ No valid target columns")
            return None
        
        # Keep only valid columns
        df = df[['date'] + valid_cols]
        
        # Check duplicates
        dup_count = df.duplicated(subset=['date']).sum()
        if dup_count > 0:
            st.warning(f"⚠️ {dup_count} duplicate dates (keeping last)")
            df = df.drop_duplicates(subset=['date'], keep='last')
        
        # Set index
        df = df.set_index('date').sort_index()
        
        # Statistics
        st.write("---")
        st.write("### ✅ Multi-Target Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Observations", len(df))
        with col2:
            st.metric("🎯 Targets", len(df.columns))
        with col3:
            st.metric("📅 Start", df.index.min().strftime('%Y-%m'))
        with col4:
            st.metric("📅 End", df.index.max().strftime('%Y-%m'))
        
        # Show statistics for each target
        st.write("**Target Statistics:**")
        
        stats_df = df.describe().T
        stats_df['coverage'] = df.notna().mean()
        
        st.dataframe(
            stats_df.style.format({
                'mean': '{:.2f}',
                'std': '{:.2f}',
                'min': '{:.2f}',
                'max': '{:.2f}',
                'coverage': '{:.1%}'
            }),
            use_container_width=True
        )
        
        # Validation
        if len(df) < MIN_OBSERVATIONS:
            st.error(f"❌ Too few observations: {len(df)} (minimum: {MIN_OBSERVATIONS})")
            return None
        
        if len(df) < RECOMMENDED_OBSERVATIONS:
            st.warning(f"⚠️ Few observations: {len(df)} (recommended: ≥{RECOMMENDED_OBSERVATIONS})")
        
        st.success(f"✅ **{len(df.columns)} targets loaded successfully!**")
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading targets: {type(e).__name__}")
        st.error(str(e))
        
        with st.expander("🐛 Full Traceback", expanded=False):
            st.code(traceback.format_exc())
        
        return None

# =============================================================================
# DATA LOADING - MONTHLY INDICATORS
# =============================================================================

def load_monthly_data(file) -> Optional[pd.DataFrame]:
    """
    Load direct monthly indicators (no aggregation needed)
    
    Examples:
    - Industrial production
    - Retail sales
    - Consumer confidence
    - PMI indices
    - Inflation rates
    
    Returns:
        DataFrame with date column and indicators
    """
    file_name = file.name if hasattr(file, 'name') else 'uploaded file'
    
    try:
        df = pd.read_csv(file)
        
        if df.empty:
            st.warning(f"⚠️ {file_name}: Empty file")
            return None
        
        # Detect date
        date_col = detect_date_column(df)
        df = df.rename(columns={date_col: 'date'})
        
        # Convert date
        df['date'] = end_of_month(df['date'])
        df = df.dropna(subset=['date'])
        
        if df.empty:
            st.warning(f"⚠️ {file_name}: No valid dates")
            return None
        
        # Get numeric columns
        numeric_cols = []
        for col in df.columns:
            if col == 'date':
                continue
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if df[col].notna().sum() > 0:
                numeric_cols.append(col)
        
        if not numeric_cols:
            st.warning(f"⚠️ {file_name}: No valid numeric columns")
            return None
        
        # Keep only date and numeric
        df = df[['date'] + numeric_cols].sort_values('date')
        
        # Rename with prefix
        file_prefix = slugify(Path(file_name).stem)
        rename_map = {col: f"monthly_{file_prefix}__{slugify(col)}" for col in numeric_cols}
        df = df.rename(columns=rename_map)
        
        st.success(f"✅ {file_name}: {len(df)} months, {len(numeric_cols)} indicators")
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading {file_name}: {str(e)}")
        return None

# =============================================================================
# DATA LOADING - QUARTERLY
# =============================================================================

def load_quarterly_data(file) -> Optional[pd.DataFrame]:
    """
    Load quarterly data (will be interpolated to monthly)
    
    Examples:
    - GDP and components
    - Quarterly labor force survey
    - Government fiscal data
    
    Returns:
        DataFrame with quarter-end dates
    """
    file_name = file.name if hasattr(file, 'name') else 'uploaded file'
    
    try:
        df = pd.read_csv(file)
        
        if df.empty:
            st.warning(f"⚠️ {file_name}: Empty file")
            return None
        
        # Detect date
        date_col = detect_date_column(df)
        df = df.rename(columns={date_col: 'date'})
        
        # Try to detect if already quarterly
        df['date'] = to_datetime_safe(df['date'])
        
        # Check frequency
        freq = detect_frequency(df['date'].dropna())
        
        if freq == 'monthly':
            st.warning(f"⚠️ {file_name}: Data appears to be monthly, not quarterly")
            st.info("💡 Consider uploading to 'Monthly Indicators' section instead")
        
        # Convert to quarter-end
        df['date'] = end_of_quarter(df['date'])
        df = df.dropna(subset=['date'])
        
        if df.empty:
            st.warning(f"⚠️ {file_name}: No valid dates")
            return None
        
        # Get numeric columns
        numeric_cols = []
        for col in df.columns:
            if col == 'date':
                continue
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if df[col].notna().sum() > 0:
                numeric_cols.append(col)
        
        if not numeric_cols:
            st.warning(f"⚠️ {file_name}: No valid numeric columns")
            return None
        
        # Keep only date and numeric
        df = df[['date'] + numeric_cols].sort_values('date')
        
        # Rename with prefix
        file_prefix = slugify(Path(file_name).stem)
        rename_map = {col: f"quarterly_{file_prefix}__{slugify(col)}" for col in numeric_cols}
        df = df.rename(columns=rename_map)
        
        st.success(f"✅ {file_name}: {len(df)} quarters, {len(numeric_cols)} indicators")
        st.info(f"ℹ️ Frequency detected: {freq}")
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading {file_name}: {str(e)}")
        return None

# =============================================================================
# DAILY & TRENDS (same as before)
# =============================================================================

def load_daily_data(file) -> Optional[pd.DataFrame]:
    """Load daily data (will be aggregated to monthly)"""
    file_name = file.name if hasattr(file, 'name') else 'uploaded file'
    
    try:
        df = pd.read_csv(file)
        
        if df.empty:
            return None
        
        date_col = detect_date_column(df)
        df = df.rename(columns={date_col: 'date'})
        df['date'] = to_datetime_safe(df['date'])
        df = df.dropna(subset=['date'])
        
        if df.empty:
            return None
        
        numeric_cols = []
        for col in df.columns:
            if col == 'date':
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().sum() > 0:
                numeric_cols.append(col)
        
        if not numeric_cols:
            return None
        
        df = df[['date'] + numeric_cols].sort_values('date')
        
        file_prefix = slugify(Path(file_name).stem)
        rename_map = {col: f"daily_{file_prefix}__{slugify(col)}" for col in numeric_cols}
        df = df.rename(columns=rename_map)
        
        st.success(f"✅ {file_name}: {len(df)} days, {len(numeric_cols)} series")
        
        return df
    
    except Exception as e:
        st.error(f"❌ {file_name}: {str(e)}")
        return None


def load_google_trends(files: List) -> Optional[pd.DataFrame]:
    """Load Google Trends"""
    if not files:
        return None
    
    frames = []
    
    for file in files:
        file_name = file.name if hasattr(file, 'name') else 'file'
        
        try:
            df = pd.read_csv(file)
            
            date_col = None
            for col_name in ['Week', 'Month', 'Date', 'date']:
                if col_name in df.columns:
                    date_col = col_name
                    break
            
            if date_col is None:
                date_col = detect_date_column(df)
            
            series_cols = [c for c in df.columns if c != date_col]
            
            if not series_cols:
                continue
            
            df = df[[date_col] + series_cols].copy()
            df = df.rename(columns={date_col: 'date'})
            df['date'] = to_datetime_safe(df['date'])
            df = df.dropna(subset=['date'])
            
            if df.empty:
                continue
            
            new_cols = [f"trends__{slugify(col)}" for col in series_cols]
            df.columns = ['date'] + new_cols
            
            for col in new_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            frames.append(df)
            st.success(f"✅ {file_name}: {len(series_cols)} trends")
        
        except Exception as e:
            st.warning(f"⚠️ {file_name}: {str(e)}")
            continue
    
    if not frames:
        return None
    
    result = frames[0]
    for df in frames[1:]:
        result = pd.merge(result, df, on='date', how='outer')
    
    result = result.sort_values('date').reset_index(drop=True)
    
    st.info(f"ℹ️ Merged {len(frames)} trend files")
    
    return result

# =============================================================================
# AGGREGATION & INTERPOLATION
# =============================================================================

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
        before = len(df)
        df = df[df['date'].dt.dayofweek < 5]
        st.info(f"ℹ️ Business days: {len(df)}/{before}")
    
    df = df.set_index('date')
    
    counts = df.resample('M').count()
    
    if method == 'sum':
        monthly = df.resample('M').sum(min_count=1)
    elif method == 'last':
        monthly = df.resample('M').last()
    else:
        monthly = df.resample('M').mean()
    
    for col in monthly.columns:
        monthly.loc[counts[col] < min_days, col] = np.nan
    
    monthly = monthly.reset_index()
    monthly['date'] = end_of_month(monthly['date'])
    
    return monthly


def interpolate_quarterly_to_monthly(
    df: pd.DataFrame,
    method: str = 'forward_fill'
) -> pd.DataFrame:
    """
    Convert quarterly data to monthly frequency
    
    Args:
        df: DataFrame with quarter-end dates as index
        method: 'forward_fill', 'linear', or 'cubic'
    
    Returns:
        Monthly DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    st.write(f"📅 Interpolating quarterly → monthly (method: {method})...")
    
    # Ensure date index
    if 'date' in df.columns:
        df = df.set_index('date')
    
    # Create monthly range
    monthly_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='M'
    )
    
    # Reindex to monthly
    df_monthly = df.reindex(monthly_range)
    
    if method == 'forward_fill':
        # Repeat quarterly value for 3 months
        df_monthly = df_monthly.fillna(method='ffill')
    
    elif method == 'linear':
        # Linear interpolation
        df_monthly = df_monthly.interpolate(method='linear')
    
    elif method == 'cubic':
        # Cubic spline
        df_monthly = df_monthly.interpolate(method='cubic')
    
    df_monthly = df_monthly.reset_index()
    df_monthly.columns = ['date'] + list(df.columns)
    df_monthly['date'] = end_of_month(df_monthly['date'])
    
    st.success(f"✅ Interpolated to {len(df_monthly)} months")
    
    return df_monthly

# =============================================================================
# PANEL BUILDING
# =============================================================================

def build_comprehensive_panel(
    targets_df: Optional[pd.DataFrame],
    daily_frames: List[pd.DataFrame],
    monthly_frames: List[pd.DataFrame],
    quarterly_frames: List[pd.DataFrame],
    trends_df: Optional[pd.DataFrame],
    daily_method: str,
    business_days: bool,
    min_days: int,
    quarterly_method: str
) -> pd.DataFrame:
    """
    Build comprehensive panel from all sources
    
    Returns:
        Unified monthly panel with all features
    """
    st.write("---")
    st.markdown("### 🔨 Building Comprehensive Panel")
    
    panel = None
    
    # 1. Start with targets if available
    if targets_df is not None and not targets_df.empty:
        st.write("🎯 **Step 1:** Adding target variables...")
        panel = targets_df.reset_index()
        panel.columns = ['date'] + list(targets_df.columns)
        st.success(f"   ✅ {len(targets_df.columns)} targets, {len(panel)} months")
    
    # 2. Add direct monthly data
    if monthly_frames:
        st.write(f"📊 **Step 2:** Adding {len(monthly_frames)} monthly indicator files...")
        
        for i, df in enumerate(monthly_frames, 1):
            if df is None or df.empty:
                continue
            
            if panel is None:
                panel = df
            else:
                panel = pd.merge(panel, df, on='date', how='outer')
            
            st.write(f"   ✅ File {i}: {len(df.columns)-1} indicators added")
    
    # 3. Aggregate daily data
    if daily_frames:
        st.write(f"📈 **Step 3:** Aggregating {len(daily_frames)} daily files...")
        
        for i, df in enumerate(daily_frames, 1):
            if df is None or df.empty:
                continue
            
            monthly = aggregate_to_monthly(df, daily_method, business_days, min_days)
            
            if panel is None:
                panel = monthly
            else:
                panel = pd.merge(panel, monthly, on='date', how='outer')
            
            st.write(f"   ✅ File {i}: {len(monthly.columns)-1} series aggregated")
    
    # 4. Interpolate quarterly data
    if quarterly_frames:
        st.write(f"📅 **Step 4:** Interpolating {len(quarterly_frames)} quarterly files...")
        
        for i, df in enumerate(quarterly_frames, 1):
            if df is None or df.empty:
                continue
            
            monthly = interpolate_quarterly_to_monthly(df, quarterly_method)
            
            if panel is None:
                panel = monthly
            else:
                panel = pd.merge(panel, monthly, on='date', how='outer')
            
            st.write(f"   ✅ File {i}: {len(monthly.columns)-1} quarterly indicators")
    
    # 5. Add Google Trends
    if trends_df is not None and not trends_df.empty:
        st.write("🔍 **Step 5:** Adding Google Trends...")
        
        gt_monthly = trends_df.set_index('date').resample('M').mean().reset_index()
        gt_monthly['date'] = end_of_month(gt_monthly['date'])
        
        if panel is None:
            panel = gt_monthly
        else:
            panel = pd.merge(panel, gt_monthly, on='date', how='outer')
        
        st.success(f"   ✅ {len(gt_monthly.columns)-1} trend series")
    
    if panel is None or panel.empty:
        st.error("❌ No data to build panel")
        return pd.DataFrame()
    
    # Finalize
    st.write("🔧 **Step 6:** Finalizing panel...")
    
    panel = panel.sort_values('date').set_index('date')
    
    # Ensure numeric
    for col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors='coerce')
    
    st.success(f"✅ **Panel ready:** {len(panel)} months × {len(panel.columns)} features")
    
    return panel

# =============================================================================
# CLEANING (same as before)
# =============================================================================

def remove_constant_columns(df: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    """Remove constant columns"""
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
    
    return df[keep_cols] if keep_cols else pd.DataFrame()


def remove_correlated_duplicates(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove correlated columns"""
    if df is None or df.empty or len(df.columns) < 2:
        return df
    
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    if to_drop:
        st.info(f"🧹 Removed {len(to_drop)} correlated columns (>{threshold:.0%})")
    
    return df.drop(columns=to_drop, errors='ignore')

# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_multi_targets(targets_df: pd.DataFrame):
    """Plot all targets on one chart"""
    fig = go.Figure()
    
    for col in targets_df.columns:
        fig.add_trace(go.Scatter(
            x=targets_df.index,
            y=targets_df[col],
            mode='lines+markers',
            name=col,
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title='All Unemployment Targets',
        xaxis_title='Date',
        yaxis_title='Unemployment Rate (%)',
        template='plotly_white',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        )
    )
    
    return fig


def plot_data_coverage(panel: pd.DataFrame, n_months: int = 60):
    """Coverage heatmap"""
    presence = panel.notna().astype(int).tail(n_months)
    
    fig = px.imshow(
        presence.T,
        aspect='auto',
        color_continuous_scale=['#EF4444', '#10B981'],
        labels={'color': 'Present'},
        title=f'Data Coverage (Last {n_months} Months)'
    )
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        xaxis_title='Date',
        yaxis_title='Feature'
    )
    
    return fig


def plot_source_breakdown(panel: pd.DataFrame):
    """Bar chart showing feature count by source"""
    
    sources = {
        'Targets': 0,
        'Daily (aggregated)': 0,
        'Monthly': 0,
        'Quarterly (interpolated)': 0,
        'Trends': 0
    }
    
    for col in panel.columns:
        if 'daily_' in col:
            sources['Daily (aggregated)'] += 1
        elif 'monthly_' in col:
            sources['Monthly'] += 1
        elif 'quarterly_' in col:
            sources['Quarterly (interpolated)'] += 1
        elif 'trends__' in col:
            sources['Trends'] += 1
        else:
            sources['Targets'] += 1
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(sources.keys()),
            y=list(sources.values()),
            marker=dict(
                color=['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899']
            ),
            text=list(sources.values()),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Features by Data Source',
        xaxis_title='Source Type',
        yaxis_title='Number of Features',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

# =============================================================================
# EXPORT
# =============================================================================

def export_to_excel(
    panel_monthly: pd.DataFrame,
    targets_df: Optional[pd.DataFrame],
    config: dict
) -> bytes:
    """Export to Excel"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if panel_monthly is not None and not panel_monthly.empty:
            panel_monthly.reset_index().to_excel(writer, sheet_name='Monthly_Panel', index=False)
        
        if targets_df is not None and not targets_df.empty:
            targets_df.reset_index().to_excel(writer, sheet_name='Targets', index=False)
        
        config_df = pd.DataFrame([config])
        config_df.to_excel(writer, sheet_name='Configuration', index=False)
    
    return output.getvalue()

# =============================================================================
# MAIN UI - NO st.set_page_config
# =============================================================================

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
    .step-header {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .data-source-box {
        background: white;
        border: 2px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🧱 Advanced Data Aggregation</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multi-source, multi-frequency panel builder for unemployment nowcasting</p>', unsafe_allow_html=True)

# Info box
st.info("""
**📊 Supported Data Sources:**
- 🎯 **Multi-Target**: Multiple unemployment rates (total, male, female, youth, etc.)
- 📊 **Monthly Indicators**: Industrial production, retail sales, PMI, etc.
- 📈 **Daily Data**: Stock prices, VIX, financial indicators → aggregated to monthly
- 📅 **Quarterly Data**: GDP, labor force survey → interpolated to monthly
- 🔍 **Google Trends**: Weekly/monthly search trends
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Aggregation Settings")
    
    st.subheader("📈 Daily → Monthly")
    daily_agg_method = st.selectbox(
        "Method:",
        options=['mean', 'sum', 'last'],
        index=0
    )
    
    use_business_days = st.checkbox("Business days only", value=False)
    min_days = st.slider("Min days/month", 1, 28, 10)
    
    st.markdown("---")
    st.subheader("📅 Quarterly → Monthly")
    
    quarterly_method = st.selectbox(
        "Interpolation:",
        options=list(QUARTERLY_METHODS.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        index=0
    )
    
    with st.expander("ℹ️ Interpolation Methods", expanded=False):
        for key, desc in QUARTERLY_METHODS.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {desc}")
    
    st.markdown("---")
    st.subheader("🧹 Cleaning")
    
    drop_constant = st.checkbox("Remove constant", value=True)
    drop_correlated = st.checkbox("Remove correlated", value=False)
    
    if drop_correlated:
        corr_threshold = st.slider("Correlation threshold", 0.80, 0.99, 0.95, 0.01, format="%.2f")
    else:
        corr_threshold = 0.95
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All", use_container_width=True):
        state.y_monthly = None
        state.targets_monthly = None
        state.panel_monthly = None
        state.panel_quarterly = None
        state.raw_daily = []
        state.raw_monthly = []
        state.raw_quarterly = []
        state.google_trends = None
        st.success("✅ Cleared!")
        st.rerun()

# =============================================================================
# STEP 1: UPLOAD - MULTI-SOURCE
# =============================================================================

st.markdown('<div class="step-header">📤 Step 1: Upload Multi-Source Data</div>', unsafe_allow_html=True)

# Create tabs for different data sources
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Targets",
    "📊 Monthly",
    "📈 Daily",
    "📅 Quarterly",
    "🔍 Trends"
])

# TAB 1: TARGETS
with tab1:
    st.markdown("""
    ### 🎯 Multiple Unemployment Targets
    
    Upload a CSV file with **multiple unemployment rates**:
    - Total unemployment
    - Male/Female unemployment
    - Youth/Adult unemployment
    - Long-term unemployment
    - By education level
    - By region
    
    **Format:** First column = date, other columns = different unemployment rates
    """)
    
    target_file = st.file_uploader(
        "Upload Multi-Target CSV",
        type=['csv'],
        key='multi_target',
        help="CSV with date + multiple unemployment columns"
    )
    
    if target_file:
        with st.spinner("🔄 Loading targets..."):
            targets_df = load_multi_target(target_file)
            
            if targets_df is not None:
                state.targets_monthly = targets_df
                
                # Auto-select primary target
                if 'total' in [c.lower() for c in targets_df.columns]:
                    primary_col = [c for c in targets_df.columns if 'total' in c.lower()][0]
                else:
                    primary_col = targets_df.columns[0]
                
                state.y_monthly = targets_df[primary_col]
                
                st.success(f"✅ Primary target set to: **{primary_col}**")
                
                # Plot all targets
                fig = plot_multi_targets(targets_df)
                st.plotly_chart(fig, use_container_width=True)

# TAB 2: MONTHLY
with tab2:
    st.markdown("""
    ### 📊 Direct Monthly Indicators
    
    Upload monthly data that **doesn't need aggregation**:
    - Industrial production index
    - Retail sales
    - Consumer confidence index
    - PMI (Manufacturing/Services)
    - CPI / Inflation rates
    - Building permits
    - New orders
    
    **Format:** CSV with date column + indicator columns
    """)
    
    monthly_files = st.file_uploader(
        "Upload Monthly Indicator CSV(s)",
        type=['csv'],
        accept_multiple_files=True,
        key='monthly_indicators'
    )
    
    if monthly_files:
        with st.spinner("🔄 Loading monthly indicators..."):
            state.raw_monthly = []
            
            for file in monthly_files:
                df = load_monthly_data(file)
                if df is not None:
                    state.raw_monthly.append(df)
            
            if state.raw_monthly:
                total_indicators = sum(len(df.columns)-1 for df in state.raw_monthly)
                st.success(f"✅ Loaded {len(state.raw_monthly)} files, {total_indicators} indicators")

# TAB 3: DAILY
with tab3:
    st.markdown("""
    ### 📈 Daily Financial Data
    
    Upload daily data (will be **aggregated to monthly**):
    - Stock indices (S&P500, FTSE, DAX, FTSE MIB)
    - VIX (Volatility index)
    - Exchange rates
    - Commodity prices
    - Bond yields
    - Banking sector indices
    
    **Format:** CSV with date + numeric columns
    """)
    
    daily_files = st.file_uploader(
        "Upload Daily CSV(s)",
        type=['csv'],
        accept_multiple_files=True,
        key='daily_data'
    )
    
    if daily_files:
        with st.spinner("🔄 Loading daily files..."):
            state.raw_daily = []
            
            for file in daily_files:
                df = load_daily_data(file)
                if df is not None:
                    state.raw_daily.append(df)
            
            if state.raw_daily:
                total_series = sum(len(df.columns)-1 for df in state.raw_daily)
                st.success(f"✅ Loaded {len(state.raw_daily)} files, {total_series} series")

# TAB 4: QUARTERLY
with tab4:
    st.markdown("""
    ### 📅 Quarterly Economic Data
    
    Upload quarterly data (will be **interpolated to monthly**):
    - GDP and components
    - Government spending
    - Investment
    - Trade balance
    - Quarterly labor force survey details
    - Productivity measures
    
    **Format:** CSV with quarter dates + indicator columns
    
    ℹ️ *Interpolation method can be selected in sidebar*
    """)
    
    quarterly_files = st.file_uploader(
        "Upload Quarterly CSV(s)",
        type=['csv'],
        accept_multiple_files=True,
        key='quarterly_data'
    )
    
    if quarterly_files:
        with st.spinner("🔄 Loading quarterly files..."):
            state.raw_quarterly = []
            
            for file in quarterly_files:
                df = load_quarterly_data(file)
                if df is not None:
                    state.raw_quarterly.append(df)
            
            if state.raw_quarterly:
                total_indicators = sum(len(df.columns)-1 for df in state.raw_quarterly)
                st.success(f"✅ Loaded {len(state.raw_quarterly)} files, {total_indicators} indicators")

# TAB 5: TRENDS
with tab5:
    st.markdown("""
    ### 🔍 Google Trends
    
    Upload Google Trends data:
    - Search volume for unemployment-related terms
    - Industry-specific searches
    - Regional job search trends
    
    **Format:** Google Trends CSV export (weekly or monthly)
    """)
    
    trends_files = st.file_uploader(
        "Upload Google Trends CSV(s)",
        type=['csv'],
        accept_multiple_files=True,
        key='trends'
    )
    
    if trends_files:
        with st.spinner("🔄 Loading trends..."):
            state.google_trends = load_google_trends(trends_files)

# =============================================================================
# STEP 2: BUILD PANEL
# =============================================================================

st.markdown('<div class="step-header">🔨 Step 2: Build Comprehensive Panel</div>', unsafe_allow_html=True)

# Show what's loaded
has_data = (
    state.targets_monthly is not None or
    state.raw_monthly or
    state.raw_daily or
    state.raw_quarterly or
    state.google_trends is not None
)

if not has_data:
    st.info("📌 Upload data in at least one category above to build panel")
else:
    # Summary of loaded data
    st.markdown("### 📋 Data Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        target_count = len(state.targets_monthly.columns) if state.targets_monthly is not None else 0
        st.metric("🎯 Targets", target_count)
    
    with col2:
        monthly_count = len(state.raw_monthly)
        st.metric("📊 Monthly Files", monthly_count)
    
    with col3:
        daily_count = len(state.raw_daily)
        st.metric("📈 Daily Files", daily_count)
    
    with col4:
        quarterly_count = len(state.raw_quarterly)
        st.metric("📅 Quarterly Files", quarterly_count)
    
    with col5:
        trends_count = len(state.google_trends.columns)-1 if state.google_trends is not None else 0
        st.metric("🔍 Trend Series", trends_count)
    
    st.markdown("---")
    
    # Build button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        build_button = st.button(
            "🚀 Build Comprehensive Panel",
            use_container_width=True,
            type="primary"
        )
    
    if build_button:
        progress = st.progress(0)
        
        with st.container():
            # Build panel
            panel = build_comprehensive_panel(
                state.targets_monthly,
                state.raw_daily,
                state.raw_monthly,
                state.raw_quarterly,
                state.google_trends,
                daily_agg_method,
                use_business_days,
                min_days,
                quarterly_method
            )
            
            progress.progress(0.5)
            
            if panel.empty:
                st.error("❌ Panel is empty")
                st.stop()
            
            # Clean
            st.write("🧹 **Cleaning panel...**")
            
            if drop_constant:
                panel = remove_constant_columns(panel)
            
            if drop_correlated:
                panel = remove_correlated_duplicates(panel, corr_threshold)
            
            progress.progress(0.8)
            
            # Align with primary target if exists
            if state.y_monthly is not None:
                st.write("🎯 **Aligning with primary target...**")
                panel = panel.loc[
                    (panel.index >= state.y_monthly.index.min()) &
                    (panel.index <= state.y_monthly.index.max())
                ]
            
            state.panel_monthly = panel
            
            progress.progress(1.0)
            progress.empty()
            
            # Summary
            st.markdown("---")
            st.markdown("### ✅ Panel Build Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📅 Months", len(panel))
                st.metric("📊 Features", len(panel.columns))
            
            with col2:
                coverage = panel.notna().mean().mean()
                st.metric("✅ Coverage", f"{coverage:.1%}")
                
                nulls = panel.isna().sum().sum()
                st.metric("❌ Missing", f"{nulls:,}")
            
            with col3:
                if state.targets_monthly is not None:
                    target_features = len([c for c in panel.columns if c in state.targets_monthly.columns])
                    st.metric("🎯 Target Features", target_features)
                
                memory = panel.memory_usage(deep=True).sum() / 1024**2
                st.metric("💾 Memory", f"{memory:.1f} MB")
            
            # Source breakdown
            fig_breakdown = plot_source_breakdown(panel)
            st.plotly_chart(fig_breakdown, use_container_width=True)
            
            st.success("🎉 **Panel ready!** Go to **Feature Engineering** or **Backtesting**")

# =============================================================================
# STEP 3: ANALYSIS
# =============================================================================

if state.panel_monthly is not None and not state.panel_monthly.empty:
    st.markdown('<div class="step-header">📊 Step 3: Panel Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Data", "🔍 Coverage", "💾 Export"])
    
    with tab1:
        st.subheader("Panel Preview (Last 24 Months)")
        st.dataframe(
            state.panel_monthly.tail(24).style.format("{:.4f}"),
            use_container_width=True,
            height=500
        )
    
    with tab2:
        st.subheader("Data Coverage Analysis")
        
        # Coverage table
        coverage_df = pd.DataFrame({
            'Feature': state.panel_monthly.columns,
            'Coverage': state.panel_monthly.notna().mean(),
            'Missing': state.panel_monthly.isna().sum()
        }).sort_values('Coverage', ascending=False)
        
        st.dataframe(
            coverage_df.style.format({'Coverage': '{:.1%}'}),
            use_container_width=True,
            height=400
        )
        
        # Heatmap
        fig_coverage = plot_data_coverage(state.panel_monthly, 60)
        st.plotly_chart(fig_coverage, use_container_width=True)
    
    with tab3:
        st.subheader("💾 Export Panel")
        
        config = {
            'daily_method': daily_agg_method,
            'quarterly_method': quarterly_method,
            'business_days': use_business_days,
            'min_days': min_days,
            'drop_constant': drop_constant,
            'drop_correlated': drop_correlated,
            'corr_threshold': corr_threshold,
            'created': datetime.now().isoformat(),
            'shape': f"{state.panel_monthly.shape[0]} × {state.panel_monthly.shape[1]}"
        }
        
        with st.expander("⚙️ Configuration"):
            st.json(config)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = state.panel_monthly.to_csv().encode('utf-8')
            st.download_button(
                "📥 CSV",
                csv,
                "panel.csv",
                use_container_width=True
            )
        
        with col2:
            excel = export_to_excel(state.panel_monthly, state.targets_monthly, config)
            st.download_button(
                "📥 Excel (All Sheets)",
                excel,
                "panel.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.caption("💻 Built with Streamlit Pro v3.0 | 🎯 Advanced Multi-Source Nowcasting")
