"""
🧱 Data Aggregation Pro v2.1 (FIXED)
================================
Professional data intake & panel builder for time series forecasting.
Features: Multi-source upload, smart aggregation, automated cleaning, visual diagnostics.

Author: AI Assistant
Date: October 2025
Version: 2.1 - Enhanced error handling and user feedback

IMPROVEMENTS:
- Better error messages with actionable tips
- Detailed progress logging
- Robust data validation
- Enhanced quarterly panel creation
- Debug-friendly output
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
import traceback

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
MAX_FILE_SIZE_MB = 50

# Default settings
DEFAULT_AGG_METHOD = 'mean'
DEFAULT_MIN_DAYS = 10
DEFAULT_CORR_THRESHOLD = 0.95

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_datetime_safe(series: pd.Series) -> pd.Series:
    """
    Convert to datetime and normalize to date only (tz-naive)
    
    Args:
        series: Series with date values
    
    Returns:
        Normalized datetime series
    """
    return pd.to_datetime(series, errors='coerce').dt.tz_localize(None).dt.normalize()


def end_of_month(series: pd.Series) -> pd.Series:
    """
    Align dates to end of month
    
    Args:
        series: Series with dates
    
    Returns:
        Series with dates aligned to month-end
    """
    dt = to_datetime_safe(series)
    return (dt + pd.offsets.MonthEnd(0)).dt.normalize()


def slugify(name: str) -> str:
    """
    Convert name to clean column identifier
    
    Args:
        name: Original column name
    
    Returns:
        Clean identifier (lowercase, no spaces/special chars)
    """
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
    """
    Smart detection of date column
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Name of the date column
    """
    # Common date column names (case-insensitive)
    common_names = ['date', 'Date', 'DATE', 'ds', 'time', 'Time', 
                    'period', 'Period', 'Week', 'Month', 'Day', 
                    'timestamp', 'Timestamp']
    
    for name in common_names:
        if name in df.columns:
            return name
    
    # Check for datetime types
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    # Check for parseable date strings in first column
    try:
        first_col = df.columns[0]
        pd.to_datetime(df[first_col].dropna().iloc[0])
        return first_col
    except:
        pass
    
    # Default to first column
    return df.columns[0]


def validate_dataframe(df: pd.DataFrame, name: str) -> Tuple[bool, str]:
    """
    Validate uploaded dataframe
    
    Args:
        df: DataFrame to validate
        name: Name for error messages
    
    Returns:
        (is_valid, error_message)
    """
    if df is None:
        return False, f"{name} is None"
    
    if df.empty:
        return False, f"{name} is empty"
    
    if len(df.columns) < 2:
        return False, f"{name} must have at least 2 columns (date + value). Found: {len(df.columns)}"
    
    return True, "Valid"


def show_dataframe_info(df: pd.DataFrame, title: str = "Data Info"):
    """
    Display detailed DataFrame information
    
    Args:
        df: DataFrame to display
        title: Title for expander
    """
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
        st.write("**Data Types:**")
        st.dataframe(
            pd.DataFrame(df.dtypes, columns=['Type']).T,
            use_container_width=True
        )
        
        st.write("**Preview (first 5 rows):**")
        st.dataframe(df.head(), use_container_width=True)

# =============================================================================
# DATA LOADING FUNCTIONS - ENHANCED
# =============================================================================

def load_target_series(file) -> Optional[pd.Series]:
    """
    Load monthly target variable from CSV with comprehensive validation
    
    Args:
        file: Uploaded file object
    
    Returns:
        Series with date index and values, or None if failed
    """
    file_name = file.name if hasattr(file, 'name') else 'uploaded file'
    
    try:
        # Check file size
        if hasattr(file, 'size'):
            size_mb = file.size / 1024**2
            if size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File too large: {size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)")
                return None
        
        # Step 1: Try multiple encodings
        df = None
        encoding_tried = []
        
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                file.seek(0)  # Reset file pointer
                df = pd.read_csv(file, encoding=encoding)
                
                if encoding != 'utf-8':
                    st.info(f"ℹ️ File loaded with {encoding} encoding")
                
                encoding_tried.append(encoding)
                break
                
            except UnicodeDecodeError:
                encoding_tried.append(encoding)
                continue
            except Exception as e:
                if encoding == encoding_tried[-1]:  # Last attempt
                    st.error(f"❌ Cannot read file: {str(e)}")
                    return None
        
        if df is None:
            st.error(f"❌ Could not read file with any encoding. Tried: {encoding_tried}")
            return None
        
        # Step 2: Basic validation
        valid, msg = validate_dataframe(df, f"Target file '{file_name}'")
        if not valid:
            st.error(f"❌ {msg}")
            return None
        
        # Show original data
        show_dataframe_info(df, f"Original Data: {file_name}")
        
        # Step 3: Detect columns
        date_col = detect_date_column(df)
        value_cols = [c for c in df.columns if c != date_col]
        
        if not value_cols:
            st.error("❌ No value column found")
            st.info(f"💡 Detected date column: **'{date_col}'**")
            st.info(f"💡 All columns: {list(df.columns)}")
            st.info("💡 File should have at least 2 columns: date + value")
            return None
        
        value_col = value_cols[0]
        
        if len(value_cols) > 1:
            st.warning(f"⚠️ Multiple value columns found. Using first: **'{value_col}'**")
            st.info(f"💡 Other columns ignored: {value_cols[1:]}")
        
        st.success(f"✅ Column mapping: date=**'{date_col}'** → value=**'{value_col}'**")
        
        # Step 4: Process data
        df = df[[date_col, value_col]].copy()
        df.columns = ['date', 'value']
        
        # Step 5: Convert dates with detailed error reporting
        st.write("**🔄 Processing dates...**")
        
        original_dates = df['date'].copy()
        df['date'] = end_of_month(df['date'])
        
        # Check for invalid dates (NaT)
        nat_mask = df['date'].isna()
        nat_count = nat_mask.sum()
        
        if nat_count > 0:
            st.error(f"❌ Found **{nat_count}/{len(df)}** invalid dates")
            
            # Show examples
            invalid_examples = original_dates[nat_mask].head(10).tolist()
            st.write("**Examples of invalid dates:**")
            for i, date in enumerate(invalid_examples, 1):
                st.write(f"{i}. `{date}`")
            
            st.info("""
            💡 **Expected date formats:**
            - YYYY-MM-DD (e.g., 2020-01-31)
            - MM/DD/YYYY (e.g., 01/31/2020)
            - DD-MM-YYYY (e.g., 31-01-2020)
            - YYYY/MM/DD (e.g., 2020/01/31)
            """)
            
            return None
        
        st.success(f"✅ All dates parsed successfully")
        
        # Step 6: Convert values with validation
        st.write("**🔄 Processing values...**")
        
        original_values = df['value'].copy()
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        nan_mask = df['value'].isna()
        nan_count = nan_mask.sum()
        
        if nan_count > 0:
            st.warning(f"⚠️ Found **{nan_count}/{len(df)}** non-numeric values (will be removed)")
            
            # Show examples
            invalid_values = pd.DataFrame({
                'date': original_dates[nan_mask],
                'original_value': original_values[nan_mask]
            }).head(10)
            
            with st.expander("🔍 View invalid values", expanded=False):
                st.dataframe(invalid_values, use_container_width=True)
        
        # Step 7: Remove NaN
        df_before = len(df)
        df = df.dropna()
        df_after = len(df)
        
        if df_after < df_before:
            removed = df_before - df_after
            st.info(f"ℹ️ Removed **{removed}** rows with missing data")
        
        if df.empty:
            st.error("❌ No valid data remaining after cleaning")
            st.info("💡 Check your file for valid date-value pairs")
            return None
        
        # Step 8: Check for duplicates
        dup_mask = df.duplicated(subset=['date'], keep=False)
        dup_count = dup_mask.sum()
        
        if dup_count > 0:
            unique_dup_dates = df[dup_mask]['date'].nunique()
            st.warning(f"⚠️ Found **{dup_count}** duplicate entries for **{unique_dup_dates}** dates")
            
            # Show duplicates
            duplicates = df[dup_mask].sort_values('date')
            with st.expander("🔍 View all duplicates", expanded=False):
                st.dataframe(duplicates, use_container_width=True)
            
            st.info("ℹ️ Keeping **last** value for each duplicate date")
            df = df.drop_duplicates(subset=['date'], keep='last')
        
        # Step 9: Create series
        series = df.set_index('date')['value'].sort_index()
        
        # Step 10: Final validation and statistics
        st.write("---")
        st.write("### ✅ Target Series Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Observations", len(series))
        
        with col2:
            date_range = f"{series.index.min().strftime('%Y-%m')}"
            st.metric("📅 Start", date_range)
        
        with col3:
            date_range = f"{series.index.max().strftime('%Y-%m')}"
            st.metric("📅 End", date_range)
        
        with col4:
            span_years = (series.index.max() - series.index.min()).days / 365.25
            st.metric("⏱️ Span", f"{span_years:.1f} years")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Mean", f"{series.mean():.2f}")
        
        with col2:
            st.metric("📊 Std Dev", f"{series.std():.2f}")
        
        with col3:
            st.metric("⬇️ Min", f"{series.min():.2f}")
        
        with col4:
            st.metric("⬆️ Max", f"{series.max():.2f}")
        
        # Warnings
        if len(series) < MIN_OBSERVATIONS:
            st.error(f"❌ Too few observations: {len(series)} (minimum: {MIN_OBSERVATIONS})")
            st.stop()
        
        if len(series) < RECOMMENDED_OBSERVATIONS:
            st.warning(f"⚠️ Few observations: {len(series)} (recommended: ≥{RECOMMENDED_OBSERVATIONS})")
        
        # Check for gaps
        expected_dates = pd.date_range(
            series.index.min(),
            series.index.max(),
            freq='M'
        )
        
        missing_dates = expected_dates.difference(series.index)
        
        if len(missing_dates) > 0:
            st.warning(f"⚠️ Found **{len(missing_dates)}** missing months in the series")
            
            with st.expander("🔍 View missing months", expanded=False):
                missing_df = pd.DataFrame({
                    'Missing Month': missing_dates.strftime('%Y-%m')
                })
                st.dataframe(missing_df, use_container_width=True)
        
        st.success(f"✅ **Target series loaded successfully!** Ready for modeling.")
        
        return series
    
    except pd.errors.ParserError as e:
        st.error(f"❌ CSV parsing error: {str(e)}")
        st.info("💡 Common causes:")
        st.write("- File is not properly formatted CSV")
        st.write("- Inconsistent number of columns")
        st.write("- Special characters in data")
        return None
    
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_name}")
        return None
    
    except MemoryError:
        st.error(f"❌ File too large to load (out of memory)")
        st.info(f"💡 Try reducing file size or use sampling")
        return None
    
    except Exception as e:
        st.error(f"❌ Unexpected error loading target")
        st.error(f"**Error type:** {type(e).__name__}")
        st.error(f"**Error message:** {str(e)}")
        
        with st.expander("🐛 Full Error Traceback (for debugging)", expanded=False):
            st.code(traceback.format_exc(), language='python')
        
        return None


def load_daily_data(file) -> Optional[pd.DataFrame]:
    """
    Load daily time series data from CSV with validation
    
    Args:
        file: Uploaded file object
    
    Returns:
        DataFrame with 'date' column and numeric features
    """
    file_name = file.name if hasattr(file, 'name') else 'uploaded file'
    
    try:
        # Read file
        df = pd.read_csv(file)
        
        valid, msg = validate_dataframe(df, f"Daily file '{file_name}'")
        if not valid:
            st.warning(f"⚠️ {msg}")
            return None
        
        # Detect date column
        date_col = detect_date_column(df)
        df = df.rename(columns={date_col: 'date'})
        
        # Convert date
        df['date'] = to_datetime_safe(df['date'])
        
        # Check for invalid dates
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"⚠️ {file_name}: {invalid_dates} invalid dates removed")
        
        df = df.dropna(subset=['date'])
        
        if df.empty:
            st.warning(f"⚠️ {file_name}: No valid dates found")
            return None
        
        # Process numeric columns
        numeric_cols = []
        for col in df.columns:
            if col == 'date':
                continue
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            valid_count = df[col].notna().sum()
            if valid_count > 0:
                numeric_cols.append(col)
        
        if not numeric_cols:
            st.warning(f"⚠️ {file_name}: No valid numeric columns")
            return None
        
        # Keep only date and numeric columns
        df = df[['date'] + numeric_cols].sort_values('date')
        
        # Rename columns with file prefix
        file_prefix = slugify(Path(file_name).stem)
        rename_map = {col: f"{file_prefix}__{slugify(col)}" for col in numeric_cols}
        df = df.rename(columns=rename_map)
        
        st.success(f"✅ {file_name}: {len(df)} days, {len(numeric_cols)} series")
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading {file_name}: {str(e)}")
        return None


def load_google_trends(files: List) -> Optional[pd.DataFrame]:
    """
    Load and merge multiple Google Trends CSV files
    
    Args:
        files: List of uploaded file objects
    
    Returns:
        Merged DataFrame with all trends
    """
    if not files:
        return None
    
    frames = []
    
    for file in files:
        file_name = file.name if hasattr(file, 'name') else 'uploaded file'
        
        try:
            df = pd.read_csv(file)
            
            # Detect date column (Week or Month for GT)
            date_col = None
            for col_name in ['Week', 'Month', 'Date', 'date']:
                if col_name in df.columns:
                    date_col = col_name
                    break
            
            if date_col is None:
                date_col = detect_date_column(df)
            
            # Get series columns
            series_cols = [c for c in df.columns if c != date_col]
            
            if not series_cols:
                st.warning(f"⚠️ {file_name}: No data columns found")
                continue
            
            # Process
            df = df[[date_col] + series_cols].copy()
            df = df.rename(columns={date_col: 'date'})
            df['date'] = to_datetime_safe(df['date'])
            
            # Remove invalid dates
            df = df.dropna(subset=['date'])
            
            if df.empty:
                st.warning(f"⚠️ {file_name}: No valid dates")
                continue
            
            # Rename with gt__ prefix
            new_cols = [f"gt__{slugify(col)}" for col in series_cols]
            df.columns = ['date'] + new_cols
            
            # Convert to numeric
            for col in new_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            frames.append(df)
            st.success(f"✅ {file_name}: {len(series_cols)} trends")
        
        except Exception as e:
            st.warning(f"⚠️ Error loading {file_name}: {str(e)}")
            continue
    
    if not frames:
        return None
    
    # Merge all Google Trends files
    result = frames[0]
    for i, df in enumerate(frames[1:], 1):
        result = pd.merge(result, df, on='date', how='outer')
    
    result = result.sort_values('date').reset_index(drop=True)
    
    st.info(f"ℹ️ Merged {len(frames)} Google Trends files → {len(result.columns)-1} total series")
    
    return result

# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_to_monthly(
    df: pd.DataFrame,
    method: str = 'mean',
    business_days_only: bool = False,
    min_days: int = 10
) -> pd.DataFrame:
    """
    Aggregate daily data to monthly with various methods
    
    Args:
        df: Daily DataFrame with 'date' column
        method: Aggregation method ('mean', 'sum', 'last')
        business_days_only: If True, exclude weekends
        min_days: Minimum days required per month
    
    Returns:
        Monthly aggregated DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df['date'] = to_datetime_safe(df['date'])
    
    # Filter business days if requested
    if business_days_only:
        before_count = len(df)
        df = df[df['date'].dt.dayofweek < 5]
        after_count = len(df)
        
        if after_count < before_count:
            st.info(f"ℹ️ Filtered to business days: {after_count}/{before_count} days kept")
    
    # Set index and resample
    df = df.set_index('date')
    
    # Count observations per month
    counts = df.resample('M').count()
    
    # Aggregate
    if method == 'sum':
        monthly = df.resample('M').sum(min_count=1)
    elif method == 'last':
        monthly = df.resample('M').last()
    else:  # mean
        monthly = df.resample('M').mean()
    
    # Apply minimum days filter
    for col in monthly.columns:
        monthly.loc[counts[col] < min_days, col] = np.nan
    
    # Count filtered months
    filtered_months = (counts < min_days).any(axis=1).sum()
    if filtered_months > 0:
        st.info(f"ℹ️ Set {filtered_months} months to NaN (fewer than {min_days} days)")
    
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
    """
    Build unified monthly panel from all data sources
    
    Args:
        daily_frames: List of daily DataFrames
        trends_df: Google Trends DataFrame
        method: Aggregation method
        business_days: Use business days only
        min_days: Minimum days per month
    
    Returns:
        Unified monthly panel
    """
    panel = None
    
    # Process daily files
    for i, df in enumerate(daily_frames, 1):
        if df is None or df.empty:
            continue
        
        st.write(f"📊 Processing daily file {i}/{len(daily_frames)}...")
        
        monthly = aggregate_to_monthly(df, method, business_days, min_days)
        
        if panel is None:
            panel = monthly
        else:
            panel = pd.merge(panel, monthly, on='date', how='outer')
        
        st.success(f"✅ File {i}: Added {len(monthly.columns)-1} series")
    
    # Process Google Trends
    if trends_df is not None and not trends_df.empty:
        st.write("🔍 Processing Google Trends...")
        
        # GT data is already weekly/monthly, resample to monthly mean
        gt_monthly = trends_df.set_index('date').resample('M').mean().reset_index()
        gt_monthly['date'] = end_of_month(gt_monthly['date'])
        
        if panel is None:
            panel = gt_monthly
        else:
            panel = pd.merge(panel, gt_monthly, on='date', how='outer')
        
        st.success(f"✅ Google Trends: Added {len(gt_monthly.columns)-1} series")
    
    if panel is None or panel.empty:
        return pd.DataFrame()
    
    # Finalize
    panel = panel.sort_values('date').set_index('date')
    
    # Ensure all columns are numeric
    for col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors='coerce')
    
    return panel


def create_quarterly_panel(monthly_panel: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    """
    Aggregate monthly panel to quarterly with validation
    
    Args:
        monthly_panel: Monthly DataFrame with DatetimeIndex
        method: Aggregation method ('mean', 'sum', 'last')
    
    Returns:
        Quarterly aggregated DataFrame
    """
    if monthly_panel is None or monthly_panel.empty:
        st.warning("⚠️ Cannot create quarterly panel: monthly panel is empty")
        return pd.DataFrame()
    
    # Validate we have enough data
    if len(monthly_panel) < 3:
        st.warning(f"⚠️ Need at least 3 months for quarterly panel (have {len(monthly_panel)})")
        return pd.DataFrame()
    
    try:
        # Ensure DatetimeIndex
        if not isinstance(monthly_panel.index, pd.DatetimeIndex):
            st.error("❌ Monthly panel index must be DatetimeIndex for quarterly aggregation")
            return pd.DataFrame()
        
        st.write(f"📅 Aggregating {len(monthly_panel)} months to quarterly...")
        
        # Resample
        if method == 'last':
            quarterly = monthly_panel.resample('Q').last()
        elif method == 'sum':
            quarterly = monthly_panel.resample('Q').sum(min_count=1)
        else:  # mean
            quarterly = monthly_panel.resample('Q').mean()
        
        # Drop completely empty quarters
        quarterly = quarterly.dropna(how='all')
        
        if quarterly.empty:
            st.warning("⚠️ Quarterly aggregation resulted in empty DataFrame")
            return pd.DataFrame()
        
        # Count valid values per quarter
        coverage = quarterly.notna().mean().mean()
        
        st.success(f"✅ Created quarterly panel: {len(quarterly)} quarters, {coverage:.1%} coverage")
        
        return quarterly
    
    except Exception as e:
        st.error(f"❌ Error creating quarterly panel: {str(e)}")
        
        with st.expander("🐛 Error Details", expanded=False):
            st.code(traceback.format_exc(), language='python')
        
        return pd.DataFrame()

# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def remove_constant_columns(df: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    """
    Remove columns with zero or near-zero variance
    
    Args:
        df: DataFrame to clean
        tolerance: Variance tolerance threshold
    
    Returns:
        DataFrame with constant columns removed
    """
    if df is None or df.empty:
        return df
    
    keep_cols = []
    removed_cols = []
    
    for col in df.columns:
        values = df[col].dropna().values
        
        if len(values) == 0:
            removed_cols.append(col)
            continue
        
        # Peak-to-peak range (max - min)
        if np.ptp(values) > tolerance:
            keep_cols.append(col)
        else:
            removed_cols.append(col)
    
    if removed_cols:
        st.info(f"🧹 Removed {len(removed_cols)} constant columns")
        
        with st.expander("🔍 View removed constant columns", expanded=False):
            for col in removed_cols[:20]:  # Show first 20
                st.write(f"- `{col}`")
            if len(removed_cols) > 20:
                st.write(f"... and {len(removed_cols)-20} more")
    
    return df[keep_cols] if keep_cols else pd.DataFrame()


def remove_correlated_duplicates(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove highly correlated duplicate columns
    
    Args:
        df: DataFrame to clean
        threshold: Correlation threshold (0-1)
    
    Returns:
        DataFrame with correlated columns removed
    """
    if df is None or df.empty:
        return df
    
    if len(df.columns) < 2:
        return df
    
    st.write(f"🔍 Checking for correlations > {threshold:.0%}...")
    
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Get upper triangle (avoid double-counting)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find columns to drop
    to_drop = []
    dropped_pairs = []
    
    for col in upper.columns:
        correlated = upper[col][upper[col] > threshold]
        if len(correlated) > 0:
            to_drop.append(col)
            for other_col, corr_val in correlated.items():
                dropped_pairs.append((col, other_col, corr_val))
    
    if to_drop:
        st.info(f"🧹 Removed {len(to_drop)} highly correlated columns (>{threshold:.0%})")
        
        with st.expander(f"🔍 View {len(dropped_pairs)} correlation pairs", expanded=False):
            pairs_df = pd.DataFrame(dropped_pairs, columns=['Column 1', 'Column 2', 'Correlation'])
            pairs_df = pairs_df.sort_values('Correlation', ascending=False)
            st.dataframe(
                pairs_df.style.format({'Correlation': '{:.4f}'}),
                use_container_width=True
            )
    else:
        st.success(f"✅ No highly correlated columns found (threshold: {threshold:.0%})")
    
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
            monthly_panel.reset_index().to_excel(
                writer, 
                sheet_name='Monthly_Panel', 
                index=False
            )
        
        # Quarterly panel
        if quarterly_panel is not None and not quarterly_panel.empty:
            quarterly_panel.reset_index().to_excel(
                writer, 
                sheet_name='Quarterly_Panel', 
                index=False
            )
        
        # Target
        if target is not None and not target.empty:
            target.reset_index().to_excel(
                writer, 
                sheet_name='Target', 
                index=False
            )
        
        # Config
        config_df = pd.DataFrame([config])
        config_df.to_excel(
            writer, 
            sheet_name='Configuration', 
            index=False
        )
    
    return output.getvalue()

# =============================================================================
# UI CONFIGURATION - NO st.set_page_config (handled in app.py)
# =============================================================================

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
    .step-header {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
    }
    .success-box {
        background: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: #DBEAFE;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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
        index=0,
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
        value=DEFAULT_MIN_DAYS,
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
            value=DEFAULT_CORR_THRESHOLD,
            step=0.01,
            format="%.2f",
            help="Columns with correlation > threshold will be removed"
        )
    else:
        corr_threshold = DEFAULT_CORR_THRESHOLD
    
    st.markdown("---")
    st.subheader("💾 Export")
    export_format = st.radio(
        "Format:", 
        ['CSV', 'Excel'],
        help="Excel includes all data in separate sheets"
    )
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All Data", use_container_width=True):
        state.y_monthly = None
        state.panel_monthly = None
        state.panel_quarterly = None
        state.raw_daily = []
        state.google_trends = None
        st.success("✅ All data cleared!")
        st.rerun()

# =============================================================================
# STEP 1: UPLOAD DATA
# =============================================================================

st.markdown('<div class="step-header"><h2>📤 Step 1: Upload Data</h2></div>', unsafe_allow_html=True)

st.info("""
📌 **Upload Requirements:**
- **Target**: Monthly unemployment data (CSV with date + value columns)
- **Daily Data**: Financial indicators, stock prices, etc. (optional)
- **Google Trends**: Weekly/monthly trends data (optional)

All files should be in CSV format with proper date columns.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 🎯 Target Variable")
    target_file = st.file_uploader(
        "Monthly target (CSV)",
        type=['csv'],
        help="CSV with date and unemployment rate",
        key='target'
    )
    
    if target_file:
        with st.spinner("🔄 Loading target..."):
            loaded_target = load_target_series(target_file)
            
            if loaded_target is not None:
                state.y_monthly = loaded_target
            else:
                st.markdown("""
                <div class="warning-box">
                <strong>⚠️ Target loading failed</strong><br>
                Check the error messages above for details.
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.markdown("##### 📊 Daily Data")
    daily_files = st.file_uploader(
        "Daily time series (CSV)",
        type=['csv'],
        accept_multiple_files=True,
        help="Stock prices, VIX, financial indicators, etc.",
        key='daily'
    )
    
    if daily_files:
        with st.spinner("🔄 Loading daily files..."):
            state.raw_daily = []
            
            for file in daily_files:
                df = load_daily_data(file)
                if df is not None:
                    state.raw_daily.append(df)
            
            if state.raw_daily:
                total_cols = sum(len(df.columns) - 1 for df in state.raw_daily)
                st.success(f"✅ Loaded {len(state.raw_daily)} files with {total_cols} total series")

with col3:
    st.markdown("##### 🔍 Google Trends")
    trends_files = st.file_uploader(
        "Google Trends (CSV)",
        type=['csv'],
        accept_multiple_files=True,
        help="Weekly or monthly trends data",
        key='trends'
    )
    
    if trends_files:
        with st.spinner("🔄 Loading trends..."):
            state.google_trends = load_google_trends(trends_files)
            
            if state.google_trends is not None:
                n_series = len(state.google_trends.columns) - 1
                st.success(f"✅ Loaded {n_series} trend series")

# Preview target
if state.y_monthly is not None and not state.y_monthly.empty:
    st.markdown("---")
    st.markdown("#### 📈 Target Preview")
    
    fig = plot_target_series(state.y_monthly)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Observations", len(state.y_monthly))
    col2.metric("📅 Start", state.y_monthly.index.min().strftime('%Y-%m'))
    col3.metric("📅 End", state.y_monthly.index.max().strftime('%Y-%m'))
    col4.metric("📈 Mean", f"{state.y_monthly.mean():.2f}")

# =============================================================================
# STEP 2: BUILD PANEL
# =============================================================================

st.markdown('<div class="step-header"><h2>🔨 Step 2: Build Panel</h2></div>', unsafe_allow_html=True)

if not state.raw_daily and state.google_trends is None:
    st.info("📌 Upload daily data or Google Trends to build a panel")
else:
    st.info(f"""
    **Ready to build panel:**
    - Daily files: {len(state.raw_daily)}
    - Google Trends: {'Yes' if state.google_trends is not None else 'No'}
    - Target: {'Yes' if state.y_monthly is not None else 'No'}
    
    Click the button below to aggregate daily data to monthly and create the panel.
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        build_button = st.button(
            "🚀 Build Panel", 
            use_container_width=True, 
            type="primary"
        )
    
    if build_button:
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Container for detailed logs
        with st.container():
            st.markdown("### 📋 Build Process Log")
            log_placeholder = st.empty()
            
            logs = []
            
            try:
                # Step 1: Aggregate
                logs.append("📊 **Step 1/4:** Aggregating daily data to monthly...")
                log_placeholder.markdown("\n\n".join(logs))
                status.text("📊 Aggregating daily data...")
                progress_bar.progress(0.2)
                
                panel = build_panel(
                    state.raw_daily,
                    state.google_trends,
                    agg_method,
                    use_business_days,
                    min_days
                )
                
                if panel.empty:
                    logs.append("❌ **Failed:** Panel is empty")
                    log_placeholder.markdown("\n\n".join(logs))
                    
                    st.error("❌ Failed to build panel")
                    st.info("""
                    **Possible reasons:**
                    - No overlapping dates between files
                    - All values filtered out by min_days threshold
                    - Date format incompatibilities
                    """)
                    st.stop()
                
                logs.append(f"✅ **Initial panel:** {panel.shape[0]} months × {panel.shape[1]} features")
                log_placeholder.markdown("\n\n".join(logs))
                progress_bar.progress(0.4)
                
                # Step 2: Clean
                logs.append("🧹 **Step 2/4:** Cleaning panel...")
                log_placeholder.markdown("\n\n".join(logs))
                status.text("🧹 Cleaning panel...")
                
                if drop_constant:
                    original_cols = panel.shape[1]
                    panel = remove_constant_columns(panel)
                    removed = original_cols - panel.shape[1]
                    
                    if removed > 0:
                        logs.append(f"   🧹 Removed {removed} constant columns")
                    else:
                        logs.append(f"   ✅ No constant columns found")
                    
                    log_placeholder.markdown("\n\n".join(logs))
                
                progress_bar.progress(0.6)
                
                if drop_correlated:
                    original_cols = panel.shape[1]
                    panel = remove_correlated_duplicates(panel, corr_threshold)
                    removed = original_cols - panel.shape[1]
                    
                    if removed > 0:
                        logs.append(f"   🧹 Removed {removed} correlated columns (>{corr_threshold:.0%})")
                    else:
                        logs.append(f"   ✅ No highly correlated columns (threshold: {corr_threshold:.0%})")
                    
                    log_placeholder.markdown("\n\n".join(logs))
                
                progress_bar.progress(0.7)
                
                # Step 3: Align with target
                logs.append("🎯 **Step 3/4:** Aligning with target period...")
                log_placeholder.markdown("\n\n".join(logs))
                status.text("🎯 Aligning with target...")
                
                if state.y_monthly is not None:
                    original_len = len(panel)
                    panel = panel.loc[
                        (panel.index >= state.y_monthly.index.min()) &
                        (panel.index <= state.y_monthly.index.max())
                    ]
                    
                    filtered = original_len - len(panel)
                    logs.append(f"   ✅ Aligned to target: {len(panel)} months (filtered {filtered} months)")
                else:
                    logs.append(f"   ℹ️ No target loaded - keeping all dates")
                
                log_placeholder.markdown("\n\n".join(logs))
                progress_bar.progress(0.85)
                
                # Step 4: Create quarterly
                logs.append("📅 **Step 4/4:** Creating quarterly panel...")
                log_placeholder.markdown("\n\n".join(logs))
                status.text("📅 Creating quarterly panel...")
                
                state.panel_monthly = panel
                state.panel_quarterly = create_quarterly_panel(panel, agg_method)
                
                if state.panel_quarterly is not None and not state.panel_quarterly.empty:
                    logs.append(f"   ✅ Quarterly panel: {state.panel_quarterly.shape[0]} quarters × {state.panel_quarterly.shape[1]} features")
                else:
                    logs.append(f"   ⚠️ Quarterly panel empty (need ≥3 months)")
                
                log_placeholder.markdown("\n\n".join(logs))
                progress_bar.progress(1.0)
                
                # Clear progress indicators
                status.empty()
                progress_bar.empty()
                
                # Final summary
                st.markdown("---")
                st.markdown("### ✅ Panel Build Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Monthly Panel:**")
                    st.metric("Months", panel.shape[0])
                    st.metric("Features", panel.shape[1])
                    coverage = panel.notna().mean().mean()
                    st.metric("Coverage", f"{coverage:.1%}")
                
                with col2:
                    st.markdown("**Quarterly Panel:**")
                    if state.panel_quarterly is not None and not state.panel_quarterly.empty:
                        st.metric("Quarters", state.panel_quarterly.shape[0])
                        st.metric("Features", state.panel_quarterly.shape[1])
                        q_coverage = state.panel_quarterly.notna().mean().mean()
                        st.metric("Coverage", f"{q_coverage:.1%}")
                    else:
                        st.info("Not available")
                
                with col3:
                    st.markdown("**Data Quality:**")
                    if state.y_monthly is not None:
                        overlap = panel.index.intersection(state.y_monthly.index)
                        st.metric("Target Overlap", f"{len(overlap)} months")
                    
                    nulls = panel.isna().sum().sum()
                    total = panel.size
                    st.metric("Missing Values", f"{nulls:,} ({nulls/total*100:.1f}%)")
                
                st.success("🎉 Panel is ready! Proceed to **Feature Engineering** or **Backtesting**")
            
            except Exception as e:
                status.empty()
                progress_bar.empty()
                
                logs.append(f"❌ **Error:** {type(e).__name__}")
                logs.append(f"```\n{str(e)}\n```")
                log_placeholder.markdown("\n\n".join(logs))
                
                st.error(f"❌ Error building panel: {type(e).__name__}")
                
                with st.expander("🐛 Full Error Traceback", expanded=True):
                    st.code(traceback.format_exc(), language='python')

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
        if state.panel_quarterly is not None and not state.panel_quarterly.empty:
            st.metric("📅 Quarters", state.panel_quarterly.shape[0])
        else:
            st.metric("📅 Quarters", "N/A")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Preview", 
        "🔍 Coverage Analysis", 
        "📈 Correlations", 
        "💾 Export"
    ])
    
    with tab1:
        st.subheader("Monthly Panel (Last 24 Months)")
        st.dataframe(
            state.panel_monthly.tail(24).style.format("{:.4f}"),
            use_container_width=True,
            height=400
        )
        
        if state.panel_quarterly is not None and not state.panel_quarterly.empty:
            st.markdown("---")
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
            'Missing': state.panel_monthly.isna().sum().values,
            'Valid': state.panel_monthly.notna().sum().values
        }).sort_values('Coverage', ascending=False)
        
        st.dataframe(
            coverage_df.style.format({
                'Coverage': '{:.1%}',
                'Missing': '{:,.0f}',
                'Valid': '{:,.0f}'
            }).background_gradient(subset=['Coverage'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # Heatmap
        st.markdown("---")
        st.subheader("Data Presence Heatmap")
        heatmap_fig = plot_coverage_heatmap(state.panel_monthly, n_months=60)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with tab3:
        if state.y_monthly is not None and not state.y_monthly.empty:
            st.subheader("Correlation with Target")
            
            corr_fig = plot_correlation_with_target(
                state.panel_monthly,
                state.y_monthly,
                top_n=20
            )
            
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Top correlations table
                y_aligned, X_aligned = state.y_monthly.align(
                    state.panel_monthly.select_dtypes(include=[np.number]),
                    join='inner'
                )
                
                if not X_aligned.empty:
                    correlations = X_aligned.corrwith(y_aligned).sort_values(ascending=False)
                    
                    st.markdown("---")
                    st.subheader("Top 10 Positive and Negative Correlations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Most Positive:**")
                        top_pos = correlations.head(10)
                        st.dataframe(
                            pd.DataFrame(top_pos, columns=['Correlation']).style.format('{:.4f}'),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.markdown("**Most Negative:**")
                        top_neg = correlations.tail(10)
                        st.dataframe(
                            pd.DataFrame(top_neg, columns=['Correlation']).style.format('{:.4f}'),
                            use_container_width=True
                        )
            else:
                st.info("No overlapping data with target for correlation analysis")
        else:
            st.info("💡 Upload target variable to see correlations with features")
    
    with tab4:
        st.subheader("💾 Export Data")
        
        # Configuration summary
        config = {
            'aggregation_method': agg_method,
            'business_days_only': use_business_days,
            'min_days_per_month': min_days,
            'drop_constant': drop_constant,
            'drop_correlated': drop_correlated,
            'correlation_threshold': corr_threshold,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'monthly_shape': f"{state.panel_monthly.shape[0]} × {state.panel_monthly.shape[1]}",
            'quarterly_shape': f"{state.panel_quarterly.shape[0]} × {state.panel_quarterly.shape[1]}" if state.panel_quarterly is not None else "N/A"
        }
        
        with st.expander("⚙️ Configuration Summary", expanded=False):
            st.json(config)
        
        st.markdown("---")
        
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
                
                if state.panel_quarterly is not None and not state.panel_quarterly.empty:
                    csv_q = state.panel_quarterly.to_csv().encode('utf-8')
                    st.download_button(
                        "📥 Download Quarterly Panel (CSV)",
                        csv_q,
                        "quarterly_panel.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:  # Excel
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
            st.info("""
            **💡 Export Tips:**
            
            - **CSV**: Individual files, easy to read
            - **Excel**: All data in one file with multiple sheets:
              - Monthly Panel
              - Quarterly Panel
              - Target Series
              - Configuration
            
            Use Excel for complete project backup!
            """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    if state.panel_monthly is not None and not state.panel_monthly.empty:
        st.caption(f"📊 Panel: {state.panel_monthly.shape[1]} features ready")
    else:
        st.caption("📊 No panel built yet")

with col2:
    if state.y_monthly is not None and not state.y_monthly.empty:
        st.caption(f"🎯 Target: {len(state.y_monthly)} months loaded")
    else:
        st.caption("🎯 No target loaded")

with col3:
    st.caption("💻 Built with Streamlit Pro v2.1")

st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.875rem; margin-top: 1rem;'>
    <strong>Next Steps:</strong> Go to <strong>Feature Engineering</strong> to create features, 
    then <strong>Backtesting</strong> to train models
</div>
""", unsafe_allow_html=True)
