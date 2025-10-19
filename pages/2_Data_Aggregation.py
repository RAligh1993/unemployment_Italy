"""
📊 Data Aggregation Pro MAX v4.0
===================================
Professional multi-format data intake & panel builder for Italian unemployment nowcasting.
Optimized for ISTAT, Eurostat, and custom data sources.

Features:
✅ Excel Support (.xlsx, .xls, .xlsm, .xlsb) + CSV
✅ AI-Powered Auto-Detection (columns, frequency, data type)
✅ ISTAT Italia Optimization (regions, categories, formats)
✅ Smart Preprocessing (outliers, missing data, seasonality)
✅ Multi-target & Multi-frequency (daily, weekly, monthly, quarterly)
✅ Interactive UI with drag-and-drop mapping
✅ Professional visualizations

Author: AI Assistant for ISTAT Nowcasting Lab
Date: October 2025
Version: 4.0.0 - Complete Rewrite
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO
import json
import re
import traceback
from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Excel libraries
try:
    import openpyxl
    HAS_OPENPYXL = True
except:
    HAS_OPENPYXL = False

try:
    import xlrd
    HAS_XLRD = True
except:
    HAS_XLRD = False

# Optional advanced features
try:
    from scipy import stats
    HAS_SCIPY = True
except:
    HAS_SCIPY = False

try:
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False


# =============================================================================
# 🎯 CONFIGURATION & CONSTANTS
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data processing"""
    # File settings
    max_file_size_mb: int = 200
    supported_formats: List[str] = field(default_factory=lambda: [
        '.csv', '.xlsx', '.xls', '.xlsm', '.xlsb', '.ods'
    ])
    encoding_attempts: List[str] = field(default_factory=lambda: [
        'utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16'
    ])
    
    # Data validation
    min_observations: int = 12
    max_missing_pct: float = 0.5
    min_coverage: float = 0.3
    
    # Processing
    outlier_std_threshold: float = 3.5
    correlation_threshold: float = 0.95
    constant_threshold: float = 1e-12


# Italian regions (20 regioni)
ITALIAN_REGIONS = {
    'piemonte': 'Piemonte',
    'valle_aosta': "Valle d'Aosta",
    'lombardia': 'Lombardia',
    'trentino': 'Trentino-Alto Adige',
    'veneto': 'Veneto',
    'friuli': 'Friuli-Venezia Giulia',
    'liguria': 'Liguria',
    'emilia': 'Emilia-Romagna',
    'toscana': 'Toscana',
    'umbria': 'Umbria',
    'marche': 'Marche',
    'lazio': 'Lazio',
    'abruzzo': 'Abruzzo',
    'molise': 'Molise',
    'campania': 'Campania',
    'puglia': 'Puglia',
    'basilicata': 'Basilicata',
    'calabria': 'Calabria',
    'sicilia': 'Sicilia',
    'sardegna': 'Sardegna'
}

# Unemployment categories with multilingual support
UNEMPLOYMENT_KEYWORDS = {
    'total': ['total', 'totale', 'unemployment', 'disoccupazione', 'tasso'],
    'male': ['male', 'uomini', 'maschi', 'men', 'masculino'],
    'female': ['female', 'donne', 'femmine', 'women', 'femenino'],
    'youth': ['youth', 'giovani', 'young', '15-24', '<25'],
    'adult': ['adult', 'adulti', '25+', '25-64'],
    'long_term': ['long', 'lungo', 'structural', 'strutturale', '>12'],
    'educated': ['education', 'istruzione', 'tertiary', 'terziaria', 'laurea'],
    'rate': ['rate', 'tasso', 'percentage', 'percentuale', '%']
}

# Date patterns
DATE_PATTERNS = {
    'italian': r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # 31/12/2023
    'iso': r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',      # 2023-12-31
    'year_month': r'(\d{4})[/-](\d{1,2})',            # 2023-12
    'quarter': r'(\d{4})[Qq]([1-4])',                 # 2023Q4
    'month_year': r'(\w+)\s+(\d{4})',                 # December 2023
}

# Known ISTAT/Eurostat column names
ISTAT_COLUMNS = {
    'date': ['time', 'periodo', 'periodo_riferimento', 'data', 'anno', 'mese', 'trimestre'],
    'value': ['valore', 'value', 'obs_value', 'dato'],
    'geo': ['geo', 'territorio', 'regione', 'area'],
    'category': ['categoria', 'tipologia', 'tipo', 'classification']
}


# =============================================================================
# 🛠️ CORE UTILITY FUNCTIONS
# =============================================================================

class ColorLogger:
    """Beautiful console logging for Streamlit"""
    
    @staticmethod
    def info(msg: str):
        st.markdown(f"ℹ️ {msg}")
    
    @staticmethod
    def success(msg: str):
        st.success(f"✅ {msg}")
    
    @staticmethod
    def warning(msg: str):
        st.warning(f"⚠️ {msg}")
    
    @staticmethod
    def error(msg: str):
        st.error(f"❌ {msg}")
    
    @staticmethod
    def debug(msg: str):
        with st.expander("🐛 Debug Info", expanded=False):
            st.code(msg)


def slugify(text: str) -> str:
    """Convert text to clean identifier"""
    text = str(text).lower().strip()
    
    # Italian character replacements
    replacements = {
        'à': 'a', 'è': 'e', 'é': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
        'ñ': 'n', 'ç': 'c'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Clean special characters
    text = re.sub(r'[^\w\s-]', '_', text)
    text = re.sub(r'[-\s]+', '_', text)
    text = re.sub(r'_+', '_', text)
    
    return text.strip('_')


def calculate_file_hash(file_content: bytes) -> str:
    """Calculate file hash for caching"""
    import hashlib
    return hashlib.md5(file_content).hexdigest()


def format_bytes(size: int) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def detect_decimal_separator(series: pd.Series) -> str:
    """Detect if comma or dot is decimal separator"""
    sample = series.dropna().astype(str).head(100)
    
    comma_count = sum(',' in x and '.' not in x for x in sample)
    dot_count = sum('.' in x and ',' not in x for x in sample)
    
    if comma_count > dot_count:
        return ','
    return '.'


def italian_to_numeric(value: Any) -> float:
    """Convert Italian number format to float"""
    if pd.isna(value):
        return np.nan
    
    value = str(value).strip()
    
    # Remove currency symbols
    value = value.replace('€', '').replace('EUR', '').strip()
    
    # Italian format: 1.234.567,89 -> 1234567.89
    if ',' in value and '.' in value:
        value = value.replace('.', '').replace(',', '.')
    elif ',' in value:
        value = value.replace(',', '.')
    
    try:
        return float(value)
    except:
        return np.nan


# =============================================================================
# 📊 EXCEL PROCESSING ENGINE
# =============================================================================

class ExcelProcessor:
    """Advanced Excel file processor"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = ColorLogger()
    
    def read_excel_file(self, file) -> Dict[str, pd.DataFrame]:
        """
        Read Excel file and return all sheets
        
        Returns:
            Dict of {sheet_name: DataFrame}
        """
        file_name = file.name if hasattr(file, 'name') else 'uploaded_file'
        file_ext = Path(file_name).suffix.lower()
        
        self.logger.info(f"Reading Excel file: **{file_name}** ({file_ext})")
        
        # Check size
        if hasattr(file, 'size'):
            size_mb = file.size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                self.logger.error(
                    f"File too large: {size_mb:.1f}MB (max: {self.config.max_file_size_mb}MB)"
                )
                return {}
        
        sheets = {}
        
        try:
            # Reset file pointer
            file.seek(0)
            
            # Try different engines
            engines = []
            if file_ext in ['.xlsx', '.xlsm']:
                engines = ['openpyxl', 'xlrd', None]
            elif file_ext == '.xls':
                engines = ['xlrd', None]
            elif file_ext == '.xlsb':
                engines = ['pyxlsb', None]
            else:
                engines = [None]
            
            last_error = None
            
            for engine in engines:
                try:
                    file.seek(0)
                    
                    # Get all sheet names
                    excel_file = pd.ExcelFile(file, engine=engine)
                    sheet_names = excel_file.sheet_names
                    
                    self.logger.success(
                        f"Found {len(sheet_names)} sheets: {', '.join(sheet_names)}"
                    )
                    
                    # Read all sheets
                    for sheet_name in sheet_names:
                        try:
                            df = pd.read_excel(
                                excel_file,
                                sheet_name=sheet_name,
                                header=None  # We'll detect header ourselves
                            )
                            
                            if not df.empty:
                                sheets[sheet_name] = df
                                self.logger.info(
                                    f"  📄 {sheet_name}: {df.shape[0]} × {df.shape[1]}"
                                )
                        
                        except Exception as e:
                            self.logger.warning(f"  ⚠️ Could not read sheet '{sheet_name}': {str(e)}")
                            continue
                    
                    if sheets:
                        break
                
                except Exception as e:
                    last_error = e
                    continue
            
            if not sheets:
                if last_error:
                    raise last_error
                else:
                    raise ValueError("No sheets could be read")
        
        except Exception as e:
            self.logger.error(f"Error reading Excel: {type(e).__name__}")
            self.logger.error(str(e))
            
            with st.expander("🐛 Full Error Traceback", expanded=False):
                st.code(traceback.format_exc())
            
            return {}
        
        return sheets
    
    def detect_header_row(self, df: pd.DataFrame) -> int:
        """
        Detect which row contains the header
        
        Returns:
            Row index (0-based)
        """
        if df.empty:
            return 0
        
        max_rows_to_check = min(10, len(df))
        
        best_row = 0
        best_score = 0
        
        for i in range(max_rows_to_check):
            row = df.iloc[i]
            
            score = 0
            
            # Count non-null values
            score += row.notna().sum()
            
            # Check for text (good for headers)
            text_count = sum(isinstance(x, str) for x in row)
            score += text_count * 2
            
            # Check for known keywords
            row_str = ' '.join(str(x).lower() for x in row if pd.notna(x))
            
            for keywords in [ISTAT_COLUMNS['date'], UNEMPLOYMENT_KEYWORDS['total']]:
                if any(kw in row_str for kw in keywords):
                    score += 5
            
            # Penalize if mostly numbers (probably data, not header)
            numeric_count = sum(isinstance(x, (int, float)) for x in row)
            if numeric_count > len(row) * 0.7:
                score -= 10
            
            if score > best_score:
                best_score = score
                best_row = i
        
        return best_row
    
    def process_sheet(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """
        Process a single sheet
        
        Steps:
        1. Detect and set header
        2. Remove empty rows/columns
        3. Clean column names
        4. Basic type inference
        """
        self.logger.info(f"Processing sheet: **{sheet_name}**")
        
        # Step 1: Detect header
        header_row = self.detect_header_row(df)
        
        if header_row > 0:
            self.logger.info(f"  🔍 Header detected at row {header_row + 1}")
            df = df.iloc[header_row:].reset_index(drop=True)
        
        # Set first row as header
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        
        # Step 2: Remove empty rows/columns
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        # Step 3: Clean column names
        df.columns = [
            slugify(str(col)) if pd.notna(col) else f'col_{i}'
            for i, col in enumerate(df.columns)
        ]
        
        # Remove duplicate column names
        cols = df.columns.tolist()
        seen = {}
        for i, col in enumerate(cols):
            if col in seen:
                seen[col] += 1
                cols[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
        df.columns = cols
        
        self.logger.success(f"  ✅ Processed: {len(df)} rows × {len(df.columns)} columns")
        
        return df


# =============================================================================
# 🤖 AI-POWERED AUTO-DETECTION
# =============================================================================

class DataDetector:
    """Intelligent data type and structure detector"""
    
    def __init__(self):
        self.logger = ColorLogger()
    
    def detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect date column with high confidence
        
        Returns:
            Column name or None
        """
        candidates = []
        
        for col in df.columns:
            score = 0
            col_lower = str(col).lower()
            
            # Check column name
            date_keywords = ['date', 'data', 'time', 'periodo', 'anno', 'mese', 
                           'year', 'month', 'quarter', 'trimestre', 'settimana']
            
            if any(kw in col_lower for kw in date_keywords):
                score += 10
            
            # Check if ISTAT known column
            if any(kw == col_lower for kw in ISTAT_COLUMNS['date']):
                score += 20
            
            # Check data type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                score += 30
            
            # Try to parse as date
            try:
                sample = df[col].dropna().head(20)
                parsed = pd.to_datetime(sample, errors='coerce')
                parse_rate = parsed.notna().mean()
                
                if parse_rate > 0.8:
                    score += int(parse_rate * 20)
            except:
                pass
            
            # Check for date patterns in text
            try:
                sample_str = df[col].dropna().astype(str).head(20)
                pattern_matches = sum(
                    bool(re.search(pattern, s))
                    for s in sample_str
                    for pattern in DATE_PATTERNS.values()
                )
                
                if pattern_matches > 10:
                    score += 15
            except:
                pass
            
            if score > 0:
                candidates.append((col, score))
        
        if not candidates:
            return None
        
        # Return highest scoring
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_col, best_score = candidates[0]
        
        if best_score >= 20:
            self.logger.success(f"Date column detected: **{best_col}** (confidence: {best_score})")
            return best_col
        
        self.logger.warning(f"Uncertain date column: {best_col} (confidence: {best_score})")
        return best_col
    
    def detect_unemployment_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect unemployment-related columns
        
        Returns:
            Dict of {category: column_name}
        """
        unemployment_cols = {}
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check each category
            for category, keywords in UNEMPLOYMENT_KEYWORDS.items():
                if any(kw in col_lower for kw in keywords):
                    
                    # Also check if column contains numeric data
                    try:
                        numeric_rate = pd.to_numeric(df[col], errors='coerce').notna().mean()
                        
                        if numeric_rate > 0.5:
                            unemployment_cols[category] = col
                            self.logger.info(f"  📊 {category}: **{col}**")
                    except:
                        pass
        
        if unemployment_cols:
            self.logger.success(f"Detected {len(unemployment_cols)} unemployment categories")
        
        return unemployment_cols
    
    def detect_regional_data(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect if data contains Italian regional breakdown
        
        Returns:
            Region column name or None
        """
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check column name
            if any(kw in col_lower for kw in ['geo', 'region', 'territorio', 'area']):
                
                # Check if contains Italian regions
                try:
                    values = df[col].dropna().astype(str).str.lower()
                    region_matches = sum(
                        any(region in v for region in ITALIAN_REGIONS.keys())
                        for v in values.head(50)
                    )
                    
                    if region_matches > 5:
                        self.logger.success(f"Regional data detected: **{col}**")
                        return col
                except:
                    pass
        
        return None
    
    def detect_frequency(self, dates: pd.Series) -> str:
        """
        Detect time series frequency
        
        Returns:
            'daily', 'weekly', 'monthly', 'quarterly', 'annual', or 'unknown'
        """
        if len(dates) < 2:
            return 'unknown'
        
        dates = pd.to_datetime(dates, errors='coerce').dropna().sort_values()
        
        if len(dates) < 2:
            return 'unknown'
        
        # Calculate differences
        diffs = dates.diff().dt.days.dropna()
        
        if len(diffs) == 0:
            return 'unknown'
        
        median_diff = diffs.median()
        mode_diff = diffs.mode()[0] if len(diffs.mode()) > 0 else median_diff
        
        # Classify
        if mode_diff <= 1:
            freq = 'daily'
        elif 5 <= mode_diff <= 9:
            freq = 'weekly'
        elif 28 <= mode_diff <= 31:
            freq = 'monthly'
        elif 85 <= mode_diff <= 95:
            freq = 'quarterly'
        elif 360 <= mode_diff <= 370:
            freq = 'annual'
        else:
            freq = 'unknown'
        
        self.logger.info(f"Frequency detected: **{freq}** (median: {median_diff:.0f} days)")
        
        return freq
    
    def detect_istat_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect if this is ISTAT formatted data
        
        Returns:
            Dict with format info
        """
        format_info = {
            'is_istat': False,
            'format_type': 'unknown',
            'columns_found': {}
        }
        
        # Check for ISTAT column patterns
        for col_type, keywords in ISTAT_COLUMNS.items():
            for col in df.columns:
                if str(col).lower() in keywords:
                    format_info['columns_found'][col_type] = col
        
        # If found 3+ ISTAT columns, likely ISTAT format
        if len(format_info['columns_found']) >= 3:
            format_info['is_istat'] = True
            format_info['format_type'] = 'istat_standard'
            
            self.logger.success("✅ ISTAT format detected!")
            self.logger.info(f"  Found columns: {format_info['columns_found']}")
        
        return format_info


# =============================================================================
# 🔄 DATA TRANSFORMATION ENGINE
# =============================================================================

class DataTransformer:
    """Transform and clean data"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = ColorLogger()
    
    def parse_dates(self, series: pd.Series, freq_hint: str = None) -> pd.Series:
        """
        Intelligent date parsing
        
        Args:
            series: Date column
            freq_hint: 'monthly', 'quarterly', etc.
        """
        self.logger.info("Parsing dates...")
        
        # Try standard pandas parsing first
        try:
            parsed = pd.to_datetime(series, errors='coerce')
            
            if parsed.notna().mean() > 0.8:
                parsed = parsed.dt.tz_localize(None).dt.normalize()
                self.logger.success(f"  ✅ {parsed.notna().sum()}/{len(parsed)} dates parsed")
                return parsed
        except:
            pass
        
        # Try custom patterns
        parsed = pd.Series(index=series.index, dtype='datetime64[ns]')
        
        for pattern_name, pattern in DATE_PATTERNS.items():
            for i, value in series.items():
                if pd.notna(parsed[i]):
                    continue
                
                try:
                    match = re.search(pattern, str(value))
                    
                    if match:
                        if pattern_name == 'italian':
                            day, month, year = match.groups()
                            parsed[i] = pd.Timestamp(f"{year}-{month}-{day}")
                        
                        elif pattern_name == 'iso':
                            year, month, day = match.groups()
                            parsed[i] = pd.Timestamp(f"{year}-{month}-{day}")
                        
                        elif pattern_name == 'year_month':
                            year, month = match.groups()
                            parsed[i] = pd.Timestamp(f"{year}-{month}-01")
                        
                        elif pattern_name == 'quarter':
                            year, quarter = match.groups()
                            month = int(quarter) * 3
                            parsed[i] = pd.Timestamp(f"{year}-{month:02d}-01")
                
                except:
                    continue
        
        parsed = parsed.dt.tz_localize(None).dt.normalize()
        
        # Align to end of month/quarter if needed
        if freq_hint == 'monthly':
            parsed = parsed + pd.offsets.MonthEnd(0)
        elif freq_hint == 'quarterly':
            parsed = parsed + pd.offsets.QuarterEnd(0)
        
        success_rate = parsed.notna().mean()
        self.logger.info(f"  📅 Parse rate: {success_rate:.1%}")
        
        return parsed
    
    def convert_to_numeric(self, series: pd.Series, handle_italian: bool = True) -> pd.Series:
        """
        Convert to numeric, handling Italian formats
        """
        if handle_italian:
            converted = series.apply(italian_to_numeric)
        else:
            converted = pd.to_numeric(series, errors='coerce')
        
        return converted
    
    def remove_outliers(self, series: pd.Series, method: str = 'mad') -> pd.Series:
        """
        Remove or cap outliers
        
        Args:
            method: 'mad' (Median Absolute Deviation) or 'iqr'
        """
        if not HAS_SCIPY:
            return series
        
        values = series.dropna()
        
        if len(values) < 10:
            return series
        
        if method == 'mad':
            median = values.median()
            mad = np.median(np.abs(values - median))
            
            if mad == 0:
                return series
            
            z_scores = 0.6745 * (values - median) / mad
            outliers = np.abs(z_scores) > self.config.outlier_std_threshold
        
        else:  # IQR
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            outliers = (values < lower) | (values > upper)
        
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            self.logger.info(f"  🔍 {outlier_count} outliers detected")
            
            # Replace outliers with NaN
            cleaned = series.copy()
            cleaned.loc[outliers[outliers].index] = np.nan
            
            return cleaned
        
        return series
    
    def impute_missing(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Impute missing values
        
        Args:
            method: 'interpolate', 'ffill', 'knn'
        """
        missing_pct = df.isna().sum().sum() / df.size
        
        if missing_pct == 0:
            return df
        
        self.logger.info(f"Imputing missing values ({missing_pct:.1%})...")
        
        if method == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both')
        
        elif method == 'ffill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        elif method == 'knn' and HAS_SKLEARN:
            imputer = KNNImputer(n_neighbors=5)
            df_imputed = imputer.fit_transform(df.select_dtypes(include=[np.number]))
            df.loc[:, df.select_dtypes(include=[np.number]).columns] = df_imputed
        
        self.logger.success(f"  ✅ Imputed")
        
        return df


# =============================================================================
# 📊 VISUALIZATION ENGINE
# =============================================================================

class DataVisualizer:
    """Create professional visualizations"""
    
    @staticmethod
    def plot_time_series(df: pd.DataFrame, date_col: str, value_cols: List[str], 
                        title: str = "Time Series") -> go.Figure:
        """Plot multiple time series"""
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for i, col in enumerate(value_cols):
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(width=2, color=colors[i % len(colors)]),
                marker=dict(size=4),
                hovertemplate=f'<b>{col}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
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
    
    @staticmethod
    def plot_coverage_heatmap(df: pd.DataFrame, max_cols: int = 50) -> go.Figure:
        """Plot data coverage heatmap"""
        
        # Limit columns
        if len(df.columns) > max_cols:
            df = df.iloc[:, :max_cols]
        
        presence = df.notna().astype(int).T
        
        fig = px.imshow(
            presence,
            aspect='auto',
            color_continuous_scale=['#EF4444', '#10B981'],
            labels={'color': 'Present'},
            title=f'Data Coverage ({len(df)} rows × {len(df.columns)} columns)'
        )
        
        fig.update_layout(
            height=max(400, len(df.columns) * 15),
            template='plotly_white',
            xaxis_title='Row Index',
            yaxis_title='Column'
        )
        
        return fig
    
    @staticmethod
    def plot_regional_map(df: pd.DataFrame, region_col: str, value_col: str) -> go.Figure:
        """Plot Italian regional data on map"""
        
        # This would need geojson data for Italian regions
        # Placeholder implementation
        
        fig = go.Figure(data=go.Bar(
            x=df[region_col],
            y=df[value_col],
            marker=dict(
                color=df[value_col],
                colorscale='RdYlGn_r',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title='Regional Unemployment',
            xaxis_title='Region',
            yaxis_title='Unemployment Rate (%)',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_data_quality(df: pd.DataFrame) -> go.Figure:
        """Plot data quality metrics"""
        
        metrics = {
            'Column': [],
            'Coverage': [],
            'Missing': [],
            'Unique': []
        }
        
        for col in df.columns:
            metrics['Column'].append(col)
            metrics['Coverage'].append(df[col].notna().mean())
            metrics['Missing'].append(df[col].isna().sum())
            
            try:
                metrics['Unique'].append(df[col].nunique())
            except:
                metrics['Unique'].append(0)
        
        metrics_df = pd.DataFrame(metrics)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Coverage %', 'Missing Count'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['Column'],
                y=metrics_df['Coverage'] * 100,
                name='Coverage',
                marker_color='#10B981'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['Column'],
                y=metrics_df['Missing'],
                name='Missing',
                marker_color='#EF4444'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig


# =============================================================================
# 🎯 STATE MANAGEMENT (Compatible with existing code)
# =============================================================================

try:
    from utils.state import AppState
except Exception:
    class _State:
        def __init__(self):
            # Targets
            self.y_monthly: Optional[pd.Series] = None
            self.targets_monthly: Optional[pd.DataFrame] = None
            
            # Panels
            self.panel_monthly: Optional[pd.DataFrame] = None
            self.panel_quarterly: Optional[pd.DataFrame] = None
            
            # Raw sources
            self.raw_daily: List[pd.DataFrame] = []
            self.raw_monthly: List[pd.DataFrame] = []
            self.raw_quarterly: List[pd.DataFrame] = []
            self.google_trends: Optional[pd.DataFrame] = None
            
            # Metadata
            self.data_metadata: Dict[str, Any] = {}
    
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
# 🎨 MAIN UI - PROFESSIONAL DATA UPLOAD
# =============================================================================

def main():
    """Main application"""
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-title {
            font-size: 3.5rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(120deg, #1e3a8a, #3b82f6, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .feature-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        .step-header {
            background: linear-gradient(90deg, #1e40af, #3b82f6);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            margin: 2rem 0 1rem 0;
            font-size: 1.5rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-title">📊 Data Aggregation Pro MAX v4.0</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🇮🇹 Advanced Excel & Multi-format Data Processing for Italian Unemployment Nowcasting</p>', unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📁 Excel Support</h3>
            <p>.xlsx, .xls, .xlsm, .xlsb<br/>Multi-sheet processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 Auto-Detection</h3>
            <p>Smart column recognition<br/>ISTAT format support</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>🇮🇹 Italian Optimized</h3>
            <p>20 regions support<br/>Italian number formats</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-box">
            <h3>🔧 Smart Processing</h3>
            <p>Outlier detection<br/>Missing data imputation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize components
    config = DataConfig()
    logger = ColorLogger()
    excel_processor = ExcelProcessor(config)
    detector = DataDetector()
    transformer = DataTransformer(config)
    visualizer = DataVisualizer()
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Processing Settings")
        
        st.subheader("🔍 Detection")
        auto_detect = st.checkbox("Auto-detect columns", value=True)
        handle_italian = st.checkbox("Italian number formats", value=True)
        
        st.subheader("🧹 Cleaning")
        remove_outliers = st.checkbox("Remove outliers", value=False)
        impute_missing = st.checkbox("Impute missing", value=False)
        
        if impute_missing:
            impute_method = st.selectbox(
                "Method:",
                ['interpolate', 'ffill', 'knn' if HAS_SKLEARN else 'interpolate']
            )
        
        st.subheader("📊 Output")
        align_to_month_end = st.checkbox("Align to month-end", value=True)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear All Data", use_container_width=True):
            for attr in ['y_monthly', 'targets_monthly', 'panel_monthly', 
                        'raw_daily', 'raw_monthly', 'raw_quarterly', 'google_trends']:
                if hasattr(state, attr):
                    if isinstance(getattr(state, attr), list):
                        setattr(state, attr, [])
                    else:
                        setattr(state, attr, None)
            st.success("✅ Cleared!")
            st.rerun()
    
    # Main upload area
    st.markdown('<div class="step-header">📤 Step 1: Upload Data Files</div>', unsafe_allow_html=True)
    
    st.info("""
    **📋 Supported Formats:**
    - 📊 **Excel**: .xlsx, .xls, .xlsm, .xlsb (multi-sheet support)
    - 📄 **CSV**: .csv (all encodings)
    - 🌐 **ISTAT/Eurostat**: Auto-detected formats
    
    **🎯 What We'll Detect:**
    - Date columns (multiple formats)
    - Unemployment categories (total, male, female, youth, etc.)
    - Italian regions (20 regioni)
    - Time frequency (daily, monthly, quarterly)
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📁 Upload your data files",
        type=['csv', 'xlsx', 'xls', 'xlsm', 'xlsb'],
        accept_multiple_files=True,
        help="You can upload multiple files at once"
    )
    
    if uploaded_files:
        st.markdown("---")
        st.markdown("### 🔄 Processing Files")
        
        all_dataframes = []
        
        for file in uploaded_files:
            file_name = file.name
            file_ext = Path(file_name).suffix.lower()
            
            with st.expander(f"📄 {file_name}", expanded=True):
                
                try:
                    # Process based on file type
                    if file_ext == '.csv':
                        # CSV processing
                        logger.info(f"Reading CSV: **{file_name}**")
                        
                        # Try multiple encodings
                        df = None
                        for encoding in config.encoding_attempts:
                            try:
                                file.seek(0)
                                df = pd.read_csv(file, encoding=encoding)
                                if encoding != 'utf-8':
                                    logger.info(f"  Using {encoding} encoding")
                                break
                            except:
                                continue
                        
                        if df is None:
                            logger.error("Could not read CSV with any encoding")
                            continue
                    
                    else:
                        # Excel processing
                        sheets = excel_processor.read_excel_file(file)
                        
                        if not sheets:
                            logger.error("No sheets could be read")
                            continue
                        
                        # Let user select sheet
                        if len(sheets) > 1:
                            selected_sheet = st.selectbox(
                                "Select sheet:",
                                options=list(sheets.keys()),
                                key=f"sheet_{file_name}"
                            )
                        else:
                            selected_sheet = list(sheets.keys())[0]
                        
                        df = sheets[selected_sheet]
                        
                        # Process sheet
                        df = excel_processor.process_sheet(df, selected_sheet)
                    
                    if df.empty:
                        logger.warning("Empty dataframe")
                        continue
                    
                    # Auto-detection
                    if auto_detect:
                        st.markdown("#### 🔍 Auto-Detection Results")
                        
                        # Detect date column
                        date_col = detector.detect_date_column(df)
                        
                        if date_col:
                            st.success(f"📅 Date column: **{date_col}**")
                            
                            # Parse dates
                            df[date_col] = transformer.parse_dates(df[date_col])
                            
                            # Detect frequency
                            freq = detector.detect_frequency(df[date_col])
                            st.info(f"⏱️ Frequency: **{freq}**")
                        
                        # Detect unemployment columns
                        unemp_cols = detector.detect_unemployment_columns(df)
                        
                        if unemp_cols:
                            st.success(f"📊 Found {len(unemp_cols)} unemployment indicators")
                        
                        # Detect regional data
                        region_col = detector.detect_regional_data(df)
                        
                        # Detect ISTAT format
                        istat_info = detector.detect_istat_format(df)
                        
                        if istat_info['is_istat']:
                            st.success("✅ ISTAT format confirmed!")
                    
                    # Convert numeric columns
                    st.markdown("#### 🔢 Converting to Numeric")
                    
                    numeric_cols = []
                    for col in df.columns:
                        if col == date_col:
                            continue
                        
                        df[col] = transformer.convert_to_numeric(df[col], handle_italian)
                        
                        valid_pct = df[col].notna().mean()
                        if valid_pct > 0.5:
                            numeric_cols.append(col)
                            st.write(f"  ✅ {col}: {valid_pct:.1%} valid")
                    
                    # Cleaning
                    if remove_outliers:
                        st.markdown("#### 🧹 Removing Outliers")
                        for col in numeric_cols:
                            df[col] = transformer.remove_outliers(df[col])
                    
                    if impute_missing and numeric_cols:
                        st.markdown("#### 🔧 Imputing Missing Values")
                        df[numeric_cols] = transformer.impute_missing(df[numeric_cols], impute_method)
                    
                    # Show preview
                    st.markdown("#### 👀 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        coverage = df.notna().mean().mean()
                        st.metric("Coverage", f"{coverage:.1%}")
                    with col4:
                        memory = df.memory_usage(deep=True).sum() / 1024**2
                        st.metric("Size", f"{memory:.1f} MB")
                    
                    # Visualize
                    if date_col and numeric_cols:
                        st.markdown("#### 📊 Visualization")
                        
                        # Select columns to plot
                        cols_to_plot = st.multiselect(
                            "Select columns to plot:",
                            options=numeric_cols,
                            default=numeric_cols[:min(5, len(numeric_cols))],
                            key=f"plot_{file_name}"
                        )
                        
                        if cols_to_plot:
                            fig = visualizer.plot_time_series(
                                df.dropna(subset=[date_col]),
                                date_col,
                                cols_to_plot,
                                title=f"{file_name} - Time Series"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Add to collection
                    all_dataframes.append({
                        'name': file_name,
                        'data': df,
                        'date_col': date_col,
                        'numeric_cols': numeric_cols,
                        'frequency': freq if auto_detect and date_col else 'unknown'
                    })
                    
                    logger.success(f"✅ {file_name} processed successfully!")
                
                except Exception as e:
                    logger.error(f"Error processing {file_name}")
                    logger.error(str(e))
                    
                    with st.expander("🐛 Error Details", expanded=False):
                        st.code(traceback.format_exc())
        
        # Build panel
        if all_dataframes:
            st.markdown("---")
            st.markdown('<div class="step-header">🔨 Step 2: Build Unified Panel</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Build Panel", type="primary", use_container_width=True):
                
                with st.spinner("Building panel..."):
                    
                    # Start with first dataframe
                    panel = all_dataframes[0]['data'].copy()
                    panel_date_col = all_dataframes[0]['date_col']
                    
                    # Merge others
                    for i, df_info in enumerate(all_dataframes[1:], 1):
                        df = df_info['data']
                        date_col = df_info['date_col']
                        
                        if date_col:
                            # Rename columns to avoid conflicts
                            rename_map = {
                                col: f"{df_info['name']}_{col}"
                                for col in df.columns
                                if col != date_col
                            }
                            df = df.rename(columns=rename_map)
                            
                            # Merge
                            panel = pd.merge(
                                panel,
                                df,
                                left_on=panel_date_col,
                                right_on=date_col,
                                how='outer'
                            )
                    
                    # Clean
                    panel = panel.sort_values(panel_date_col)
                    panel = panel.set_index(panel_date_col)
                    
                    # Save to state
                    state.panel_monthly = panel
                    
                    logger.success("✅ Panel built successfully!")
                    
                    # Show summary
                    st.markdown("### 📊 Panel Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Rows", len(panel))
                    with col2:
                        st.metric("Columns", len(panel.columns))
                    with col3:
                        coverage = panel.notna().mean().mean()
                        st.metric("Coverage", f"{coverage:.1%}")
                    
                    # Visualizations
                    tab1, tab2, tab3 = st.tabs(["📋 Data", "🔍 Coverage", "📊 Quality"])
                    
                    with tab1:
                        st.dataframe(panel.head(20), use_container_width=True, height=400)
                    
                    with tab2:
                        fig = visualizer.plot_coverage_heatmap(panel)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        fig = visualizer.plot_data_quality(panel)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Export
                    st.markdown("### 💾 Export Panel")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = panel.to_csv().encode('utf-8')
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "panel_monthly.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Excel export
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            panel.to_excel(writer, sheet_name='Panel')
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            "📥 Download Excel",
                            excel_data,
                            "panel_monthly.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
    
    else:
        st.info("👆 Upload files to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 2rem;'>
        <p><strong>📊 Data Aggregation Pro MAX v4.0</strong></p>
        <p>🇮🇹 Optimized for Italian Unemployment Nowcasting | 🎯 ISTAT & Eurostat Compatible</p>
        <p>💻 Built with Streamlit | 🚀 Production Ready | 📦 GitHub Compatible</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 🚀 RUN
# =============================================================================

if __name__ == "__main__":
    main()
