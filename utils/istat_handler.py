"""
🇮🇹 ISTAT Format Handler Module
=================================

Specialized processor for ISTAT (Italian National Statistics Institute) data.
Converts long-format panel data to time series suitable for nowcasting.

Format Example:
    Input (Long Format):
    ┌──────────┬──────┬──────┬────────┬───────┐
    │Territory │ Sex  │ Age  │ Time   │ Value │
    ├──────────┼──────┼──────┼────────┼───────┤
    │ Italy    │ Male │Y15-74│2020-Q1 │ 1230  │
    │ Italy    │ Male │Y15-74│2020-Q2 │ 1003  │
    └──────────┴──────┴──────┴────────┴───────┘
    
    Output (Time Series):
    ┌────────────┬───────┐
    │ Date       │ Value │
    ├────────────┼───────┤
    │ 2020-03-31 │ 1230  │
    │ 2020-06-30 │ 1003  │
    └────────────┴───────┘

Features:
    - Auto-detection of ISTAT column structure
    - Multi-criteria filtering (territory, sex, age)
    - Quarterly to monthly conversion
    - Multiple interpolation methods
    - Italian region support (20 regions)

Author: ISTAT Nowcasting Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re


# =============================================================================
# Configuration Classes
# =============================================================================

class InterpolationMethod(Enum):
    """Supported interpolation methods for frequency conversion"""
    LINEAR = 'linear'
    CUBIC = 'cubic'
    QUADRATIC = 'quadratic'
    SPLINE = 'spline'
    PCHIP = 'pchip'


@dataclass
class ISTATColumns:
    """Detected ISTAT column names"""
    territory: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[str] = None
    time: Optional[str] = None
    value: Optional[str] = None
    indicator: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if all required columns detected"""
        return self.time is not None and self.value is not None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# =============================================================================
# Main Handler Class
# =============================================================================

class ISTATHandler:
    """
    ISTAT format data processor with auto-detection and conversion.
    
    This class handles the complete pipeline:
    1. Format detection
    2. Column identification  
    3. Data filtering
    4. Time series conversion
    5. Frequency conversion (quarterly → monthly)
    
    Usage:
        Basic:
            >>> handler = ISTATHandler()
            >>> if handler.is_istat_format(df):
            ...     cols = handler.detect_columns(df)
            ...     ts = handler.convert_to_timeseries(df, cols)
        
        With filtering:
            >>> filtered = handler.filter_data(
            ...     df, cols,
            ...     territory='Italy',
            ...     sex='Males',
            ...     age='Y15-74'
            ... )
            >>> ts = handler.convert_to_timeseries(filtered, cols)
        
        Quarterly to monthly:
            >>> monthly = handler.quarterly_to_monthly(
            ...     ts,
            ...     method=InterpolationMethod.CUBIC
            ... )
    
    Attributes:
        column_patterns: Dictionary of known ISTAT column name patterns
        italian_regions: List of 20 Italian regions
        quarter_patterns: Regex patterns for quarterly dates
    """
    
    # Column name patterns (English, Italian, Eurostat)
    COLUMN_PATTERNS = {
        'territory': [
            'territory', 'ref_area', 'geo', 'area', 'region',
            'regione', 'territorio', 'zona'
        ],
        'sex': [
            'sex', 'gender', 'sesso', 'genere'
        ],
        'age': [
            'age', 'age_group', 'age_class', 'eta', 'classe_eta'
        ],
        'time': [
            'time_period', 'time', 'period', 'date',
            'periodo', 'data', 'tempo'
        ],
        'value': [
            'obs_value', 'observation', 'value', 'valore',
            'dato', 'obs'
        ],
        'indicator': [
            'indicator', 'indicatore', 'measure', 'misura'
        ]
    }
    
    # Italian regions (20 + national)
    ITALIAN_REGIONS = [
        'Italy', 'Italia',
        'Piemonte', 'Valle d\'Aosta', 'Lombardia',
        'Trentino-Alto Adige', 'Veneto', 'Friuli-Venezia Giulia',
        'Liguria', 'Emilia-Romagna', 'Toscana', 'Umbria',
        'Marche', 'Lazio', 'Abruzzo', 'Molise', 'Campania',
        'Puglia', 'Basilicata', 'Calabria', 'Sicilia', 'Sardegna'
    ]
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize ISTAT handler.
        
        Args:
            strict_mode: If True, requires all columns to be present
        """
        self.strict_mode = strict_mode
        self._cache: Dict[str, any] = {}
    
    # =========================================================================
    # Format Detection
    # =========================================================================
    
    def is_istat_format(self, df: pd.DataFrame, min_matches: int = 4) -> bool:
        """
        Detect if DataFrame has ISTAT format structure.
        
        Args:
            df: Input DataFrame
            min_matches: Minimum column matches required (default: 4/6)
        
        Returns:
            bool: True if ISTAT format detected
            
        Example:
            >>> df = pd.read_excel('unemployment_data.xlsx')
            >>> if handler.is_istat_format(df):
            ...     print("ISTAT format confirmed!")
        """
        if df.empty or len(df.columns) < 3:
            return False
        
        # Normalize column names for comparison
        df_cols_lower = {
            col.lower().replace('_', '').replace(' ', ''): col
            for col in df.columns
        }
        
        matches = 0
        matched_types = set()
        
        # Check each column type
        for col_type, patterns in self.COLUMN_PATTERNS.items():
            for pattern in patterns:
                clean_pattern = pattern.replace('_', '').replace(' ', '')
                
                if clean_pattern in df_cols_lower:
                    matches += 1
                    matched_types.add(col_type)
                    break
        
        # Must have at least time and value
        has_required = 'time' in matched_types and 'value' in matched_types
        
        return has_required and matches >= min_matches
    
    def detect_columns(self, df: pd.DataFrame) -> ISTATColumns:
        """
        Detect and map ISTAT column names.
        
        Args:
            df: Input DataFrame
        
        Returns:
            ISTATColumns: Detected column mappings
            
        Raises:
            ValueError: If required columns (time, value) not found
            
        Example:
            >>> cols = handler.detect_columns(df)
            >>> print(f"Time column: {cols.time}")
            >>> print(f"Value column: {cols.value}")
        """
        # Normalize for matching
        df_cols_map = {
            col.lower().replace('_', '').replace(' ', ''): col
            for col in df.columns
        }
        
        detected = ISTATColumns()
        
        # Match each column type
        for col_type, patterns in self.COLUMN_PATTERNS.items():
            for pattern in patterns:
                clean_pattern = pattern.replace('_', '').replace(' ', '')
                
                if clean_pattern in df_cols_map:
                    setattr(detected, col_type, df_cols_map[clean_pattern])
                    break
        
        # Validate required columns
        if not detected.is_complete():
            missing = []
            if detected.time is None:
                missing.append('time')
            if detected.value is None:
                missing.append('value')
            
            raise ValueError(
                f"Required ISTAT columns not found: {', '.join(missing)}\n"
                f"Available columns: {list(df.columns)}"
            )
        
        return detected
    
    # =========================================================================
    # Data Analysis
    # =========================================================================
    
    def get_filter_options(self, 
                          df: pd.DataFrame,
                          cols: ISTATColumns) -> Dict[str, List[str]]:
        """
        Extract unique values for each filterable column.
        
        Args:
            df: Input DataFrame
            cols: Detected column mappings
        
        Returns:
            dict: Options for each dimension {column_type: [values]}
            
        Example:
            >>> options = handler.get_filter_options(df, cols)
            >>> print(f"Available territories: {options['territory']}")
            >>> print(f"Available sexes: {options['sex']}")
        """
        options = {}
        
        for col_type in ['territory', 'sex', 'age', 'indicator']:
            col_name = getattr(cols, col_type)
            
            if col_name and col_name in df.columns:
                unique_vals = df[col_name].dropna().unique()
                options[col_type] = sorted(unique_vals.tolist())
        
        return options
    
    def analyze_coverage(self,
                        df: pd.DataFrame,
                        cols: ISTATColumns) -> Dict[str, any]:
        """
        Analyze data coverage and quality.
        
        Args:
            df: Input DataFrame
            cols: Detected column mappings
        
        Returns:
            dict: Coverage statistics
            
        Example:
            >>> coverage = handler.analyze_coverage(df, cols)
            >>> print(f"Time range: {coverage['time_range']}")
            >>> print(f"Missing values: {coverage['missing_pct']:.1%}")
        """
        analysis = {
            'total_rows': len(df),
            'time_range': None,
            'value_stats': {},
            'missing_pct': 0.0,
            'dimensions': {}
        }
        
        # Time range
        if cols.time:
            time_vals = pd.to_datetime(df[cols.time], errors='coerce').dropna()
            if not time_vals.empty:
                analysis['time_range'] = (
                    time_vals.min().strftime('%Y-%m-%d'),
                    time_vals.max().strftime('%Y-%m-%d')
                )
        
        # Value statistics
        if cols.value:
            val_series = pd.to_numeric(df[cols.value], errors='coerce')
            analysis['value_stats'] = {
                'count': val_series.notna().sum(),
                'mean': float(val_series.mean()),
                'std': float(val_series.std()),
                'min': float(val_series.min()),
                'max': float(val_series.max())
            }
            analysis['missing_pct'] = val_series.isna().mean()
        
        # Dimension cardinality
        for dim in ['territory', 'sex', 'age']:
            col_name = getattr(cols, dim)
            if col_name:
                analysis['dimensions'][dim] = df[col_name].nunique()
        
        return analysis
    
    # =========================================================================
    # Data Filtering
    # =========================================================================
    
    def filter_data(self,
                   df: pd.DataFrame,
                   cols: ISTATColumns,
                   territory: Optional[str] = None,
                   sex: Optional[str] = None,
                   age: Optional[str] = None,
                   indicator: Optional[str] = None,
                   time_start: Optional[str] = None,
                   time_end: Optional[str] = None) -> pd.DataFrame:
        """
        Filter DataFrame by specified criteria.
        
        Args:
            df: Input DataFrame
            cols: Column mappings
            territory: Territory filter (e.g., 'Italy', 'Lombardia')
            sex: Sex filter (e.g., 'Males', 'Females', 'Total')
            age: Age group filter (e.g., 'Y15-74')
            indicator: Indicator filter
            time_start: Start date (inclusive)
            time_end: End date (inclusive)
        
        Returns:
            pd.DataFrame: Filtered data
            
        Example:
            >>> filtered = handler.filter_data(
            ...     df, cols,
            ...     territory='Italy',
            ...     sex='Males',
            ...     age='Y15-74',
            ...     time_start='2020-01-01'
            ... )
            >>> print(f"Filtered to {len(filtered)} rows")
        """
        result = df.copy()
        
        # Dimension filters
        filters = {
            'territory': territory,
            'sex': sex,
            'age': age,
            'indicator': indicator
        }
        
        for col_type, value in filters.items():
            if value is not None:
                col_name = getattr(cols, col_type)
                
                if col_name and col_name in result.columns:
                    result = result[result[col_name] == value]
        
        # Time filter
        if (time_start or time_end) and cols.time:
            time_series = pd.to_datetime(result[cols.time], errors='coerce')
            
            if time_start:
                start_dt = pd.to_datetime(time_start)
                result = result[time_series >= start_dt]
            
            if time_end:
                end_dt = pd.to_datetime(time_end)
                result = result[time_series <= end_dt]
        
        return result.reset_index(drop=True)
    
    # =========================================================================
    # Time Series Conversion
    # =========================================================================
    
    def convert_to_timeseries(self,
                             df: pd.DataFrame,
                             cols: ISTATColumns,
                             name: Optional[str] = 'unemployment_rate') -> pd.Series:
        """
        Convert filtered ISTAT data to time series.
        
        Args:
            df: Filtered DataFrame (single series)
            cols: Column mappings
            name: Name for resulting Series
        
        Returns:
            pd.Series: Time series with datetime index
            
        Raises:
            ValueError: If conversion fails
            
        Example:
            >>> ts = handler.convert_to_timeseries(filtered_df, cols)
            >>> print(f"Created series: {len(ts)} observations")
            >>> print(f"Date range: {ts.index.min()} to {ts.index.max()}")
        """
        if df.empty:
            raise ValueError("Cannot convert empty DataFrame")
        
        # Extract time and value columns
        time_col = cols.time
        value_col = cols.value
        
        ts_df = df[[time_col, value_col]].copy()
        
        # Parse dates
        ts_df['parsed_date'] = self._parse_time_column(ts_df[time_col])
        
        # Convert values to numeric
        ts_df['numeric_value'] = pd.to_numeric(ts_df[value_col], errors='coerce')
        
        # Drop NaN
        ts_df = ts_df.dropna(subset=['parsed_date', 'numeric_value'])
        
        if ts_df.empty:
            raise ValueError("No valid data after date parsing and numeric conversion")
        
        # Handle duplicates (take mean if multiple values per date)
        if ts_df['parsed_date'].duplicated().any():
            ts_df = ts_df.groupby('parsed_date')['numeric_value'].mean().reset_index()
            ts_df.columns = ['parsed_date', 'numeric_value']
        
        # Sort and create Series
        ts_df = ts_df.sort_values('parsed_date')
        
        result = pd.Series(
            ts_df['numeric_value'].values,
            index=pd.DatetimeIndex(ts_df['parsed_date']),
            name=name
        )
        
        return result
    
    def _parse_time_column(self, time_series: pd.Series) -> pd.Series:
        """
        Parse various time formats to datetime.
        
        Supports:
            - Quarterly: 2020-Q1, 2020Q1, 2020 Q1
            - Monthly: 2020-01, 2020-01-31
            - ISO dates: 2020-01-15
        
        Args:
            time_series: Series with time strings
        
        Returns:
            pd.Series: Parsed datetime (end of period)
        """
        def parse_single(val):
            if pd.isna(val):
                return pd.NaT
            
            val_str = str(val).strip().upper()
            
            # Quarterly patterns
            quarterly_patterns = [
                r'(\d{4})[- ]?Q(\d)',  # 2020-Q1, 2020Q1, 2020 Q1
                r'Q(\d)[- ]?(\d{4})',  # Q1-2020, Q1 2020
            ]
            
            for pattern in quarterly_patterns:
                match = re.search(pattern, val_str)
                if match:
                    groups = match.groups()
                    
                    # Determine year and quarter
                    if len(groups[0]) == 4:  # year first
                        year, quarter = int(groups[0]), int(groups[1])
                    else:  # quarter first
                        quarter, year = int(groups[0]), int(groups[1])
                    
                    if 1 <= quarter <= 4:
                        month = quarter * 3  # Q1→3, Q2→6, Q3→9, Q4→12
                        date = pd.Timestamp(year=year, month=month, day=1)
                        return date + pd.offsets.MonthEnd(0)
            
            # Standard datetime parsing
            try:
                date = pd.to_datetime(val_str, errors='coerce')
                if pd.notna(date):
                    # Align to end of month
                    return date + pd.offsets.MonthEnd(0)
            except:
                pass
            
            return pd.NaT
        
        return time_series.apply(parse_single)
    
    # =========================================================================
    # Frequency Conversion
    # =========================================================================
    
    def quarterly_to_monthly(self,
                            quarterly: pd.Series,
                            method: Union[str, InterpolationMethod] = InterpolationMethod.LINEAR,
                            extrapolate: bool = False) -> pd.Series:
        """
        Convert quarterly data to monthly using interpolation.
        
        Args:
            quarterly: Quarterly time series
            method: Interpolation method ('linear', 'cubic', 'quadratic', etc.)
            extrapolate: Whether to extrapolate beyond data range
        
        Returns:
            pd.Series: Monthly time series
            
        Example:
            >>> monthly = handler.quarterly_to_monthly(
            ...     quarterly_series,
            ...     method='cubic'
            ... )
            >>> print(f"Converted {len(quarterly)} quarters to {len(monthly)} months")
        
        Note:
            Quarterly values are assumed to represent the end of quarter.
            Interpolation fills in the intermediate months.
        """
        if quarterly.empty:
            return pd.Series(dtype=float)
        
        # Convert method to string
        if isinstance(method, InterpolationMethod):
            method = method.value
        
        # Create monthly date range
        start = quarterly.index.min()
        end = quarterly.index.max()
        
        monthly_dates = pd.date_range(start, end, freq='M')
        
        # Combine quarterly and monthly dates
        all_dates = quarterly.index.union(monthly_dates).sort_values()
        
        # Reindex to all dates
        reindexed = quarterly.reindex(all_dates)
        
        # Interpolate
        limit = None if extrapolate else 0
        interpolated = reindexed.interpolate(
            method=method,
            limit_direction='both' if extrapolate else 'forward',
            limit=limit
        )
        
        # Keep only monthly dates
        result = interpolated[interpolated.index.isin(monthly_dates)]
        
        return result
    
    def detect_frequency(self, time_series: pd.Series) -> str:
        """
        Detect time series frequency.
        
        Args:
            time_series: Series with datetime index
        
        Returns:
            str: Frequency ('daily', 'weekly', 'monthly', 'quarterly', 'annual', 'irregular')
            
        Example:
            >>> freq = handler.detect_frequency(ts)
            >>> print(f"Detected frequency: {freq}")
        """
        if len(time_series) < 2:
            return 'irregular'
        
        # Calculate time differences
        diffs = time_series.index.to_series().diff().dt.days.dropna()
        
        if len(diffs) == 0:
            return 'irregular'
        
        median_diff = diffs.median()
        
        # Classify
        if median_diff <= 2:
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
            return 'irregular'


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_process(df: pd.DataFrame,
                 territory: str = 'Italy',
                 sex: str = 'Total',
                 age: str = 'Y15-74') -> pd.Series:
    """
    Quick one-liner to process ISTAT data.
    
    Args:
        df: ISTAT format DataFrame
        territory: Territory filter
        sex: Sex filter
        age: Age filter
    
    Returns:
        pd.Series: Processed time series
        
    Example:
        >>> df = pd.read_excel('istat_data.xlsx')
        >>> ts = quick_process(df, territory='Italy', sex='Males')
    """
    handler = ISTATHandler()
    
    if not handler.is_istat_format(df):
        raise ValueError("Not ISTAT format")
    
    cols = handler.detect_columns(df)
    filtered = handler.filter_data(df, cols, territory=territory, sex=sex, age=age)
    ts = handler.convert_to_timeseries(filtered, cols)
    
    return ts


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing ISTAT handler module...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Territory': ['Italy'] * 8,
        'Sex': ['Males'] * 8,
        'Age': ['Y15-74'] * 8,
        'TIME_PERIOD': ['2020-Q1', '2020-Q2', '2020-Q3', '2020-Q4',
                        '2021-Q1', '2021-Q2', '2021-Q3', '2021-Q4'],
        'Observation': [1230.5, 1003.8, 1326.8, 1293.8,
                       1396.3, 1229.4, 1099.5, 1219.7]
    })
    
    handler = ISTATHandler()
    
    # Test format detection
    assert handler.is_istat_format(sample_data), "Format detection failed"
    print("✓ Format detection passed")
    
    # Test column detection
    cols = handler.detect_columns(sample_data)
    assert cols.is_complete(), "Column detection failed"
    print(f"✓ Column detection passed: {cols.to_dict()}")
    
    # Test conversion
    ts = handler.convert_to_timeseries(sample_data, cols)
    assert len(ts) == 8, "Conversion failed"
    print(f"✓ Conversion passed: {len(ts)} observations")
    
    # Test quarterly to monthly
    monthly = handler.quarterly_to_monthly(ts)
    assert len(monthly) > len(ts), "Frequency conversion failed"
    print(f"✓ Frequency conversion passed: {len(ts)} → {len(monthly)}")
    
    print("\n✅ All tests passed!")
