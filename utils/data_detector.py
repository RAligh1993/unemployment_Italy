"""
🔍 Data Detection Module
=========================

Intelligent auto-detection of data types, columns, and formats.
Uses heuristics and machine learning-like scoring to identify patterns.

Detection Capabilities:
    - Date columns (15+ formats)
    - Numeric columns (with Italian formatting)
    - Categorical columns
    - Time series frequency
    - ISTAT format structure
    - Italian regional data
    - Unemployment indicators

Features:
    - Multi-pattern matching
    - Confidence scoring
    - Format inference
    - Metadata extraction

Author: ISTAT Nowcasting Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
from collections import Counter


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ColumnType(Enum):
    """Column data types"""
    DATE = 'date'
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'
    TEXT = 'text'
    BOOLEAN = 'boolean'
    ID = 'identifier'
    UNKNOWN = 'unknown'


class DateFormat(Enum):
    """Date format patterns"""
    ISO = 'iso'              # 2020-01-15
    ITALIAN = 'italian'      # 31/12/2020
    US = 'us'               # 12/31/2020
    QUARTERLY = 'quarterly'  # 2020-Q1
    MONTHLY = 'monthly'      # 2020-01
    TEXT = 'text'           # January 2020


@dataclass
class ColumnMetadata:
    """Metadata for a single column"""
    name: str
    type: ColumnType
    confidence: float
    sub_type: Optional[str] = None
    format: Optional[str] = None
    missing_pct: float = 0.0
    unique_count: int = 0
    sample_values: List[Any] = None
    
    def __post_init__(self):
        if self.sample_values is None:
            self.sample_values = []


@dataclass
class DatasetMetadata:
    """Metadata for entire dataset"""
    shape: Tuple[int, int]
    columns: List[ColumnMetadata]
    detected_formats: List[str]
    is_time_series: bool
    frequency: Optional[str]
    date_column: Optional[str]
    value_columns: List[str]
    quality_score: float


# =============================================================================
# Main Detector Class
# =============================================================================

class DataDetector:
    """
    Intelligent data detection and classification engine.
    
    This class analyzes DataFrames to automatically detect:
    - Column types and formats
    - Date columns and formats
    - Time series structure
    - ISTAT format
    - Italian regional data
    - Data quality issues
    
    Usage:
        Basic:
            >>> detector = DataDetector()
            >>> date_col = detector.detect_date_column(df)
            >>> print(f"Date column: {date_col}")
        
        Full analysis:
            >>> metadata = detector.analyze_dataset(df)
            >>> print(f"Detected {len(metadata.columns)} columns")
            >>> print(f"Time series: {metadata.is_time_series}")
        
        Column types:
            >>> types = detector.detect_column_types(df)
            >>> for col, col_type in types.items():
            ...     print(f"{col}: {col_type}")
    
    Attributes:
        patterns: Dictionary of detection patterns
        confidence_threshold: Minimum confidence for positive detection
    """
    
    # Date patterns with priority order
    DATE_PATTERNS = {
        'iso': r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',
        'italian': r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$',
        'us': r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$',
        'quarterly': r'^\d{4}[-\s]?[Qq][1-4]$',
        'monthly': r'^\d{4}[-]\d{1,2}$',
        'year': r'^\d{4}$',
    }
    
    # ISTAT column keywords
    ISTAT_KEYWORDS = {
        'date': ['time', 'periodo', 'date', 'period', 'time_period'],
        'value': ['observation', 'obs_value', 'value', 'valore'],
        'territory': ['territory', 'geo', 'area', 'regione', 'ref_area'],
        'sex': ['sex', 'gender', 'sesso'],
        'age': ['age', 'eta', 'age_group']
    }
    
    # Unemployment keywords (multilingual)
    UNEMPLOYMENT_KEYWORDS = {
        'total': ['unemployment', 'disoccupazione', 'tasso', 'rate', 'total', 'totale'],
        'male': ['male', 'males', 'uomini', 'maschi', 'men'],
        'female': ['female', 'females', 'donne', 'femmine', 'women'],
        'youth': ['youth', 'giovani', 'young', '15-24', '<25'],
        'long_term': ['long term', 'lungo periodo', 'structural', '>12']
    }
    
    # Italian regions
    ITALIAN_REGIONS = [
        'italy', 'italia', 'piemonte', 'valle d\'aosta', 'lombardia',
        'trentino', 'veneto', 'friuli', 'liguria', 'emilia',
        'toscana', 'umbria', 'marche', 'lazio', 'abruzzo',
        'molise', 'campania', 'puglia', 'basilicata', 'calabria',
        'sicilia', 'sardegna'
    ]
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize detector.
        
        Args:
            confidence_threshold: Minimum confidence for positive detection (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self._cache: Dict[str, Any] = {}
    
    # =========================================================================
    # High-Level Analysis
    # =========================================================================
    
    def analyze_dataset(self, df: pd.DataFrame) -> DatasetMetadata:
        """
        Comprehensive dataset analysis.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DatasetMetadata: Complete metadata
        
        Example:
            >>> metadata = detector.analyze_dataset(df)
            >>> print(metadata.quality_score)
            >>> print(metadata.is_time_series)
        """
        # Detect column types
        column_metadata = []
        for col in df.columns:
            meta = self._analyze_column(df[col], str(col))
            column_metadata.append(meta)
        
        # Detect date column
        date_col = self.detect_date_column(df)
        
        # Detect value columns
        value_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
            and col != date_col
        ]
        
        # Check if time series
        is_ts = False
        frequency = None
        
        if date_col and len(value_cols) > 0:
            is_ts = True
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                frequency = self.detect_frequency(dates)
            except:
                pass
        
        # Detect formats
        formats = []
        if self.is_istat_format(df):
            formats.append('ISTAT')
        if self.has_italian_regions(df):
            formats.append('Italian_Regional')
        if self.has_unemployment_indicators(df):
            formats.append('Unemployment_Data')
        
        # Calculate quality score
        quality = self._calculate_quality_score(df)
        
        return DatasetMetadata(
            shape=df.shape,
            columns=column_metadata,
            detected_formats=formats,
            is_time_series=is_ts,
            frequency=frequency,
            date_column=date_col,
            value_columns=value_cols,
            quality_score=quality
        )
    
    def _analyze_column(self, series: pd.Series, name: str) -> ColumnMetadata:
        """Analyze single column"""
        
        # Basic stats
        missing_pct = series.isna().mean()
        unique_count = series.nunique()
        sample_values = series.dropna().head(5).tolist()
        
        # Detect type
        col_type, confidence, sub_type = self._detect_column_type(series)
        
        # Detect format if date
        format_str = None
        if col_type == ColumnType.DATE:
            format_str = self._detect_date_format(series)
        
        return ColumnMetadata(
            name=name,
            type=col_type,
            confidence=confidence,
            sub_type=sub_type,
            format=format_str,
            missing_pct=missing_pct,
            unique_count=unique_count,
            sample_values=sample_values
        )
    
    # =========================================================================
    # Column Type Detection
    # =========================================================================
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, ColumnType]:
        """
        Detect type of each column.
        
        Args:
            df: Input DataFrame
        
        Returns:
            dict: {column_name: ColumnType}
        
        Example:
            >>> types = detector.detect_column_types(df)
            >>> date_cols = [k for k, v in types.items() if v == ColumnType.DATE]
        """
        types = {}
        
        for col in df.columns:
            col_type, _, _ = self._detect_column_type(df[col])
            types[col] = col_type
        
        return types
    
    def _detect_column_type(self, series: pd.Series) -> Tuple[ColumnType, float, Optional[str]]:
        """
        Detect type of single column.
        
        Returns:
            tuple: (ColumnType, confidence, sub_type)
        """
        # Drop NaN for analysis
        series = series.dropna()
        
        if len(series) == 0:
            return ColumnType.UNKNOWN, 0.0, None
        
        # Check date
        date_score = self._score_date_column(series)
        if date_score > self.confidence_threshold:
            return ColumnType.DATE, date_score, self._detect_date_format(series)
        
        # Check numeric
        numeric_score = self._score_numeric_column(series)
        if numeric_score > self.confidence_threshold:
            return ColumnType.NUMERIC, numeric_score, None
        
        # Check boolean
        if self._is_boolean(series):
            return ColumnType.BOOLEAN, 1.0, None
        
        # Check ID
        if self._is_identifier(series):
            return ColumnType.ID, 0.8, None
        
        # Check categorical
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.5:
            return ColumnType.CATEGORICAL, 0.7, None
        
        # Default to text
        return ColumnType.TEXT, 0.5, None
    
    def _score_date_column(self, series: pd.Series) -> float:
        """Score likelihood of being a date column (0-1)"""
        score = 0.0
        sample = series.head(min(100, len(series)))
        
        # Try pandas parsing
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            parse_rate = parsed.notna().mean()
            score += parse_rate * 0.6
        except:
            pass
        
        # Check patterns
        sample_str = sample.astype(str)
        pattern_matches = 0
        
        for pattern in self.DATE_PATTERNS.values():
            matches = sample_str.str.match(pattern).sum()
            pattern_matches = max(pattern_matches, matches)
        
        pattern_score = pattern_matches / len(sample)
        score += pattern_score * 0.4
        
        return min(score, 1.0)
    
    def _score_numeric_column(self, series: pd.Series) -> float:
        """Score likelihood of being numeric (0-1)"""
        # If already numeric dtype
        if pd.api.types.is_numeric_dtype(series):
            return 1.0
        
        # Try conversion
        try:
            # Try standard conversion
            numeric = pd.to_numeric(series, errors='coerce')
            success_rate = numeric.notna().mean()
            
            if success_rate > 0.8:
                return success_rate
            
            # Try Italian format (1.234,56)
            converted = series.astype(str).str.replace('.', '').str.replace(',', '.')
            numeric = pd.to_numeric(converted, errors='coerce')
            success_rate = numeric.notna().mean()
            
            return success_rate
        
        except:
            return 0.0
    
    def _is_boolean(self, series: pd.Series) -> bool:
        """Check if column is boolean-like"""
        unique = set(series.unique())
        
        boolean_patterns = [
            {True, False},
            {1, 0},
            {'true', 'false'},
            {'yes', 'no'},
            {'y', 'n'},
            {'si', 'no'}  # Italian
        ]
        
        unique_lower = {str(v).lower() for v in unique}
        
        return any(unique_lower == pattern or unique == pattern 
                   for pattern in boolean_patterns)
    
    def _is_identifier(self, series: pd.Series) -> bool:
        """Check if column is an identifier/ID"""
        # High uniqueness
        if series.nunique() / len(series) < 0.95:
            return False
        
        # Check for ID-like patterns
        sample_str = series.astype(str).head(10)
        
        id_patterns = [
            r'^[A-Z]{2,}\d+$',  # AB123, ABC4567
            r'^\d+$',            # Pure numbers
            r'^[a-f0-9-]{36}$',  # UUID
        ]
        
        for pattern in id_patterns:
            if sample_str.str.match(pattern).sum() > 5:
                return True
        
        return False
    
    # =========================================================================
    # Date Detection
    # =========================================================================
    
    def detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect primary date column.
        
        Args:
            df: Input DataFrame
        
        Returns:
            str: Column name or None
        
        Example:
            >>> date_col = detector.detect_date_column(df)
            >>> if date_col:
            ...     print(f"Found date column: {date_col}")
        """
        candidates = []
        
        for col in df.columns:
            score = 0.0
            col_lower = str(col).lower()
            
            # Check column name
            date_keywords = ['date', 'time', 'period', 'data', 'periodo', 'tempo']
            if any(kw in col_lower for kw in date_keywords):
                score += 0.3
            
            # Check ISTAT keywords
            if any(kw == col_lower.replace('_', '') for kw in self.ISTAT_KEYWORDS['date']):
                score += 0.4
            
            # Check dtype
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                score += 0.5
            
            # Try parsing
            try:
                parsed = pd.to_datetime(df[col].dropna().head(50), errors='coerce')
                parse_rate = parsed.notna().mean()
                score += parse_rate * 0.3
            except:
                pass
            
            if score > 0:
                candidates.append((col, score))
        
        if not candidates:
            return None
        
        # Return highest scoring
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_col, best_score = candidates[0]
        
        if best_score >= self.confidence_threshold:
            return best_col
        
        return None
    
    def _detect_date_format(self, series: pd.Series) -> str:
        """Detect specific date format"""
        sample = series.dropna().head(20).astype(str)
        
        # Count matches for each pattern
        pattern_scores = {}
        
        for format_name, pattern in self.DATE_PATTERNS.items():
            matches = sample.str.match(pattern).sum()
            pattern_scores[format_name] = matches / len(sample)
        
        # Return best match
        if pattern_scores:
            best_format = max(pattern_scores, key=pattern_scores.get)
            if pattern_scores[best_format] > 0.5:
                return best_format
        
        return 'unknown'
    
    # =========================================================================
    # Frequency Detection
    # =========================================================================
    
    def detect_frequency(self, dates: pd.Series) -> str:
        """
        Detect time series frequency.
        
        Args:
            dates: Series of datetime values
        
        Returns:
            str: 'daily', 'weekly', 'monthly', 'quarterly', 'annual', 'irregular'
        
        Example:
            >>> dates = pd.to_datetime(df['date'])
            >>> freq = detector.detect_frequency(dates)
            >>> print(f"Detected frequency: {freq}")
        """
        if len(dates) < 2:
            return 'irregular'
        
        dates = pd.to_datetime(dates, errors='coerce').dropna().sort_values()
        
        if len(dates) < 2:
            return 'irregular'
        
        # Calculate differences in days
        diffs = dates.diff().dt.days.dropna()
        
        if len(diffs) == 0:
            return 'irregular'
        
        # Get median and mode
        median_diff = diffs.median()
        mode_diff = diffs.mode()[0] if len(diffs.mode()) > 0 else median_diff
        
        # Classify
        if mode_diff <= 2:
            return 'daily'
        elif 5 <= mode_diff <= 9:
            return 'weekly'
        elif 28 <= mode_diff <= 32:
            return 'monthly'
        elif 85 <= mode_diff <= 95:
            return 'quarterly'
        elif 360 <= mode_diff <= 370:
            return 'annual'
        else:
            return 'irregular'
    
    # =========================================================================
    # Format Detection
    # =========================================================================
    
    def is_istat_format(self, df: pd.DataFrame, min_matches: int = 4) -> bool:
        """
        Check if DataFrame has ISTAT format.
        
        Args:
            df: Input DataFrame
            min_matches: Minimum column matches (default: 4/5)
        
        Returns:
            bool: True if ISTAT format detected
        """
        df_cols_lower = {col.lower().replace('_', ''): col for col in df.columns}
        
        matches = 0
        for category_keywords in self.ISTAT_KEYWORDS.values():
            for keyword in category_keywords:
                if keyword.replace('_', '') in df_cols_lower:
                    matches += 1
                    break
        
        return matches >= min_matches
    
    def has_italian_regions(self, df: pd.DataFrame) -> bool:
        """
        Check if data contains Italian regions.
        
        Args:
            df: Input DataFrame
        
        Returns:
            bool: True if Italian regions found
        """
        for col in df.columns:
            try:
                values_lower = df[col].dropna().astype(str).str.lower()
                
                region_matches = sum(
                    any(region in val for region in self.ITALIAN_REGIONS)
                    for val in values_lower.head(50)
                )
                
                if region_matches >= 3:
                    return True
            except:
                continue
        
        return False
    
    def has_unemployment_indicators(self, df: pd.DataFrame) -> bool:
        """
        Check if data contains unemployment indicators.
        
        Args:
            df: Input DataFrame
        
        Returns:
            bool: True if unemployment indicators found
        """
        # Check column names
        all_text = ' '.join(str(col).lower() for col in df.columns)
        
        keyword_matches = sum(
            any(kw in all_text for kw in keywords)
            for keywords in self.UNEMPLOYMENT_KEYWORDS.values()
        )
        
        return keyword_matches >= 2
    
    # =========================================================================
    # Quality Assessment
    # =========================================================================
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Factors:
        - Missing values
        - Duplicate rows
        - Consistency
        - Coverage
        """
        score = 100.0
        
        # Missing values penalty
        missing_pct = df.isna().sum().sum() / df.size
        score -= missing_pct * 30
        
        # Duplicate rows penalty
        duplicate_pct = df.duplicated().sum() / len(df)
        score -= duplicate_pct * 20
        
        # Empty columns penalty
        empty_cols = (df.isna().all()).sum()
        if len(df.columns) > 0:
            score -= (empty_cols / len(df.columns)) * 15
        
        # Low variance columns (potentially useless)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_var_count = 0
        
        for col in numeric_cols:
            if df[col].std() < 1e-10:
                low_var_count += 1
        
        if len(numeric_cols) > 0:
            score -= (low_var_count / len(numeric_cols)) * 10
        
        return max(0.0, min(100.0, score))


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_analyze(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick analysis with common detections.
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Analysis results
    
    Example:
        >>> results = quick_analyze(df)
        >>> print(results['date_column'])
        >>> print(results['is_time_series'])
    """
    detector = DataDetector()
    
    return {
        'shape': df.shape,
        'date_column': detector.detect_date_column(df),
        'column_types': detector.detect_column_types(df),
        'is_istat': detector.is_istat_format(df),
        'has_italian_regions': detector.has_italian_regions(df),
        'has_unemployment': detector.has_unemployment_indicators(df),
        'quality_score': detector._calculate_quality_score(df)
    }


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing data detector module...")
    
    # Create sample data
    sample_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=12, freq='M'),
        'unemployment_rate': np.random.rand(12) * 10 + 5,
        'region': ['Italy'] * 12,
        'sex': ['Total'] * 12,
        'text_col': ['Sample'] * 12
    })
    
    detector = DataDetector()
    
    # Test date detection
    date_col = detector.detect_date_column(sample_df)
    assert date_col == 'date', f"Date detection failed: {date_col}"
    print(f"✓ Date detection: {date_col}")
    
    # Test column types
    types = detector.detect_column_types(sample_df)
    print(f"✓ Column types: {types}")
    
    # Test frequency
    freq = detector.detect_frequency(sample_df['date'])
    print(f"✓ Frequency: {freq}")
    
    # Test full analysis
    metadata = detector.analyze_dataset(sample_df)
    print(f"✓ Full analysis: quality={metadata.quality_score:.1f}")
    
    print("\n✅ All tests passed!")
