"""
Data Intelligence Engine
Auto-detection of date, target, frequency, features
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG


class DataIntelligence:
    """Intelligent data loading and analysis"""
    
    def __init__(self):
        self.config = CONFIG
    
    def load_file(self, file) -> pd.DataFrame:
        """
        Load CSV or Excel file
        Auto-detect format and encoding
        """
        try:
            # Get file extension
            filename = file.name.lower()
            
            if filename.endswith('.csv'):
                # Try different encodings
                for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(file, encoding=encoding)
                        break
                    except:
                        continue
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
            
            # Clean column names
            df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to load file: {str(e)}")
    
    def detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect date column
        Try multiple heuristics
        """
        candidates = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check by name
            if any(word in col_lower for word in ['date', 'time', 'month', 'year', 'period']):
                candidates.append((col, 100))
                continue
            
            # Check by content
            try:
                pd.to_datetime(df[col], errors='raise')
                candidates.append((col, 80))
            except:
                pass
        
        if not candidates:
            return None
        
        # Return best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def parse_dates(self, df: pd.DataFrame, date_col: str) -> pd.Series:
        """
        Parse dates with multiple format attempts
        """
        formats_to_try = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%Y%m%d',
            None  # Let pandas infer
        ]
        
        for fmt in formats_to_try:
            try:
                if fmt is None:
                    return pd.to_datetime(df[date_col])
                else:
                    return pd.to_datetime(df[date_col], format=fmt)
            except:
                continue
        
        raise ValueError(f"Could not parse dates in column '{date_col}'")
    
    def detect_frequency(self, dates: pd.Series) -> Tuple[str, Dict]:
        """
        Detect data frequency
        Returns: (frequency, diagnostics)
        """
        dates = pd.Series(dates).sort_values().reset_index(drop=True)
        
        # Calculate time deltas
        deltas = dates.diff().dt.days.dropna()
        
        if len(deltas) < 2:
            return 'unknown', {'reason': 'insufficient data'}
        
        median_delta = deltas.median()
        
        # Classify frequency
        for freq, (min_days, max_days) in self.config.FREQ_THRESHOLDS.items():
            if min_days <= median_delta <= max_days:
                diagnostics = {
                    'median_delta': float(median_delta),
                    'min_delta': float(deltas.min()),
                    'max_delta': float(deltas.max()),
                    'std_delta': float(deltas.std()),
                    'observations': len(dates),
                    'date_range': (str(dates.min()), str(dates.max()))
                }
                return freq, diagnostics
        
        return 'irregular', {
            'median_delta': float(median_delta),
            'reason': 'does not match standard frequencies'
        }
    
    def suggest_target(self, df: pd.DataFrame, date_col: str) -> Optional[str]:
        """
        Suggest target variable
        Based on name and characteristics
        """
        exclude = [date_col]
        candidates = []
        
        for col in df.columns:
            if col in exclude:
                continue
            
            col_lower = str(col).lower()
            
            # Check if numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Check for target keywords
            keywords = ['unemployment', 'gdp', 'sales', 'revenue', 'inflation', 
                       'rate', 'index', 'growth', 'unemp', 'target', 'y']
            
            score = 0
            for kw in keywords:
                if kw in col_lower:
                    score += 10
            
            # Check variation (not constant)
            if df[col].nunique() > len(df) * 0.1:
                score += 5
            
            # Check missing values (prefer fewer)
            missing_pct = df[col].isna().sum() / len(df)
            score += (1 - missing_pct) * 5
            
            if score > 0:
                candidates.append((col, score))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def analyze_column(self, series: pd.Series) -> Dict:
        """
        Analyze single column
        Statistics and quality checks
        """
        clean = series.dropna()
        
        if len(clean) == 0:
            return {'type': 'empty'}
        
        analysis = {
            'dtype': str(series.dtype),
            'count': len(series),
            'missing': int(series.isna().sum()),
            'missing_pct': float(series.isna().sum() / len(series) * 100)
        }
        
        if pd.api.types.is_numeric_dtype(series):
            analysis.update({
                'type': 'numeric',
                'mean': float(clean.mean()),
                'std': float(clean.std()),
                'min': float(clean.min()),
                'max': float(clean.max()),
                'median': float(clean.median()),
                'q25': float(clean.quantile(0.25)),
                'q75': float(clean.quantile(0.75))
            })
            
            # Outliers
            z_scores = np.abs(stats.zscore(clean))
            outliers = (z_scores > self.config.OUTLIER_THRESHOLD).sum()
            analysis['outliers'] = int(outliers)
            analysis['outliers_pct'] = float(outliers / len(clean) * 100)
            
            # Stationarity (simplified ADF)
            if len(clean) >= 10:
                try:
                    from statsmodels.tsa.stattools import adfuller
                    adf_result = adfuller(clean, autolag='AIC')
                    analysis['adf_statistic'] = float(adf_result[0])
                    analysis['adf_pvalue'] = float(adf_result[1])
                    analysis['is_stationary'] = adf_result[1] < 0.05
                except:
                    analysis['is_stationary'] = None
        
        else:
            analysis['type'] = 'categorical'
            analysis['unique'] = int(series.nunique())
        
        return analysis
    
    def validate_dataset(self, df: pd.DataFrame, date_col: str, target_col: str) -> Dict:
        """
        Validate dataset for nowcasting
        Returns validation report
        """
        report = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check minimum observations
        if len(df) < self.config.MIN_OBSERVATIONS:
            report['errors'].append(
                f"Insufficient data: {len(df)} rows (minimum: {self.config.MIN_OBSERVATIONS})"
            )
            report['valid'] = False
        
        # Check target missing values
        target_missing = df[target_col].isna().sum() / len(df)
        if target_missing > self.config.MAX_MISSING_PCT:
            report['errors'].append(
                f"Too many missing values in target: {target_missing*100:.1f}%"
            )
            report['valid'] = False
        
        # Check date gaps
        dates = pd.to_datetime(df[date_col]).sort_values()
        deltas = dates.diff().dt.days.dropna()
        if deltas.std() > deltas.median() * 0.5:
            report['warnings'].append(
                "Irregular date spacing detected - may affect frequency alignment"
            )
        
        # Check for duplicates
        if df[date_col].duplicated().any():
            report['errors'].append("Duplicate dates found")
            report['valid'] = False
        
        return report


class DatasetInfo:
    """Container for dataset information"""
    
    def __init__(self, df: pd.DataFrame, date_col: str, target_col: str, 
                 frequency: str, freq_diag: Dict, validation: Dict):
        self.df = df
        self.date_col = date_col
        self.target_col = target_col
        self.frequency = frequency
        self.freq_diagnostics = freq_diag
        self.validation = validation
        self.n_obs = len(df)
        self.date_range = (df[date_col].min(), df[date_col].max())
        self.feature_cols = [c for c in df.columns if c not in [date_col, target_col]]
    
    def summary(self) -> Dict:
        """Get summary dict"""
        return {
            'observations': self.n_obs,
            'date_range': [str(d) for d in self.date_range],
            'frequency': self.frequency,
            'target': self.target_col,
            'n_features': len(self.feature_cols),
            'valid': self.validation['valid']
        }
