"""
Feature Engineering Factory
Automatic feature generation based on frequency and data characteristics
Real implementation - production-ready
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG


class FeatureFactory:
    """
    Automatic feature engineering based on data frequency
    Creates lags, differences, moving averages, seasonal features
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.created_features = []
        self.feature_metadata = {}
    
    def create_baseline_features(self,
                                 df: pd.DataFrame,
                                 target_col: str,
                                 date_col: str,
                                 frequency: str,
                                 max_lags: Optional[int] = None) -> pd.DataFrame:
        """
        Create baseline autoregressive features
        
        Args:
            df: Input dataframe
            target_col: Target variable column
            date_col: Date column
            frequency: Data frequency (daily, weekly, monthly, etc.)
            max_lags: Maximum lag order (auto if None)
        
        Returns:
            DataFrame with lag features added
        """
        result = df.copy()
        
        # Determine appropriate lags based on frequency
        lags = self._get_default_lags(frequency, max_lags)
        
        # Create lag features
        for lag in lags:
            lag_name = f"{target_col}_lag{lag}"
            result[lag_name] = result[target_col].shift(lag)
            
            self.created_features.append(lag_name)
            self.feature_metadata[lag_name] = {
                'type': 'lag',
                'source': target_col,
                'lag_order': lag,
                'frequency': frequency
            }
        
        return result
    
    def create_difference_features(self,
                                   df: pd.DataFrame,
                                   target_col: str,
                                   orders: List[int] = [1, 2]) -> pd.DataFrame:
        """
        Create differenced features (Δ, Δ²)
        
        Args:
            df: Input dataframe
            target_col: Target variable
            orders: Difference orders to create
        
        Returns:
            DataFrame with difference features
        """
        result = df.copy()
        
        for order in orders:
            if order == 1:
                diff_name = f"{target_col}_diff"
                result[diff_name] = result[target_col].diff(1)
            else:
                diff_name = f"{target_col}_diff{order}"
                result[diff_name] = result[target_col].diff(order)
            
            self.created_features.append(diff_name)
            self.feature_metadata[diff_name] = {
                'type': 'difference',
                'source': target_col,
                'order': order
            }
        
        return result
    
    def create_moving_average_features(self,
                                      df: pd.DataFrame,
                                      target_col: str,
                                      frequency: str,
                                      windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create moving average features
        
        Args:
            df: Input dataframe
            target_col: Target variable
            frequency: Data frequency
            windows: MA windows (auto if None)
        
        Returns:
            DataFrame with MA features
        """
        result = df.copy()
        
        if windows is None:
            windows = self._get_default_ma_windows(frequency)
        
        for window in windows:
            ma_name = f"{target_col}_ma{window}"
            result[ma_name] = result[target_col].rolling(window=window, min_periods=1).mean()
            
            self.created_features.append(ma_name)
            self.feature_metadata[ma_name] = {
                'type': 'moving_average',
                'source': target_col,
                'window': window
            }
        
        return result
    
    def create_seasonal_features(self,
                                df: pd.DataFrame,
                                date_col: str,
                                frequency: str) -> pd.DataFrame:
        """
        Create seasonal/calendar features
        
        Args:
            df: Input dataframe
            date_col: Date column
            frequency: Data frequency
        
        Returns:
            DataFrame with seasonal features
        """
        result = df.copy()
        dates = pd.to_datetime(result[date_col])
        
        if frequency == 'daily':
            # Day of week
            result['dow'] = dates.dt.dayofweek
            for i in range(7):
                dow_name = f"dow_{i}"
                result[dow_name] = (dates.dt.dayofweek == i).astype(int)
                self.created_features.append(dow_name)
                self.feature_metadata[dow_name] = {
                    'type': 'seasonal',
                    'subtype': 'day_of_week',
                    'value': i
                }
            
            # Day of month
            result['dom'] = dates.dt.day
            self.created_features.append('dom')
            
            # Month
            result['month'] = dates.dt.month
            self.created_features.append('month')
        
        elif frequency == 'weekly':
            # Week of year
            result['week'] = dates.dt.isocalendar().week
            self.created_features.append('week')
            
            # Month
            result['month'] = dates.dt.month
            self.created_features.append('month')
        
        elif frequency == 'monthly':
            # Month dummies
            for m in range(1, 13):
                month_name = f"month_{m}"
                result[month_name] = (dates.dt.month == m).astype(int)
                self.created_features.append(month_name)
                self.feature_metadata[month_name] = {
                    'type': 'seasonal',
                    'subtype': 'month',
                    'value': m
                }
            
            # Quarter
            result['quarter'] = dates.dt.quarter
            self.created_features.append('quarter')
        
        elif frequency == 'quarterly':
            # Quarter dummies
            for q in range(1, 5):
                quarter_name = f"quarter_{q}"
                result[quarter_name] = (dates.dt.quarter == q).astype(int)
                self.created_features.append(quarter_name)
                self.feature_metadata[quarter_name] = {
                    'type': 'seasonal',
                    'subtype': 'quarter',
                    'value': q
                }
        
        # Year (for all frequencies)
        result['year'] = dates.dt.year
        self.created_features.append('year')
        
        return result
    
    def create_yoy_features(self,
                           df: pd.DataFrame,
                           target_col: str,
                           frequency: str) -> pd.DataFrame:
        """
        Create year-over-year change features
        
        Args:
            df: Input dataframe
            target_col: Target variable
            frequency: Data frequency
        
        Returns:
            DataFrame with YoY features
        """
        result = df.copy()
        
        # Determine YoY lag based on frequency
        yoy_lag_map = {
            'daily': 365,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'annual': 1
        }
        
        yoy_lag = yoy_lag_map.get(frequency, 12)
        
        # YoY level change
        yoy_name = f"{target_col}_yoy"
        result[yoy_name] = result[target_col] - result[target_col].shift(yoy_lag)
        
        self.created_features.append(yoy_name)
        self.feature_metadata[yoy_name] = {
            'type': 'yoy',
            'source': target_col,
            'lag': yoy_lag
        }
        
        # YoY percentage change
        yoy_pct_name = f"{target_col}_yoy_pct"
        result[yoy_pct_name] = (result[target_col] / result[target_col].shift(yoy_lag) - 1) * 100
        
        self.created_features.append(yoy_pct_name)
        self.feature_metadata[yoy_pct_name] = {
            'type': 'yoy_pct',
            'source': target_col,
            'lag': yoy_lag
        }
        
        return result
    
    def create_interaction_features(self,
                                   df: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features (multiplicative)
        
        Args:
            df: Input dataframe
            feature_pairs: List of (col1, col2) tuples
        
        Returns:
            DataFrame with interaction features
        """
        result = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 not in result.columns or col2 not in result.columns:
                continue
            
            interaction_name = f"{col1}_x_{col2}"
            result[interaction_name] = result[col1] * result[col2]
            
            self.created_features.append(interaction_name)
            self.feature_metadata[interaction_name] = {
                'type': 'interaction',
                'sources': [col1, col2]
            }
        
        return result
    
    def create_growth_rate_features(self,
                                   df: pd.DataFrame,
                                   cols: List[str],
                                   periods: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
        """
        Create growth rate features
        
        Args:
            df: Input dataframe
            cols: Columns to compute growth rates
            periods: Periods for growth calculation
        
        Returns:
            DataFrame with growth rate features
        """
        result = df.copy()
        
        for col in cols:
            if col not in result.columns:
                continue
            
            for period in periods:
                growth_name = f"{col}_growth{period}"
                result[growth_name] = (result[col] / result[col].shift(period) - 1) * 100
                
                self.created_features.append(growth_name)
                self.feature_metadata[growth_name] = {
                    'type': 'growth_rate',
                    'source': col,
                    'period': period
                }
        
        return result
    
    def create_volatility_features(self,
                                  df: pd.DataFrame,
                                  cols: List[str],
                                  windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """
        Create rolling volatility (standard deviation) features
        
        Args:
            df: Input dataframe
            cols: Columns to compute volatility
            windows: Rolling windows
        
        Returns:
            DataFrame with volatility features
        """
        result = df.copy()
        
        for col in cols:
            if col not in result.columns:
                continue
            
            for window in windows:
                vol_name = f"{col}_vol{window}"
                result[vol_name] = result[col].rolling(window=window, min_periods=2).std()
                
                self.created_features.append(vol_name)
                self.feature_metadata[vol_name] = {
                    'type': 'volatility',
                    'source': col,
                    'window': window
                }
        
        return result
    
    def auto_engineer_features(self,
                              df: pd.DataFrame,
                              target_col: str,
                              date_col: str,
                              frequency: str,
                              exog_cols: Optional[List[str]] = None,
                              include_seasonal: bool = True,
                              include_yoy: bool = True,
                              include_ma: bool = True) -> pd.DataFrame:
        """
        Automatic comprehensive feature engineering
        
        Args:
            df: Input dataframe
            target_col: Target variable
            date_col: Date column
            frequency: Data frequency
            exog_cols: Additional exogenous columns to engineer
            include_seasonal: Include seasonal features
            include_yoy: Include YoY features
            include_ma: Include moving averages
        
        Returns:
            DataFrame with all engineered features
        """
        result = df.copy()
        
        # 1. Baseline lags
        result = self.create_baseline_features(result, target_col, date_col, frequency)
        
        # 2. Differences
        result = self.create_difference_features(result, target_col, orders=[1])
        
        # 3. Moving averages
        if include_ma:
            result = self.create_moving_average_features(result, target_col, frequency)
        
        # 4. Seasonal features
        if include_seasonal:
            result = self.create_seasonal_features(result, date_col, frequency)
        
        # 5. Year-over-year
        if include_yoy and frequency in ['monthly', 'quarterly']:
            result = self.create_yoy_features(result, target_col, frequency)
        
        # 6. Engineer exogenous variables
        if exog_cols:
            for col in exog_cols:
                if col not in result.columns:
                    continue
                
                # Lags for exogenous
                exog_lags = self._get_exog_lags(frequency)
                for lag in exog_lags:
                    lag_name = f"{col}_lag{lag}"
                    result[lag_name] = result[col].shift(lag)
                    self.created_features.append(lag_name)
                
                # Differences for exogenous
                diff_name = f"{col}_diff"
                result[diff_name] = result[col].diff(1)
                self.created_features.append(diff_name)
        
        return result
    
    def select_features_by_correlation(self,
                                      df: pd.DataFrame,
                                      target_col: str,
                                      feature_cols: List[str],
                                      threshold: float = 0.1,
                                      top_k: Optional[int] = None) -> List[str]:
        """
        Select features based on correlation with target
        
        Args:
            df: Input dataframe
            target_col: Target variable
            feature_cols: Candidate features
            threshold: Minimum absolute correlation
            top_k: Return top K features (None = all above threshold)
        
        Returns:
            Selected feature names
        """
        # Clean data
        valid_cols = [c for c in feature_cols if c in df.columns]
        subset = df[[target_col] + valid_cols].dropna()
        
        if len(subset) < 10:
            return []
        
        # Calculate correlations
        correlations = subset[valid_cols].corrwith(subset[target_col]).abs()
        correlations = correlations.dropna()
        
        # Filter by threshold
        selected = correlations[correlations >= threshold].sort_values(ascending=False)
        
        # Top K
        if top_k is not None:
            selected = selected.head(top_k)
        
        return selected.index.tolist()
    
    def remove_constant_features(self, df: pd.DataFrame, 
                                 feature_cols: List[str],
                                 variance_threshold: float = 0.0) -> List[str]:
        """
        Remove features with zero or near-zero variance
        
        Args:
            df: Input dataframe
            feature_cols: Feature columns to check
            variance_threshold: Minimum variance
        
        Returns:
            Features to keep
        """
        keep = []
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            values = df[col].dropna()
            if len(values) < 2:
                continue
            
            variance = values.var()
            if variance > variance_threshold:
                keep.append(col)
        
        return keep
    
    def remove_highly_correlated_features(self,
                                         df: pd.DataFrame,
                                         feature_cols: List[str],
                                         threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features (multicollinearity)
        
        Args:
            df: Input dataframe
            feature_cols: Feature columns
            threshold: Correlation threshold (remove if > threshold)
        
        Returns:
            Features to keep
        """
        subset = df[feature_cols].dropna()
        
        if len(subset) < 10:
            return feature_cols
        
        # Correlation matrix
        corr_matrix = subset.corr().abs()
        
        # Upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        
        keep = [col for col in feature_cols if col not in to_drop]
        
        return keep
    
    def _get_default_lags(self, frequency: str, max_lags: Optional[int]) -> List[int]:
        """Get default lag structure based on frequency"""
        if max_lags is not None:
            return list(range(1, max_lags + 1))
        
        lag_map = {
            'daily': [1, 7, 14, 30],
            'weekly': [1, 2, 4, 8, 52],
            'monthly': [1, 2, 3, 6, 12],
            'quarterly': [1, 2, 4],
            'annual': [1, 2]
        }
        
        return lag_map.get(frequency, [1, 2, 3])
    
    def _get_exog_lags(self, frequency: str) -> List[int]:
        """Get appropriate lags for exogenous variables"""
        exog_lag_map = {
            'daily': [1, 7],
            'weekly': [1, 2, 4],
            'monthly': [1, 2, 3],
            'quarterly': [1, 2],
            'annual': [1]
        }
        
        return exog_lag_map.get(frequency, [1])
    
    def _get_default_ma_windows(self, frequency: str) -> List[int]:
        """Get default moving average windows based on frequency"""
        ma_map = {
            'daily': [7, 30, 90],
            'weekly': [4, 8, 13],
            'monthly': [3, 6, 12],
            'quarterly': [2, 4],
            'annual': [2, 3]
        }
        
        return ma_map.get(frequency, [3, 6])
    
    def get_feature_importance_info(self, feature_name: str) -> Dict:
        """
        Get metadata about a feature
        Useful for interpretation
        """
        return self.feature_metadata.get(feature_name, {'type': 'unknown'})
    
    def get_created_features_by_type(self, feature_type: str) -> List[str]:
        """Get all created features of a specific type"""
        return [
            feat for feat, meta in self.feature_metadata.items()
            if meta.get('type') == feature_type
        ]
    
    def reset(self):
        """Reset feature tracking"""
        self.created_features = []
        self.feature_metadata = {}


class FeatureValidator:
    """
    Validate features for quality and usability
    """
    
    @staticmethod
    def validate_features(df: pd.DataFrame, 
                         feature_cols: List[str],
                         target_col: str,
                         min_valid_pct: float = 0.7) -> Dict:
        """
        Comprehensive feature validation
        
        Args:
            df: Dataframe with features
            feature_cols: Feature columns to validate
            target_col: Target variable
            min_valid_pct: Minimum valid (non-missing) percentage
        
        Returns:
            Validation report
        """
        report = {
            'valid_features': [],
            'invalid_features': [],
            'warnings': [],
            'stats': {}
        }
        
        for col in feature_cols:
            if col not in df.columns:
                report['invalid_features'].append({
                    'feature': col,
                    'reason': 'column not found'
                })
                continue
            
            # Check valid percentage
            valid_pct = df[col].notna().sum() / len(df)
            
            if valid_pct < min_valid_pct:
                report['invalid_features'].append({
                    'feature': col,
                    'reason': f'too many missing ({(1-valid_pct)*100:.1f}%)'
                })
                continue
            
            # Check variance
            values = df[col].dropna()
            if len(values) >= 2:
                if values.var() == 0:
                    report['invalid_features'].append({
                        'feature': col,
                        'reason': 'zero variance (constant)'
                    })
                    continue
            
            # Check for inf values
            if pd.api.types.is_numeric_dtype(df[col]):
                if np.isinf(df[col]).any():
                    report['warnings'].append(f"{col}: contains infinite values")
            
            # Feature is valid
            report['valid_features'].append(col)
            report['stats'][col] = {
                'valid_pct': float(valid_pct * 100),
                'mean': float(values.mean()) if len(values) > 0 else None,
                'std': float(values.std()) if len(values) > 1 else None
            }
        
        return report
    
    @staticmethod
    def check_target_overlap(df: pd.DataFrame,
                            feature_cols: List[str],
                            target_col: str) -> Dict:
        """
        Check overlap between features and target (no look-ahead bias)
        
        Args:
            df: Dataframe
            feature_cols: Feature columns
            target_col: Target column
        
        Returns:
            Overlap analysis
        """
        # Count rows where both feature and target are available
        target_valid = df[target_col].notna()
        
        overlaps = {}
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            feature_valid = df[col].notna()
            overlap_count = (target_valid & feature_valid).sum()
            overlap_pct = overlap_count / target_valid.sum() * 100
            
            overlaps[col] = {
                'overlap_count': int(overlap_count),
                'overlap_pct': float(overlap_pct)
            }
        
        return overlaps
