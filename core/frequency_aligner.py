"""
Frequency Alignment Engine
Handles mixed-frequency data and MIDAS aggregation
Real implementation - no demo code
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG


class FrequencyAligner:
    """
    Aligns high-frequency data (daily/weekly) to low-frequency (monthly/quarterly)
    Implements multiple MIDAS weighting schemes
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.alignment_cache = {}
    
    def detect_alignment_strategy(self, target_freq: str, feature_freq: str) -> str:
        """
        Determine best alignment strategy based on frequencies
        
        Args:
            target_freq: Target variable frequency (monthly, quarterly)
            feature_freq: Feature frequency (daily, weekly, monthly)
        
        Returns:
            Strategy name: 'direct', 'simple_agg', 'midas'
        """
        # Same frequency - direct merge
        if target_freq == feature_freq:
            return 'direct'
        
        # Map frequency hierarchy
        freq_hierarchy = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'annual': 365
        }
        
        target_days = freq_hierarchy.get(target_freq, 30)
        feature_days = freq_hierarchy.get(feature_freq, 30)
        
        # High to low frequency
        if feature_days < target_days:
            ratio = target_days / feature_days
            
            # Significant frequency difference → MIDAS
            if ratio >= 3:
                return 'midas'
            else:
                return 'simple_agg'
        
        # Low to high frequency (interpolation)
        else:
            return 'interpolate'
    
    def align_datasets(self, 
                       target_df: pd.DataFrame,
                       feature_df: pd.DataFrame,
                       target_date_col: str,
                       feature_date_col: str,
                       target_freq: str,
                       feature_freq: str,
                       feature_cols: List[str],
                       cutoff_day: Optional[int] = None) -> pd.DataFrame:
        """
        Main alignment function
        
        Args:
            target_df: Target variable dataframe
            feature_df: Feature dataframe (potentially different frequency)
            target_date_col: Date column in target
            feature_date_col: Date column in features
            target_freq: Target frequency
            feature_freq: Feature frequency
            feature_cols: Columns to align
            cutoff_day: Day cutoff for nowcasting (e.g., 15 for mid-month)
        
        Returns:
            Aligned dataframe
        """
        strategy = self.detect_alignment_strategy(target_freq, feature_freq)
        
        if strategy == 'direct':
            return self._direct_merge(target_df, feature_df, 
                                     target_date_col, feature_date_col,
                                     feature_cols)
        
        elif strategy == 'midas':
            return self._midas_alignment(target_df, feature_df,
                                        target_date_col, feature_date_col,
                                        feature_cols, target_freq, feature_freq,
                                        cutoff_day)
        
        elif strategy == 'simple_agg':
            return self._simple_aggregation(target_df, feature_df,
                                           target_date_col, feature_date_col,
                                           feature_cols, target_freq)
        
        else:  # interpolate
            return self._interpolate(target_df, feature_df,
                                    target_date_col, feature_date_col,
                                    feature_cols)
    
    def _direct_merge(self, target_df, feature_df, 
                     target_date_col, feature_date_col, feature_cols):
        """Direct merge for same frequency"""
        target = target_df[[target_date_col]].copy()
        target['date_key'] = pd.to_datetime(target[target_date_col])
        
        feature = feature_df[[feature_date_col] + feature_cols].copy()
        feature['date_key'] = pd.to_datetime(feature[feature_date_col])
        
        merged = target.merge(feature, on='date_key', how='left')
        merged = merged.drop(columns=['date_key', feature_date_col])
        
        return merged
    
    def _simple_aggregation(self, target_df, feature_df,
                           target_date_col, feature_date_col,
                           feature_cols, target_freq):
        """
        Simple aggregation without MIDAS
        Average/sum high-freq data into low-freq periods
        """
        result = target_df[[target_date_col]].copy()
        result['date'] = pd.to_datetime(result[target_date_col])
        
        feature = feature_df[[feature_date_col] + feature_cols].copy()
        feature['date'] = pd.to_datetime(feature[feature_date_col])
        
        for idx, row in result.iterrows():
            target_date = row['date']
            
            # Define period
            if target_freq == 'monthly':
                period_start = pd.Timestamp(target_date.year, target_date.month, 1)
                period_end = period_start + pd.offsets.MonthEnd(0)
            elif target_freq == 'quarterly':
                quarter = (target_date.month - 1) // 3 + 1
                period_start = pd.Timestamp(target_date.year, (quarter-1)*3 + 1, 1)
                period_end = period_start + pd.offsets.QuarterEnd(0)
            else:
                continue
            
            # Get features in period
            mask = (feature['date'] >= period_start) & (feature['date'] <= period_end)
            period_data = feature.loc[mask, feature_cols]
            
            if len(period_data) > 0:
                # Mean aggregation
                for col in feature_cols:
                    result.loc[idx, f"{col}_agg"] = period_data[col].mean()
        
        result = result.drop(columns=['date'])
        return result
    
    def _midas_alignment(self, target_df, feature_df,
                        target_date_col, feature_date_col,
                        feature_cols, target_freq, feature_freq,
                        cutoff_day=None):
        """
        MIDAS alignment - REAL IMPLEMENTATION
        
        Creates multiple MIDAS features with different weighting schemes
        """
        if cutoff_day is None:
            cutoff_day = self.config.MIDAS_CUTOFF_DAY
        
        result = target_df[[target_date_col]].copy()
        result['date'] = pd.to_datetime(result[target_date_col])
        
        feature = feature_df[[feature_date_col] + feature_cols].copy()
        feature['date'] = pd.to_datetime(feature[feature_date_col])
        feature = feature.sort_values('date').reset_index(drop=True)
        
        # For each target observation
        for idx, row in result.iterrows():
            target_date = row['date']
            
            # Cutoff date (e.g., 15th of month for nowcasting)
            if target_freq == 'monthly':
                cutoff_date = pd.Timestamp(target_date.year, target_date.month, 
                                          min(cutoff_day, target_date.day))
            elif target_freq == 'quarterly':
                quarter_end = target_date + pd.offsets.QuarterEnd(0)
                cutoff_date = quarter_end - pd.Timedelta(days=15)
            else:
                cutoff_date = target_date
            
            # Get available high-freq data up to cutoff
            available = feature[feature['date'] <= cutoff_date]
            
            if len(available) == 0:
                continue
            
            # Apply MIDAS with different specifications
            for window in self.config.MIDAS_WINDOWS:
                # Take last W observations
                recent = available.tail(window)
                
                if len(recent) < 2:
                    continue
                
                # Equal weights
                weights_eq = self._compute_equal_weights(len(recent))
                for col in feature_cols:
                    values = recent[col].to_numpy(dtype=float)
                    valid_mask = np.isfinite(values)
                    
                    if valid_mask.sum() >= 2:
                        agg_value = np.sum(values[valid_mask] * weights_eq[valid_mask]) / weights_eq[valid_mask].sum()
                        result.loc[idx, f"{col}__eq__W{window}"] = float(agg_value)
                
                # Exponential weights
                for lam in self.config.MIDAS_LAMBDAS:
                    weights_exp = self._compute_exponential_weights(len(recent), lam)
                    
                    for col in feature_cols:
                        values = recent[col].to_numpy(dtype=float)
                        valid_mask = np.isfinite(values)
                        
                        if valid_mask.sum() >= 2:
                            agg_value = np.sum(values[valid_mask] * weights_exp[valid_mask]) / weights_exp[valid_mask].sum()
                            result.loc[idx, f"{col}__exp{lam}__W{window}"] = float(agg_value)
        
        result = result.drop(columns=['date'])
        return result
    
    def _compute_equal_weights(self, n: int) -> np.ndarray:
        """
        Equal weights for MIDAS
        
        Args:
            n: Number of observations
        
        Returns:
            Weight array (most recent = last element)
        """
        weights = np.ones(n, dtype=float)
        return weights / weights.sum()
    
    def _compute_exponential_weights(self, n: int, lambda_param: float) -> np.ndarray:
        """
        Exponential Almon weights for MIDAS
        More weight on recent observations
        
        Args:
            n: Number of observations
            lambda_param: Decay parameter (higher = more recent weight)
        
        Returns:
            Weight array (most recent = last element)
        """
        # j ranges from 0 (oldest) to n-1 (most recent)
        # w[j] ∝ exp(-λ * (n-1-j)) so most recent has highest weight
        j = np.arange(n, dtype=float)
        weights = np.exp(-lambda_param * (n - 1 - j))
        
        # Handle numerical issues
        weights[weights < 1e-10] = 1e-10
        
        return weights / weights.sum()
    
    def _compute_beta_weights(self, n: int, theta1: float, theta2: float) -> np.ndarray:
        """
        Beta polynomial weights for MIDAS
        Flexible weight distribution
        
        Args:
            n: Number of observations
            theta1, theta2: Beta distribution parameters
        
        Returns:
            Weight array
        """
        from scipy.special import beta as beta_func
        
        j = np.arange(n, dtype=float)
        x = (j + 1) / n
        
        weights = (x ** (theta1 - 1)) * ((1 - x) ** (theta2 - 1))
        weights = weights / beta_func(theta1, theta2)
        
        # Normalize
        return weights / weights.sum()
    
    def _interpolate(self, target_df, feature_df,
                    target_date_col, feature_date_col, feature_cols):
        """
        Interpolate low-freq features to high-freq target
        (Less common in nowcasting)
        """
        result = target_df[[target_date_col]].copy()
        result['date'] = pd.to_datetime(result[target_date_col])
        
        feature = feature_df[[feature_date_col] + feature_cols].copy()
        feature['date'] = pd.to_datetime(feature[feature_date_col])
        
        # Merge and forward-fill
        merged = result.merge(feature, on='date', how='left', suffixes=('', '_feat'))
        
        for col in feature_cols:
            merged[col] = merged[col].fillna(method='ffill')
        
        merged = merged.drop(columns=['date'] + [c for c in merged.columns if c.endswith('_feat')])
        return merged
    
    def create_midas_features(self, 
                             target_dates: pd.Series,
                             feature_df: pd.DataFrame,
                             feature_date_col: str,
                             feature_cols: List[str],
                             windows: Optional[List[int]] = None,
                             lambdas: Optional[List[float]] = None,
                             cutoff_day: int = 15) -> pd.DataFrame:
        """
        Standalone MIDAS feature creation
        Used when user explicitly wants MIDAS
        
        Args:
            target_dates: Target observation dates
            feature_df: High-frequency feature data
            feature_date_col: Date column in features
            feature_cols: Columns to aggregate
            windows: MIDAS windows to use
            lambdas: Exponential decay parameters
            cutoff_day: Nowcast cutoff day
        
        Returns:
            DataFrame with MIDAS features
        """
        if windows is None:
            windows = self.config.MIDAS_WINDOWS
        if lambdas is None:
            lambdas = self.config.MIDAS_LAMBDAS
        
        target_df = pd.DataFrame({'date': target_dates})
        
        result = self._midas_alignment(
            target_df, feature_df,
            'date', feature_date_col,
            feature_cols, 'monthly', 'weekly',
            cutoff_day
        )
        
        return result
    
    def transform_features_mom(self, df: pd.DataFrame, feature_cols: List[str],
                              clip_range: Tuple[float, float] = (-50, 50)) -> pd.DataFrame:
        """
        Transform MIDAS features to Month-over-Month (MoM) changes
        
        Args:
            df: DataFrame with MIDAS features
            feature_cols: Columns to transform
            clip_range: Clip outliers to this range
        
        Returns:
            DataFrame with MoM features added
        """
        result = df.copy()
        
        for col in feature_cols:
            if col not in result.columns:
                continue
            
            values = result[col].to_numpy(dtype=float)
            
            # Log transformation (handle negatives/zeros)
            values_safe = np.where(values > 0, values, np.nan)
            log_values = np.log1p(values_safe)
            
            # Month-over-month change
            log_lag = pd.Series(log_values).shift(1).to_numpy()
            mom = (log_values - log_lag) * 100.0
            
            # Clip outliers
            mom = np.clip(mom, clip_range[0], clip_range[1])
            
            result[f"{col}__mom"] = mom
        
        return result
    
    def get_alignment_info(self, target_freq: str, feature_freq: str) -> Dict:
        """
        Get information about alignment strategy
        Useful for UI display
        """
        strategy = self.detect_alignment_strategy(target_freq, feature_freq)
        
        info = {
            'strategy': strategy,
            'description': '',
            'recommended_windows': [],
            'recommended_lambdas': []
        }
        
        if strategy == 'direct':
            info['description'] = "Direct merge - frequencies match"
        
        elif strategy == 'simple_agg':
            info['description'] = "Simple averaging - moderate frequency difference"
        
        elif strategy == 'midas':
            info['description'] = "MIDAS aggregation - significant frequency difference"
            
            # Estimate appropriate windows
            if target_freq == 'monthly':
                if feature_freq == 'daily':
                    info['recommended_windows'] = [20, 30]  # 20-30 days
                elif feature_freq == 'weekly':
                    info['recommended_windows'] = [4, 8, 12]  # 4-12 weeks
            
            elif target_freq == 'quarterly':
                if feature_freq == 'daily':
                    info['recommended_windows'] = [60, 90]
                elif feature_freq == 'weekly':
                    info['recommended_windows'] = [12, 16]
                elif feature_freq == 'monthly':
                    info['recommended_windows'] = [3]
            
            info['recommended_lambdas'] = [0.6, 0.8]
        
        elif strategy == 'interpolate':
            info['description'] = "Interpolation - low-to-high frequency (uncommon)"
        
        return info
    
    def validate_alignment(self, aligned_df: pd.DataFrame) -> Dict:
        """
        Validate alignment quality
        Check for excessive missing values, alignment errors
        """
        report = {
            'valid': True,
            'warnings': [],
            'stats': {}
        }
        
        # Check missing values in aligned features
        feature_cols = [c for c in aligned_df.columns if '__' in c or c.endswith('_agg')]
        
        if feature_cols:
            missing_pcts = aligned_df[feature_cols].isna().sum() / len(aligned_df) * 100
            report['stats']['missing_pcts'] = missing_pcts.to_dict()
            
            # Warn if too many missing
            high_missing = missing_pcts[missing_pcts > 30]
            if len(high_missing) > 0:
                report['warnings'].append(
                    f"{len(high_missing)} features have >30% missing values after alignment"
                )
        
        # Check if any features were created
        if len(feature_cols) == 0:
            report['warnings'].append("No features created - alignment may have failed")
        
        return report


class FrequencyConverter:
    """
    Helper class for frequency conversions
    Useful utilities for date manipulation
    """
    
    @staticmethod
    def infer_period_dates(date: pd.Timestamp, freq: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get period start and end dates for a given frequency
        
        Args:
            date: Reference date
            freq: Frequency (monthly, quarterly, annual)
        
        Returns:
            (period_start, period_end)
        """
        if freq == 'monthly':
            start = pd.Timestamp(date.year, date.month, 1)
            end = start + pd.offsets.MonthEnd(0)
        
        elif freq == 'quarterly':
            quarter = (date.month - 1) // 3 + 1
            start = pd.Timestamp(date.year, (quarter-1)*3 + 1, 1)
            end = start + pd.offsets.QuarterEnd(0)
        
        elif freq == 'annual':
            start = pd.Timestamp(date.year, 1, 1)
            end = pd.Timestamp(date.year, 12, 31)
        
        else:
            start = end = date
        
        return start, end
    
    @staticmethod
    def get_frequency_ratio(high_freq: str, low_freq: str) -> float:
        """
        Calculate approximate ratio between frequencies
        """
        freq_days = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'annual': 365
        }
        
        high_days = freq_days.get(high_freq, 1)
        low_days = freq_days.get(low_freq, 30)
        
        return low_days / high_days
