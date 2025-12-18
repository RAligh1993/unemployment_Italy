"""
Feature Engineering Module
Creates MIDAS features, interactions, and transformations


Institution: ISTAT
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.special import beta as beta_func


class FeatureEngineer:
    """Advanced feature engineering for nowcasting"""
    
    def __init__(self):
        self.scaler = None
        
    @staticmethod
    def create_midas_lag_features(gt_weekly: pd.DataFrame,
                                  unemp_monthly: pd.DataFrame,
                                  keywords: List[str],
                                  n_lags: int = 4) -> pd.DataFrame:
        """
        Create MIDAS lag features from weekly GT data
        
        Args:
            gt_weekly: Weekly Google Trends data
            unemp_monthly: Monthly unemployment data (for dates)
            keywords: List of keywords to create lags for
            n_lags: Number of weekly lags
            
        Returns:
            DataFrame with lag features for each keyword
        """
        results = []
        
        for idx, row in unemp_monthly.iterrows():
            month_end = row['date']
            
            # Get last n_lags weeks up to month_end
            mask = gt_weekly['date'] <= month_end
            last_weeks = gt_weekly[mask].tail(n_lags)
            
            if len(last_weeks) < n_lags:
                # Insufficient data
                features = {}
                for kw in keywords:
                    for lag in range(n_lags):
                        features[f'{kw}_w{lag}'] = np.nan
            else:
                # Create lag features (lag 0 = most recent week)
                features = {}
                for kw in keywords:
                    for lag in range(n_lags):
                        # Most recent = lag 0, oldest = lag n_lags-1
                        features[f'{kw}_w{lag}'] = last_weeks.iloc[-(lag+1)][kw]
            
            features['date'] = month_end
            results.append(features)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def exponential_weights(n_lags: int, theta: float) -> np.ndarray:
        """
        Create exponential Almon MIDAS weights
        
        Args:
            n_lags: Number of lags
            theta: Decay parameter (higher = faster decay)
            
        Returns:
            Normalized weight array
        """
        weights = np.exp(-theta * np.arange(n_lags))
        return weights / weights.sum()
    
    @staticmethod
    def beta_weights(n_lags: int, theta1: float, theta2: float) -> np.ndarray:
        """
        Create Beta polynomial MIDAS weights
        
        Args:
            n_lags: Number of lags
            theta1, theta2: Beta distribution parameters
            
        Returns:
            Normalized weight array
        """
        lags = np.arange(1, n_lags + 1)
        weights = []
        
        for i in lags:
            eps_i = i / n_lags
            w = (eps_i ** (theta1 - 1) * (1 - eps_i) ** (theta2 - 1) / 
                 beta_func(theta1, theta2))
            weights.append(w)
        
        weights = np.array(weights)
        
        # Reverse (most recent first) and normalize
        weights = weights[::-1]
        return weights / weights.sum()
    
    @staticmethod
    def apply_midas_weights(lag_features: pd.DataFrame,
                           keywords: List[str],
                           weights: np.ndarray,
                           n_lags: int) -> pd.DataFrame:
        """
        Apply MIDAS weights to lag features
        
        Args:
            lag_features: DataFrame with lag columns (kw_w0, kw_w1, ...)
            keywords: List of keywords
            weights: Weight array
            n_lags: Number of lags
            
        Returns:
            DataFrame with weighted MIDAS features
        """
        result = lag_features[['date']].copy()
        
        for kw in keywords:
            weighted_sum = 0
            
            for lag in range(n_lags):
                col = f'{kw}_w{lag}'
                if col in lag_features.columns:
                    weighted_sum += lag_features[col].values * weights[lag]
            
            result[f'{kw}_midas'] = weighted_sum
        
        return result
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame,
                                    feature_cols: List[str]) -> pd.DataFrame:
        """
        Create interaction features (pairwise products)
        
        Args:
            df: Input dataframe
            feature_cols: Columns to create interactions for
            
        Returns:
            DataFrame with interaction features
        """
        result = df.copy()
        
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                col1, col2 = feature_cols[i], feature_cols[j]
                
                if col1 in df.columns and col2 in df.columns:
                    result[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        return result
    
    @staticmethod
    def create_polynomial_features(df: pd.DataFrame,
                                   feature_cols: List[str],
                                   degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            df: Input dataframe
            feature_cols: Columns to create polynomials for
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        result = df.copy()
        
        for col in feature_cols:
            if col in df.columns:
                for d in range(2, degree + 1):
                    result[f'{col}_pow{d}'] = df[col] ** d
        
        return result
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame,
                               feature_cols: List[str],
                               windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            df: Input dataframe (must be sorted by date)
            feature_cols: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        result = df.copy()
        
        for col in feature_cols:
            if col in df.columns:
                for window in windows:
                    result[f'{col}_roll_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    result[f'{col}_roll_std_{window}'] = df[col].rolling(window, min_periods=1).std()
        
        return result
    
    def select_top_keywords(self,
                           df: pd.DataFrame,
                           keyword_cols: List[str],
                           target_col: str = 'target',
                           top_k: int = 6) -> List[str]:
        """
        Select top keywords based on correlation with target
        
        Args:
            df: Training dataframe
            keyword_cols: List of keyword columns
            target_col: Target column name
            top_k: Number of keywords to select
            
        Returns:
            List of top keyword names
        """
        correlations = {}
        
        for kw in keyword_cols:
            if kw in df.columns:
                corr = abs(df[kw].corr(df[target_col]))
                if not np.isnan(corr):
                    correlations[kw] = corr
        
        # Sort by correlation
        sorted_kw = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        return [kw for kw, _ in sorted_kw[:top_k]]
