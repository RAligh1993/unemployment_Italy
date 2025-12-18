"""
Data Loader Module
Handles data upload, validation, cleaning, and preprocessing

Institution: ISTAT
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Comprehensive data loading and preprocessing"""
    
    def __init__(self):
        self.df_unemployment = None
        self.df_gt_merged = None
        self.df_exog = None
        self.gt_quality = {}
        
    def load_demo_data(self) -> pd.DataFrame:
        """Generate demo unemployment data for testing"""
        dates = pd.date_range('2016-01-01', '2025-08-01', freq='MS')
        n = len(dates)
        
        # Simulate realistic unemployment data
        trend = np.linspace(10, 6, n)
        seasonal = 0.5 * np.sin(np.arange(n) * 2 * np.pi / 12)
        covid_shock = np.zeros(n)
        covid_start = 50  # Around 2020
        covid_shock[covid_start:covid_start+12] = np.array([2, 3.5, 4, 3, 2, 1.5, 1, 0.5, 0.3, 0.2, 0.1, 0])
        
        noise = np.random.randn(n) * 0.3
        unemp = trend + seasonal + covid_shock + noise
        unemp = np.clip(unemp, 5, 13)
        
        # Youth unemployment (higher and more volatile)
        unemp_youth = unemp * 1.8 + np.random.randn(n) * 0.5
        
        df = pd.DataFrame({
            'date': dates,
            'unemp': unemp,
            'unemp(25-34)': unemp_youth
        })
        
        return df
    
    def validate_unemployment_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate unemployment data structure and quality
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        if 'date' not in df.columns:
            errors.append("Missing 'date' column")
        
        if 'unemp' not in df.columns:
            errors.append("Missing 'unemp' column")
        
        if len(errors) > 0:
            return False, errors
        
        # Check date format
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            errors.append("Cannot parse 'date' column as datetime")
        
        # Check for missing values
        if df['unemp'].isna().sum() > 0:
            errors.append(f"Found {df['unemp'].isna().sum()} missing values in unemployment")
        
        # Check value range (unemployment should be 0-100%)
        if df['unemp'].min() < 0 or df['unemp'].max() > 100:
            errors.append("Unemployment values outside reasonable range [0, 100]")
        
        # Check for duplicates
        if df['date'].duplicated().any():
            errors.append("Found duplicate dates")
        
        # Check sample size
        if len(df) < 36:
            errors.append(f"Insufficient data: {len(df)} months (minimum 36 required)")
        
        return len(errors) == 0, errors
    
    def load_google_trends(self, files: List) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and merge multiple Google Trends segments
        
        Args:
            files: List of uploaded Excel files
            
        Returns:
            Tuple of (merged_dataframe, quality_summary)
        """
        segments = {}
        
        # Load each segment
        for i, file in enumerate(files, 1):
            try:
                df = pd.read_excel(file)
                
                # Standardize date column
                if 'Week' in df.columns:
                    df['date'] = pd.to_datetime(df['Week'])
                elif 'week' in df.columns:
                    df['date'] = pd.to_datetime(df['week'])
                else:
                    df['date'] = pd.to_datetime(df.iloc[:, 0])
                
                # Handle '<1' values
                for col in df.columns:
                    if col not in ['Week', 'week', 'date']:
                        df[col] = pd.to_numeric(
                            df[col].astype(str).str.replace('<1', '0.5'),
                            errors='coerce'
                        )
                
                segments[i] = df
                
            except Exception as e:
                print(f"Error loading segment {i}: {e}")
        
        if len(segments) == 0:
            return None, {}
        
        # If only one segment, return as-is
        if len(segments) == 1:
            return segments[1], {'GOOD': list(segments[1].columns.drop('date'))}
        
        # Merge multiple segments
        merged, quality = self._merge_gt_segments(segments)
        
        return merged, quality
    
    def _merge_gt_segments(self, segments: Dict[int, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
        """
        Merge overlapping GT segments with quality assessment
        
        Args:
            segments: Dictionary of segment DataFrames
            
        Returns:
            Tuple of (merged_df, quality_dict)
        """
        # Get common keywords
        common_keywords = set(segments[1].columns)
        for seg in segments.values():
            common_keywords = common_keywords.intersection(seg.columns)
        common_keywords = common_keywords - {'date', 'Week', 'week'}
        common_keywords = list(common_keywords)
        
        # Compute scaling factors
        scaling_factors = {}
        
        for i in range(1, len(segments)):
            seg_i = segments[i]
            seg_i_plus_1 = segments[i+1]
            
            # Find overlap
            overlap_start = max(seg_i['date'].min(), seg_i_plus_1['date'].min())
            overlap_end = min(seg_i['date'].max(), seg_i_plus_1['date'].max())
            
            overlap_i = seg_i[(seg_i['date'] >= overlap_start) & (seg_i['date'] <= overlap_end)].set_index('date')
            overlap_i_plus_1 = seg_i_plus_1[(seg_i_plus_1['date'] >= overlap_start) & (seg_i_plus_1['date'] <= overlap_end)].set_index('date')
            
            for kw in common_keywords:
                if kw not in scaling_factors:
                    scaling_factors[kw] = {}
                
                # Compute median ratio
                ratios = overlap_i[kw] / overlap_i_plus_1[kw]
                ratios = ratios[(ratios > 0) & (ratios < np.inf)]
                
                if len(ratios) >= 20:
                    scaling_factors[kw][f'{i}_to_{i+1}'] = ratios.median()
        
        # Chain factors
        chained_factors = {}
        for kw in common_keywords:
            chained_factors[kw] = {1: 1.0}
            
            for j in range(2, len(segments) + 1):
                factor = 1.0
                for i in range(1, j):
                    pair_factor = scaling_factors.get(kw, {}).get(f'{i}_to_{i+1}', np.nan)
                    if not np.isnan(pair_factor):
                        factor *= pair_factor
                    else:
                        factor = np.nan
                        break
                
                chained_factors[kw][j] = factor
        
        # Quality assessment
        quality = {'GOOD': [], 'CAUTION': [], 'WARNING': []}
        
        for kw in common_keywords:
            factors = [chained_factors[kw][j] for j in range(1, len(segments) + 1)]
            factors = [f for f in factors if not np.isnan(f)]
            
            if len(factors) == 0:
                continue
            
            max_factor = max(factors)
            min_factor = min(factors)
            
            if max_factor > 3.0 or min_factor < 0.33:
                quality['WARNING'].append(kw)
            elif max_factor > 2.0 or min_factor < 0.5:
                quality['CAUTION'].append(kw)
            else:
                quality['GOOD'].append(kw)
        
        # Merge segments
        use_keywords = quality['GOOD'] + quality['CAUTION']
        
        all_data = []
        for seg_id, seg_data in segments.items():
            seg_copy = seg_data.copy()
            
            for kw in use_keywords:
                if kw in seg_copy.columns:
                    factor = chained_factors[kw][seg_id]
                    if not np.isnan(factor):
                        seg_copy[kw] = seg_copy[kw] / factor
            
            all_data.append(seg_copy[['date'] + use_keywords])
        
        # Concatenate and deduplicate
        merged = pd.concat(all_data, ignore_index=True)
        merged = merged.sort_values('date').drop_duplicates('date', keep='first')
        merged = merged.reset_index(drop=True)
        
        return merged, quality
    
    def aggregate_gt_to_monthly(self, 
                                gt_weekly: pd.DataFrame,
                                unemp_monthly: pd.DataFrame,
                                window: int = 12) -> pd.DataFrame:
        """
        Aggregate weekly GT to monthly using simple moving average
        
        Args:
            gt_weekly: Weekly GT data
            unemp_monthly: Monthly unemployment data (for dates)
            window: Number of weeks to average
            
        Returns:
            Monthly GT features
        """
        results = []
        
        keywords = [col for col in gt_weekly.columns if col != 'date']
        
        for idx, row in unemp_monthly.iterrows():
            month_end = row['date']
            
            # Get last 'window' weeks up to month_end
            mask = gt_weekly['date'] <= month_end
            available = gt_weekly[mask].tail(window)
            
            if len(available) < window:
                # Insufficient data
                gt_feat = {kw: np.nan for kw in keywords}
            else:
                gt_feat = {kw: available[kw].mean() for kw in keywords}
            
            gt_feat['date'] = month_end
            results.append(gt_feat)
        
        return pd.DataFrame(results)
    
    def create_lags(self, df: pd.DataFrame, 
                   lag_cols: List[str],
                   n_lags: int = 2) -> pd.DataFrame:
        """
        Create lagged features
        
        Args:
            df: Input dataframe
            lag_cols: Columns to create lags for
            n_lags: Number of lags
            
        Returns:
            DataFrame with lag features
        """
        result = df.copy()
        
        for col in lag_cols:
            if col in result.columns:
                for lag in range(1, n_lags + 1):
                    result[f'{col}_lag{lag}'] = result[col].shift(lag)
        
        return result
    
    def process_data(self,
                    df_unemployment: pd.DataFrame,
                    gt_files: Optional[List] = None,
                    exog_file: Optional = None) -> pd.DataFrame:
        """
        Complete data processing pipeline
        
        Args:
            df_unemployment: Unemployment data
            gt_files: Google Trends files (optional)
            exog_file: Exogenous variables file (optional)
            
        Returns:
            Clean, merged dataframe ready for modeling
        """
        # Validate unemployment data
        is_valid, errors = self.validate_unemployment_data(df_unemployment)
        if not is_valid:
            raise ValueError(f"Invalid unemployment data: {', '.join(errors)}")
        
        # Ensure date is datetime
        df_unemployment['date'] = pd.to_datetime(df_unemployment['date'])
        df_unemployment = df_unemployment.sort_values('date').reset_index(drop=True)
        
        self.df_unemployment = df_unemployment
        
        # Start with unemployment
        df = df_unemployment.copy()
        
        # Load Google Trends
        if gt_files and len(gt_files) > 0:
            gt_merged, quality = self.load_google_trends(gt_files)
            
            if gt_merged is not None:
                self.df_gt_merged = gt_merged
                self.gt_quality = quality
                
                # Aggregate to monthly
                gt_monthly = self.aggregate_gt_to_monthly(gt_merged, df)
                
                # Merge
                df = df.merge(gt_monthly, on='date', how='left')
        
        # Load exogenous variables
        if exog_file:
            try:
                df_exog = pd.read_csv(exog_file)
                df_exog['date'] = pd.to_datetime(df_exog['date'])
                
                self.df_exog = df_exog
                
                # Merge
                df = df.merge(df_exog, on='date', how='left')
                
            except Exception as e:
                print(f"Warning: Could not load exogenous data: {e}")
        
        # Create target (first difference)
        df['target'] = df['unemp'].diff()
        
        # Create lags
        lag_cols = ['unemp']
        if 'unemp(25-34)' in df.columns:
            lag_cols.append('unemp(25-34)')
        
        # Add CCI, HICP if available
        if 'CCI' in df.columns:
            lag_cols.append('CCI')
        if 'PRC-HICP' in df.columns:
            lag_cols.append('PRC-HICP')
        
        df = self.create_lags(df, lag_cols, n_lags=2)
        
        # Rename for consistency
        if 'unemp(25-34)_lag1' in df.columns:
            df['unemp_youth_lag1'] = df['unemp(25-34)_lag1']
        
        if 'CCI_lag1' in df.columns:
            df['CCI_lag1'] = df['CCI_lag1']
        
        if 'PRC-HICP_lag1' in df.columns:
            df['HICP_lag1'] = df['PRC-HICP_lag1']
        
        # Drop rows with NaN (from differencing and lagging)
        df_clean = df.dropna().reset_index(drop=True)
        
        return df_clean
