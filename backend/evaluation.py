"""
Evaluation Module
Comprehensive performance metrics and statistical tests

Institution: ISTAT
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class Evaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.results = {}
        
    def compute_metrics(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       baseline_pred: Optional[np.ndarray] = None,
                       model_name: str = "Model") -> Dict:
        """
        Compute comprehensive performance metrics
        
        Args:
            y_true: Actual values
            y_pred: Model predictions
            baseline_pred: Baseline predictions (for comparison)
            model_name: Name of model
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        
        # Bias
        metrics['Bias'] = np.mean(y_pred - y_true)
        metrics['Abs_Bias'] = np.abs(metrics['Bias'])
        
        # Median absolute error (robust to outliers)
        metrics['MedAE'] = np.median(np.abs(y_true - y_pred))
        
        # Quantile losses
        metrics['Q90_Error'] = np.quantile(np.abs(y_true - y_pred), 0.9)
        metrics['Max_Error'] = np.max(np.abs(y_true - y_pred))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['R2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Direction accuracy (for changes)
        if len(y_true) > 1:
            correct_direction = np.sum(np.sign(y_pred) == np.sign(y_true))
            metrics['Direction_Accuracy'] = correct_direction / len(y_true) * 100
        
        # Comparison with baseline
        if baseline_pred is not None:
            baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
            metrics['Baseline_RMSE'] = baseline_rmse
            
            # Improvement percentage
            improvement = (1 - metrics['RMSE'] / baseline_rmse) * 100
            metrics['Improvement_pct'] = improvement
            
            # Win rate (percentage of periods where model beats baseline)
            model_errors = np.abs(y_true - y_pred)
            baseline_errors = np.abs(y_true - baseline_pred)
            wins = np.sum(model_errors < baseline_errors)
            metrics['Win_Rate_pct'] = wins / len(y_true) * 100
            
            # Statistical tests
            try:
                cw_stat, cw_pval = self.clark_west_test(
                    y_true, baseline_pred, y_pred
                )
                metrics['Clark_West_stat'] = cw_stat
                metrics['p_value'] = cw_pval
                
                dm_stat, dm_pval = self.diebold_mariano_test(
                    y_true, baseline_pred, y_pred
                )
                metrics['DM_stat'] = dm_stat
                metrics['DM_pvalue'] = dm_pval
                
            except Exception as e:
                print(f"Warning: Could not compute statistical tests: {e}")
                metrics['p_value'] = np.nan
        
        return metrics
    
    def clark_west_test(self,
                       y_true: np.ndarray,
                       pred_baseline: np.ndarray,
                       pred_extended: np.ndarray,
                       bandwidth: Optional[int] = None) -> Tuple[float, float]:
        """
        Clark-West test for nested models
        
        Tests H0: equal forecast accuracy vs H1: extended model better
        
        Args:
            y_true: Actual values
            pred_baseline: Baseline (restricted) predictions
            pred_extended: Extended (unrestricted) predictions
            bandwidth: HAC bandwidth (default: n^(1/3))
            
        Returns:
            Tuple of (test_statistic, p_value)
        """
        n = len(y_true)
        
        # Forecast errors
        e_baseline = y_true - pred_baseline
        e_extended = y_true - pred_extended
        
        # Adjusted loss differential
        f_t = e_baseline**2 - (e_extended**2 - (pred_baseline - pred_extended)**2)
        f_bar = f_t.mean()
        
        # HAC standard error (Newey-West)
        if bandwidth is None:
            bandwidth = int(np.ceil(n ** (1/3)))
        
        # Demean
        f_demean = f_t - f_bar
        
        # Variance calculation with Bartlett kernel
        gamma_0 = (f_demean ** 2).mean()
        
        gamma_sum = 0
        for lag in range(1, min(bandwidth + 1, n)):
            if lag < n:
                gamma_lag = (f_demean[lag:] * f_demean[:-lag]).mean()
                weight = 1 - lag / (bandwidth + 1)  # Bartlett kernel
                gamma_sum += 2 * weight * gamma_lag
        
        var_hac = gamma_0 + gamma_sum
        se_hac = np.sqrt(var_hac / n)
        
        # Test statistic
        if se_hac > 0:
            cw_stat = f_bar / se_hac
        else:
            cw_stat = 0
        
        # p-value (one-tailed)
        p_value = 1 - stats.norm.cdf(cw_stat)
        
        return cw_stat, p_value
    
    def diebold_mariano_test(self,
                            y_true: np.ndarray,
                            pred_1: np.ndarray,
                            pred_2: np.ndarray) -> Tuple[float, float]:
        """
        Diebold-Mariano test for equal predictive accuracy
        
        Tests H0: equal expected loss (two-tailed)
        
        Args:
            y_true: Actual values
            pred_1: First model predictions
            pred_2: Second model predictions
            
        Returns:
            Tuple of (test_statistic, p_value)
        """
        # Loss differential (squared errors)
        e1 = y_true - pred_1
        e2 = y_true - pred_2
        
        d = e1**2 - e2**2
        d_bar = d.mean()
        
        # Variance
        n = len(d)
        d_var = d.var(ddof=1) / n
        
        # Test statistic
        if d_var > 0:
            dm_stat = d_bar / np.sqrt(d_var)
        else:
            dm_stat = 0
        
        # p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        return dm_stat, p_value
    
    def period_wise_evaluation(self,
                              y_true: pd.Series,
                              y_pred: np.ndarray,
                              y_baseline: np.ndarray,
                              dates: pd.Series,
                              period: str = 'year') -> pd.DataFrame:
        """
        Evaluate performance by time period
        
        Args:
            y_true: Actual values
            y_pred: Model predictions
            y_baseline: Baseline predictions
            dates: Date series
            period: Grouping period ('year', 'quarter', 'month')
            
        Returns:
            DataFrame with period-wise metrics
        """
        df = pd.DataFrame({
            'date': dates.values,
            'y_true': y_true.values,
            'y_pred': y_pred,
            'y_baseline': y_baseline
        })
        
        # Create period column
        if period == 'year':
            df['period'] = pd.to_datetime(df['date']).dt.year
        elif period == 'quarter':
            df['period'] = pd.to_datetime(df['date']).dt.to_period('Q').astype(str)
        elif period == 'month':
            df['period'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
        else:
            df['period'] = 'All'
        
        results = []
        
        for per, group in df.groupby('period'):
            rmse_model = np.sqrt(mean_squared_error(
                group['y_true'], group['y_pred']
            ))
            
            rmse_baseline = np.sqrt(mean_squared_error(
                group['y_true'], group['y_baseline']
            ))
            
            mae_model = mean_absolute_error(
                group['y_true'], group['y_pred']
            )
            
            improvement = (1 - rmse_model / rmse_baseline) * 100
            
            # Win rate
            model_errors = np.abs(group['y_true'] - group['y_pred'])
            baseline_errors = np.abs(group['y_true'] - group['y_baseline'])
            win_rate = (model_errors < baseline_errors).sum() / len(group) * 100
            
            results.append({
                'Period': per,
                'N': len(group),
                'Baseline_RMSE': rmse_baseline,
                'Model_RMSE': rmse_model,
                'Model_MAE': mae_model,
                'Improvement_pct': improvement,
                'Win_Rate_pct': win_rate
            })
        
        return pd.DataFrame(results)
    
    def backtesting_analysis(self,
                            train_data: pd.DataFrame,
                            test_data: pd.DataFrame,
                            model_class,
                            expanding_window: bool = True,
                            min_train_size: int = 36) -> Dict:
        """
        Perform backtesting analysis with walk-forward validation
        
        Args:
            train_data: Initial training data
            test_data: Test data
            model_class: Model class to instantiate
            expanding_window: Use expanding vs rolling window
            min_train_size: Minimum training window size
            
        Returns:
            Dictionary with backtesting results
        """
        predictions = []
        actuals = []
        dates = []
        
        full_data = pd.concat([train_data, test_data], ignore_index=True)
        train_end_idx = len(train_data)
        
        for i in range(train_end_idx, len(full_data)):
            if expanding_window:
                # Expanding window: use all data from start
                train_subset = full_data.iloc[:i]
            else:
                # Rolling window: use fixed size
                train_start = max(0, i - min_train_size)
                train_subset = full_data.iloc[train_start:i]
            
            if len(train_subset) < min_train_size:
                continue
            
            # Train model
            try:
                model = model_class()
                model.fit(train_subset)
                
                # Predict one step ahead
                test_row = full_data.iloc[[i]]
                pred = model.predict(test_row)[0]
                
                predictions.append(pred)
                actuals.append(full_data.iloc[i]['target'])
                dates.append(full_data.iloc[i]['date'])
                
            except Exception as e:
                print(f"Warning: Backtest failed at step {i}: {e}")
                continue
        
        if len(predictions) == 0:
            return {'error': 'No predictions generated'}
        
        # Compute metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates,
            'RMSE': rmse,
            'MAE': mae,
            'n_steps': len(predictions)
        }


class ResultsAnalyzer:
    """Analyze and summarize model results"""
    
    @staticmethod
    def create_comparison_table(results_list: list) -> pd.DataFrame:
        """
        Create formatted comparison table
        
        Args:
            results_list: List of result dictionaries
            
        Returns:
            Formatted DataFrame
        """
        df = pd.DataFrame(results_list)
        
        # Sort by RMSE
        df = df.sort_values('RMSE').reset_index(drop=True)
        
        # Add rank
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        return df
    
    @staticmethod
    def identify_best_model(results_list: list,
                           metric: str = 'RMSE',
                           minimize: bool = True) -> Dict:
        """
        Identify best model based on metric
        
        Args:
            results_list: List of result dictionaries
            metric: Metric to optimize
            minimize: Whether to minimize (True) or maximize (False)
            
        Returns:
            Dictionary with best model info
        """
        if minimize:
            best_idx = np.argmin([r[metric] for r in results_list])
        else:
            best_idx = np.argmax([r[metric] for r in results_list])
        
        return results_list[best_idx]
    
    @staticmethod
    def statistical_summary(results_df: pd.DataFrame) -> Dict:
        """
        Generate statistical summary of results
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Summary dictionary
        """
        summary = {
            'n_models': len(results_df),
            'best_model': results_df.loc[results_df['RMSE'].idxmin(), 'Model'],
            'best_rmse': results_df['RMSE'].min(),
            'worst_rmse': results_df['RMSE'].max(),
            'mean_rmse': results_df['RMSE'].mean(),
            'std_rmse': results_df['RMSE'].std(),
            'n_significant': (results_df['p_value'] < 0.05).sum() if 'p_value' in results_df else 0
        }
        
        return summary
