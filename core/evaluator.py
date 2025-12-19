"""
Comprehensive Evaluation Framework
Statistical tests, backtesting, performance metrics
Production-ready implementations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TestResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    is_significant: bool
    direction: str  # 'one-tailed' or 'two-tailed'
    null_hypothesis: str
    interpretation: str
    metadata: Dict


@dataclass
class BacktestResult:
    """Container for backtest results"""
    model_name: str
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    predictions: np.ndarray
    actual: np.ndarray
    metrics: Dict
    model_params: Dict


# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """
    Calculate comprehensive performance metrics
    """
    
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Root Mean Squared Error"""
        actual = np.asarray(actual, float).ravel()
        predicted = np.asarray(predicted, float).ravel()
        
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not mask.any():
            return np.nan
        
        errors = actual[mask] - predicted[mask]
        return float(np.sqrt(np.mean(errors ** 2)))
    
    @staticmethod
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Error"""
        actual = np.asarray(actual, float).ravel()
        predicted = np.asarray(predicted, float).ravel()
        
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not mask.any():
            return np.nan
        
        errors = np.abs(actual[mask] - predicted[mask])
        return float(np.mean(errors))
    
    @staticmethod
    def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        actual = np.asarray(actual, float).ravel()
        predicted = np.asarray(predicted, float).ravel()
        
        mask = np.isfinite(actual) & np.isfinite(predicted) & (actual != 0)
        if not mask.any():
            return np.nan
        
        errors = np.abs((actual[mask] - predicted[mask]) / actual[mask])
        return float(np.mean(errors) * 100)
    
    @staticmethod
    def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Squared Error"""
        actual = np.asarray(actual, float).ravel()
        predicted = np.asarray(predicted, float).ravel()
        
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not mask.any():
            return np.nan
        
        errors = actual[mask] - predicted[mask]
        return float(np.mean(errors ** 2))
    
    @staticmethod
    def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Bias (mean error)"""
        actual = np.asarray(actual, float).ravel()
        predicted = np.asarray(predicted, float).ravel()
        
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not mask.any():
            return np.nan
        
        errors = actual[mask] - predicted[mask]
        return float(np.mean(errors))
    
    @staticmethod
    def direction_accuracy(actual: np.ndarray, predicted: np.ndarray, 
                          reference: Optional[np.ndarray] = None) -> float:
        """
        Direction accuracy (% correct sign of change)
        
        Args:
            actual: Actual values
            predicted: Predicted values
            reference: Reference values (e.g., lag-1). If None, use lagged actual
        """
        actual = np.asarray(actual, float).ravel()
        predicted = np.asarray(predicted, float).ravel()
        
        if reference is None:
            reference = np.roll(actual, 1)
            reference[0] = np.nan
        else:
            reference = np.asarray(reference, float).ravel()
        
        # Direction of change
        actual_dir = np.sign(actual - reference)
        pred_dir = np.sign(predicted - reference)
        
        mask = np.isfinite(actual_dir) & np.isfinite(pred_dir)
        if not mask.any():
            return np.nan
        
        correct = (actual_dir[mask] == pred_dir[mask]).sum()
        total = mask.sum()
        
        return float(correct / total * 100)
    
    @staticmethod
    def theil_u(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Theil's U statistic
        U < 1: better than naive forecast
        """
        actual = np.asarray(actual, float).ravel()
        predicted = np.asarray(predicted, float).ravel()
        
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not mask.any() or mask.sum() < 2:
            return np.nan
        
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        numerator = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
        
        # Naive forecast (no-change)
        naive = np.roll(actual_clean, 1)
        naive[0] = actual_clean[0]
        denominator = np.sqrt(np.mean((actual_clean - naive) ** 2))
        
        if denominator == 0:
            return np.nan
        
        return float(numerator / denominator)
    
    @staticmethod
    def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
        """R-squared coefficient"""
        actual = np.asarray(actual, float).ravel()
        predicted = np.asarray(predicted, float).ravel()
        
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not mask.any():
            return np.nan
        
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        ss_res = np.sum((actual_clean - predicted_clean) ** 2)
        ss_tot = np.sum((actual_clean - actual_clean.mean()) ** 2)
        
        if ss_tot == 0:
            return np.nan
        
        return float(1 - ss_res / ss_tot)
    
    @staticmethod
    def calculate_all(actual: np.ndarray, predicted: np.ndarray,
                     reference: Optional[np.ndarray] = None) -> Dict:
        """Calculate all metrics"""
        metrics = {
            'rmse': MetricsCalculator.rmse(actual, predicted),
            'mae': MetricsCalculator.mae(actual, predicted),
            'mape': MetricsCalculator.mape(actual, predicted),
            'mse': MetricsCalculator.mse(actual, predicted),
            'bias': MetricsCalculator.bias(actual, predicted),
            'direction_accuracy': MetricsCalculator.direction_accuracy(actual, predicted, reference),
            'theil_u': MetricsCalculator.theil_u(actual, predicted),
            'r_squared': MetricsCalculator.r_squared(actual, predicted)
        }
        
        return metrics


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

class StatisticalTests:
    """
    Comprehensive statistical tests for forecast comparison
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def diebold_mariano_test(self, 
                            errors_1: np.ndarray, 
                            errors_2: np.ndarray,
                            alternative: str = 'greater') -> TestResult:
        """
        Diebold-Mariano test for equal predictive accuracy
        
        H0: MSE(model1) = MSE(model2)
        H1: MSE(model1) < MSE(model2) (if alternative='greater')
        
        Args:
            errors_1: Forecast errors from model 1
            errors_2: Forecast errors from model 2
            alternative: 'two-sided', 'greater' (model1 better), 'less'
        
        Returns:
            TestResult
        """
        e1 = np.asarray(errors_1, float).ravel()
        e2 = np.asarray(errors_2, float).ravel()
        
        # Remove NaNs
        mask = np.isfinite(e1) & np.isfinite(e2)
        e1 = e1[mask]
        e2 = e2[mask]
        
        if len(e1) < 8:
            return TestResult(
                test_name="Diebold-Mariano",
                statistic=np.nan,
                p_value=np.nan,
                critical_value=None,
                is_significant=False,
                direction=alternative,
                null_hypothesis="Equal predictive accuracy",
                interpretation="Insufficient data",
                metadata={'n': len(e1)}
            )
        
        # Loss differential: d_t = e2^2 - e1^2
        d = e2**2 - e1**2
        d_bar = np.mean(d)
        
        # HAC standard error (Newey-West)
        bandwidth = int(np.ceil(len(d) ** (1/3)))
        variance = self._newey_west_variance(d, bandwidth)
        
        if not np.isfinite(variance) or variance <= 0:
            return TestResult(
                test_name="Diebold-Mariano",
                statistic=np.nan,
                p_value=np.nan,
                critical_value=None,
                is_significant=False,
                direction=alternative,
                null_hypothesis="Equal predictive accuracy",
                interpretation="Variance estimation failed",
                metadata={'n': len(e1)}
            )
        
        # DM statistic
        dm_stat = d_bar / np.sqrt(variance / len(d))
        
        # P-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        elif alternative == 'greater':
            # H1: model 1 better (d_bar > 0)
            p_value = 1 - stats.norm.cdf(dm_stat)
        else:  # 'less'
            p_value = stats.norm.cdf(dm_stat)
        
        is_significant = p_value < self.significance_level
        
        interpretation = self._interpret_dm_result(dm_stat, p_value, alternative)
        
        return TestResult(
            test_name="Diebold-Mariano",
            statistic=float(dm_stat),
            p_value=float(p_value),
            critical_value=stats.norm.ppf(1 - self.significance_level),
            is_significant=is_significant,
            direction=alternative,
            null_hypothesis="Equal predictive accuracy",
            interpretation=interpretation,
            metadata={'n': len(e1), 'bandwidth': bandwidth, 'd_bar': float(d_bar)}
        )
    
    def clark_west_test(self,
                       errors_full: np.ndarray,
                       errors_restricted: np.ndarray,
                       forecast_full: np.ndarray,
                       forecast_restricted: np.ndarray) -> TestResult:
        """
        Clark-West test for nested models
        
        Tests if full model (with additional features) is better than restricted
        
        H0: MSE(full) = MSE(restricted)
        H1: MSE(full) < MSE(restricted)
        
        Args:
            errors_full: Errors from full model
            errors_restricted: Errors from restricted (nested) model
            forecast_full: Forecasts from full model
            forecast_restricted: Forecasts from restricted model
        
        Returns:
            TestResult
        """
        e_full = np.asarray(errors_full, float).ravel()
        e_rest = np.asarray(errors_restricted, float).ravel()
        f_full = np.asarray(forecast_full, float).ravel()
        f_rest = np.asarray(forecast_restricted, float).ravel()
        
        mask = (np.isfinite(e_full) & np.isfinite(e_rest) & 
                np.isfinite(f_full) & np.isfinite(f_rest))
        
        e_full = e_full[mask]
        e_rest = e_rest[mask]
        f_full = f_full[mask]
        f_rest = f_rest[mask]
        
        if len(e_full) < 8:
            return TestResult(
                test_name="Clark-West",
                statistic=np.nan,
                p_value=np.nan,
                critical_value=None,
                is_significant=False,
                direction='one-tailed',
                null_hypothesis="Nested models have equal accuracy",
                interpretation="Insufficient data",
                metadata={'n': len(e_full)}
            )
        
        # Adjusted loss differential
        f_diff = f_rest - f_full
        adjusted = e_rest**2 - (e_full**2 - f_diff**2)
        
        adj_bar = np.mean(adjusted)
        
        # HAC standard error
        bandwidth = int(np.ceil(len(adjusted) ** (1/3)))
        variance = self._newey_west_variance(adjusted, bandwidth)
        
        if not np.isfinite(variance) or variance <= 0:
            return TestResult(
                test_name="Clark-West",
                statistic=np.nan,
                p_value=np.nan,
                critical_value=None,
                is_significant=False,
                direction='one-tailed',
                null_hypothesis="Nested models have equal accuracy",
                interpretation="Variance estimation failed",
                metadata={'n': len(e_full)}
            )
        
        # CW statistic
        cw_stat = adj_bar / np.sqrt(variance / len(adjusted))
        
        # One-tailed p-value (H1: full better)
        p_value = 1 - stats.norm.cdf(cw_stat)
        
        is_significant = p_value < self.significance_level
        
        if is_significant:
            interpretation = f"Full model significantly better (p={p_value:.4f})"
        else:
            interpretation = f"No significant improvement (p={p_value:.4f})"
        
        return TestResult(
            test_name="Clark-West",
            statistic=float(cw_stat),
            p_value=float(p_value),
            critical_value=stats.norm.ppf(1 - self.significance_level),
            is_significant=is_significant,
            direction='one-tailed',
            null_hypothesis="Nested models have equal accuracy",
            interpretation=interpretation,
            metadata={'n': len(e_full), 'bandwidth': bandwidth}
        )
    
    def harvey_leybourne_newbold_correction(self,
                                           dm_statistic: float,
                                           n: int) -> Tuple[float, float]:
        """
        Harvey-Leybourne-Newbold small sample correction for DM test
        
        Args:
            dm_statistic: Original DM statistic
            n: Sample size
        
        Returns:
            (corrected_statistic, corrected_p_value)
        """
        if n < 8:
            return np.nan, np.nan
        
        # Correction factor
        correction = np.sqrt((n + 1 - 2 * 1 + 1 * (1 - 1) / n) / n)
        
        corrected_stat = dm_statistic * correction
        corrected_p = 2 * (1 - stats.t.cdf(abs(corrected_stat), df=n-1))
        
        return float(corrected_stat), float(corrected_p)
    
    def giacomini_white_test(self,
                            errors_1: np.ndarray,
                            errors_2: np.ndarray,
                            conditioning_vars: Optional[np.ndarray] = None) -> TestResult:
        """
        Giacomini-White test for conditional predictive ability
        
        Tests if forecast performance differs conditional on information set
        
        Args:
            errors_1: Errors from model 1
            errors_2: Errors from model 2
            conditioning_vars: Conditioning variables (if None, unconditional)
        
        Returns:
            TestResult
        """
        e1 = np.asarray(errors_1, float).ravel()
        e2 = np.asarray(errors_2, float).ravel()
        
        mask = np.isfinite(e1) & np.isfinite(e2)
        e1 = e1[mask]
        e2 = e2[mask]
        
        if len(e1) < 10:
            return TestResult(
                test_name="Giacomini-White",
                statistic=np.nan,
                p_value=np.nan,
                critical_value=None,
                is_significant=False,
                direction='two-sided',
                null_hypothesis="Equal conditional predictive ability",
                interpretation="Insufficient data",
                metadata={'n': len(e1)}
            )
        
        # Loss differential
        d = e2**2 - e1**2
        
        # If no conditioning variables, reduce to DM
        if conditioning_vars is None:
            h = np.ones((len(d), 1))
        else:
            h = np.asarray(conditioning_vars)
            if h.ndim == 1:
                h = h.reshape(-1, 1)
            h = h[mask]
        
        # Regression: d_t = h_t' β + error
        try:
            from scipy.linalg import lstsq
            beta, _, _, _ = lstsq(h, d)
            fitted = h @ beta
            residuals = d - fitted
            
            # Test statistic (Wald)
            n = len(d)
            k = h.shape[1]
            
            # Variance estimate
            S = (h.T @ np.diag(residuals**2) @ h) / n
            
            # Wald statistic
            wald_stat = n * (beta.T @ np.linalg.inv(S) @ beta)
            
            # Chi-squared distribution
            p_value = 1 - stats.chi2.cdf(wald_stat, df=k)
            
            is_significant = p_value < self.significance_level
            
            return TestResult(
                test_name="Giacomini-White",
                statistic=float(wald_stat),
                p_value=float(p_value),
                critical_value=stats.chi2.ppf(1 - self.significance_level, df=k),
                is_significant=is_significant,
                direction='two-sided',
                null_hypothesis="Equal conditional predictive ability",
                interpretation=f"Conditional test: {'Significant' if is_significant else 'Not significant'}",
                metadata={'n': n, 'k': k}
            )
            
        except Exception as e:
            return TestResult(
                test_name="Giacomini-White",
                statistic=np.nan,
                p_value=np.nan,
                critical_value=None,
                is_significant=False,
                direction='two-sided',
                null_hypothesis="Equal conditional predictive ability",
                interpretation=f"Test failed: {str(e)}",
                metadata={'n': len(e1)}
            )
    
    def _newey_west_variance(self, series: np.ndarray, bandwidth: int) -> float:
        """
        Calculate Newey-West HAC variance estimate
        
        Args:
            series: Time series
            bandwidth: Bandwidth (number of lags)
        
        Returns:
            Variance estimate
        """
        series = np.asarray(series, float).ravel()
        series = series[np.isfinite(series)]
        
        if len(series) == 0:
            return np.nan
        
        # De-mean
        u = series - np.mean(series)
        
        # Variance (lag 0)
        gamma_0 = np.mean(u ** 2)
        
        # Autocovariances
        gamma_sum = 0.0
        for lag in range(1, min(bandwidth + 1, len(u))):
            weight = 1.0 - lag / (bandwidth + 1.0)
            gamma_lag = np.mean(u[lag:] * u[:-lag])
            gamma_sum += 2.0 * weight * gamma_lag
        
        variance = gamma_0 + gamma_sum
        
        return float(variance)
    
    def _interpret_dm_result(self, statistic: float, p_value: float, 
                            alternative: str) -> str:
        """Interpret DM test result"""
        if alternative == 'greater':
            if p_value < 0.01:
                return f"Model 1 significantly better (p={p_value:.4f}, ***)"
            elif p_value < 0.05:
                return f"Model 1 significantly better (p={p_value:.4f}, **)"
            elif p_value < 0.10:
                return f"Model 1 marginally better (p={p_value:.4f}, *)"
            else:
                return f"No significant difference (p={p_value:.4f})"
        else:
            if p_value < 0.05:
                return f"Significant difference (p={p_value:.4f})"
            else:
                return f"No significant difference (p={p_value:.4f})"


# ============================================================================
# BOOTSTRAP METHODS
# ============================================================================

class BootstrapMethods:
    """
    Bootstrap confidence intervals and hypothesis tests
    """
    
    def __init__(self, n_iterations: int = 2000, random_seed: int = 42):
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def moving_block_bootstrap(self,
                               errors_1: np.ndarray,
                               errors_2: np.ndarray,
                               block_size: int = 6,
                               statistic_func: Callable = None) -> Dict:
        """
        Moving block bootstrap for dependent data
        
        Args:
            errors_1: Errors from model 1
            errors_2: Errors from model 2
            block_size: Block size for bootstrap
            statistic_func: Function to compute statistic (default: MSE difference)
        
        Returns:
            Bootstrap results with CI
        """
        e1 = np.asarray(errors_1, float).ravel()
        e2 = np.asarray(errors_2, float).ravel()
        
        mask = np.isfinite(e1) & np.isfinite(e2)
        e1 = e1[mask]
        e2 = e2[mask]
        
        n = len(e1)
        
        if n < block_size * 2:
            return {
                'mean': np.nan,
                'std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'bootstrap_dist': []
            }
        
        if statistic_func is None:
            # Default: MSE difference
            statistic_func = lambda e1, e2: np.mean(e2**2) - np.mean(e1**2)
        
        # Observed statistic
        observed = statistic_func(e1, e2)
        
        # Bootstrap
        bootstrap_stats = []
        
        for _ in range(self.n_iterations):
            # Resample blocks
            n_blocks = int(np.ceil(n / block_size))
            starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
            
            boot_e1 = []
            boot_e2 = []
            
            for start in starts:
                boot_e1.extend(e1[start:start + block_size])
                boot_e2.extend(e2[start:start + block_size])
            
            boot_e1 = np.array(boot_e1[:n])
            boot_e2 = np.array(boot_e2[:n])
            
            # Calculate statistic
            boot_stat = statistic_func(boot_e1, boot_e2)
            bootstrap_stats.append(boot_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        bootstrap_stats = bootstrap_stats[np.isfinite(bootstrap_stats)]
        
        if len(bootstrap_stats) < 10:
            return {
                'mean': np.nan,
                'std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'bootstrap_dist': []
            }
        
        # Confidence interval (percentile method)
        ci_lower = np.percentile(bootstrap_stats, 2.5)
        ci_upper = np.percentile(bootstrap_stats, 97.5)
        
        return {
            'observed': float(observed),
            'mean': float(np.mean(bootstrap_stats)),
            'std': float(np.std(bootstrap_stats)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'includes_zero': ci_lower <= 0 <= ci_upper,
            'bootstrap_dist': bootstrap_stats.tolist()
        }
    
    def stationary_bootstrap(self,
                            errors_1: np.ndarray,
                            errors_2: np.ndarray,
                            avg_block_size: float = 6.0) -> Dict:
        """
        Stationary bootstrap (random block length)
        
        Args:
            errors_1: Errors from model 1
            errors_2: Errors from model 2
            avg_block_size: Average block size
        
        Returns:
            Bootstrap results
        """
        e1 = np.asarray(errors_1, float).ravel()
        e2 = np.asarray(errors_2, float).ravel()
        
        mask = np.isfinite(e1) & np.isfinite(e2)
        e1 = e1[mask]
        e2 = e2[mask]
        
        n = len(e1)
        p = 1.0 / avg_block_size  # Geometric distribution parameter
        
        bootstrap_stats = []
        
        for _ in range(self.n_iterations):
            boot_e1 = []
            boot_e2 = []
            
            while len(boot_e1) < n:
                # Random start
                start = np.random.randint(0, n)
                
                # Random block length (geometric)
                block_len = np.random.geometric(p)
                block_len = min(block_len, n - start, n - len(boot_e1))
                
                boot_e1.extend(e1[start:start + block_len])
                boot_e2.extend(e2[start:start + block_len])
            
            boot_e1 = np.array(boot_e1[:n])
            boot_e2 = np.array(boot_e2[:n])
            
            # MSE difference
            stat = np.mean(boot_e2**2) - np.mean(boot_e1**2)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        bootstrap_stats = bootstrap_stats[np.isfinite(bootstrap_stats)]
        
        if len(bootstrap_stats) < 10:
            return {
                'mean': np.nan,
                'std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan
            }
        
        return {
            'mean': float(np.mean(bootstrap_stats)),
            'std': float(np.std(bootstrap_stats)),
            'ci_lower': float(np.percentile(bootstrap_stats, 2.5)),
            'ci_upper': float(np.percentile(bootstrap_stats, 97.5))
        }


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """
    Comprehensive backtesting framework
    Supports rolling-origin, expanding window, walk-forward
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.results = []
    
    def rolling_origin_backtest(self,
                                data: pd.DataFrame,
                                target_col: str,
                                feature_cols: List[str],
                                model_factory: Callable,
                                n_splits: int = 10,
                                test_size: int = None,
                                min_train_size: int = 24) -> List[BacktestResult]:
        """
        Rolling-origin backtesting
        
        Train window moves forward, test size fixed
        
        Args:
            data: Full dataset
            target_col: Target column
            feature_cols: Feature columns
            model_factory: Function that returns a fresh model instance
            n_splits: Number of splits
            test_size: Test set size (if None, auto-calculate)
            min_train_size: Minimum training size
        
        Returns:
            List of BacktestResult
        """
        results = []
        
        # Prepare data
        df = data[[target_col] + feature_cols].dropna().reset_index(drop=True)
        
        if len(df) < min_train_size + 10:
            raise ValueError(f"Insufficient data: {len(df)} rows")
        
        # Calculate test size if not provided
        if test_size is None:
            test_size = max(6, int((len(df) - min_train_size) / n_splits))
        
        # Generate split points
        max_train_end = len(df) - test_size
        train_ends = np.linspace(min_train_size, max_train_end, n_splits, dtype=int)
        
        for split_id, train_end in enumerate(train_ends):
            test_start = train_end
            test_end = min(test_start + test_size, len(df))
            
            # Split data
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            if len(test_df) < 2:
                continue
            
            # Prepare X, y
            X_train = train_df[feature_cols].to_numpy(float)
            y_train = train_df[target_col].to_numpy(float)
            X_test = test_df[feature_cols].to_numpy(float)
            y_test = test_df[target_col].to_numpy(float)
            
            # Train and predict
            try:
                model = model_factory()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Metrics
                metrics = MetricsCalculator.calculate_all(y_test, y_pred)
                
                # Store result
                result = BacktestResult(
                    model_name=model.name if hasattr(model, 'name') else 'Unknown',
                    split_id=split_id,
                    train_start=pd.Timestamp('2000-01-01'),  # Placeholder
                    train_end=pd.Timestamp('2000-01-01'),
                    test_start=pd.Timestamp('2000-01-01'),
                    test_end=pd.Timestamp('2000-01-01'),
                    n_train=len(train_df),
                    n_test=len(test_df),
                    predictions=y_pred,
                    actual=y_test,
                    metrics=metrics,
                    model_params={}
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"Split {split_id} failed: {str(e)}")
                continue
        
        return results
    
    def expanding_window_backtest(self,
                                  data: pd.DataFrame,
                                  target_col: str,
                                  feature_cols: List[str],
                                  model_factory: Callable,
                                  initial_train_size: int = 24,
                                  step_size: int = 1) -> List[BacktestResult]:
        """
        Expanding window backtesting
        
        Train window expands, predict one/multiple steps ahead
        
        Args:
            data: Full dataset
            target_col: Target column
            feature_cols: Feature columns
            model_factory: Function that returns a fresh model instance
            initial_train_size: Initial training size
            step_size: How many obs to predict before retraining
        
        Returns:
            List of BacktestResult
        """
        results = []
        
        df = data[[target_col] + feature_cols].dropna().reset_index(drop=True)
        
        if len(df) < initial_train_size + 10:
            raise ValueError(f"Insufficient data: {len(df)} rows")
        
        train_end = initial_train_size
        split_id = 0
        
        while train_end < len(df) - step_size:
            test_start = train_end
            test_end = min(test_start + step_size, len(df))
            
            # Split
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            # Prepare X, y
            X_train = train_df[feature_cols].to_numpy(float)
            y_train = train_df[target_col].to_numpy(float)
            X_test = test_df[feature_cols].to_numpy(float)
            y_test = test_df[target_col].to_numpy(float)
            
            # Train and predict
            try:
                model = model_factory()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                metrics = MetricsCalculator.calculate_all(y_test, y_pred)
                
                result = BacktestResult(
                    model_name=model.name if hasattr(model, 'name') else 'Unknown',
                    split_id=split_id,
                    train_start=pd.Timestamp('2000-01-01'),
                    train_end=pd.Timestamp('2000-01-01'),
                    test_start=pd.Timestamp('2000-01-01'),
                    test_end=pd.Timestamp('2000-01-01'),
                    n_train=len(train_df),
                    n_test=len(test_df),
                    predictions=y_pred,
                    actual=y_test,
                    metrics=metrics,
                    model_params={}
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"Split {split_id} failed: {str(e)}")
            
            # Move forward
            train_end = test_end
            split_id += 1
        
        return results
    
    def walk_forward_validation(self,
                               data: pd.DataFrame,
                               target_col: str,
                               feature_cols: List[str],
                               model_factory: Callable,
                               train_size: int = 60,
                               test_size: int = 12) -> List[BacktestResult]:
        """
        Walk-forward validation (rolling window)
        
        Fixed train/test size, both windows move forward
        
        Args:
            data: Full dataset
            target_col: Target column
            feature_cols: Feature columns
            model_factory: Function that returns model
            train_size: Training window size
            test_size: Test window size
        
        Returns:
            List of BacktestResult
        """
        results = []
        
        df = data[[target_col] + feature_cols].dropna().reset_index(drop=True)
        
        n = len(df)
        split_id = 0
        
        start = 0
        while start + train_size + test_size <= n:
            train_end = start + train_size
            test_end = train_end + test_size
            
            # Split
            train_df = df.iloc[start:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()
            
            X_train = train_df[feature_cols].to_numpy(float)
            y_train = train_df[target_col].to_numpy(float)
            X_test = test_df[feature_cols].to_numpy(float)
            y_test = test_df[target_col].to_numpy(float)
            
            try:
                model = model_factory()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                metrics = MetricsCalculator.calculate_all(y_test, y_pred)
                
                result = BacktestResult(
                    model_name=model.name if hasattr(model, 'name') else 'Unknown',
                    split_id=split_id,
                    train_start=pd.Timestamp('2000-01-01'),
                    train_end=pd.Timestamp('2000-01-01'),
                    test_start=pd.Timestamp('2000-01-01'),
                    test_end=pd.Timestamp('2000-01-01'),
                    n_train=len(train_df),
                    n_test=len(test_df),
                    predictions=y_pred,
                    actual=y_test,
                    metrics=metrics,
                    model_params={}
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"Split {split_id} failed: {str(e)}")
            
            # Move window forward
            start += test_size
            split_id += 1
        
        return results
    
    def summarize_backtest_results(self, results: List[BacktestResult]) -> Dict:
        """
        Summarize backtest results
        
        Args:
            results: List of BacktestResult
        
        Returns:
            Summary statistics
        """
        if not results:
            return {}
        
        # Extract metrics across splits
        rmse_list = [r.metrics['rmse'] for r in results if np.isfinite(r.metrics['rmse'])]
        mae_list = [r.metrics['mae'] for r in results if np.isfinite(r.metrics['mae'])]
        
        summary = {
            'n_splits': len(results),
            'mean_rmse': float(np.mean(rmse_list)) if rmse_list else np.nan,
            'std_rmse': float(np.std(rmse_list)) if rmse_list else np.nan,
            'min_rmse': float(np.min(rmse_list)) if rmse_list else np.nan,
            'max_rmse': float(np.max(rmse_list)) if rmse_list else np.nan,
            'mean_mae': float(np.mean(mae_list)) if mae_list else np.nan,
            'std_mae': float(np.std(mae_list)) if mae_list else np.nan,
        }
        
        return summary


# ============================================================================
# COMPREHENSIVE EVALUATOR
# ============================================================================

class ComprehensiveEvaluator:
    """
    Main evaluator class combining all functionality
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.metrics_calc = MetricsCalculator()
        self.stat_tests = StatisticalTests(significance_level=config.SIGNIFICANCE_LEVEL if config else 0.05)
        self.bootstrap = BootstrapMethods()
        self.backtest_engine = BacktestEngine(config)
    
    def compare_forecasts(self,
                         actual: np.ndarray,
                         predictions_1: np.ndarray,
                         predictions_2: np.ndarray,
                         model1_name: str = "Model 1",
                         model2_name: str = "Model 2",
                         reference: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive comparison of two forecasts
        
        Returns:
            Complete comparison results
        """
        # Metrics
        metrics_1 = self.metrics_calc.calculate_all(actual, predictions_1, reference)
        metrics_2 = self.metrics_calc.calculate_all(actual, predictions_2, reference)
        
        # Errors
        errors_1 = actual - predictions_1
        errors_2 = actual - predictions_2
        
        # Statistical tests
        dm_result = self.stat_tests.diebold_mariano_test(errors_1, errors_2, alternative='greater')
        
        # Bootstrap CI
        bootstrap_result = self.bootstrap.moving_block_bootstrap(errors_1, errors_2)
        
        comparison = {
            'model1': {
                'name': model1_name,
                'metrics': metrics_1
            },
            'model2': {
                'name': model2_name,
                'metrics': metrics_2
            },
            'tests': {
                'diebold_mariano': {
                    'statistic': dm_result.statistic,
                    'p_value': dm_result.p_value,
                    'is_significant': dm_result.is_significant,
                    'interpretation': dm_result.interpretation
                }
            },
            'bootstrap': bootstrap_result,
            'improvement': {
                'rmse_pct': ((metrics_2['rmse'] - metrics_1['rmse']) / metrics_2['rmse'] * 100) if metrics_2['rmse'] > 0 else np.nan,
                'mae_pct': ((metrics_2['mae'] - metrics_1['mae']) / metrics_2['mae'] * 100) if metrics_2['mae'] > 0 else np.nan
            }
        }
        
        return comparison
