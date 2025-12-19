"""
Model Library
Complete implementations of nowcasting models
Real production-ready code - no demos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG


@dataclass
class ModelResult:
    """Container for model results"""
    model_name: str
    predictions: np.ndarray
    actual: np.ndarray
    fitted_model: Any
    scaler: Optional[StandardScaler]
    feature_names: List[str]
    train_score: float
    test_score: float
    hyperparameters: Dict
    metadata: Dict
    
    def get_errors(self) -> np.ndarray:
        """Get prediction errors"""
        return self.actual - self.predictions
    
    def get_metrics(self) -> Dict:
        """Calculate performance metrics"""
        errors = self.get_errors()
        valid_mask = np.isfinite(errors)
        
        if not valid_mask.any():
            return {}
        
        errors_clean = errors[valid_mask]
        actual_clean = self.actual[valid_mask]
        pred_clean = self.predictions[valid_mask]
        
        metrics = {
            'rmse': float(np.sqrt(np.mean(errors_clean ** 2))),
            'mae': float(np.mean(np.abs(errors_clean))),
            'mse': float(np.mean(errors_clean ** 2)),
            'mape': float(np.mean(np.abs(errors_clean / actual_clean)) * 100) if (actual_clean != 0).all() else np.nan,
            'bias': float(np.mean(errors_clean)),
            'std_error': float(np.std(errors_clean))
        }
        
        return metrics


class BaseNowcastModel(ABC):
    """
    Abstract base class for nowcasting models
    All models inherit from this
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.feature_names = []
        self.metadata = {}
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'BaseNowcastModel':
        """Fit model to training data"""
        pass
    
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        pass
    
    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray) -> np.ndarray:
        """Fit and predict in one call"""
        self.fit(X_train, y_train)
        return self.predict(X_test)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (if supported)"""
        return None


class PersistenceModel(BaseNowcastModel):
    """
    Persistence (Random Walk) Model
    Forecast: ŷ_t = y_{t-1}
    
    This is the strongest baseline for many economic series
    """
    
    def __init__(self):
        super().__init__("Persistence")
        self.last_value = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'PersistenceModel':
        """
        For persistence, we just store the last training value
        X_train should contain lag-1 values
        """
        if X_train.shape[1] > 0:
            # Assume first column is lag-1
            self.last_value = X_train[-1, 0]
        else:
            self.last_value = y_train[-1]
        
        self.is_fitted = True
        self.metadata['last_train_value'] = float(self.last_value)
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict using lag-1 values from X_test
        If X_test contains lag-1, use it; otherwise use last_value
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        n_test = X_test.shape[0]
        
        if X_test.shape[1] > 0:
            # Use lag-1 from test data
            predictions = X_test[:, 0].copy()
        else:
            # Use last train value for all predictions
            predictions = np.full(n_test, self.last_value)
        
        return predictions


class HistoricalMeanModel(BaseNowcastModel):
    """
    Historical Mean Model
    Forecast: ŷ_t = mean(y_train)
    """
    
    def __init__(self):
        super().__init__("Historical Mean")
        self.mean_value = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'HistoricalMeanModel':
        """Calculate training mean"""
        self.mean_value = np.mean(y_train[np.isfinite(y_train)])
        self.is_fitted = True
        self.metadata['train_mean'] = float(self.mean_value)
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict mean for all test observations"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        return np.full(X_test.shape[0], self.mean_value)


class AutoRegressiveModel(BaseNowcastModel):
    """
    Autoregressive Model AR(p)
    Uses LinearRegression on lagged values
    """
    
    def __init__(self, order: int = 1):
        super().__init__(f"AR({order})")
        self.order = order
        self.model = LinearRegression()
        self.scaler = StandardScaler()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'AutoRegressiveModel':
        """
        Fit AR model
        X_train should contain first 'order' columns as lags
        """
        if X_train.shape[1] < self.order:
            raise ValueError(f"Need at least {self.order} features for AR({self.order})")
        
        # Use only first 'order' columns (lags)
        X_ar = X_train[:, :self.order]
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X_ar)
        
        # Fit
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        
        self.metadata['coefficients'] = self.model.coef_.tolist()
        self.metadata['intercept'] = float(self.model.intercept_)
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_ar = X_test[:, :self.order]
        X_scaled = self.scaler.transform(X_ar)
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get AR coefficients"""
        if not self.is_fitted:
            return {}
        
        importance = {}
        for i, coef in enumerate(self.model.coef_):
            importance[f"lag_{i+1}"] = float(coef)
        
        return importance


class RidgeModel(BaseNowcastModel):
    """
    Ridge Regression with L2 regularization
    Handles multicollinearity well
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__(f"Ridge(α={alpha})")
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=CONFIG.RANDOM_SEED)
        self.scaler = StandardScaler()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'RidgeModel':
        """Fit Ridge model"""
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Fit
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        
        self.metadata['alpha'] = self.alpha
        self.metadata['n_features'] = X_train.shape[1]
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get Ridge coefficients (scaled)"""
        if not self.is_fitted:
            return {}
        
        importance = {}
        for i, coef in enumerate(self.model.coef_):
            importance[f"feature_{i}"] = float(coef)
        
        return importance


class LassoModel(BaseNowcastModel):
    """
    Lasso Regression with L1 regularization
    Performs feature selection (some coefficients → 0)
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__(f"Lasso(α={alpha})")
        self.alpha = alpha
        self.model = Lasso(alpha=alpha, random_state=CONFIG.RANDOM_SEED, max_iter=10000)
        self.scaler = StandardScaler()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'LassoModel':
        """Fit Lasso model"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        
        # Count non-zero coefficients
        n_nonzero = np.sum(self.model.coef_ != 0)
        
        self.metadata['alpha'] = self.alpha
        self.metadata['n_features'] = X_train.shape[1]
        self.metadata['n_selected'] = int(n_nonzero)
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get Lasso coefficients (many will be zero)"""
        if not self.is_fitted:
            return {}
        
        importance = {}
        for i, coef in enumerate(self.model.coef_):
            if coef != 0:  # Only non-zero
                importance[f"feature_{i}"] = float(coef)
        
        return importance


class ElasticNetModel(BaseNowcastModel):
    """
    Elastic Net: L1 + L2 regularization
    Compromise between Ridge and Lasso
    """
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5):
        super().__init__(f"ElasticNet(α={alpha},l1={l1_ratio})")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, 
                               random_state=CONFIG.RANDOM_SEED, max_iter=10000)
        self.scaler = StandardScaler()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'ElasticNetModel':
        """Fit ElasticNet model"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        
        self.metadata['alpha'] = self.alpha
        self.metadata['l1_ratio'] = self.l1_ratio
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)


class DeltaCorrectionModel(BaseNowcastModel):
    """
    Delta-Correction Model
    
    Strategy:
    1. Predict Δ = y_t - y_{t-1} using Ridge/Lasso
    2. Forecast: ŷ_t = y_{t-1} + w * Δ̂_t
    
    Where w ∈ [0, 1] is blending weight
    """
    
    def __init__(self, base_model: BaseNowcastModel, blend_weight: float = 1.0):
        super().__init__(f"DeltaCorrection({base_model.name})")
        self.base_model = base_model
        self.blend_weight = blend_weight
        self.lag_column_idx = 0  # Assume first column is lag-1
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'DeltaCorrectionModel':
        """
        Fit on delta (changes) instead of levels
        X_train should contain y_{t-1} as first column
        """
        if X_train.shape[1] == 0:
            raise ValueError("X_train must contain at least lag-1")
        
        # Extract lag-1
        y_lag1 = X_train[:, self.lag_column_idx]
        
        # Compute delta target
        delta_target = y_train - y_lag1
        
        # Fit base model on delta
        self.base_model.fit(X_train, delta_target)
        self.is_fitted = True
        
        self.metadata['blend_weight'] = self.blend_weight
        self.metadata['base_model'] = self.base_model.name
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict using delta correction
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Extract lag-1 from test
        y_lag1_test = X_test[:, self.lag_column_idx]
        
        # Predict delta
        delta_pred = self.base_model.predict(X_test)
        
        # Apply correction: ŷ = lag1 + w * Δ̂
        y_pred = y_lag1_test + self.blend_weight * delta_pred
        
        return y_pred


class MIDASModel(BaseNowcastModel):
    """
    MIDAS Regression Model
    
    Assumes features are already MIDAS-aggregated
    (by FrequencyAligner)
    
    Uses Ridge regression on MIDAS features
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__(f"MIDAS(α={alpha})")
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=CONFIG.RANDOM_SEED)
        self.scaler = StandardScaler()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'MIDASModel':
        """Fit MIDAS model"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        
        self.metadata['alpha'] = self.alpha
        self.metadata['note'] = "Features assumed pre-aggregated via MIDAS"
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)


class EnsembleModel(BaseNowcastModel):
    """
    Ensemble of multiple models
    Simple averaging or weighted
    """
    
    def __init__(self, models: List[BaseNowcastModel], weights: Optional[List[float]] = None):
        super().__init__("Ensemble")
        self.models = models
        
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'EnsembleModel':
        """Fit all models"""
        for model in self.models:
            model.fit(X_train, y_train)
        
        self.is_fitted = True
        self.metadata['n_models'] = len(self.models)
        self.metadata['model_names'] = [m.name for m in self.models]
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Weighted average of predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        predictions = np.zeros((X_test.shape[0], len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_test)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=1, weights=self.weights)
        
        return ensemble_pred


class ModelLibrary:
    """
    Factory for creating and managing models
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.available_models = {
            'persistence': PersistenceModel,
            'mean': HistoricalMeanModel,
            'ar1': lambda: AutoRegressiveModel(order=1),
            'ar2': lambda: AutoRegressiveModel(order=2),
            'ridge': RidgeModel,
            'lasso': LassoModel,
            'elastic': ElasticNetModel,
            'midas': MIDASModel
        }
    
    def create_model(self, model_type: str, **kwargs) -> BaseNowcastModel:
        """
        Create a model instance
        
        Args:
            model_type: Type of model ('ridge', 'lasso', etc.)
            **kwargs: Model-specific parameters
        
        Returns:
            Model instance
        """
        if model_type not in self.available_models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.available_models[model_type]
        
        if callable(model_class):
            if kwargs:
                return model_class(**kwargs)
            else:
                return model_class()
        else:
            return model_class
    
    def create_benchmark_suite(self) -> List[BaseNowcastModel]:
        """
        Create standard benchmark models
        Always included in comparison
        """
        return [
            PersistenceModel(),
            HistoricalMeanModel(),
            AutoRegressiveModel(order=1),
            AutoRegressiveModel(order=2)
        ]
    
    def create_ridge_grid(self, alphas: Optional[List[float]] = None) -> List[RidgeModel]:
        """Create Ridge models with different alphas"""
        if alphas is None:
            alphas = self.config.RIDGE_ALPHAS
        
        return [RidgeModel(alpha=alpha) for alpha in alphas]
    
    def create_lasso_grid(self, alphas: Optional[List[float]] = None) -> List[LassoModel]:
        """Create Lasso models with different alphas"""
        if alphas is None:
            alphas = self.config.LASSO_ALPHAS
        
        return [LassoModel(alpha=alpha) for alpha in alphas]
    
    def create_delta_correction_models(self, 
                                       base_models: List[BaseNowcastModel],
                                       blend_weights: Optional[List[float]] = None) -> List[DeltaCorrectionModel]:
        """
        Create delta-correction variants
        """
        if blend_weights is None:
            blend_weights = [0.7, 0.8, 0.9, 1.0]
        
        delta_models = []
        for base_model in base_models:
            for w in blend_weights:
                delta_models.append(DeltaCorrectionModel(base_model, blend_weight=w))
        
        return delta_models
    
    def get_model_info(self, model_type: str) -> Dict:
        """Get information about a model type"""
        info = {
            'persistence': {
                'name': 'Persistence (Random Walk)',
                'description': 'Forecast = previous value',
                'complexity': 'very_low',
                'interpretable': True
            },
            'mean': {
                'name': 'Historical Mean',
                'description': 'Forecast = training average',
                'complexity': 'very_low',
                'interpretable': True
            },
            'ar1': {
                'name': 'AR(1)',
                'description': 'First-order autoregressive',
                'complexity': 'low',
                'interpretable': True
            },
            'ar2': {
                'name': 'AR(2)',
                'description': 'Second-order autoregressive',
                'complexity': 'low',
                'interpretable': True
            },
            'ridge': {
                'name': 'Ridge Regression',
                'description': 'L2 regularization, handles multicollinearity',
                'complexity': 'medium',
                'interpretable': True
            },
            'lasso': {
                'name': 'Lasso Regression',
                'description': 'L1 regularization, feature selection',
                'complexity': 'medium',
                'interpretable': True
            },
            'elastic': {
                'name': 'Elastic Net',
                'description': 'L1 + L2 regularization',
                'complexity': 'medium',
                'interpretable': True
            },
            'midas': {
                'name': 'MIDAS Regression',
                'description': 'Mixed-frequency data sampling',
                'complexity': 'medium',
                'interpretable': True
            }
        }
        
        return info.get(model_type, {'name': model_type, 'description': 'Unknown model'})


class ModelSelector:
    """
    Automatic model selection via cross-validation
    """
    
    def __init__(self, library: ModelLibrary, cv_splits: int = 3):
        self.library = library
        self.cv_splits = cv_splits
        self.results = []
    
    def select_best_model(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         models: List[BaseNowcastModel],
                         metric: str = 'rmse') -> Tuple[BaseNowcastModel, Dict]:
        """
        Select best model using time series cross-validation
        
        Args:
            X: Features
            y: Target
            models: List of models to compare
            metric: Metric to optimize ('rmse', 'mae')
        
        Returns:
            (best_model, results)
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        scores = {model.name: [] for model in models}
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            for model in models:
                # Create fresh instance to avoid state issues
                model_instance = type(model)() if hasattr(model, '__class__') else model
                
                try:
                    model_instance.fit(X_train, y_train)
                    y_pred = model_instance.predict(X_val)
                    
                    if metric == 'rmse':
                        score = np.sqrt(mean_squared_error(y_val, y_pred))
                    elif metric == 'mae':
                        score = mean_absolute_error(y_val, y_pred)
                    else:
                        score = mean_squared_error(y_val, y_pred)
                    
                    scores[model.name].append(score)
                    
                except Exception as e:
                    # Model failed - assign high penalty
                    scores[model.name].append(1e10)
        
        # Average scores
        avg_scores = {name: np.mean(s) for name, s in scores.items()}
        
        # Find best
        best_name = min(avg_scores, key=avg_scores.get)
        best_model = next(m for m in models if m.name == best_name)
        
        results = {
            'scores': avg_scores,
            'best_model': best_name,
            'best_score': avg_scores[best_name]
        }
        
        return best_model, results
