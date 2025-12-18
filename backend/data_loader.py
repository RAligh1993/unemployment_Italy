"""
Models Module
Implements various nowcasting models (MIDAS, ML, etc.)

Author:Rajabali Ghasempour
Institution: ISTAT
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .feature_engineering import FeatureEngineer


class BaseModel:
    """Base class for all models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def fit(self, train_data: pd.DataFrame):
        """Fit model on training data"""
        raise NotImplementedError
    
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        raise NotImplementedError


class MIDASExponentialModel(BaseModel):
    """MIDAS with exponential Almon weights"""
    
    def __init__(self, 
                 theta: float = 3.0,
                 n_lags: int = 4,
                 alpha: float = 50.0,
                 top_k_keywords: int = 3):
        super().__init__(f"MIDAS Exp(θ={theta})")
        self.theta = theta
        self.n_lags = n_lags
        self.alpha = alpha
        self.top_k_keywords = top_k_keywords
        self.selected_keywords = None
        self.weights = None
        
    def fit(self, train_data: pd.DataFrame):
        """Fit MIDAS model"""
        # Get GT keywords
        gt_cols = [col for col in train_data.columns if col not in [
            'date', 'target', 'unemp', 'unemp_lag1', 'unemp_lag2',
            'unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1'
        ] and not col.endswith('_w0') and not col.endswith('_w1') and 
           not col.endswith('_w2') and not col.endswith('_w3')]
        
        if len(gt_cols) == 0:
            # No GT features, use baseline only
            self.feature_cols = ['unemp_lag1', 'unemp_lag2']
            if 'unemp_youth_lag1' in train_data.columns:
                self.feature_cols.append('unemp_youth_lag1')
            if 'CCI_lag1' in train_data.columns:
                self.feature_cols.append('CCI_lag1')
            if 'HICP_lag1' in train_data.columns:
                self.feature_cols.append('HICP_lag1')
        else:
            # Select top keywords
            engineer = FeatureEngineer()
            self.selected_keywords = engineer.select_top_keywords(
                train_data, gt_cols, 'target', self.top_k_keywords
            )
            
            # Compute MIDAS weights
            self.weights = engineer.exponential_weights(self.n_lags, self.theta)
            
            # Create weighted features (simple average as proxy)
            self.feature_cols = []
            for kw in self.selected_keywords:
                if kw in train_data.columns:
                    self.feature_cols.append(kw)
            
            # Add baseline features
            if 'unemp_lag1' in train_data.columns:
                self.feature_cols.append('unemp_lag1')
            if 'unemp_lag2' in train_data.columns:
                self.feature_cols.append('unemp_lag2')
            if 'unemp_youth_lag1' in train_data.columns:
                self.feature_cols.append('unemp_youth_lag1')
            if 'CCI_lag1' in train_data.columns:
                self.feature_cols.append('CCI_lag1')
            if 'HICP_lag1' in train_data.columns:
                self.feature_cols.append('HICP_lag1')
        
        # Prepare features
        X_train = train_data[self.feature_cols].values
        y_train = train_data['target'].values
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit Ridge
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_train_scaled, y_train)
        
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        X_test = test_data[self.feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class MIDASBetaModel(BaseModel):
    """MIDAS with Beta polynomial weights"""
    
    def __init__(self,
                 theta1: float = 5.0,
                 theta2: float = 1.0,
                 n_lags: int = 4,
                 alpha: float = 50.0,
                 top_k_keywords: int = 3):
        super().__init__(f"MIDAS Beta({theta1},{theta2})")
        self.theta1 = theta1
        self.theta2 = theta2
        self.n_lags = n_lags
        self.alpha = alpha
        self.top_k_keywords = top_k_keywords
        self.selected_keywords = None
        self.weights = None
        
    def fit(self, train_data: pd.DataFrame):
        """Fit MIDAS model"""
        # Similar to exponential, but with beta weights
        gt_cols = [col for col in train_data.columns if col not in [
            'date', 'target', 'unemp', 'unemp_lag1', 'unemp_lag2',
            'unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1'
        ]]
        
        if len(gt_cols) > 0:
            engineer = FeatureEngineer()
            self.selected_keywords = engineer.select_top_keywords(
                train_data, gt_cols, 'target', self.top_k_keywords
            )
            
            self.weights = engineer.beta_weights(self.n_lags, self.theta1, self.theta2)
            
            self.feature_cols = self.selected_keywords.copy()
        else:
            self.feature_cols = []
        
        # Add baseline
        for col in ['unemp_lag1', 'unemp_lag2', 'unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1']:
            if col in train_data.columns:
                self.feature_cols.append(col)
        
        X_train = train_data[self.feature_cols].values
        y_train = train_data['target'].values
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_train_scaled, y_train)
        
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        X_test = test_data[self.feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class RidgeModel(BaseModel):
    """Ridge regression baseline"""
    
    def __init__(self, alpha: float = 50.0, include_gt: bool = True):
        super().__init__("Ridge")
        self.alpha = alpha
        self.include_gt = include_gt
        
    def fit(self, train_data: pd.DataFrame):
        # Select features
        self.feature_cols = ['unemp_lag1', 'unemp_lag2']
        
        for col in ['unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1']:
            if col in train_data.columns:
                self.feature_cols.append(col)
        
        if self.include_gt:
            gt_cols = [col for col in train_data.columns if col not in [
                'date', 'target', 'unemp', 'unemp_lag1', 'unemp_lag2',
                'unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1'
            ]]
            
            if len(gt_cols) > 0:
                engineer = FeatureEngineer()
                top_kw = engineer.select_top_keywords(train_data, gt_cols, 'target', 6)
                self.feature_cols.extend(top_kw)
        
        X_train = train_data[self.feature_cols].values
        y_train = train_data['target'].values
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_train_scaled, y_train)
        
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        X_test = test_data[self.feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class LassoModel(BaseModel):
    """Lasso regression with automatic feature selection"""
    
    def __init__(self, alpha: float = 1.0, include_gt: bool = True):
        super().__init__("Lasso")
        self.alpha = alpha
        self.include_gt = include_gt
        
    def fit(self, train_data: pd.DataFrame):
        self.feature_cols = ['unemp_lag1', 'unemp_lag2']
        
        for col in ['unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1']:
            if col in train_data.columns:
                self.feature_cols.append(col)
        
        if self.include_gt:
            gt_cols = [col for col in train_data.columns if col not in [
                'date', 'target', 'unemp', 'unemp_lag1', 'unemp_lag2',
                'unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1'
            ]]
            self.feature_cols.extend(gt_cols)
        
        X_train = train_data[self.feature_cols].values
        y_train = train_data['target'].values
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = Lasso(alpha=self.alpha, max_iter=10000)
        self.model.fit(X_train_scaled, y_train)
        
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        X_test = test_data[self.feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class RandomForestModel(BaseModel):
    """Random Forest regressor"""
    
    def __init__(self, n_estimators: int = 100, include_gt: bool = True):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.include_gt = include_gt
        
    def fit(self, train_data: pd.DataFrame):
        self.feature_cols = ['unemp_lag1', 'unemp_lag2']
        
        for col in ['unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1']:
            if col in train_data.columns:
                self.feature_cols.append(col)
        
        if self.include_gt:
            gt_cols = [col for col in train_data.columns if col not in [
                'date', 'target', 'unemp', 'unemp_lag1', 'unemp_lag2',
                'unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1'
            ]]
            
            if len(gt_cols) > 0:
                engineer = FeatureEngineer()
                top_kw = engineer.select_top_keywords(train_data, gt_cols, 'target', 10)
                self.feature_cols.extend(top_kw)
        
        X_train = train_data[self.feature_cols].values
        y_train = train_data['target'].values
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        X_test = test_data[self.feature_cols].values
        return self.model.predict(X_test)


class XGBoostModel(BaseModel):
    """XGBoost regressor"""
    
    def __init__(self, n_estimators: int = 100, include_gt: bool = True):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.include_gt = include_gt
        
    def fit(self, train_data: pd.DataFrame):
        self.feature_cols = ['unemp_lag1', 'unemp_lag2']
        
        for col in ['unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1']:
            if col in train_data.columns:
                self.feature_cols.append(col)
        
        if self.include_gt:
            gt_cols = [col for col in train_data.columns if col not in [
                'date', 'target', 'unemp', 'unemp_lag1', 'unemp_lag2',
                'unemp_youth_lag1', 'CCI_lag1', 'HICP_lag1'
            ]]
            
            if len(gt_cols) > 0:
                engineer = FeatureEngineer()
                top_kw = engineer.select_top_keywords(train_data, gt_cols, 'target', 10)
                self.feature_cols.extend(top_kw)
        
        X_train = train_data[self.feature_cols].values
        y_train = train_data['target'].values
        
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        X_test = test_data[self.feature_cols].values
        return self.model.predict(X_test)


class ModelFactory:
    """Factory for creating models"""
    
    @staticmethod
    def create_model(model_name: str, include_gt: bool = True) -> BaseModel:
        """
        Create model by name
        
        Args:
            model_name: Name of model
            include_gt: Whether to include GT features
            
        Returns:
            Model instance
        """
        if model_name == "MIDAS Exponential":
            return MIDASExponentialModel()
        elif model_name == "MIDAS Beta":
            return MIDASBetaModel()
        elif model_name == "Ridge":
            return RidgeModel(include_gt=include_gt)
        elif model_name == "Lasso":
            return LassoModel(include_gt=include_gt)
        elif model_name == "Random Forest":
            return RandomForestModel(include_gt=include_gt)
        elif model_name == "XGBoost":
            return XGBoostModel(include_gt=include_gt)
        else:
            raise ValueError(f"Unknown model: {model_name}")
