"""
Configuration for Nowcasting Platform
Production-ready settings
"""

from dataclasses import dataclass
from typing import List, Tuple
import os

@dataclass
class AppConfig:
    """Main application configuration"""
    
    # App info
    APP_NAME: str = "Nowcasting Platform"
    VERSION: str = "1.0.0"
    AUTHOR: str = "ISTAT"
    
    # Paths
    OUTPUT_DIR: str = "outputs"
    FIGURES_DIR: str = "figures"
    TEMP_DIR: str = "temp"
    
    # Data validation
    MIN_OBSERVATIONS: int = 24
    MAX_MISSING_PCT: float = 0.3
    OUTLIER_THRESHOLD: float = 3.0
    
    # Frequency detection
    FREQ_THRESHOLDS = {
        'daily': (0.8, 1.5),      # days
        'weekly': (6, 8),         # days
        'monthly': (28, 32),      # days
        'quarterly': (89, 93),    # days
        'annual': (360, 370)      # days
    }
    
    # Feature engineering
    LAGS_MONTHLY: List[int] = [1, 2, 3, 6, 12]
    LAGS_WEEKLY: List[int] = [1, 2, 4, 8, 52]
    LAGS_DAILY: List[int] = [1, 7, 30]
    
    # MIDAS
    MIDAS_WINDOWS: List[int] = [4, 8, 12]
    MIDAS_LAMBDAS: List[float] = [0.6, 0.8]
    MIDAS_CUTOFF_DAY: int = 15
    
    # Models
    RIDGE_ALPHAS: List[float] = [1, 10, 50, 100]
    LASSO_ALPHAS: List[float] = [0.01, 0.1, 1.0]
    
    # Evaluation
    TRAIN_SPLIT: float = 0.7
    CV_SPLITS: int = 3
    RANDOM_SEED: int = 42
    
    # Statistical tests
    SIGNIFICANCE_LEVEL: float = 0.05
    BOOTSTRAP_ITERATIONS: int = 2000
    
    # UI
    PRIMARY_COLOR: str = "#003366"     # ISTAT Blue
    SECONDARY_COLOR: str = "#FF6B35"   # Accent Orange
    SUCCESS_COLOR: str = "#28A745"     # Green
    WARNING_COLOR: str = "#FFC107"     # Yellow
    DANGER_COLOR: str = "#DC3545"      # Red
    
    def __post_init__(self):
        """Create directories"""
        for dir_path in [self.OUTPUT_DIR, self.FIGURES_DIR, self.TEMP_DIR]:
            os.makedirs(dir_path, exist_ok=True)


# Global config instance
CONFIG = AppConfig()
