"""
Nowcasting Platform - Core Module
Production-ready forecasting framework
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .data_intelligence import DataIntelligence, DatasetInfo
from .frequency_aligner import FrequencyAligner, FrequencyConverter
from .feature_factory import FeatureFactory, FeatureValidator
from .model_library import (
    ModelLibrary,
    BaseNowcastModel,
    PersistenceModel,
    HistoricalMeanModel,
    AutoRegressiveModel,
    RidgeModel,
    LassoModel,
    ElasticNetModel,
    DeltaCorrectionModel,
    MIDASModel,
    EnsembleModel,
    ModelResult
)
from .evaluator import (
    MetricsCalculator,
    StatisticalTests,
    BootstrapMethods,
    BacktestEngine,
    ComprehensiveEvaluator,
    TestResult,
    BacktestResult
)
from .exporter import (
    DataExporter,
    FigureExporter,
    ReportGenerator,
    PackageExporter,
    StreamlitDownloader
)

# Define what's available for "from core import *"
__all__ = [
    # Data Intelligence
    'DataIntelligence',
    'DatasetInfo',
    
    # Frequency Alignment
    'FrequencyAligner',
    'FrequencyConverter',
    
    # Feature Engineering
    'FeatureFactory',
    'FeatureValidator',
    
    # Models
    'ModelLibrary',
    'BaseNowcastModel',
    'PersistenceModel',
    'HistoricalMeanModel',
    'AutoRegressiveModel',
    'RidgeModel',
    'LassoModel',
    'ElasticNetModel',
    'DeltaCorrectionModel',
    'MIDASModel',
    'EnsembleModel',
    'ModelResult',
    
    # Evaluation
    'MetricsCalculator',
    'StatisticalTests',
    'BootstrapMethods',
    'BacktestEngine',
    'ComprehensiveEvaluator',
    'TestResult',
    'BacktestResult',
    
    # Export
    'DataExporter',
    'FigureExporter',
    'ReportGenerator',
    'PackageExporter',
    'StreamlitDownloader',
]

# Version info
def get_version():
    """Get package version"""
    return __version__

def print_info():
    """Print package information"""
    info = f"""
    ╔═══════════════════════════════════════════════════════╗
    ║       NOWCASTING PLATFORM - CORE MODULE               ║
    ╠═══════════════════════════════════════════════════════╣
    ║  Version:  {__version__:<40}      ║
    ║  Author:   {__author__:<40}      ║
    ║  Email:    {__email__:<40}      ║
    ╠═══════════════════════════════════════════════════════╣
    ║  Modules:                                             ║
    ║    ✓ Data Intelligence (auto-detection)              ║
    ║    ✓ Frequency Aligner (MIDAS)                       ║
    ║    ✓ Feature Factory (engineering)                   ║
    ║    ✓ Model Library (8+ models)                       ║
    ║    ✓ Evaluator (tests + backtesting)                 ║
    ║    ✓ Exporter (CSV, JSON, ZIP)                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(info)

# Module-level initialization
def _initialize():
    """Initialize core module"""
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

_initialize()
