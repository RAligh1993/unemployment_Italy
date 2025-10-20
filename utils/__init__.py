"""
🔧 ISTAT Unemployment Nowcasting Lab - Utility Modules
=====================================================

Professional utility package for Italian unemployment nowcasting.

Modules:
    - state: Centralized application state management
    - istat_handler: ISTAT format data processor
    - excel_processor: Advanced Excel file reader
    - data_detector: Intelligent column/format detection
    - visualizer: Professional chart generation

Author: ISTAT Nowcasting Team
Version: 1.0.0
Python: >=3.8
"""

__version__ = "1.0.0"
__author__ = "ISTAT Nowcasting Team"
__all__ = [
    "AppState",
    "ISTATHandler", 
    "ExcelProcessor",
    "DataDetector",
    "Visualizer"
]

# =============================================================================
# Core Imports with Graceful Fallbacks
# =============================================================================

# State Management
try:
    from .state import AppState
    _HAS_STATE = True
except ImportError as e:
    _HAS_STATE = False
    _STATE_ERROR = str(e)

# ISTAT Handler
try:
    from .istat_handler import ISTATHandler
    _HAS_ISTAT = True
except ImportError as e:
    _HAS_ISTAT = False
    _ISTAT_ERROR = str(e)

# Excel Processor
try:
    from .excel_processor import ExcelProcessor
    _HAS_EXCEL = True
except ImportError as e:
    _HAS_EXCEL = False
    _EXCEL_ERROR = str(e)

# Data Detector
try:
    from .data_detector import DataDetector
    _HAS_DETECTOR = True
except ImportError as e:
    _HAS_DETECTOR = False
    _DETECTOR_ERROR = str(e)

# Visualizer
try:
    from .visualizer import Visualizer
    _HAS_VISUALIZER = True
except ImportError as e:
    _HAS_VISUALIZER = False
    _VISUALIZER_ERROR = str(e)

# =============================================================================
# Health Check Function
# =============================================================================

def check_health() -> dict:
    """
    Check availability of all utility modules.
    
    Returns:
        dict: Status of each module
        
    Example:
        >>> from utils import check_health
        >>> status = check_health()
        >>> print(status)
        {'state': True, 'istat': True, ...}
    """
    return {
        'state': _HAS_STATE,
        'istat_handler': _HAS_ISTAT,
        'excel_processor': _HAS_EXCEL,
        'data_detector': _HAS_DETECTOR,
        'visualizer': _HAS_VISUALIZER
    }

def get_errors() -> dict:
    """
    Get import error messages for failed modules.
    
    Returns:
        dict: Error messages for failed imports
    """
    errors = {}
    
    if not _HAS_STATE:
        errors['state'] = _STATE_ERROR
    if not _HAS_ISTAT:
        errors['istat_handler'] = _ISTAT_ERROR
    if not _HAS_EXCEL:
        errors['excel_processor'] = _EXCEL_ERROR
    if not _HAS_DETECTOR:
        errors['data_detector'] = _DETECTOR_ERROR
    if not _HAS_VISUALIZER:
        errors['visualizer'] = _VISUALIZER_ERROR
    
    return errors

# =============================================================================
# Module Info
# =============================================================================

def info():
    """Print package information"""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  ISTAT Unemployment Nowcasting Lab - Utils Package      ║
╠══════════════════════════════════════════════════════════╣
║  Version: {__version__:<46} ║
║  Author:  {__author__:<46} ║
╠══════════════════════════════════════════════════════════╣
║  Available Modules:                                      ║
║    {'✓' if _HAS_STATE else '✗'} state            - State Management                  ║
║    {'✓' if _HAS_ISTAT else '✗'} istat_handler    - ISTAT Format Processor           ║
║    {'✓' if _HAS_EXCEL else '✗'} excel_processor  - Excel File Reader                ║
║    {'✓' if _HAS_DETECTOR else '✗'} data_detector   - Auto Detection Engine           ║
║    {'✓' if _HAS_VISUALIZER else '✗'} visualizer      - Chart Generator                  ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    errors = get_errors()
    if errors:
        print("\n⚠️  Import Errors:")
        for module, error in errors.items():
            print(f"   • {module}: {error}")

# =============================================================================
# Convenience Functions
# =============================================================================

def get_state():
    """
    Quick access to application state.
    
    Returns:
        AppState instance
        
    Raises:
        ImportError: If state module unavailable
        
    Example:
        >>> from utils import get_state
        >>> state = get_state()
        >>> state.y_monthly = my_series
    """
    if not _HAS_STATE:
        raise ImportError(f"State module unavailable: {_STATE_ERROR}")
    return AppState.get()

def create_istat_handler():
    """
    Create new ISTAT handler instance.
    
    Returns:
        ISTATHandler instance
        
    Raises:
        ImportError: If istat_handler module unavailable
    """
    if not _HAS_ISTAT:
        raise ImportError(f"ISTAT handler unavailable: {_ISTAT_ERROR}")
    return ISTATHandler()

def create_excel_processor():
    """
    Create new Excel processor instance.
    
    Returns:
        ExcelProcessor instance
        
    Raises:
        ImportError: If excel_processor module unavailable
    """
    if not _HAS_EXCEL:
        raise ImportError(f"Excel processor unavailable: {_EXCEL_ERROR}")
    return ExcelProcessor()

def create_detector():
    """
    Create new data detector instance.
    
    Returns:
        DataDetector instance
        
    Raises:
        ImportError: If data_detector module unavailable
    """
    if not _HAS_DETECTOR:
        raise ImportError(f"Data detector unavailable: {_DETECTOR_ERROR}")
    return DataDetector()

def create_visualizer():
    """
    Create new visualizer instance.
    
    Returns:
        Visualizer instance
        
    Raises:
        ImportError: If visualizer module unavailable
    """
    if not _HAS_VISUALIZER:
        raise ImportError(f"Visualizer unavailable: {_VISUALIZER_ERROR}")
    return Visualizer()

# =============================================================================
# Auto-check on import (optional, can be disabled)
# =============================================================================

import os
_AUTO_CHECK = os.getenv('UTILS_AUTO_CHECK', 'false').lower() == 'true'

if _AUTO_CHECK:
    health = check_health()
    failed = [k for k, v in health.items() if not v]
    
    if failed:
        import warnings
        warnings.warn(
            f"Some utils modules failed to load: {', '.join(failed)}. "
            f"Run utils.info() for details.",
            ImportWarning
        )
