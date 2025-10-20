"""
📊 State Management Module
===========================

Centralized state management for Streamlit multi-page applications.
Provides persistent storage across page navigation.

Features:
    - Type-safe state variables
    - Automatic initialization
    - Clear/reset functionality
    - State inspection utilities
    - Thread-safe access

Author: ISTAT Nowcasting Team
Version: 1.0.0
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import streamlit as st


# =============================================================================
# State Data Class
# =============================================================================

@dataclass
class _AppState:
    """
    Internal state container with type annotations.
    
    Attributes:
        Target Data:
            y_monthly: Primary target variable (monthly unemployment rate)
            targets_monthly: Multiple target variables (DataFrame)
        
        Feature Panels:
            panel_monthly: Monthly aggregated features
            panel_quarterly: Quarterly aggregated features
            panel_daily: Daily features (not aggregated)
        
        Raw Data Sources:
            raw_daily: List of daily DataFrames from upload
            raw_monthly: List of monthly DataFrames
            raw_quarterly: List of quarterly DataFrames
            google_trends: Google Trends data
        
        Model Results:
            bt_results: Backtesting predictions {model_name: Series}
            bt_metrics: Model performance metrics (DataFrame)
            forecasts: Out-of-sample forecasts
        
        Feature Engineering:
            feature_recipe: List of transformations applied
            feature_importance: Feature importance scores
        
        Metadata:
            data_metadata: General metadata dictionary
            upload_history: History of uploaded files
            last_update: Timestamp of last update
    """
    
    # Target variables
    y_monthly: Optional[pd.Series] = None
    targets_monthly: Optional[pd.DataFrame] = None
    
    # Feature panels
    panel_monthly: Optional[pd.DataFrame] = None
    panel_quarterly: Optional[pd.DataFrame] = None
    panel_daily: Optional[pd.DataFrame] = None
    
    # Raw data sources
    raw_daily: List[pd.DataFrame] = field(default_factory=list)
    raw_monthly: List[pd.DataFrame] = field(default_factory=list)
    raw_quarterly: List[pd.DataFrame] = field(default_factory=list)
    google_trends: Optional[pd.DataFrame] = None
    
    # Model results
    bt_results: Dict[str, pd.Series] = field(default_factory=dict)
    bt_metrics: Optional[pd.DataFrame] = None
    forecasts: Dict[str, pd.Series] = field(default_factory=dict)
    
    # Feature engineering
    feature_recipe: List[Dict[str, Any]] = field(default_factory=list)
    feature_importance: Optional[pd.DataFrame] = None
    
    # Metadata
    data_metadata: Dict[str, Any] = field(default_factory=dict)
    upload_history: List[Dict[str, Any]] = field(default_factory=list)
    last_update: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize timestamps"""
        if self.last_update is None:
            self.last_update = datetime.now()


# =============================================================================
# State Manager Class
# =============================================================================

class AppState:
    """
    Application state manager with Streamlit integration.
    
    This class provides a singleton pattern for accessing application state
    across different pages in a Streamlit multi-page app.
    
    Usage:
        Basic:
            >>> from utils.state import AppState
            >>> state = AppState.get()
            >>> state.y_monthly = pd.Series([1, 2, 3])
        
        With initialization:
            >>> state = AppState.init()
            >>> print(state.y_monthly)
        
        Clear state:
            >>> AppState.clear()
            >>> state = AppState.get()  # Fresh state
    
    Thread Safety:
        Streamlit handles session state in a thread-safe manner.
        Each user session has its own isolated state.
    """
    
    _STATE_KEY = "_app_state_v1"
    
    @classmethod
    def init(cls) -> _AppState:
        """
        Initialize state in Streamlit session.
        
        Creates new state if doesn't exist, otherwise returns existing.
        
        Returns:
            _AppState: State instance
            
        Example:
            >>> state = AppState.init()
            >>> state.y_monthly = my_data
        """
        if cls._STATE_KEY not in st.session_state:
            st.session_state[cls._STATE_KEY] = _AppState()
        
        return st.session_state[cls._STATE_KEY]
    
    @classmethod
    def get(cls) -> _AppState:
        """
        Get current state (initializes if needed).
        
        Returns:
            _AppState: State instance
            
        Example:
            >>> state = AppState.get()
            >>> if state.y_monthly is not None:
            ...     print(f"Target has {len(state.y_monthly)} observations")
        """
        return cls.init()
    
    @classmethod
    def clear(cls) -> _AppState:
        """
        Clear all state and return fresh instance.
        
        Returns:
            _AppState: New empty state
            
        Warning:
            This will delete ALL data in the state!
            
        Example:
            >>> AppState.clear()  # Everything deleted
            >>> state = AppState.get()  # Fresh state
        """
        if cls._STATE_KEY in st.session_state:
            del st.session_state[cls._STATE_KEY]
        
        return cls.init()
    
    @classmethod
    def exists(cls) -> bool:
        """
        Check if state has been initialized.
        
        Returns:
            bool: True if state exists
            
        Example:
            >>> if AppState.exists():
            ...     state = AppState.get()
        """
        return cls._STATE_KEY in st.session_state
    
    @classmethod
    def update_timestamp(cls):
        """
        Update last_update timestamp to now.
        
        Example:
            >>> state = AppState.get()
            >>> state.y_monthly = new_data
            >>> AppState.update_timestamp()
        """
        state = cls.get()
        state.last_update = datetime.now()
    
    @classmethod
    def summary(cls) -> Dict[str, Any]:
        """
        Get summary of current state contents.
        
        Returns:
            dict: Summary statistics
            
        Example:
            >>> summary = AppState.summary()
            >>> print(f"Target loaded: {summary['has_target']}")
            >>> print(f"Models trained: {summary['num_models']}")
        """
        state = cls.get()
        
        return {
            # Target info
            'has_target': state.y_monthly is not None and not state.y_monthly.empty,
            'target_length': len(state.y_monthly) if state.y_monthly is not None else 0,
            
            # Panel info
            'has_panel_monthly': state.panel_monthly is not None and not state.panel_monthly.empty,
            'panel_monthly_shape': state.panel_monthly.shape if state.panel_monthly is not None else (0, 0),
            
            'has_panel_quarterly': state.panel_quarterly is not None and not state.panel_quarterly.empty,
            'panel_quarterly_shape': state.panel_quarterly.shape if state.panel_quarterly is not None else (0, 0),
            
            # Raw data
            'num_daily_files': len(state.raw_daily),
            'num_monthly_files': len(state.raw_monthly),
            'has_google_trends': state.google_trends is not None and not state.google_trends.empty,
            
            # Models
            'num_models': len(state.bt_results),
            'model_names': list(state.bt_results.keys()),
            'has_metrics': state.bt_metrics is not None and not state.bt_metrics.empty,
            
            # Feature engineering
            'num_recipe_steps': len(state.feature_recipe),
            'has_feature_importance': state.feature_importance is not None,
            
            # Metadata
            'num_uploads': len(state.upload_history),
            'last_update': state.last_update.isoformat() if state.last_update else None
        }
    
    @classmethod
    def validate(cls) -> List[str]:
        """
        Validate state and return list of issues.
        
        Returns:
            list: List of validation warnings/errors
            
        Example:
            >>> issues = AppState.validate()
            >>> if issues:
            ...     for issue in issues:
            ...         st.warning(issue)
        """
        state = cls.get()
        issues = []
        
        # Check target
        if state.y_monthly is None or state.y_monthly.empty:
            issues.append("⚠️ No target variable loaded")
        elif len(state.y_monthly) < 24:
            issues.append(f"⚠️ Target has only {len(state.y_monthly)} observations (recommended: ≥24)")
        
        # Check panel
        if state.panel_monthly is None or state.panel_monthly.empty:
            issues.append("⚠️ No monthly panel built")
        
        # Check alignment
        if (state.y_monthly is not None and not state.y_monthly.empty and
            state.panel_monthly is not None and not state.panel_monthly.empty):
            
            y_dates = set(state.y_monthly.index)
            x_dates = set(state.panel_monthly.index)
            overlap = y_dates.intersection(x_dates)
            
            if len(overlap) == 0:
                issues.append("❌ No date overlap between target and panel!")
            elif len(overlap) < 12:
                issues.append(f"⚠️ Only {len(overlap)} dates overlap between target and panel")
        
        # Check models
        if state.bt_results and not state.bt_metrics:
            issues.append("⚠️ Models exist but no metrics computed")
        
        return issues
    
    @classmethod
    def export_summary(cls) -> Dict[str, Any]:
        """
        Export complete state summary for debugging/logging.
        
        Returns:
            dict: Complete state information
            
        Example:
            >>> import json
            >>> summary = AppState.export_summary()
            >>> with open('state_snapshot.json', 'w') as f:
            ...     json.dump(summary, f, indent=2, default=str)
        """
        state = cls.get()
        
        def safe_shape(obj):
            """Get shape safely"""
            if obj is None:
                return None
            try:
                return obj.shape
            except:
                return (len(obj),) if hasattr(obj, '__len__') else None
        
        return {
            'timestamp': datetime.now().isoformat(),
            'state_exists': cls.exists(),
            
            'target': {
                'y_monthly': {
                    'exists': state.y_monthly is not None,
                    'length': len(state.y_monthly) if state.y_monthly is not None else 0,
                    'date_range': [
                        state.y_monthly.index.min().isoformat(),
                        state.y_monthly.index.max().isoformat()
                    ] if state.y_monthly is not None and not state.y_monthly.empty else None
                },
                'targets_monthly': {
                    'exists': state.targets_monthly is not None,
                    'shape': safe_shape(state.targets_monthly)
                }
            },
            
            'panels': {
                'monthly': {
                    'exists': state.panel_monthly is not None,
                    'shape': safe_shape(state.panel_monthly)
                },
                'quarterly': {
                    'exists': state.panel_quarterly is not None,
                    'shape': safe_shape(state.panel_quarterly)
                },
                'daily': {
                    'exists': state.panel_daily is not None,
                    'shape': safe_shape(state.panel_daily)
                }
            },
            
            'raw_data': {
                'daily_files': len(state.raw_daily),
                'monthly_files': len(state.raw_monthly),
                'quarterly_files': len(state.raw_quarterly),
                'google_trends': state.google_trends is not None
            },
            
            'models': {
                'bt_results_count': len(state.bt_results),
                'model_names': list(state.bt_results.keys()),
                'has_metrics': state.bt_metrics is not None,
                'forecasts_count': len(state.forecasts)
            },
            
            'feature_engineering': {
                'recipe_steps': len(state.feature_recipe),
                'has_importance': state.feature_importance is not None
            },
            
            'metadata': {
                'uploads': len(state.upload_history),
                'last_update': state.last_update.isoformat() if state.last_update else None
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def get_state() -> _AppState:
    """
    Shorthand for AppState.get()
    
    Returns:
        _AppState: Current state
        
    Example:
        >>> from utils.state import get_state
        >>> state = get_state()
    """
    return AppState.get()

def clear_state() -> _AppState:
    """
    Shorthand for AppState.clear()
    
    Returns:
        _AppState: Fresh state
        
    Example:
        >>> from utils.state import clear_state
        >>> clear_state()
    """
    return AppState.clear()


# =============================================================================
# State Context Manager (Advanced)
# =============================================================================

class StateContext:
    """
    Context manager for temporary state modifications.
    
    Usage:
        >>> with StateContext() as state:
        ...     state.y_monthly = test_data
        ...     run_analysis()
        >>> # Original state restored after context
    """
    
    def __init__(self):
        self._backup = None
    
    def __enter__(self) -> _AppState:
        """Save current state and return it"""
        import copy
        self._backup = copy.deepcopy(AppState.get())
        return AppState.get()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original state"""
        if self._backup is not None:
            st.session_state[AppState._STATE_KEY] = self._backup
        return False


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing state module...")
    
    # Test initialization
    state = AppState.init()
    print(f"✓ State initialized: {type(state)}")
    
    # Test data assignment
    state.y_monthly = pd.Series([1, 2, 3], name='test')
    print(f"✓ Data assigned: {len(state.y_monthly)} values")
    
    # Test summary
    summary = AppState.summary()
    print(f"✓ Summary generated: {summary['has_target']}")
    
    # Test validation
    issues = AppState.validate()
    print(f"✓ Validation completed: {len(issues)} issues")
    
    # Test export
    export = AppState.export_summary()
    print(f"✓ Export generated: {len(export)} keys")
    
    print("\n✅ All tests passed!")
