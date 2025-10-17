"""
🧪 Feature Engineering Pro v2.0
===================================
Advanced feature transformation engine for time series forecasting.
Features: Smart transforms, lags, rolling stats, recipe management, visual diagnostics.

Author: AI Assistant
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Optional, List, Dict, Set, Tuple

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

try:
    from utils.state import AppState
except Exception:
    class _State:
        def __init__(self):
            self.panel_monthly: Optional[pd.DataFrame] = None
            self.y_monthly: Optional[pd.Series] = None
            self._panel_monthly_orig: Optional[pd.DataFrame] = None
    
    class AppState:
        @staticmethod
        def init():
            if "_app" not in st.session_state:
                st.session_state["_app"] = _State()
            return st.session_state["_app"]
        
        @staticmethod
        def get():
            return AppState.init()

state = AppState.init()

# Initialize recipe in session state
if 'fe_recipe' not in st.session_state:
    st.session_state.fe_recipe = []

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def slugify(name: str) -> str:
    """Convert name to clean identifier"""
    name = str(name).strip().lower()
    for char in [' ', '/', '(', ')', '-', '%', ':', ',', '.', '__']:
        name = name.replace(char, '_')
    while '__' in name:
        name = name.replace('__', '_')
    return name.strip('_')

def make_unique_name(base: str, existing: Set[str]) -> str:
    """Generate unique column name"""
    if base not in existing:
        existing.add(base)
        return base
    
    counter = 1
    while f"{base}_{counter}" in existing:
        counter += 1
    
    new_name = f"{base}_{counter}"
    existing.add(new_name)
    return new_name

def is_numeric_column(series: pd.Series) -> bool:
    """Check if column is numeric"""
    return pd.api.types.is_numeric_dtype(series)

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns"""
    return [col for col in df.columns if is_numeric_column(df[col])]

def calculate_memory_usage(df: pd.DataFrame) -> str:
    """Calculate and format memory usage"""
    bytes_usage = df.memory_usage(deep=True).sum()
    
    if bytes_usage < 1024:
        return f"{bytes_usage:.0f} B"
    elif bytes_usage < 1024**2:
        return f"{bytes_usage/1024:.1f} KB"
    elif bytes_usage < 1024**3:
        return f"{bytes_usage/1024**2:.1f} MB"
    else:
        return f"{bytes_usage/1024**3:.2f} GB"

# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def safe_log_transform(series: pd.Series) -> pd.Series:
    """Safe logarithm transform handling negative and zero values"""
    series = series.astype(float)
    
    # Find minimum positive value
    positive_values = series[series > 0]
    if len(positive_values) > 0:
        epsilon = np.nanmin(positive_values) * 1e-6
    else:
        epsilon = 1e-9
    
    return np.log(series + epsilon)

def winsorize_series(series: pd.Series, lower_pct: float, upper_pct: float) -> pd.Series:
    """Winsorize series at specified percentiles"""
    lower_val = np.nanpercentile(series, lower_pct)
    upper_val = np.nanpercentile(series, upper_pct)
    return series.clip(lower_val, upper_val)

def apply_transforms(
    df: pd.DataFrame,
    columns: List[str],
    diff: bool = False,
    pct_change: bool = False,
    log: bool = False,
    winsorize: bool = False,
    winsor_lower: float = 1.0,
    winsor_upper: float = 99.0
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply selected transforms to columns"""
    
    result_df = df.copy()
    existing_cols = set(result_df.columns)
    new_columns = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        series = pd.to_numeric(df[col], errors='coerce')
        
        if diff:
            new_col = make_unique_name(f"diff1__{slugify(col)}", existing_cols)
            result_df[new_col] = series.diff(1)
            new_columns.append(new_col)
        
        if pct_change:
            new_col = make_unique_name(f"pct1__{slugify(col)}", existing_cols)
            result_df[new_col] = series.pct_change(1)
            new_columns.append(new_col)
        
        if log:
            new_col = make_unique_name(f"log__{slugify(col)}", existing_cols)
            result_df[new_col] = safe_log_transform(series)
            new_columns.append(new_col)
        
        if winsorize:
            new_col = make_unique_name(
                f"win__{slugify(col)}_{int(winsor_lower)}_{int(winsor_upper)}", 
                existing_cols
            )
            result_df[new_col] = winsorize_series(series, winsor_lower, winsor_upper)
            new_columns.append(new_col)
    
    return result_df, new_columns

def apply_lags(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int]
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply lag features"""
    
    result_df = df.copy()
    existing_cols = set(result_df.columns)
    new_columns = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        series = pd.to_numeric(df[col], errors='coerce')
        
        for lag in lags:
            new_col = make_unique_name(f"lag{lag}__{slugify(col)}", existing_cols)
            result_df[new_col] = series.shift(lag)
            new_columns.append(new_col)
    
    return result_df, new_columns

def apply_rolling_stats(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    stats: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply rolling statistics"""
    
    result_df = df.copy()
    existing_cols = set(result_df.columns)
    new_columns = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        series = pd.to_numeric(df[col], errors='coerce')
        
        for window in windows:
            min_periods = max(1, window // 2)
            rolling = series.rolling(window, min_periods=min_periods)
            
            if 'mean' in stats:
                new_col = make_unique_name(f"roll{window}_mean__{slugify(col)}", existing_cols)
                result_df[new_col] = rolling.mean()
                new_columns.append(new_col)
            
            if 'std' in stats:
                new_col = make_unique_name(f"roll{window}_std__{slugify(col)}", existing_cols)
                result_df[new_col] = rolling.std(ddof=0)
                new_columns.append(new_col)
            
            if 'min' in stats:
                new_col = make_unique_name(f"roll{window}_min__{slugify(col)}", existing_cols)
                result_df[new_col] = rolling.min()
                new_columns.append(new_col)
            
            if 'max' in stats:
                new_col = make_unique_name(f"roll{window}_max__{slugify(col)}", existing_cols)
                result_df[new_col] = rolling.max()
                new_columns.append(new_col)
    
    return result_df, new_columns

def apply_standardization(
    df: pd.DataFrame,
    columns: List[str],
    expanding: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply z-score standardization"""
    
    result_df = df.copy()
    existing_cols = set(result_df.columns)
    new_columns = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        series = pd.to_numeric(df[col], errors='coerce')
        
        if expanding:
            mean = series.expanding().mean()
            std = series.expanding().std(ddof=0)
        else:
            mean = series.mean()
            std = series.std(ddof=0)
        
        new_col = make_unique_name(f"{slugify(col)}__z", existing_cols)
        result_df[new_col] = (series - mean) / (std + 1e-9)
        new_columns.append(new_col)
    
    return result_df, new_columns

def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'none',
    drop_leading: bool = True
) -> pd.DataFrame:
    """Handle missing values"""
    
    result_df = df.copy()
    
    if method == 'ffill':
        result_df = result_df.ffill()
    elif method == 'bfill':
        result_df = result_df.bfill()
    elif method == 'ffill_then_bfill':
        result_df = result_df.ffill().bfill()
    
    if drop_leading:
        # Remove leading rows that are all NaN
        result_df = result_df.loc[~result_df.isna().all(axis=1)]
    
    return result_df

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_feature_comparison(
    original: pd.Series,
    transformed: pd.Series,
    feature_name: str
):
    """Compare original and transformed feature"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Original', 'Transformed'),
        vertical_spacing=0.15
    )
    
    # Original
    fig.add_trace(
        go.Scatter(
            x=original.index,
            y=original.values,
            mode='lines',
            name='Original',
            line=dict(color='#3B82F6', width=2)
        ),
        row=1, col=1
    )
    
    # Transformed
    fig.add_trace(
        go.Scatter(
            x=transformed.index,
            y=transformed.values,
            mode='lines',
            name='Transformed',
            line=dict(color='#10B981', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'Feature Comparison: {feature_name}',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def plot_missing_values_heatmap(df: pd.DataFrame, n_rows: int = 60):
    """Plot missing values heatmap"""
    
    presence = df.tail(n_rows).notna().astype(int)
    
    fig = px.imshow(
        presence.T,
        aspect='auto',
        color_continuous_scale=['#EF4444', '#10B981'],
        labels={'color': 'Present'},
        title=f'Data Presence (Last {n_rows} Months)'
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        xaxis_title='Date',
        yaxis_title='Feature',
        coloraxis_showscale=True
    )
    
    return fig

def plot_correlation_with_target(
    panel: pd.DataFrame,
    target: pd.Series,
    top_n: int = 15
):
    """Plot correlation with target"""
    
    y_aligned, X_aligned = target.align(panel, join='inner')
    
    if X_aligned.empty:
        return None
    
    correlations = X_aligned.corrwith(y_aligned).sort_values()
    
    # Get top positive and negative correlations
    top_corr = pd.concat([
        correlations.head(top_n // 2),
        correlations.tail(top_n // 2)
    ])
    
    colors = ['#EF4444' if x < 0 else '#10B981' for x in top_corr.values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_corr.values,
        y=top_corr.index,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{x:.3f}' for x in top_corr.values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Correlations with Target',
        xaxis_title='Correlation',
        yaxis_title='Feature',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig

# =============================================================================
# UI CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Feature Engineering Pro",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(120deg, #7c3aed, #c026d3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(90deg, #7c3aed, #c026d3);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed 0%, #c026d3 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Header
st.markdown('<h1 class="main-title">🧪 Feature Engineering Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform, lag, and enhance your time series features</p>', unsafe_allow_html=True)

# Check if panel exists
if state.panel_monthly is None or state.panel_monthly.empty:
    st.error("⚠️ **No monthly panel found!**")
    st.info("👉 Please go to **Data & Aggregation** page first to build your panel.")
    st.stop()

# Create original snapshot if not exists
if state._panel_monthly_orig is None:
    state._panel_monthly_orig = state.panel_monthly.copy()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("🔍 Display Options")
    show_last_n = st.slider(
        "Preview last N months",
        min_value=12,
        max_value=240,
        value=60,
        step=12
    )
    
    show_memory = st.checkbox("Show memory usage", value=False)
    
    st.markdown("---")
    st.subheader("📊 Panel Info")
    
    current_shape = state.panel_monthly.shape
    original_shape = state._panel_monthly_orig.shape if state._panel_monthly_orig is not None else (0, 0)
    
    st.metric("Rows", current_shape[0])
    st.metric("Features", current_shape[1])
    
    if original_shape[1] > 0:
        new_features = current_shape[1] - original_shape[1]
        st.metric("New Features", new_features, delta=f"+{new_features}")
    
    if show_memory:
        memory = calculate_memory_usage(state.panel_monthly)
        st.metric("Memory", memory)
    
    st.markdown("---")
    st.subheader("🔄 Reset")
    
    if st.button("🔙 Reset to Original", use_container_width=True, type="secondary"):
        if state._panel_monthly_orig is not None:
            state.panel_monthly = state._panel_monthly_orig.copy()
            st.session_state.fe_recipe = []
            st.success("✅ Reset to original panel!")
            st.rerun()
    
    if st.button("🗑️ Clear Recipe", use_container_width=True):
        st.session_state.fe_recipe = []
        st.success("✅ Recipe cleared!")

# Get numeric columns
numeric_cols = get_numeric_columns(state.panel_monthly)

if not numeric_cols:
    st.error("⚠️ No numeric columns found in panel!")
    st.stop()

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔄 Transforms",
    "⏱️ Lags & Rolling",
    "🔧 Missing Values",
    "📊 Standardize",
    "📋 Recipe"
])

# =============================================================================
# TAB 1: TRANSFORMS
# =============================================================================

with tab1:
    st.markdown('<div class="step-header"><h3>📐 Deterministic Transforms</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_cols = st.multiselect(
            "Select features to transform:",
            options=numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            key='transform_cols'
        )
    
    with col2:
        st.markdown("#### Transform Options")
        do_diff = st.checkbox("📉 Difference (1st)", value=False)
        do_pct = st.checkbox("📊 Percent Change", value=False)
        do_log = st.checkbox("📈 Logarithm", value=False)
        do_winsor = st.checkbox("✂️ Winsorize", value=False)
    
    if do_winsor:
        col1, col2 = st.columns(2)
        with col1:
            winsor_lower = st.slider("Lower percentile", 0.0, 10.0, 1.0, 0.5)
        with col2:
            winsor_upper = st.slider("Upper percentile", 90.0, 100.0, 99.0, 0.5)
    else:
        winsor_lower = 1.0
        winsor_upper = 99.0
    
    st.info("💡 **Tip:** New columns will have prefixes like `diff1__`, `pct1__`, `log__`, `win__`")
    
    if st.button("🚀 Apply Transforms", type="primary", use_container_width=True):
        if not selected_cols:
            st.warning("⚠️ Please select at least one feature")
        elif not any([do_diff, do_pct, do_log, do_winsor]):
            st.warning("⚠️ Please select at least one transform")
        else:
            with st.spinner("Applying transforms..."):
                state.panel_monthly, new_cols = apply_transforms(
                    state.panel_monthly,
                    selected_cols,
                    diff=do_diff,
                    pct_change=do_pct,
                    log=do_log,
                    winsorize=do_winsor,
                    winsor_lower=winsor_lower,
                    winsor_upper=winsor_upper
                )
                
                # Update recipe
                st.session_state.fe_recipe.append({
                    'operation': 'transform',
                    'columns': selected_cols,
                    'params': {
                        'diff': do_diff,
                        'pct_change': do_pct,
                        'log': do_log,
                        'winsorize': do_winsor,
                        'winsor_lower': winsor_lower,
                        'winsor_upper': winsor_upper
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
                st.success(f"✅ Created {len(new_cols)} new features!")
                st.rerun()

# =============================================================================
# TAB 2: LAGS & ROLLING
# =============================================================================

with tab2:
    st.markdown('<div class="step-header"><h3>⏱️ Temporal Features</h3></div>', unsafe_allow_html=True)
    
    selected_cols_lag = st.multiselect(
        "Select features for lags/rolling:",
        options=numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        key='lag_cols'
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📅 Lags")
        lag_input = st.text_input(
            "Lag periods (comma-separated)",
            value="1,3,6,12",
            help="e.g., 1,3,6,12 for 1, 3, 6, and 12 month lags"
        )
        
        # Parse lags
        lags = []
        if lag_input.strip():
            for val in lag_input.split(','):
                try:
                    lag = int(val.strip())
                    if lag > 0:
                        lags.append(lag)
                except:
                    pass
        lags = sorted(set(lags))
    
    with col2:
        st.markdown("#### 📊 Rolling Windows")
        window_input = st.text_input(
            "Window sizes (comma-separated)",
            value="3,6,12",
            help="e.g., 3,6,12 for 3, 6, and 12 month windows"
        )
        
        # Parse windows
        windows = []
        if window_input.strip():
            for val in window_input.split(','):
                try:
                    window = int(val.strip())
                    if window > 0:
                        windows.append(window)
                except:
                    pass
        windows = sorted(set(windows))
    
    with col3:
        st.markdown("#### 📈 Statistics")
        roll_stats = st.multiselect(
            "Rolling statistics:",
            options=['mean', 'std', 'min', 'max'],
            default=['mean', 'std']
        )
    
    st.info(f"💡 **Will create:** {len(selected_cols_lag)} cols × ({len(lags)} lags + {len(windows)} windows × {len(roll_stats)} stats) = {len(selected_cols_lag) * (len(lags) + len(windows) * len(roll_stats))} features")
    
    if st.button("🚀 Generate Features", type="primary", use_container_width=True):
        if not selected_cols_lag:
            st.warning("⚠️ Please select at least one feature")
        elif not lags and not windows:
            st.warning("⚠️ Please specify lags or rolling windows")
        else:
            progress = st.progress(0)
            status = st.empty()
            
            all_new_cols = []
            
            # Apply lags
            if lags:
                status.text("Creating lag features...")
                progress.progress(0.3)
                state.panel_monthly, lag_cols = apply_lags(
                    state.panel_monthly,
                    selected_cols_lag,
                    lags
                )
                all_new_cols.extend(lag_cols)
            
            # Apply rolling
            if windows and roll_stats:
                status.text("Creating rolling features...")
                progress.progress(0.6)
                state.panel_monthly, roll_cols = apply_rolling_stats(
                    state.panel_monthly,
                    selected_cols_lag,
                    windows,
                    roll_stats
                )
                all_new_cols.extend(roll_cols)
            
            progress.progress(1.0)
            status.empty()
            progress.empty()
            
            # Update recipe
            st.session_state.fe_recipe.append({
                'operation': 'lags_rolling',
                'columns': selected_cols_lag,
                'params': {
                    'lags': lags,
                    'windows': windows,
                    'stats': roll_stats
                },
                'timestamp': datetime.now().isoformat()
            })
            
            st.success(f"✅ Created {len(all_new_cols)} new features!")
            st.rerun()

# =============================================================================
# TAB 3: MISSING VALUES
# =============================================================================

with tab3:
    st.markdown('<div class="step-header"><h3>🔧 Handle Missing Values</h3></div>', unsafe_allow_html=True)
    
    # Calculate current missing stats
    total_values = state.panel_monthly.size
    missing_values = state.panel_monthly.isna().sum().sum()
    missing_pct = (missing_values / total_values * 100) if total_values > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Values", f"{total_values:,}")
    col2.metric("Missing Values", f"{missing_values:,}")
    col3.metric("Missing %", f"{missing_pct:.2f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fill_method = st.selectbox(
            "Fill method:",
            options=['none', 'ffill', 'bfill', 'ffill_then_bfill'],
            format_func=lambda x: {
                'none': 'None (Keep as is)',
                'ffill': 'Forward Fill (Use last valid)',
                'bfill': 'Backward Fill (Use next valid)',
                'ffill_then_bfill': 'Forward then Backward Fill'
            }[x]
        )
    
    with col2:
        drop_leading = st.checkbox(
            "Drop leading empty rows",
            value=True,
            help="Remove rows at the start that are completely empty"
        )
    
    if st.button("🚀 Apply Fill Method", type="primary", use_container_width=True):
        with st.spinner("Handling missing values..."):
            state.panel_monthly = handle_missing_values(
                state.panel_monthly,
                method=fill_method,
                drop_leading=drop_leading
            )
            
            # Update recipe
            st.session_state.fe_recipe.append({
                'operation': 'fill_missing',
                'params': {
                    'method': fill_method,
                    'drop_leading': drop_leading
                },
                'timestamp': datetime.now().isoformat()
            })
            
            st.success("✅ Missing values handled!")
            st.rerun()

# =============================================================================
# TAB 4: STANDARDIZE
# =============================================================================

with tab4:
    st.markdown('<div class="step-header"><h3>📊 Z-Score Standardization</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_cols_std = st.multiselect(
            "Select features to standardize:",
            options=numeric_cols,
            default=[],
            key='std_cols'
        )
    
    with col2:
        st.markdown("#### Options")
        use_expanding = st.checkbox(
            "📈 Expanding window",
            value=False,
            help="Use expanding mean/std to avoid look-ahead bias (important for time series!)"
        )
    
    st.info("💡 **New columns** will have `__z` suffix. Z-score = (x - mean) / std")
    
    if st.button("🚀 Apply Standardization", type="primary", use_container_width=True):
        if not selected_cols_std:
            st.warning("⚠️ Please select at least one feature")
        else:
            with st.spinner("Standardizing features..."):
                state.panel_monthly, new_cols = apply_standardization(
                    state.panel_monthly,
                    selected_cols_std,
                    expanding=use_expanding
                )
                
                # Update recipe
                st.session_state.fe_recipe.append({
                    'operation': 'standardize',
                    'columns': selected_cols_std,
                    'params': {
                        'expanding': use_expanding
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
                st.success(f"✅ Created {len(new_cols)} standardized features!")
                st.rerun()

# =============================================================================
# TAB 5: RECIPE
# =============================================================================

with tab5:
    st.markdown('<div class="step-header"><h3>📋 Feature Engineering Recipe</h3></div>', unsafe_allow_html=True)
    
    recipe = st.session_state.fe_recipe
    
    if not recipe:
        st.info("📝 No operations recorded yet. Apply transforms to build your recipe!")
    else:
        st.markdown(f"**Total operations:** {len(recipe)}")
        
        # Display recipe
        recipe_json = json.dumps(recipe, indent=2)
        st.code(recipe_json, language='json')
        
        # Export recipe
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Download Recipe (JSON)",
                data=recipe_json.encode('utf-8'),
                file_name='feature_recipe.json',
                mime='application/json',
                use_container_width=True
            )
        
        with col2:
            if st.button("🗑️ Clear Recipe", use_container_width=True):
                st.session_state.fe_recipe = []
                st.success("✅ Recipe cleared!")
                st.rerun()
        
        with col3:
            if st.button("📊 Recipe Summary", use_container_width=True):
                st.session_state['show_recipe_summary'] = True
    
    st.markdown("---")
    
    # Import recipe
    st.markdown("### 📤 Import Recipe")
    
    recipe_input = st.text_area(
        "Paste JSON recipe here:",
        height=200,
        placeholder='[{"operation": "transform", ...}]'
    )
    
    if st.button("🚀 Apply Imported Recipe", type="primary"):
        try:
            imported_recipe = json.loads(recipe_input)
            
            if not isinstance(imported_recipe, list):
                st.error("❌ Recipe must be a JSON array")
            else:
                st.session_state.fe_recipe = imported_recipe
                st.success(f"✅ Imported {len(imported_recipe)} operations!")
                st.info("💡 Recipe loaded. You may need to reapply operations.")
        
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON: {str(e)}")
    
    # Recipe summary
    if recipe and st.session_state.get('show_recipe_summary', False):
        st.markdown("---")
        st.markdown("### 📊 Recipe Summary")
        
        # Count operations by type
        op_counts = {}
        for step in recipe:
            op = step.get('operation', 'unknown')
            op_counts[op] = op_counts.get(op, 0) + 1
        
        summary_df = pd.DataFrame(list(op_counts.items()), columns=['Operation', 'Count'])
        st.dataframe(summary_df, use_container_width=True)

# =============================================================================
# PREVIEW & DIAGNOSTICS
# =============================================================================

st.markdown("---")
st.markdown('<div class="step-header"><h2>🔍 Panel Preview & Diagnostics</h2></div>', unsafe_allow_html=True)

panel_preview = state.panel_monthly.tail(show_last_n)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("📅 Total Months", state.panel_monthly.shape[0])
col2.metric("📊 Total Features", state.panel_monthly.shape[1])

coverage = panel_preview.notna().mean().mean()
col3.metric("✅ Avg Coverage", f"{coverage:.1%}")

if state.y_monthly is not None:
    y_aligned, X_aligned = state.y_monthly.align(state.panel_monthly, join='inner')
    col4.metric("🎯 Overlap with Target", f"{len(y_aligned)} months")

# Data preview
st.markdown("### 📋 Data Preview")
st.dataframe(
    panel_preview.tail(24).style.format("{:.4f}"),
    use_container_width=True,
    height=400
)

# Missing values heatmap
st.markdown("### 🔍 Data Coverage Heatmap")
heatmap_fig = plot_missing_values_heatmap(state.panel_monthly, show_last_n)
st.plotly_chart(heatmap_fig, use_container_width=True)

# Correlation with target
if state.y_monthly is not None and not state.y_monthly.empty:
    st.markdown("### 📈 Correlation with Target")
    
    corr_fig = plot_correlation_with_target(
        state.panel_monthly,
        state.y_monthly,
        top_n=20
    )
    
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"🔧 {len(st.session_state.fe_recipe)} operations in recipe")

with col2:
    st.caption("💾 Changes persisted in session state")

with col3:
    st.caption("💻 Built with Streamlit Pro")
