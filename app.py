"""
Nowcasting Platform - Main Application
Professional Streamlit interface for time series nowcasting
Production-ready implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add core modules to path
sys.path.append(str(Path(__file__).parent))

# Core imports
from config import CONFIG
from core.data_intelligence import DataIntelligence, DatasetInfo
from core.frequency_aligner import FrequencyAligner
from core.feature_factory import FeatureFactory
from core.model_library import (
    ModelLibrary, PersistenceModel, HistoricalMeanModel, 
    AutoRegressiveModel, RidgeModel, LassoModel, DeltaCorrectionModel
)
from core.evaluator import (
    MetricsCalculator, StatisticalTests, BootstrapMethods, 
    BacktestEngine, ComprehensiveEvaluator
)
from core.exporter import DataExporter, FigureExporter, ReportGenerator, PackageExporter

# UI imports
from ui.styles import get_custom_css, get_header_html
from ui.components import (
    DashboardComponents, WelcomeScreen, DataPreviewComponent,
    ModelConfigurationComponent, ResultsDashboard
)
from ui.charts import NowcastingCharts


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Nowcasting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/nowcasting-platform',
        'Report a bug': 'https://github.com/yourusername/nowcasting-platform/issues',
        'About': """
        # Nowcasting Platform v1.0
        
        Professional forecasting tool for economists and data scientists.
        
        Features:
        - Auto-detection of data frequency
        - Multiple model comparison
        - Statistical significance tests
        - Mixed-frequency data handling (MIDAS)
        - Comprehensive export capabilities
        """
    }
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(get_custom_css(), unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    
    if 'step' not in st.session_state:
        st.session_state.step = 0
    
    if 'target_data' not in st.session_state:
        st.session_state.target_data = None
    
    if 'exog_data' not in st.session_state:
        st.session_state.exog_data = None
    
    if 'alt_data' not in st.session_state:
        st.session_state.alt_data = []
    
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = None
    
    if 'aligned_data' not in st.session_state:
        st.session_state.aligned_data = None
    
    if 'engineered_data' not in st.session_state:
        st.session_state.engineered_data = None
    
    if 'model_config' not in st.session_state:
        st.session_state.model_config = {}
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if 'figures' not in st.session_state:
        st.session_state.figures = {}

initialize_session_state()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def reset_workflow():
    """Reset workflow to beginning"""
    for key in list(st.session_state.keys()):
        if key != 'step':
            del st.session_state[key]
    st.session_state.step = 0
    st.rerun()


def advance_step():
    """Advance to next step"""
    st.session_state.step += 1
    st.rerun()


def go_to_step(step_number: int):
    """Go to specific step"""
    st.session_state.step = step_number
    st.rerun()


# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    get_header_html(
        "📈 Nowcasting Platform",
        "Professional Time Series Forecasting with Mixed-Frequency Data"
    ),
    unsafe_allow_html=True
)


# ============================================================================
# SIDEBAR - NAVIGATION
# ============================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/200x80/003366/FFFFFF?text=NOWCAST", use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Workflow")
    
    # Progress indicator
    steps = [
        "1️⃣ Upload Data",
        "2️⃣ Configure Models",
        "3️⃣ Run Analysis",
        "4️⃣ View Results",
        "5️⃣ Export"
    ]
    
    current_step = st.session_state.step
    
    for idx, step in enumerate(steps):
        if idx < current_step:
            st.markdown(f"✅ ~~{step}~~")
        elif idx == current_step:
            st.markdown(f"**👉 {step}**")
        else:
            st.markdown(f"⚪ {step}")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Reset Workflow", use_container_width=True):
        reset_workflow()
    
    if st.session_state.results:
        if st.button("📊 Jump to Results", use_container_width=True):
            go_to_step(3)
    
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Advanced Settings"):
        train_split = st.slider("Train/Test Split", 0.5, 0.9, 0.7, 0.05)
        random_seed = st.number_input("Random Seed", 0, 9999, 42)
        significance_level = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)
        
        st.session_state.train_split = train_split
        st.session_state.random_seed = random_seed
        st.session_state.significance_level = significance_level
    
    st.markdown("---")
    st.caption("v1.0.0 | Made with ❤️ for Economists")


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Step 0: Welcome Screen
if st.session_state.step == 0:
    
    WelcomeScreen.render()
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Nowcasting", use_container_width=True, type="primary"):
            advance_step()


# Step 1: Data Upload
elif st.session_state.step == 1:
    
    st.markdown("## 1️⃣ Data Upload")
    
    st.info("""
    📌 **Instructions:**
    - Upload your **target variable** (the series you want to forecast)
    - Optionally add **exogenous variables** (economic indicators)
    - Optionally add **alternative data** sources (Google Trends, etc.)
    """)
    
    # Target variable
    st.markdown("### 📊 Target Variable (Required)")
    
    target_file = st.file_uploader(
        "Upload target series (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        key='target_upload',
        help="Must contain a date column and target variable"
    )
    
    if target_file:
        try:
            with st.spinner("📖 Reading file..."):
                intel = DataIntelligence()
                df_target = intel.load_file(target_file)
            
            st.success(f"✅ Loaded {len(df_target)} rows, {len(df_target.columns)} columns")
            
            # Show preview
            with st.expander("👁️ Preview Data"):
                st.dataframe(df_target.head(10), use_container_width=True)
            
            # Detect date column
            date_col = intel.detect_date_column(df_target)
            if date_col:
                st.success(f"✅ Detected date column: **{date_col}**")
            else:
                st.warning("⚠️ Could not auto-detect date column")
                date_col = st.selectbox("Select date column:", df_target.columns)
            
            # Parse dates
            df_target['date_parsed'] = intel.parse_dates(df_target, date_col)
            
            # Detect frequency
            freq, freq_diag = intel.detect_frequency(df_target['date_parsed'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frequency", freq.upper())
            with col2:
                st.metric("Observations", freq_diag.get('observations', 0))
            with col3:
                median_delta = freq_diag.get('median_delta', 0)
                st.metric("Median Gap", f"{median_delta:.1f} days")
            
            # Detect target
            suggested_target = intel.suggest_target(df_target, date_col)
            
            if suggested_target:
                st.info(f"💡 Suggested target: **{suggested_target}**")
                use_suggested = st.checkbox("Use suggested target", value=True)
                if use_suggested:
                    target_col = suggested_target
                else:
                    numeric_cols = df_target.select_dtypes(include=[np.number]).columns.tolist()
                    target_col = st.selectbox("Select target variable:", numeric_cols)
            else:
                numeric_cols = df_target.select_dtypes(include=[np.number]).columns.tolist()
                target_col = st.selectbox("Select target variable:", numeric_cols)
            
            # Validate
            validation = intel.validate_dataset(df_target, date_col, target_col)
            
            if validation['valid']:
                st.success("✅ Data validation passed")
            else:
                st.error("❌ Data validation failed:")
                for error in validation['errors']:
                    st.error(error)
            
            for warning in validation.get('warnings', []):
                st.warning(warning)
            
            # Create DatasetInfo
            dataset_info = DatasetInfo(
                df=df_target,
                date_col=date_col,
                target_col=target_col,
                frequency=freq,
                freq_diag=freq_diag,
                validation=validation
            )
            
            st.session_state.target_data = dataset_info
            st.session_state.dataset_info = dataset_info
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    st.markdown("---")
    
    # Exogenous variables (optional)
    st.markdown("### 📈 Exogenous Variables (Optional)")
    
    with st.expander("ℹ️ What are exogenous variables?"):
        st.markdown("""
        Exogenous variables are external factors that may help predict your target:
        - **Economic indicators:** GDP, inflation, interest rates
        - **Sentiment indices:** Consumer confidence, business surveys
        - **Policy variables:** Government spending, tax rates
        
        These should be at the **same or higher frequency** as your target.
        """)
    
    exog_file = st.file_uploader(
        "Upload exogenous variables (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        key='exog_upload'
    )
    
    if exog_file:
        try:
            intel = DataIntelligence()
            df_exog = intel.load_file(exog_file)
            st.success(f"✅ Loaded {len(df_exog)} rows, {len(df_exog.columns)} columns")
            
            with st.expander("👁️ Preview Exogenous Data"):
                st.dataframe(df_exog.head(10), use_container_width=True)
            
            st.session_state.exog_data = df_exog
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    st.markdown("---")
    
    # Alternative data (optional)
    st.markdown("### 🌐 Alternative Data (Optional)")
    
    with st.expander("ℹ️ What is alternative data?"):
        st.markdown("""
        Alternative data sources include:
        - **Google Trends:** Search volume for keywords
        - **Social media:** Twitter sentiment, mentions
        - **Web traffic:** Page views, clicks
        - **Satellite data:** Parking lot occupancy, shipping activity
        
        ⚠️ **Important:** Alternative data may contain measurement bias.
        Platform will apply MIDAS aggregation for frequency alignment.
        """)
    
    include_alt = st.checkbox("Include alternative data sources")
    
    if include_alt:
        st.warning("⚠️ Alternative data may contain measurement bias. Use with caution.")
        
        alt_files = st.file_uploader(
            "Upload alternative data files",
            type=['csv', 'xlsx', 'xls'],
            key='alt_upload',
            accept_multiple_files=True
        )
        
        if alt_files:
            st.session_state.alt_data = []
            for idx, file in enumerate(alt_files):
                try:
                    intel = DataIntelligence()
                    df_alt = intel.load_file(file)
                    st.success(f"✅ File {idx+1}: {file.name} - {len(df_alt)} rows")
                    st.session_state.alt_data.append(df_alt)
                except Exception as e:
                    st.error(f"❌ Error loading {file.name}: {str(e)}")
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            go_to_step(0)
    
    with col3:
        if st.session_state.target_data and st.session_state.target_data.validation['valid']:
            if st.button("Next ➡️", use_container_width=True, type="primary"):
                advance_step()
        else:
            st.button("Next ➡️", use_container_width=True, disabled=True)
            st.caption("Upload valid target data to continue")


# Step 2: Model Configuration
elif st.session_state.step == 2:
    
    st.markdown("## 2️⃣ Model Configuration")
    
    # Show data summary
    if st.session_state.dataset_info:
        DataPreviewComponent.render(st.session_state.dataset_info)
    
    st.markdown("---")
    
    # Model selection
    st.markdown("### 🤖 Select Models")
    
    tab1, tab2, tab3 = st.tabs(["📊 Benchmarks", "🔬 Machine Learning", "⚙️ Advanced"])
    
    with tab1:
        st.markdown("**Benchmark Models** (Always Included)")
        
        st.info("""
        The following baseline models are always included for comparison:
        - **Persistence:** ŷ_t = y_{t-1} (Random Walk)
        - **Historical Mean:** ŷ_t = mean(y_train)
        - **AR(1):** First-order autoregressive
        - **AR(2):** Second-order autoregressive
        """)
        
        st.markdown("✅ All benchmark models will be included automatically")
    
    with tab2:
        st.markdown("**Machine Learning Models**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_ridge = st.checkbox("✅ Ridge Regression", value=True)
            if use_ridge:
                ridge_alphas = st.multiselect(
                    "Ridge regularization (α)",
                    [1, 10, 50, 100, 200],
                    default=[10, 50, 100],
                    help="Higher α = more regularization"
                )
            else:
                ridge_alphas = []
            
            use_lasso = st.checkbox("✅ Lasso Regression", value=True)
            if use_lasso:
                lasso_alphas = st.multiselect(
                    "Lasso regularization (α)",
                    [0.001, 0.01, 0.1, 1.0],
                    default=[0.01, 0.1],
                    help="Lasso performs feature selection"
                )
            else:
                lasso_alphas = []
        
        with col2:
            use_elastic = st.checkbox("Elastic Net", value=False)
            if use_elastic:
                elastic_alpha = st.select_slider("ElasticNet α", [0.01, 0.1, 1.0], value=0.1)
                elastic_l1_ratio = st.slider("L1 ratio", 0.0, 1.0, 0.5, 0.1)
            
            use_delta = st.checkbox("✅ Delta-Correction", value=True)
            if use_delta:
                st.info("Predicts changes (Δ) instead of levels, then corrects: ŷ = lag1 + w·Δ̂")
                blend_weights = st.multiselect(
                    "Blending weights (w)",
                    [0.5, 0.7, 0.8, 0.9, 1.0],
                    default=[0.9, 1.0],
                    help="w=1.0 means full delta, w<1.0 blends with persistence"
                )
            else:
                blend_weights = []
        
        # Store config
        st.session_state.model_config = {
            'use_ridge': use_ridge,
            'ridge_alphas': ridge_alphas,
            'use_lasso': use_lasso,
            'lasso_alphas': lasso_alphas,
            'use_elastic': use_elastic,
            'use_delta': use_delta,
            'blend_weights': blend_weights
        }
    
    with tab3:
        st.markdown("**Advanced Settings**")
        
        # MIDAS (if alternative data)
        if st.session_state.alt_data:
            st.markdown("#### Mixed-Frequency Settings (MIDAS)")
            
            use_midas = st.checkbox("Enable MIDAS aggregation", value=True)
            
            if use_midas:
                midas_windows = st.multiselect(
                    "MIDAS windows (W)",
                    [4, 8, 12, 16],
                    default=[4, 8],
                    help="Number of high-freq observations to aggregate"
                )
                
                midas_lambdas = st.multiselect(
                    "Exponential decay (λ)",
                    [0.5, 0.6, 0.7, 0.8, 0.9],
                    default=[0.6, 0.8],
                    help="Higher λ = more weight on recent data"
                )
                
                cutoff_day = st.number_input(
                    "Nowcast cutoff day",
                    1, 28, 15,
                    help="Use data up to day D of nowcast month"
                )
                
                st.session_state.model_config.update({
                    'use_midas': use_midas,
                    'midas_windows': midas_windows,
                    'midas_lambdas': midas_lambdas,
                    'cutoff_day': cutoff_day
                })
        
        # Feature engineering
        st.markdown("#### Feature Engineering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_lags = st.checkbox("Include lags", value=True)
            if include_lags:
                max_lags = st.number_input("Maximum lag order", 1, 24, 12)
            
            include_ma = st.checkbox("Include moving averages", value=True)
        
        with col2:
            include_seasonal = st.checkbox("Include seasonal dummies", value=True)
            include_yoy = st.checkbox("Include year-over-year", value=True)
        
        st.session_state.model_config.update({
            'include_lags': include_lags,
            'max_lags': max_lags if include_lags else None,
            'include_ma': include_ma,
            'include_seasonal': include_seasonal,
            'include_yoy': include_yoy
        })
        
        # Evaluation settings
        st.markdown("#### Evaluation Settings")
        
        enable_backtest = st.checkbox("Enable rolling-origin backtest", value=True)
        if enable_backtest:
            n_splits = st.slider("Number of splits", 5, 20, 10)
            st.session_state.model_config['enable_backtest'] = True
            st.session_state.model_config['n_splits'] = n_splits
        else:
            st.session_state.model_config['enable_backtest'] = False
        
        enable_bootstrap = st.checkbox("Enable bootstrap CI", value=True)
        if enable_bootstrap:
            bootstrap_iterations = st.number_input("Bootstrap iterations", 500, 5000, 2000, 500)
            st.session_state.model_config['enable_bootstrap'] = True
            st.session_state.model_config['bootstrap_iterations'] = bootstrap_iterations
        else:
            st.session_state.model_config['enable_bootstrap'] = False
    
    st.markdown("---")
    
    # Summary
    with st.expander("📋 Configuration Summary", expanded=True):
        config = st.session_state.model_config
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Models to Train:**")
            models_count = 4  # Benchmarks
            if config.get('use_ridge') and config.get('ridge_alphas'):
                models_count += len(config['ridge_alphas'])
            if config.get('use_lasso') and config.get('lasso_alphas'):
                models_count += len(config['lasso_alphas'])
            if config.get('use_delta') and config.get('blend_weights'):
                models_count *= len(config['blend_weights'])
            
            st.metric("Total Models", models_count)
        
        with col2:
            st.markdown("**Evaluation:**")
            eval_methods = []
            if config.get('enable_backtest'):
                eval_methods.append(f"Rolling ({config.get('n_splits')} splits)")
            if config.get('enable_bootstrap'):
                eval_methods.append(f"Bootstrap ({config.get('bootstrap_iterations')} iter)")
            eval_methods.append("DM Test")
            eval_methods.append("CW Test")
            
            st.write(", ".join(eval_methods))
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            go_to_step(1)
    
    with col3:
        if st.button("Run Analysis ▶️", use_container_width=True, type="primary"):
            advance_step()


# Step 3: Run Analysis
elif st.session_state.step == 3:
    
    st.markdown("## 3️⃣ Running Analysis")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Feature engineering
        status_text.text("🔧 Engineering features...")
        progress_bar.progress(10)
        
        dataset_info = st.session_state.dataset_info
        config = st.session_state.model_config
        
        factory = FeatureFactory()
        
        df_features = factory.auto_engineer_features(
            df=dataset_info.df,
            target_col=dataset_info.target_col,
            date_col=dataset_info.date_col,
            frequency=dataset_info.frequency,
            include_seasonal=config.get('include_seasonal', True),
            include_yoy=config.get('include_yoy', True),
            include_ma=config.get('include_ma', True)
        )
        
        progress_bar.progress(20)
        
        # Step 2: Handle alternative data (MIDAS)
        if st.session_state.alt_data and config.get('use_midas'):
            status_text.text("🌐 Aggregating alternative data (MIDAS)...")
            
            aligner = FrequencyAligner()
            
            for alt_df in st.session_state.alt_data:
                # Detect alt data frequency
                intel = DataIntelligence()
                alt_date_col = intel.detect_date_column(alt_df)
                if alt_date_col:
                    alt_dates = intel.parse_dates(alt_df, alt_date_col)
                    alt_freq, _ = intel.detect_frequency(alt_dates)
                    
                    # Align
                    alt_cols = [c for c in alt_df.columns if c != alt_date_col]
                    aligned = aligner.align_datasets(
                        df_features,
                        alt_df,
                        dataset_info.date_col,
                        alt_date_col,
                        dataset_info.frequency,
                        alt_freq,
                        alt_cols,
                        cutoff_day=config.get('cutoff_day', 15)
                    )
                    
                    # Merge
                    df_features = df_features.merge(aligned, on=dataset_info.date_col, how='left')
        
        progress_bar.progress(30)
        
        # Step 3: Prepare train/test split
        status_text.text("✂️ Splitting data...")
        
        df_clean = df_features.dropna(subset=[dataset_info.target_col])
        
        train_split = st.session_state.get('train_split', 0.7)
        split_idx = int(len(df_clean) * train_split)
        
        train_df = df_clean.iloc[:split_idx].copy()
        test_df = df_clean.iloc[split_idx:].copy()
        
        # Feature columns
        feature_cols = [c for c in df_clean.columns 
                       if c not in [dataset_info.date_col, dataset_info.target_col]]
        
        # Remove low-quality features
        feature_cols = factory.remove_constant_features(df_clean, feature_cols)
        
        X_train = train_df[feature_cols].values
        y_train = train_df[dataset_info.target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[dataset_info.target_col].values
        
        progress_bar.progress(40)
        
        # Step 4: Train models
        status_text.text("🤖 Training models...")
        
        library = ModelLibrary()
        
        # Benchmark models
        models_to_train = []
        models_to_train.extend(library.create_benchmark_suite())
        
        # Ridge
        if config.get('use_ridge') and config.get('ridge_alphas'):
            for alpha in config['ridge_alphas']:
                models_to_train.append(RidgeModel(alpha=alpha))
        
        # Lasso
        if config.get('use_lasso') and config.get('lasso_alphas'):
            for alpha in config['lasso_alphas']:
                models_to_train.append(LassoModel(alpha=alpha))
        
        progress_bar.progress(50)
        
        # Train all models
        results = {}
        predictions = {}
        
        for idx, model in enumerate(models_to_train):
            status_text.text(f"🤖 Training {model.name}...")
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                predictions[model.name] = y_pred
                
                # Calculate metrics
                calc = MetricsCalculator()
                metrics = calc.calculate_all(y_test, y_pred)
                results[model.name] = metrics
                
            except Exception as e:
                st.warning(f"⚠️ {model.name} failed: {str(e)}")
            
            progress_bar.progress(50 + int(30 * (idx+1) / len(models_to_train)))
        
        # Delta-correction variants
        if config.get('use_delta') and config.get('blend_weights'):
            status_text.text("🔄 Creating delta-correction variants...")
            
            for base_name, base_pred in list(predictions.items()):
                if 'Ridge' in base_name or 'Lasso' in base_name:
                    for w in config['blend_weights']:
                        # Recreate model
                        if 'Ridge' in base_name:
                            alpha = float(base_name.split('α=')[1].split(')')[0])
                            base_model = RidgeModel(alpha=alpha)
                        else:
                            alpha = float(base_name.split('α=')[1].split(')')[0])
                            base_model = LassoModel(alpha=alpha)
                        
                        # Delta correction
                        delta_model = DeltaCorrectionModel(base_model, blend_weight=w)
                        
                        try:
                            delta_model.fit(X_train, y_train)
                            y_pred_delta = delta_model.predict(X_test)
                            
                            delta_name = f"Δ-{base_name[:-1]},w={w})"
                            predictions[delta_name] = y_pred_delta
                            
                            metrics = MetricsCalculator().calculate_all(y_test, y_pred_delta)
                            results[delta_name] = metrics
                        except:
                            pass
        
        progress_bar.progress(80)
        
        # Step 5: Statistical tests
        status_text.text("📊 Running statistical tests...")
        
        # Find best model
        rmse_scores = {name: m['rmse'] for name, m in results.items()}
        best_model_name = min(rmse_scores, key=rmse_scores.get)
        
        # Persistence for comparison
        pers_name = 'Persistence'
        if pers_name in predictions:
            tester = StatisticalTests(significance_level=st.session_state.get('significance_level', 0.05))
            
            # DM test: Best vs Persistence
            e_best = y_test - predictions[best_model_name]
            e_pers = y_test - predictions[pers_name]
            
            dm_result = tester.diebold_mariano_test(e_best, e_pers, alternative='greater')
            
            # CW test: Best vs simpler baseline (if applicable)
            baseline_names = [n for n in predictions.keys() if 'AR(' in n or 'Mean' in n]
            if baseline_names:
                baseline_name = baseline_names[0]
                e_baseline = y_test - predictions[baseline_name]
                
                cw_result = tester.clark_west_test(
                    e_best, e_baseline,
                    predictions[best_model_name], predictions[baseline_name]
                )
            else:
                cw_result = None
            
            test_results = {
                'Diebold-Mariano': {
                    'statistic': dm_result.statistic,
                    'p_value': dm_result.p_value,
                    'is_significant': dm_result.is_significant,
                    'interpretation': dm_result.interpretation
                }
            }
            
            if cw_result:
                test_results['Clark-West'] = {
                    'statistic': cw_result.statistic,
                    'p_value': cw_result.p_value,
                    'is_significant': cw_result.is_significant,
                    'interpretation': cw_result.interpretation
                }
        else:
            test_results = {}
        
        progress_bar.progress(90)
        
        # Step 6: Bootstrap CI (if enabled)
        if config.get('enable_bootstrap'):
            status_text.text("🔁 Computing bootstrap confidence intervals...")
            
            bootstrap = BootstrapMethods(
                n_iterations=config.get('bootstrap_iterations', 2000)
            )
            
            if pers_name in predictions:
                bootstrap_result = bootstrap.moving_block_bootstrap(
                    e_best, e_pers, block_size=6
                )
            else:
                bootstrap_result = None
        else:
            bootstrap_result = None
        
        progress_bar.progress(95)
        
        # Step 7: Backtesting (if enabled)
        if config.get('enable_backtest'):
            status_text.text("🔄 Running rolling-origin backtest...")
            
            backtest_engine = BacktestEngine()
            
            # Use best model for backtest
            def model_factory():
                if 'Ridge' in best_model_name:
                    alpha = float(best_model_name.split('α=')[1].split(')')[0])
                    return RidgeModel(alpha=alpha)
                else:
                    return PersistenceModel()
            
            try:
                backtest_results = backtest_engine.rolling_origin_backtest(
                    df_clean,
                    dataset_info.target_col,
                    feature_cols,
                    model_factory,
                    n_splits=config.get('n_splits', 10)
                )
                
                backtest_summary = backtest_engine.summarize_backtest_results(backtest_results)
            except Exception as e:
                st.warning(f"⚠️ Backtest failed: {str(e)}")
                backtest_results = []
                backtest_summary = {}
        else:
            backtest_results = []
            backtest_summary = {}
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Store results
        st.session_state.results = {
            'metrics': results,
            'predictions': predictions,
            'test_actual': y_test,
            'test_dates': test_df[dataset_info.date_col].values,
            'best_model': best_model_name,
            'statistical_tests': test_results,
            'bootstrap': bootstrap_result,
            'backtest_results': backtest_results,
            'backtest_summary': backtest_summary,
            'feature_cols': feature_cols
        }
        
        # Auto-advance
        st.success("✅ Analysis complete! Generating visualizations...")
        st.balloons()
        
        import time
        time.sleep(2)
        advance_step()
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Retry", use_container_width=True):
                st.rerun()


# Step 4: Results
elif st.session_state.step == 4:
    
    st.markdown("## 4️⃣ Results")
    
    if not st.session_state.results:
        st.error("No results available. Please run analysis first.")
        if st.button("⬅️ Go Back"):
            go_to_step(2)
        st.stop()
    
    results = st.session_state.results
    
    # Results Dashboard
    ResultsDashboard.render(results)
    
    st.markdown("---")
    
    # Detailed tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Predictions",
        "📈 Performance",
        "🧪 Statistical Tests",
        "🔄 Backtest",
        "📋 Summary"
    ])
    
    with tab1:
        st.markdown("### 📊 Predictions vs Actual")
        
        # Generate chart
        charts = NowcastingCharts()
        
        dates = pd.to_datetime(results['test_dates'])
        actual = results['test_actual']
        predictions_dict = results['predictions']
        
        # Limit to top 5 models for clarity
        rmse_scores = {name: results['metrics'][name]['rmse'] 
                      for name in predictions_dict.keys() 
                      if name in results['metrics']}
        top_models = sorted(rmse_scores, key=rmse_scores.get)[:5]
        
        predictions_to_plot = {name: predictions_dict[name] 
                              for name in top_models}
        
        fig_pred = charts.plot_predictions_vs_actual(
            dates, actual, predictions_to_plot,
            title="Predictions vs Actual (Top 5 Models)"
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Errors
        st.markdown("### 📉 Forecast Errors")
        
        errors_dict = {name: actual - predictions_dict[name] 
                      for name in top_models}
        
        fig_errors = charts.plot_forecast_errors(
            dates, errors_dict,
            title="Forecast Errors Over Time"
        )
        
        st.plotly_chart(fig_errors, use_container_width=True)
        
        # Store figures
        st.session_state.figures['predictions_vs_actual'] = fig_pred
        st.session_state.figures['forecast_errors'] = fig_errors
    
    with tab2:
        st.markdown("### 📈 Model Performance Comparison")
        
        # Metrics table
        DashboardComponents.comparison_table(
            results['metrics'],
            highlight_best=True,
            metrics_to_show=['rmse', 'mae', 'mape', 'direction_accuracy']
        )
        
        # Metrics comparison chart
        fig_metrics = charts.plot_metrics_comparison(
            results['metrics'],
            metric_names=['rmse', 'mae'],
            title="RMSE and MAE Comparison"
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Error distribution
        st.markdown("### 📊 Error Distribution")
        
        errors_all = {name: actual - pred 
                     for name, pred in predictions_dict.items()}
        
        fig_dist = charts.plot_error_distribution(
            errors_all,
            title="Error Distribution Across Models"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.session_state.figures['metrics_comparison'] = fig_metrics
        st.session_state.figures['error_distribution'] = fig_dist
    
    with tab3:
        st.markdown("### 🧪 Statistical Significance Tests")
        
        test_results = results.get('statistical_tests', {})
        
        if test_results:
            for test_name, test_result in test_results.items():
                DashboardComponents.statistical_test_badge(
                    test_name,
                    test_result['p_value'],
                    test_result['statistic']
                )
                
                with st.expander(f"ℹ️ About {test_name}"):
                    st.markdown(test_result.get('interpretation', ''))
            
            # Statistical tests chart
            fig_tests = charts.plot_statistical_tests(
                test_results,
                title="Statistical Test Results"
            )
            
            st.plotly_chart(fig_tests, use_container_width=True)
            
            st.session_state.figures['statistical_tests'] = fig_tests
        else:
            st.info("No statistical tests available")
        
        # Bootstrap CI
        bootstrap = results.get('bootstrap')
        if bootstrap:
            st.markdown("### 🔁 Bootstrap Confidence Interval")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Difference", f"{bootstrap['mean']:.4f}")
            with col2:
                st.metric("95% CI Lower", f"{bootstrap['ci_lower']:.4f}")
            with col3:
                st.metric("95% CI Upper", f"{bootstrap['ci_upper']:.4f}")
            
            if bootstrap.get('includes_zero'):
                st.warning("⚠️ Confidence interval includes zero - improvement not statistically robust")
            else:
                st.success("✅ Confidence interval excludes zero - improvement is statistically robust")
    
    with tab4:
        st.markdown("### 🔄 Rolling-Origin Backtest")
        
        backtest_results = results.get('backtest_results', [])
        
        if backtest_results:
            backtest_summary = results.get('backtest_summary', {})
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Splits", backtest_summary.get('n_splits', 0))
            with col2:
                st.metric("Mean RMSE", f"{backtest_summary.get('mean_rmse', 0):.4f}")
            with col3:
                st.metric("Std RMSE", f"{backtest_summary.get('std_rmse', 0):.4f}")
            with col4:
                st.metric("Min RMSE", f"{backtest_summary.get('min_rmse', 0):.4f}")
            
            # Rolling performance chart
            fig_rolling = charts.plot_rolling_performance(
                backtest_results,
                metric='rmse',
                title="RMSE Across Rolling Windows"
            )
            
            st.plotly_chart(fig_rolling, use_container_width=True)
            
            st.session_state.figures['rolling_backtest'] = fig_rolling
            
            # Detailed results table
            with st.expander("📋 Detailed Backtest Results"):
                backtest_df = pd.DataFrame([{
                    'Split': r.split_id,
                    'Train Size': r.n_train,
                    'Test Size': r.n_test,
                    'RMSE': r.metrics['rmse'],
                    'MAE': r.metrics['mae']
                } for r in backtest_results])
                
                st.dataframe(backtest_df, use_container_width=True)
        else:
            st.info("No backtest results available")
    
    with tab5:
        st.markdown("### 📋 Complete Summary")
        
        # Best model highlight
        best_model = results['best_model']
        best_metrics = results['metrics'][best_model]
        
        st.success(f"🏆 **Best Model:** {best_model}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", f"{best_metrics['rmse']:.4f}")
        with col2:
            st.metric("MAE", f"{best_metrics['mae']:.4f}")
        with col3:
            st.metric("Direction Acc", f"{best_metrics.get('direction_accuracy', 0):.1f}%")
        with col4:
            st.metric("Theil's U", f"{best_metrics.get('theil_u', 0):.3f}")
        
        st.markdown("---")
        
        # Key findings
        st.markdown("### 🔍 Key Findings")
        
        # Compare to persistence
        if 'Persistence' in results['metrics']:
            pers_rmse = results['metrics']['Persistence']['rmse']
            best_rmse = best_metrics['rmse']
            improvement = (1 - best_rmse / pers_rmse) * 100
            
            if improvement > 0:
                st.success(f"✅ Best model improves over persistence by **{improvement:.2f}%**")
            else:
                st.warning(f"⚠️ Best model is {abs(improvement):.2f}% worse than persistence")
        
        # Statistical significance
        dm_test = results.get('statistical_tests', {}).get('Diebold-Mariano')
        if dm_test:
            if dm_test['is_significant']:
                st.success(f"✅ Improvement is statistically significant (DM p={dm_test['p_value']:.4f})")
            else:
                st.warning(f"⚠️ Improvement is not statistically significant (DM p={dm_test['p_value']:.4f})")
        
        # Feature importance (if available)
        st.markdown("### 🎯 Top Features")
        
        feature_cols = results.get('feature_cols', [])
        if feature_cols:
            st.write(f"**Total features used:** {len(feature_cols)}")
            
            with st.expander("📋 View All Features"):
                st.write(feature_cols)
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back to Config", use_container_width=True):
            go_to_step(2)
    
    with col3:
        if st.button("Export Results ➡️", use_container_width=True, type="primary"):
            advance_step()


# Step 5: Export
elif st.session_state.step == 5:
    
    st.markdown("## 5️⃣ Export Results")
    
    if not st.session_state.results:
        st.error("No results to export")
        st.stop()
    
    results = st.session_state.results
    
    st.info("📦 Download your complete nowcasting results")
    
    # Prepare exports
    data_exporter = DataExporter()
    figure_exporter = FigureExporter()
    
    # Individual downloads
    st.markdown("### 📄 Individual Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Predictions CSV
        predictions_df = pd.DataFrame({
            'date': results['test_dates'],
            'actual': results['test_actual']
        })
        
        for model_name, pred in results['predictions'].items():
            col_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            predictions_df[f'pred_{col_name}'] = pred
        
        csv_predictions = predictions_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "📊 Download Predictions CSV",
            csv_predictions,
            "predictions.csv",
            "text/csv",
            use_container_width=True
        )
        
        # Metrics CSV
        metrics_df = pd.DataFrame(results['metrics']).T
        metrics_df.index.name = 'model'
        metrics_df.reset_index(inplace=True)
        
        csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "📈 Download Metrics CSV",
            csv_metrics,
            "metrics.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Statistical tests JSON
        tests_json = json.dumps(results.get('statistical_tests', {}), indent=2).encode('utf-8')
        
        st.download_button(
            "🧪 Download Statistical Tests JSON",
            tests_json,
            "statistical_tests.json",
            "application/json",
            use_container_width=True
        )
        
        # Backtest results
        if results.get('backtest_results'):
            backtest_df = pd.DataFrame([{
                'split': r.split_id,
                'n_train': r.n_train,
                'n_test': r.n_test,
                **r.metrics
            } for r in results['backtest_results']])
            
            csv_backtest = backtest_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                "🔄 Download Backtest Results CSV",
                csv_backtest,
                "backtest_results.csv",
                "text/csv",
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Complete package
    st.markdown("### 📦 Complete Package")
    
    st.info("Download all results in a single ZIP file")
    
    if st.button("🎁 Generate Complete Package", use_container_width=True, type="primary"):
        with st.spinner("📦 Creating package..."):
            try:
                from core.exporter import StreamlitDownloader
                
                # Prepare files
                files_dict = {}
                
                # CSVs
                files_dict['predictions.csv'] = csv_predictions
                files_dict['metrics.csv'] = csv_metrics
                files_dict['statistical_tests.json'] = tests_json
                
                if results.get('backtest_results'):
                    files_dict['backtest_results.csv'] = csv_backtest
                
                # Figures (HTML)
                for fig_name, fig in st.session_state.figures.items():
                    html_buffer = io.StringIO()
                    fig.write_html(html_buffer)
                    files_dict[f'{fig_name}.html'] = html_buffer.getvalue().encode('utf-8')
                
                # Create ZIP
                zip_bytes = StreamlitDownloader.prepare_zip_download(files_dict)
                
                st.download_button(
                    "⬇️ Download ZIP Package",
                    zip_bytes,
                    f"nowcast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    "application/zip",
                    use_container_width=True
                )
                
                st.success("✅ Package ready for download!")
                
            except Exception as e:
                st.error(f"❌ Error creating package: {str(e)}")
    
    st.markdown("---")
    
    # Final actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Back to Results", use_container_width=True):
            go_to_step(4)
    
    with col2:
        if st.button("🔄 New Analysis", use_container_width=True):
            reset_workflow()
    
    with col3:
        st.markdown("### ✅ Complete!")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div class="footer">
    <p>
        <strong>Nowcasting Platform v1.0</strong><br>
        Developed for professional economists and data scientists<br>
        📧 Contact | 📚 Documentation | 🐛 Report Issues
    </p>
</div>
""", unsafe_allow_html=True)
