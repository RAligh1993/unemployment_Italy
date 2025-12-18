"""
Italian Unemployment Nowcasting System
Professional Streamlit App with Multi-Model Framework

Author: Rajabali Ghasempour
Institution: ISTAT
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Backend imports
from backend.data_loader import DataLoader
from backend.feature_engineering import FeatureEngineer
from backend.models import ModelFactory
from backend.evaluation import Evaluator
from backend.forecaster import RealTimeForecaster
from utils.visualizations import Visualizer
from utils.helpers import format_number, get_signal_status

# Page config
st.set_page_config(
    page_title="Unemployment Nowcasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=ISTAT", use_column_width=True)
    st.markdown("### 📊 Nowcasting System")
    st.markdown("---")
    
    st.markdown("#### ⚙️ Configuration")
    
    # Mode selection
    mode = st.radio(
        "Operating Mode",
        ["Default (Unemployment + GT)", "Custom Analysis"],
        help="Default mode uses Italian unemployment data with Google Trends"
    )
    
    st.markdown("---")
    
    # Data upload
    st.markdown("#### 📁 Data Upload")
    
    uploaded_unemployment = st.file_uploader(
        "Unemployment Data (CSV)",
        type=['csv'],
        help="Monthly unemployment rate time series"
    )
    
    uploaded_gt = st.file_uploader(
        "Google Trends (Excel)",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        help="Multiple 5-year segments (optional)"
    )
    
    uploaded_exog = st.file_uploader(
        "Exogenous Variables (CSV)",
        type=['csv'],
        help="Additional predictors (CCI, HICP, etc.)"
    )
    
    st.markdown("---")
    
    # Model settings
    st.markdown("#### 🎯 Model Settings")
    
    train_test_split = st.slider(
        "Train/Test Split (%)",
        min_value=50,
        max_value=90,
        value=70,
        step=5,
        help="Percentage of data for training"
    )
    
    include_gt = st.checkbox(
        "Include Google Trends",
        value=True,
        help="Use GT features in models"
    )
    
    models_to_run = st.multiselect(
        "Models to Train",
        ["MIDAS Exponential", "MIDAS Beta", "Ridge", "Lasso", "Random Forest", "XGBoost", "LSTM"],
        default=["MIDAS Exponential", "Ridge"],
        help="Select multiple models for comparison"
    )
    
    st.markdown("---")
    
    # Action buttons
    if st.button("🚀 Load & Process Data", type="primary", use_container_width=True):
        st.session_state.trigger_load = True
    
    if st.button("🤖 Train Models", disabled=not st.session_state.data_loaded, use_container_width=True):
        st.session_state.trigger_train = True
    
    st.markdown("---")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Author:** Rajabali Ghasempour")
    st.markdown("**Institution:** ISTAT")

# Main content
st.markdown('<div class="main-header">🇮🇹 Italian Unemployment Nowcasting System</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Data Explorer",
    "🤖 Models",
    "📉 Results",
    "🔮 Live Nowcast",
    "📚 Documentation"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    st.markdown("### Welcome to the Unemployment Nowcasting System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Purpose</h4>
            <p>Provide real-time unemployment estimates using high-frequency Google Trends data, 2-3 weeks before official releases.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>⚡ Key Features</h4>
            <ul>
                <li>Multi-model framework</li>
                <li>MIDAS aggregation</li>
                <li>Statistical testing</li>
                <li>Real-time nowcasting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Current Status</h4>
            <p><strong>Data Loaded:</strong> {}</p>
            <p><strong>Models Trained:</strong> {}</p>
        </div>
        """.format(
            "✅ Yes" if st.session_state.data_loaded else "❌ No",
            "✅ Yes" if st.session_state.models_trained else "❌ No"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Start Guide
    st.markdown("### 🚀 Quick Start")
    
    with st.expander("📖 How to Use This App", expanded=not st.session_state.data_loaded):
        st.markdown("""
        #### Step 1: Upload Data
        - **Unemployment Data**: Monthly unemployment rate (required)
        - **Google Trends**: Weekly search data (optional but recommended)
        - **Exogenous Variables**: CCI, HICP, etc. (optional)
        
        #### Step 2: Configure Settings
        - Choose operating mode (Default or Custom)
        - Select train/test split ratio
        - Pick models to train
        
        #### Step 3: Load & Process
        - Click "🚀 Load & Process Data" in sidebar
        - Review data quality and correlations
        
        #### Step 4: Train Models
        - Click "🤖 Train Models" to start training
        - Compare model performance
        
        #### Step 5: Generate Nowcasts
        - Navigate to "🔮 Live Nowcast" tab
        - Get real-time predictions with confidence intervals
        """)
    
    # System Architecture
    with st.expander("🏗️ System Architecture"):
        st.markdown("""
```
        Data Sources          Processing              Models                Output
        ─────────────        ──────────────         ─────────────         ────────────
        │ Unemployment   →   │ 5-Seg Merge    →    │ MIDAS Exp     →    │ Nowcast
        │ Google Trends  →   │ Feature Eng    →    │ MIDAS Beta    →    │ ± CI
        │ CCI, HICP      →   │ Lag Creation   →    │ Ridge/Lasso   →    │ Alerts
                             │ Scaling        →    │ ML Models     →    │ Viz
```
        
        **Backend Components:**
        - Data Loader: Handles uploads, cleaning, validation
        - Feature Engineer: Creates lags, MIDAS weights, interactions
        - Model Factory: Trains multiple model types
        - Evaluator: Computes metrics, statistical tests
        - Forecaster: Real-time nowcasting engine
        """)
    
    # Performance Summary (if models trained)
    if st.session_state.models_trained and 'model_results' in st.session_state.results:
        st.markdown("### 📊 Latest Performance Summary")
        
        results = st.session_state.results['model_results']
        best_model = results.loc[results['RMSE'].idxmin()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Best Model",
                best_model['Model'],
                delta=None
            )
        
        with col2:
            st.metric(
                "RMSE",
                f"{best_model['RMSE']:.4f}",
                delta=None
            )
        
        with col3:
            improvement = best_model.get('Improvement_pct', 0)
            st.metric(
                "Improvement",
                f"{improvement:+.1f}%",
                delta=f"{improvement:+.1f}%"
            )
        
        with col4:
            p_value = best_model.get('p_value', 1.0)
            st.metric(
                "Significance",
                "Yes ✅" if p_value < 0.05 else "No ❌",
                delta=f"p={p_value:.4f}"
            )

# ============================================================================
# TAB 2: DATA EXPLORER
# ============================================================================

with tab2:
    st.markdown("### 📈 Data Explorer & Quality Assessment")
    
    if not st.session_state.data_loaded:
        st.info("👆 Please upload data and click 'Load & Process Data' in the sidebar to begin.")
    else:
        data_loader = st.session_state.get('data_loader')
        
        # Data Summary
        st.markdown("#### 📊 Dataset Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h5>Unemployment Data</h5>
                <p><strong>Observations:</strong> {}</p>
                <p><strong>Date Range:</strong> {} to {}</p>
                <p><strong>Mean:</strong> {:.2f}%</p>
                <p><strong>Std Dev:</strong> {:.2f}%</p>
            </div>
            """.format(
                len(st.session_state.df_clean),
                st.session_state.df_clean['date'].min().date(),
                st.session_state.df_clean['date'].max().date(),
                st.session_state.df_clean['target'].mean(),
                st.session_state.df_clean['target'].std()
            ), unsafe_allow_html=True)
        
        with col2:
            if include_gt and 'gt_data' in st.session_state:
                gt_quality = st.session_state.get('gt_quality', {})
                st.markdown("""
                <div class="success-box">
                    <h5>Google Trends Quality</h5>
                    <p><strong>GOOD Keywords:</strong> {}</p>
                    <p><strong>CAUTION Keywords:</strong> {}</p>
                    <p><strong>WARNING Keywords:</strong> {}</p>
                    <p><strong>Total Merged Weeks:</strong> {}</p>
                </div>
                """.format(
                    len(gt_quality.get('GOOD', [])),
                    len(gt_quality.get('CAUTION', [])),
                    len(gt_quality.get('WARNING', [])),
                    len(st.session_state.gt_data)
                ), unsafe_allow_html=True)
        
        # Time Series Plot
        st.markdown("#### 📉 Time Series Visualization")
        
        viz = Visualizer()
        fig = viz.plot_time_series_interactive(
            st.session_state.df_clean,
            'date',
            'target',
            title="Unemployment Rate Changes Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Matrix
        st.markdown("#### 🔗 Correlation Analysis")
        
        target_col = 'target'
        feature_cols = [col for col in st.session_state.df_clean.columns 
                       if col not in ['date', target_col] and 
                       st.session_state.df_clean[col].dtype in ['float64', 'int64']]
        
        if len(feature_cols) > 0:
            corr_matrix = st.session_state.df_clean[feature_cols + [target_col]].corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top Correlations with Target
            st.markdown("##### Top Correlations with Target")
            
            target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Strongest Positive:**")
                positive_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False).head(5)
                for feat, corr in positive_corr.items():
                    st.write(f"- **{feat}**: {corr:+.3f}")
            
            with col2:
                st.markdown("**Strongest Negative:**")
                negative_corr = corr_matrix[target_col].drop(target_col).sort_values().head(5)
                for feat, corr in negative_corr.items():
                    st.write(f"- **{feat}**: {corr:+.3f}")
        
        # Data Table
        with st.expander("📋 View Raw Data"):
            st.dataframe(
                st.session_state.df_clean,
                use_container_width=True,
                height=400
            )

# ============================================================================
# TAB 3: MODELS
# ============================================================================

with tab3:
    st.markdown("### 🤖 Model Training & Comparison")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the sidebar.")
    else:
        # Model Configuration
        st.markdown("#### ⚙️ Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h5>Selected Models</h5>
                <ul>
                    {}
                </ul>
            </div>
            """.format("\n".join([f"<li>{m}</li>" for m in models_to_run])), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h5>Training Configuration</h5>
                <p><strong>Train Size:</strong> {}%</p>
                <p><strong>Test Size:</strong> {}%</p>
                <p><strong>GT Included:</strong> {}</p>
            </div>
            """.format(train_test_split, 100-train_test_split, "Yes" if include_gt else "No"), unsafe_allow_html=True)
        
        with col3:
            if st.session_state.models_trained:
                st.markdown("""
                <div class="success-box">
                    <h5>Training Complete ✅</h5>
                    <p>Models trained successfully</p>
                    <p>Check Results tab for details</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <h5>Not Trained Yet ⏳</h5>
                    <p>Click "Train Models" in sidebar</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Training Progress (if triggered)
        if st.session_state.get('trigger_train', False):
            with st.spinner("🤖 Training models... This may take a few minutes."):
                # Initialize components
                factory = ModelFactory()
                evaluator = Evaluator()
                
                # Get data splits
                train_data = st.session_state.train
                test_data = st.session_state.test
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_list = []
                
                for idx, model_name in enumerate(models_to_run):
                    status_text.text(f"Training {model_name}... ({idx+1}/{len(models_to_run)})")
                    
                    # Train model
                    model = factory.create_model(model_name, include_gt=include_gt)
                    model.fit(train_data)
                    
                    # Predict
                    pred = model.predict(test_data)
                    
                    # Evaluate
                    metrics = evaluator.compute_metrics(
                        test_data['target'].values,
                        pred,
                        baseline_pred=np.full(len(pred), train_data['target'].mean())
                    )
                    
                    results_list.append({
                        'Model': model_name,
                        'RMSE': metrics['RMSE'],
                        'MAE': metrics['MAE'],
                        'Improvement_pct': metrics['Improvement_pct'],
                        'p_value': metrics.get('p_value', np.nan)
                    })
                    
                    progress_bar.progress((idx + 1) / len(models_to_run))
                
                # Save results
                st.session_state.results['model_results'] = pd.DataFrame(results_list)
                st.session_state.models_trained = True
                st.session_state.trigger_train = False
                
                status_text.text("✅ Training complete!")
                st.success("All models trained successfully!")
                st.rerun()

# ============================================================================
# TAB 4: RESULTS
# ============================================================================

with tab4:
    st.markdown("### 📉 Model Results & Performance")
    
    if not st.session_state.models_trained:
        st.info("📊 Train models first to see results here.")
    else:
        results_df = st.session_state.results['model_results']
        
        # Performance Table
        st.markdown("#### 📊 Model Comparison")
        
        st.dataframe(
            results_df.style.background_gradient(subset=['RMSE'], cmap='RdYlGn_r')
                     .background_gradient(subset=['Improvement_pct'], cmap='RdYlGn')
                     .format({'RMSE': '{:.4f}', 'MAE': '{:.4f}', 
                             'Improvement_pct': '{:+.2f}%', 'p_value': '{:.4f}'}),
            use_container_width=True
        )
        
        # Best Model Highlight
        best_idx = results_df['RMSE'].idxmin()
        best_model = results_df.loc[best_idx]
        
        st.markdown(f"""
        <div class="success-box">
            <h4>🏆 Best Model: {best_model['Model']}</h4>
            <p><strong>RMSE:</strong> {best_model['RMSE']:.4f}</p>
            <p><strong>Improvement:</strong> {best_model['Improvement_pct']:+.2f}%</p>
            <p><strong>Statistical Significance:</strong> {'✅ Yes (p < 0.05)' if best_model['p_value'] < 0.05 else '❌ No'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization
        st.markdown("#### 📈 Performance Visualization")
        
        viz = Visualizer()
        fig = viz.plot_model_comparison(results_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Period-wise Performance
        if 'period_results' in st.session_state.results:
            st.markdown("#### 📅 Period-wise Performance")
            period_df = st.session_state.results['period_results']
            
            fig = viz.plot_period_performance(period_df)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 5: LIVE NOWCAST
# ============================================================================

with tab5:
    st.markdown("### 🔮 Live Unemployment Nowcast")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first to generate nowcasts.")
    else:
        # Current Nowcast
        st.markdown("#### 📊 Current Month Nowcast")
        
        forecaster = st.session_state.get('forecaster')
        
        if forecaster:
            nowcast = forecaster.generate_nowcast()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Nowcast (MIDAS)",
                    f"{nowcast['prediction']:+.2f} pp",
                    delta=nowcast.get('vs_historical', None)
                )
            
            with col2:
                st.metric(
                    "Confidence Interval",
                    f"[{nowcast['ci_lower']:+.2f}, {nowcast['ci_upper']:+.2f}]",
                    delta=None
                )
            
            with col3:
                signal_status = get_signal_status(nowcast['gt_signals'])
                st.metric(
                    "GT Signal Status",
                    signal_status['level'],
                    delta=signal_status['message']
                )
        
        # GT Signal Monitoring
        if include_gt:
            st.markdown("#### 🚨 Google Trends Early Warning Signals")
            
            gt_latest = st.session_state.get('gt_latest', {})
            
            col1, col2, col3 = st.columns(3)
            
            keywords = ['centro per l\'impiego', 'offerte di lavoro', 'curriculum']
            
            for idx, kw in enumerate(keywords):
                with [col1, col2, col3][idx]:
                    value = gt_latest.get(kw, 50)
                    baseline = 50
                    change = ((value - baseline) / baseline) * 100
                    
                    st.metric(
                        kw.title(),
                        f"{value:.0f}",
                        delta=f"{change:+.1f}%"
                    )
        
        # Forecast Timeline
        st.markdown("#### 📅 Nowcast Timeline")
        
        st.markdown("""
        <div class="info-box">
            <p><strong>Current Date:</strong> {}</p>
            <p><strong>Reference Month:</strong> {}</p>
            <p><strong>Official Release:</strong> {} (in {} days)</p>
            <p><strong>GT Data Advantage:</strong> 2-3 weeks earlier</p>
        </div>
        """.format(
            datetime.now().strftime("%Y-%m-%d"),
            "November 2025",
            "December 15, 2025",
            12
        ), unsafe_allow_html=True)

# ============================================================================
# TAB 6: DOCUMENTATION
# ============================================================================

with tab6:
    st.markdown("### 📚 Documentation & User Guide")
    
    with st.expander("📖 Complete User Guide", expanded=True):
        st.markdown("""
        # Italian Unemployment Nowcasting System
        ## User Guide
        
        ### System Overview
        
        This application provides real-time unemployment nowcasts for Italy using:
        - **Official ISTAT unemployment data** (monthly)
        - **Google Trends search data** (weekly, optional)
        - **Exogenous economic indicators** (CCI, HICP, optional)
        
        ### Key Features
        
        #### 1. Multi-Model Framework
        - **MIDAS Models**: Exponential and Beta polynomial weighting
        - **Linear Models**: Ridge, Lasso regression
        - **Machine Learning**: Random Forest, XGBoost
        - **Deep Learning**: LSTM networks (experimental)
        
        #### 2. Google Trends Integration
        - Automatic 5-segment merging with quality checks
        - Keyword selection based on economic relevance
        - Weekly-to-monthly aggregation via MIDAS
        
        #### 3. Statistical Rigor
        - Clark-West test for nested models
        - Diebold-Mariano test for general comparison
        - Backtesting with walk-forward validation
        
        ### Workflow
        
        **Step 1: Data Upload**
        - Upload unemployment CSV with 'date' and 'unemp' columns
        - Optionally upload Google Trends Excel files (5-year segments)
        - Optionally upload exogenous variables CSV
        
        **Step 2: Configuration**
        - Select operating mode (Default or Custom)
        - Choose train/test split ratio
        - Select models to train
        
        **Step 3: Processing**
        - Click "Load & Process Data"
        - Review data quality in Data Explorer tab
        - Check correlation structure
        
        **Step 4: Modeling**
        - Click "Train Models"
        - Wait for training to complete
        - Review results in Results tab
        
        **Step 5: Nowcasting**
        - Navigate to Live Nowcast tab
        - View current month prediction
        - Monitor GT early warning signals
        
        ### Interpreting Results
        
        #### RMSE (Root Mean Squared Error)
        - Lower is better
        - Measures average prediction error
        - Comparable across models
        
        #### Improvement Percentage
        - Shows gain over naive baseline (historical mean)
        - +7% means 7% RMSE reduction
        - Realistic gains: 5-10%
        
        #### p-value
        - Statistical significance test
        - p < 0.05 indicates significant improvement
        - Be cautious with borderline values
        
        #### Confidence Intervals
        - ±2 standard deviations
        - 95% probability true value is in range
        - Wider intervals = higher uncertainty
        
        ### Best Practices
        
        1. **Data Quality**: Ensure clean, consistent data with no large gaps
        2. **Sample Size**: Minimum 60 months training data recommended
        3. **Google Trends**: Use multiple keywords for robustness
        4. **Model Selection**: Start with MIDAS, add ML models for comparison
        5. **Validation**: Always check out-of-sample performance
        
        ### Limitations
        
        - **Timeliness vs Accuracy**: Early signals may be noisy
        - **Method Dependency**: GT value requires proper aggregation
        - **Structural Breaks**: Performance may degrade during crises
        - **Regional Aggregation**: National nowcasts mask local variation
        
        ### Technical Details
        
        **MIDAS Exponential Weights (θ=3.0):**
```
        w_j = exp(-θ * j) / Σ exp(-θ * i)
        
        Result: [0.950, 0.047, 0.002, 0.0001]
        → 95% weight on most recent week
```
        
        **Ridge Regression:**
```
        β̂ = argmin (Σ(y - Xβ)² + α||β||²)
        
        α = 50.0 (selected via cross-validation)
```
        
        ### Contact & Support
        
        - **Author**: Rajabali Ghasempour
        - **Institution**: ISTAT
        - **Version**: 1.0.0
        - **Last Updated**: December 2025
        
        For issues or questions, please contact ISTAT Labor Statistics Division.
        """)
    
    with st.expander("🔧 Technical Specifications"):
        st.markdown("""
        ### System Requirements
        
        **Software:**
        - Python 3.8+
        - Streamlit 1.28+
        - See requirements.txt for full dependencies
        
        **Hardware:**
        - Minimum: 4GB RAM
        - Recommended: 8GB RAM, 2+ CPU cores
        
        **Data Format:**
        - Unemployment: CSV with 'date', 'unemp' columns
        - Google Trends: Excel (.xlsx) with 'Week' and keyword columns
        - Exogenous: CSV with 'date' and variable columns
        
        ### API Reference
        
        **Backend Modules:**
```python
        from backend.data_loader import DataLoader
        from backend.models import ModelFactory
        from backend.evaluation import Evaluator
        from backend.forecaster import RealTimeForecaster
```
        
        **Key Classes:**
        - `DataLoader`: Handle data upload, cleaning, merging
        - `FeatureEngineer`: Create lags, MIDAS features
        - `ModelFactory`: Train multiple model types
        - `Evaluator`: Compute metrics, statistical tests
        - `RealTimeForecaster`: Generate live nowcasts
        
        ### Deployment
        
        **Local:**
```bash
        streamlit run app.py
```
        
        **Streamlit Cloud:**
        1. Push to GitHub
        2. Connect repository in Streamlit Cloud
        3. Configure secrets (if needed)
        4. Deploy
        
        **Docker:**
```dockerfile
        FROM python:3.9-slim
        COPY . /app
        WORKDIR /app
        RUN pip install -r requirements.txt
        CMD streamlit run app.py
```
        """)
    
    with st.expander("📊 Example Datasets"):
        st.markdown("""
        ### Sample Data Files
        
        Download example datasets to test the app:
        
        **1. Unemployment Data (unemployment.csv)**
```csv
        date,unemp,unemp(25-34)
        2020-01-01,9.8,16.2
        2020-02-01,9.7,16.0
        2020-03-01,9.9,16.5
        ...
```
        
        **2. Google Trends (segment1.xlsx)**
        | Week | lavoro | cerco lavoro | offerte di lavoro | ... |
        |------|--------|--------------|-------------------|-----|
        | 2020-07-05 | 45 | 38 | 52 | ... |
        | 2020-07-12 | 47 | 40 | 51 | ... |
        
        **3. Exogenous Variables (exog.csv)**
```csv
        date,CCI,PRC-HICP
        2020-01-01,105.2,0.2
        2020-02-01,104.8,0.1
        ...
```
        """)

# Handle data loading trigger
if st.session_state.get('trigger_load', False):
    with st.spinner("📁 Loading and processing data..."):
        try:
            loader = DataLoader()
            
            # Load unemployment
            if uploaded_unemployment:
                df_unemp = pd.read_csv(uploaded_unemployment)
            else:
                # Use default demo data
                st.warning("No data uploaded. Using demo dataset.")
                df_unemp = loader.load_demo_data()
            
            # Process
            df_clean = loader.process_data(
                df_unemp,
                gt_files=uploaded_gt if uploaded_gt else None,
                exog_file=uploaded_exog if uploaded_exog else None
            )
            
            # Train/test split
            split_idx = int(len(df_clean) * train_test_split / 100)
            train = df_clean.iloc[:split_idx]
            test = df_clean.iloc[split_idx:]
            
            # Save to session state
            st.session_state.data_loader = loader
            st.session_state.df_clean = df_clean
            st.session_state.train = train
            st.session_state.test = test
            st.session_state.data_loaded = True
            st.session_state.trigger_load = False
            
            st.success("✅ Data loaded successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.session_state.trigger_load = False
