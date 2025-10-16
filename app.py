"""
ISTAT Unemployment Nowcasting Lab - Professional Edition
Main Application File with Modern UI and Claude AI Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import json
from pathlib import Path

# ==========================================
# CONFIGURATION & THEME
# ==========================================

st.set_page_config(
    page_title="ISTAT Nowcasting Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional Unemployment Nowcasting System for Italy"
    }
)

# Professional Dark Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Dark Theme Background */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(120deg, #6366F1 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Cards - Glassmorphic Effect */
    .stMetric {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
    }
    
    /* Primary Button Special */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stTextArea > div > div > textarea {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 10px;
        color: white !important;
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    .stSuccess { border-left-color: #10B981; }
    .stWarning { border-left-color: #F59E0B; }
    .stError { border-left-color: #EF4444; }
    .stInfo { border-left-color: #6366F1; }
    
    /* Dataframe Styling */
    .dataframe {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Correlation Badge */
    .correlation-badge {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        margin: 4px;
        backdrop-filter: blur(10px);
    }
    
    .corr-high { 
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1));
        border: 1px solid #10B981;
        color: #10B981;
    }
    
    .corr-med { 
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1));
        border: 1px solid #F59E0B;
        color: #F59E0B;
    }
    
    .corr-low { 
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
        border: 1px solid #EF4444;
        color: #EF4444;
    }
    
    /* Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

class AppState:
    """Central state management"""
    
    @staticmethod
    def init():
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.data = None
            st.session_state.target_series = None
            st.session_state.correlations = {}
            st.session_state.models = {}
            st.session_state.predictions = {}
            st.session_state.panel_monthly = None
            st.session_state.panel_quarterly = None
            st.session_state.feature_importance = {}
            st.session_state.ai_messages = []
            st.session_state.news_signal = None
            st.session_state.shap_values = None
            st.session_state.backtest_results = {}
            st.session_state.ensemble_weights = {}
            
    @staticmethod
    def get(key, default=None):
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key, value):
        st.session_state[key] = value
        
    @staticmethod
    def update(key, value):
        if key in st.session_state:
            st.session_state[key].update(value)
        else:
            st.session_state[key] = value

# Initialize state
AppState.init()

# ==========================================
# INSTANT CORRELATION MODULE
# ==========================================

def calculate_instant_correlation(df, target_col):
    """Calculate and display instant correlations"""
    if df is None or target_col not in df.columns:
        return None
    
    # Calculate correlations
    corr = df.corr()[target_col].sort_values(ascending=False)
    
    # Remove self-correlation
    corr = corr[corr.index != target_col]
    
    return corr

def display_correlation_cards(correlations, top_n=6):
    """Display correlation results as beautiful cards"""
    st.markdown("### 🎯 Instant Correlation Analysis")
    
    cols = st.columns(3)
    for i, (feature, corr_value) in enumerate(correlations.head(top_n).items()):
        col_idx = i % 3
        
        # Determine correlation strength and color
        abs_corr = abs(corr_value)
        if abs_corr >= 0.7:
            strength = "Strong"
            css_class = "corr-high"
            icon = "🔥"
        elif abs_corr >= 0.4:
            strength = "Moderate"
            css_class = "corr-med"
            icon = "📊"
        else:
            strength = "Weak"
            css_class = "corr-low"
            icon = "📉"
        
        direction = "↑" if corr_value > 0 else "↓"
        
        with cols[col_idx]:
            st.markdown(f"""
            <div style="
                background: rgba(30, 41, 59, 0.5);
                backdrop-filter: blur(10px);
                border: 1px solid {'#10B981' if abs_corr >= 0.7 else '#F59E0B' if abs_corr >= 0.4 else '#EF4444'};
                border-radius: 16px;
                padding: 20px;
                margin: 10px 0;
                transition: all 0.3s ease;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 24px;">{icon}</span>
                    <span style="font-size: 20px; font-weight: bold; color: {'#10B981' if corr_value > 0 else '#EF4444'};">
                        {direction} {abs(corr_value):.3f}
                    </span>
                </div>
                <div style="margin-top: 10px;">
                    <div style="font-weight: 600; color: #E5E7EB; font-size: 14px;">
                        {feature[:30]}{'...' if len(feature) > 30 else ''}
                    </div>
                    <div style="color: #9CA3AF; font-size: 12px; margin-top: 5px;">
                        {strength} correlation
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def create_correlation_heatmap(df, target_col):
    """Create interactive correlation heatmap"""
    # Select top correlated features
    corr_with_target = df.corr()[target_col].abs().sort_values(ascending=False)
    top_features = corr_with_target.head(20).index.tolist()
    
    # Create correlation matrix for top features
    corr_matrix = df[top_features].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Top 20 Features Correlation Matrix"
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        font=dict(color='#E5E7EB'),
        title=dict(font=dict(size=20))
    )
    
    return fig

# ==========================================
# MAIN UI COMPONENTS
# ==========================================

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 24px; margin: 0;">📊 ISTAT Lab</h1>
        <p style="color: #9CA3AF; font-size: 14px; margin-top: 5px;">
            Professional Nowcasting System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Upload Section
    st.markdown("### 📁 Data Upload")
    
    target_file = st.file_uploader(
        "Target Series (Monthly)",
        type=['csv', 'xlsx'],
        help="Upload unemployment rate data"
    )
    
    feature_files = st.file_uploader(
        "Feature Data",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        help="Upload predictor variables"
    )
    
    google_trends = st.file_uploader(
        "Google Trends (Optional)",
        type=['csv'],
        accept_multiple_files=True
    )
    
    st.markdown("---")
    
    # Quick Settings
    st.markdown("### ⚙️ Quick Settings")
    
    aggregation = st.selectbox(
        "Aggregation Method",
        ["mean", "median", "sum", "last"],
        index=0
    )
    
    correlation_lag = st.slider(
        "Max Correlation Lag",
        0, 12, 3,
        help="Check correlations with lags"
    )
    
    st.markdown("---")
    
    # Model Selection
    st.markdown("### 🤖 Models")
    
    use_ai = st.checkbox("Enable Claude AI", value=True)
    use_ensemble = st.checkbox("Auto Ensemble", value=True)
    use_shap = st.checkbox("SHAP Analysis", value=False)

# Main Content Area
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 48px; font-weight: 700; margin: 0;">
        Italy Unemployment Nowcasting
    </h1>
    <p style="color: #9CA3AF; font-size: 18px; margin-top: 10px;">
        Real-time economic forecasting with AI-powered insights
    </p>
</div>
""", unsafe_allow_html=True)

# Main Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔬 Analysis", "🎯 Correlations", "📈 Predictions"])

with tab1:
    # Dashboard Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Rate",
            value="7.8%",
            delta="-0.2%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Forecast (1M)",
            value="7.6%",
            delta="-0.2%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Model Accuracy",
            value="94.2%",
            delta="+1.3%"
        )
    
    with col4:
        st.metric(
            label="Data Quality",
            value="Good",
            delta="98%"
        )
    
    # Main Chart Placeholder
    st.markdown("### 📈 Unemployment Trend & Forecast")
    
    # Create sample data for demonstration
    dates = pd.date_range('2020-01-01', '2025-01-01', freq='M')
    actual = 8 + np.sin(np.arange(len(dates)) * 0.1) + np.random.normal(0, 0.2, len(dates))
    forecast = actual[-12:] + np.random.normal(0, 0.1, 12)
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='#6366F1', width=3)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=dates[-12:],
        y=forecast,
        mode='lines',
        name='Forecast',
        line=dict(color='#10B981', width=3, dash='dot')
    ))
    
    # Confidence bands
    upper = forecast + 0.5
    lower = forecast - 0.5
    
    fig.add_trace(go.Scatter(
        x=dates[-12:],
        y=upper,
        fill=None,
        mode='lines',
        line_color='rgba(16, 185, 129, 0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=dates[-12:],
        y=lower,
        fill='tonexty',
        mode='lines',
        line_color='rgba(16, 185, 129, 0)',
        name='Confidence Band',
        fillcolor='rgba(16, 185, 129, 0.2)'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🔬 Data Analysis")
    
    if AppState.get('data') is not None:
        # Data overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Data Overview")
            st.dataframe(
                AppState.get('data').head(),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Statistical Summary")
            st.dataframe(
                AppState.get('data').describe(),
                use_container_width=True
            )
    else:
        st.info("📤 Please upload data to begin analysis")

with tab3:
    st.markdown("### 🎯 Correlation Analysis")
    
    if target_file and feature_files:
        # Process uploaded files
        if st.button("Calculate Correlations", type="primary"):
            with st.spinner("Analyzing correlations..."):
                # Load target data
                target_df = pd.read_csv(target_file)
                AppState.set('target_series', target_df)
                
                # Load feature data
                feature_dfs = []
                for file in feature_files:
                    df = pd.read_csv(file)
                    feature_dfs.append(df)
                
                # Merge all data
                combined_df = target_df.copy()
                for df in feature_dfs:
                    combined_df = pd.merge(combined_df, df, on='date', how='outer')
                
                AppState.set('data', combined_df)
                
                # Calculate correlations
                if 'unemp_rate' in combined_df.columns:
                    correlations = calculate_instant_correlation(combined_df, 'unemp_rate')
                    AppState.set('correlations', correlations)
                    
                    # Display results
                    display_correlation_cards(correlations)
                    
                    # Show heatmap
                    st.markdown("### 🗺️ Correlation Heatmap")
                    fig = create_correlation_heatmap(combined_df, 'unemp_rate')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Lag correlation analysis
                    st.markdown("### ⏱️ Lag Correlation Analysis")
                    
                    lag_results = []
                    for lag in range(1, correlation_lag + 1):
                        lagged_corr = {}
                        for col in combined_df.columns:
                            if col != 'unemp_rate' and col != 'date':
                                lagged_series = combined_df[col].shift(lag)
                                corr_value = combined_df['unemp_rate'].corr(lagged_series)
                                lagged_corr[f"{col}_lag{lag}"] = corr_value
                        
                        top_lagged = pd.Series(lagged_corr).nlargest(3)
                        for feat, corr in top_lagged.items():
                            lag_results.append({
                                'Feature': feat.split('_lag')[0],
                                'Lag': lag,
                                'Correlation': corr
                            })
                    
                    if lag_results:
                        lag_df = pd.DataFrame(lag_results)
                        st.dataframe(
                            lag_df.style.background_gradient(subset=['Correlation']),
                            use_container_width=True
                        )
    else:
        st.info("📤 Please upload both target and feature data to analyze correlations")

with tab4:
    st.markdown("### 📈 Predictions & Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Performance")
        
        # Sample model comparison data
        models_df = pd.DataFrame({
            'Model': ['Ridge', 'XGBoost', 'LSTM', 'Ensemble', 'Prophet'],
            'MAE': [0.23, 0.19, 0.21, 0.18, 0.24],
            'RMSE': [0.31, 0.27, 0.29, 0.25, 0.33],
            'R²': [0.89, 0.92, 0.90, 0.93, 0.88]
        })
        
        fig = px.bar(
            models_df,
            x='Model',
            y='MAE',
            color='MAE',
            color_continuous_scale='Viridis',
            title='Model Comparison - MAE'
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Prediction Intervals")
        
        # Sample prediction interval chart
        dates_future = pd.date_range('2025-01-01', '2025-12-01', freq='M')
        predictions = 7.5 + np.random.normal(0, 0.1, len(dates_future))
        
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=dates_future,
            y=predictions,
            mode='lines+markers',
            name='Prediction',
            line=dict(color='#10B981', width=3)
        ))
        
        # Add confidence intervals
        for conf, color, alpha in [(0.95, '#6366F1', 0.2), (0.80, '#8B5CF6', 0.3)]:
            margin = np.random.normal(0.5, 0.1) * (1 - conf + 0.5)
            
            fig.add_trace(go.Scatter(
                x=dates_future,
                y=predictions + margin,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=dates_future,
                y=predictions - margin,
                mode='lines',
                line=dict(width=0),
                name=f'{int(conf*100)}% CI',
                fill='tonexty',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})'
            ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            title='2025 Forecasts with Confidence Intervals'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #9CA3AF;">
    <p>Professional Nowcasting System | Version 2.0 | Powered by Claude AI</p>
    <p style="font-size: 12px; margin-top: 10px;">
        © 2025 ISTAT Research Lab | Real-time Economic Intelligence
    </p>
</div>
""", unsafe_allow_html=True)
