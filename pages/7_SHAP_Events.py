"""
🧭 SHAP & Events Pro v2.0
=======================================
Advanced model interpretation with SHAP values and timeline event analysis.
Features: Interactive visualizations, contribution analysis, event impact assessment.

Author: AI Assistant
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from typing import Optional, List, Dict, Tuple, Any

# Optional dependencies
try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False

try:
    from sklearn.linear_model import RidgeCV, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False
    st.error("⚠️ scikit-learn required. Install: pip install scikit-learn")
    st.stop()

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

try:
    from utils.state import AppState
except Exception:
    class _State:
        def __init__(self):
            self.y_monthly: Optional[pd.Series] = None
            self.panel_monthly: Optional[pd.DataFrame] = None
            self.bt_results: Dict[str, pd.Series] = {}
            self.bt_metrics: Optional[pd.DataFrame] = None
    
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

# Initialize events
if 'events' not in st.session_state:
    st.session_state.events = []

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SHAP & Events Analysis",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #7c3aed, #a855f7, #d946ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .section-header {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        color: white;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #E5E7EB;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .event-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #7c3aed;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .info-box {
        background: #F0F9FF;
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #FFF7ED;
        border: 2px solid #F97316;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive model metrics"""
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    }

def calculate_shap_values(model, X_train: np.ndarray, X_test: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """Calculate SHAP values with multiple fallback strategies"""
    
    if not HAS_SHAP:
        # Coefficient-based fallback
        coef = model.coef_
        if X_test is None:
            X_test = X_train
        shap_values = X_test * coef
        expected_value = model.intercept_
        return shap_values, expected_value
    
    try:
        # Try TreeExplainer for tree models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test if X_test is not None else X_train)
        expected_value = explainer.expected_value
    except:
        try:
            # Try LinearExplainer
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test if X_test is not None else X_train)
            expected_value = explainer.expected_value
        except:
            try:
                # Try KernelExplainer (slowest but most general)
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
                shap_values = explainer.shap_values(X_test if X_test is not None else X_train)
                expected_value = explainer.expected_value
            except:
                # Coefficient fallback
                coef = model.coef_
                if X_test is None:
                    X_test = X_train
                shap_values = X_test * coef
                expected_value = model.intercept_
    
    return shap_values, expected_value

def create_waterfall_chart(
    feature_names: List[str],
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    base_value: float,
    prediction: float,
    actual: float,
    top_n: int = 15
) -> go.Figure:
    """Create enhanced waterfall chart for SHAP contributions"""
    
    # Sort by absolute SHAP value
    idx = np.argsort(np.abs(shap_values))[::-1][:top_n]
    
    features = [feature_names[i] for i in idx]
    contributions = shap_values[idx]
    values = feature_values[idx]
    
    # Create waterfall data
    x_data = ['Base'] + features + ['Prediction', 'Actual']
    
    # Calculate cumulative values
    cumulative = [base_value]
    for contrib in contributions:
        cumulative.append(cumulative[-1] + contrib)
    cumulative.append(prediction)
    cumulative.append(actual)
    
    # Create figure
    fig = go.Figure()
    
    # Add base
    fig.add_trace(go.Bar(
        x=[x_data[0]],
        y=[base_value],
        marker=dict(color='lightgray'),
        name='Base',
        text=[f'{base_value:.3f}'],
        textposition='outside'
    ))
    
    # Add contributions
    for i, (feat, contrib, val) in enumerate(zip(features, contributions, values)):
        color = '#10B981' if contrib > 0 else '#EF4444'
        fig.add_trace(go.Bar(
            x=[feat],
            y=[abs(contrib)],
            base=[cumulative[i] if contrib > 0 else cumulative[i] - abs(contrib)],
            marker=dict(color=color),
            name=feat,
            text=[f'{contrib:+.3f}<br>val={val:.3f}'],
            textposition='outside',
            hovertemplate=f'<b>{feat}</b><br>Contribution: {contrib:+.4f}<br>Value: {val:.4f}<extra></extra>'
        ))
    
    # Add prediction bar
    fig.add_trace(go.Bar(
        x=['Prediction'],
        y=[prediction],
        marker=dict(color='#3B82F6'),
        name='Prediction',
        text=[f'{prediction:.3f}'],
        textposition='outside'
    ))
    
    # Add actual bar
    fig.add_trace(go.Bar(
        x=['Actual'],
        y=[actual],
        marker=dict(color='#1F2937'),
        name='Actual',
        text=[f'{actual:.3f}'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Contribution Waterfall Chart',
        xaxis_title='Features',
        yaxis_title='Value',
        template='plotly_white',
        height=600,
        showlegend=False,
        hovermode='x'
    )
    
    return fig

def create_force_plot_style(
    feature_names: List[str],
    shap_values: np.ndarray,
    base_value: float,
    prediction: float,
    top_n: int = 10
) -> go.Figure:
    """Create force plot style visualization"""
    
    # Get top positive and negative contributions
    pos_idx = np.where(shap_values > 0)[0]
    neg_idx = np.where(shap_values < 0)[0]
    
    pos_sorted = pos_idx[np.argsort(shap_values[pos_idx])[::-1]][:top_n]
    neg_sorted = neg_idx[np.argsort(shap_values[neg_idx])][:top_n]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Positive Impact', 'Negative Impact'),
        horizontal_spacing=0.1
    )
    
    # Positive contributions
    if len(pos_sorted) > 0:
        pos_features = [feature_names[i] for i in pos_sorted]
        pos_values = shap_values[pos_sorted]
        
        fig.add_trace(
            go.Bar(
                x=pos_values,
                y=pos_features,
                orientation='h',
                marker=dict(color='#10B981'),
                text=[f'{v:.4f}' for v in pos_values],
                textposition='outside'
            ),
            row=1, col=1
        )
    
    # Negative contributions
    if len(neg_sorted) > 0:
        neg_features = [feature_names[i] for i in neg_sorted]
        neg_values = shap_values[neg_sorted]
        
        fig.add_trace(
            go.Bar(
                x=neg_values,
                y=neg_features,
                orientation='h',
                marker=dict(color='#EF4444'),
                text=[f'{v:.4f}' for v in neg_values],
                textposition='outside'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=f'Force Plot: Base={base_value:.3f} → Prediction={prediction:.3f}',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig

def create_feature_interaction_plot(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature1: str,
    feature2: str
) -> go.Figure:
    """Create feature interaction scatter plot"""
    
    idx1 = X.columns.get_loc(feature1)
    idx2 = X.columns.get_loc(feature2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=X[feature1],
        y=X[feature2],
        mode='markers',
        marker=dict(
            size=10,
            color=shap_values[:, idx1],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title=f'SHAP<br>{feature1}'),
            line=dict(width=1, color='white')
        ),
        text=[f'{f1:.3f}, {f2:.3f}' for f1, f2 in zip(X[feature1], X[feature2])],
        hovertemplate='<b>%{text}</b><br>SHAP: %{marker.color:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Feature Interaction: {feature1} vs {feature2}',
        xaxis_title=feature1,
        yaxis_title=feature2,
        template='plotly_white',
        height=500
    )
    
    return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Header
st.markdown('<h1 class="main-title">🧭 SHAP & Events Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced model interpretation and timeline event analysis</p>', unsafe_allow_html=True)

# Check prerequisites
if not hasattr(state, 'y_monthly') or state.y_monthly is None or state.y_monthly.empty:
    st.error("⚠️ **No target data found**")
    st.info("👉 Please upload data in **Data & Aggregation** page first")
    st.stop()

if not hasattr(state, 'panel_monthly') or state.panel_monthly is None or state.panel_monthly.empty:
    st.error("⚠️ **No feature panel found**")
    st.info("👉 Please build feature panel in **Feature Engineering** page first")
    st.stop()

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Time window
    st.subheader("📅 Time Window")
    years = sorted(list(set(state.y_monthly.index.year)))
    
    start_year = st.select_slider(
        "Start year",
        options=years,
        value=years[max(0, len(years)-5)]
    )
    
    end_year = st.select_slider(
        "End year",
        options=years,
        value=years[-1]
    )
    
    # Feature selection
    st.subheader("📊 Features")
    numeric_cols = state.panel_monthly.select_dtypes(include=[np.number]).columns.tolist()
    
    n_features = st.slider(
        "Number of top features",
        min_value=5,
        max_value=min(50, len(numeric_cols)),
        value=min(20, len(numeric_cols))
    )
    
    selected_features = st.multiselect(
        "Select features (or auto-select top N)",
        options=numeric_cols,
        default=[]
    )
    
    # Ridge parameters
    st.subheader("🎯 Ridge Regression")
    
    alpha_min = st.number_input("Min alpha", value=0.01, format="%.4f")
    alpha_max = st.number_input("Max alpha", value=100.0, format="%.2f")
    n_alphas = st.slider("Number of alphas", 5, 50, 20)
    
    standardize = st.checkbox("Standardize features", value=True)
    
    # SHAP settings
    st.subheader("🔍 SHAP Settings")
    use_shap = st.checkbox(
        "Use SHAP" if HAS_SHAP else "SHAP not available",
        value=HAS_SHAP,
        disabled=not HAS_SHAP
    )
    
    if not HAS_SHAP:
        st.info("💡 Install shap: pip install shap")
    
    # Event settings
    st.subheader("📍 Event Analysis")
    
    if state.bt_results:
        model_for_events = st.selectbox(
            "Model for event analysis",
            options=['None'] + list(state.bt_results.keys())
        )
    else:
        model_for_events = 'None'
        st.info("No models available. Run backtesting first.")
    
    event_window_pre = st.slider("Pre-event window (months)", 1, 24, 6)
    event_window_post = st.slider("Post-event window (months)", 1, 24, 6)

# =============================================================================
# DATA PREPARATION
# =============================================================================

st.markdown('<div class="section-header">📊 Model Training & SHAP Calculation</div>', unsafe_allow_html=True)

# Filter by year
mask = (state.y_monthly.index.year >= start_year) & (state.y_monthly.index.year <= end_year)
y_filtered = state.y_monthly.loc[mask]
X_filtered = state.panel_monthly.loc[y_filtered.index]

# Feature selection
if selected_features:
    features_to_use = selected_features
else:
    # Auto-select top N features by correlation
    correlations = X_filtered.corrwith(y_filtered).abs().sort_values(ascending=False)
    features_to_use = correlations.head(n_features).index.tolist()

X_selected = X_filtered[features_to_use].select_dtypes(include=[np.number])

# Combine and drop NaN
combined = pd.concat([y_filtered.rename('target'), X_selected], axis=1).dropna()

if len(combined) < 12:
    st.error(f"❌ Insufficient data: {len(combined)} observations (need ≥12)")
    st.stop()

y_train = combined['target'].values
X_train_df = combined.drop('target', axis=1)
feature_names = X_train_df.columns.tolist()

# Standardization
if standardize:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
else:
    X_train = X_train_df.values
    scaler = None

# Train Ridge model
with st.spinner("🔄 Training Ridge model..."):
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)
    
    ridge = RidgeCV(alphas=alphas, store_cv_values=False, cv=5)
    ridge.fit(X_train, y_train)
    
    y_pred = ridge.predict(X_train)
    metrics = calculate_model_metrics(y_train, y_pred)

# Display metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("R² Score", f"{metrics['R2']:.4f}")
col2.metric("MAE", f"{metrics['MAE']:.4f}")
col3.metric("RMSE", f"{metrics['RMSE']:.4f}")
col4.metric("Selected α", f"{ridge.alpha_:.4f}")

# Calculate SHAP values
with st.spinner("🔄 Computing SHAP values..."):
    shap_values, expected_value = calculate_shap_values(ridge, X_train)

st.success(f"✅ Model trained on {len(y_train)} observations with {len(feature_names)} features")

# =============================================================================
# GLOBAL FEATURE IMPORTANCE
# =============================================================================

st.markdown('<div class="section-header">🌍 Global Feature Importance</div>', unsafe_allow_html=True)

# Calculate global importance
global_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': global_importance
}).sort_values('Importance', ascending=False)

col1, col2 = st.columns([2, 1])

with col1:
    # Bar chart
    fig_importance = go.Figure()
    
    top_features = importance_df.head(25)
    
    fig_importance.add_trace(go.Bar(
        x=top_features['Importance'],
        y=top_features['Feature'],
        orientation='h',
        marker=dict(
            color=top_features['Importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Importance')
        ),
        text=[f'{v:.4f}' for v in top_features['Importance']],
        textposition='outside'
    ))
    
    fig_importance.update_layout(
        title='Top 25 Features by Mean |SHAP|',
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='',
        template='plotly_white',
        height=700,
        showlegend=False
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)

with col2:
    st.markdown("#### 📊 Top Features Table")
    st.dataframe(
        importance_df.head(20).style.format({'Importance': '{:.6f}'}),
        use_container_width=True,
        height=650
    )

# =============================================================================
# LOCAL EXPLANATION
# =============================================================================

st.markdown('<div class="section-header">🎯 Local Explanation (Single Month)</div>', unsafe_allow_html=True)

# Month selection
dates_available = combined.index.tolist()
selected_date = st.select_slider(
    "Select month for detailed analysis:",
    options=dates_available,
    value=dates_available[-1]
)

month_idx = dates_available.index(selected_date)

# Get values for selected month
month_shap = shap_values[month_idx]
month_features = X_train_df.iloc[month_idx].values
month_pred = y_pred[month_idx]
month_actual = y_train[month_idx]

st.markdown(f"### Analysis for {selected_date.strftime('%Y-%m')}")

col1, col2, col3 = st.columns(3)
col1.metric("Actual Value", f"{month_actual:.4f}")
col2.metric("Predicted Value", f"{month_pred:.4f}")
col3.metric("Error", f"{month_actual - month_pred:+.4f}", delta=f"{((month_actual - month_pred)/month_actual)*100:+.2f}%")

# Tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["🌊 Waterfall Chart", "⚡ Force Plot", "📋 Contribution Table"])

with tab1:
    waterfall_fig = create_waterfall_chart(
        feature_names,
        month_shap,
        month_features,
        expected_value,
        month_pred,
        month_actual,
        top_n=15
    )
    st.plotly_chart(waterfall_fig, use_container_width=True)

with tab2:
    force_fig = create_force_plot_style(
        feature_names,
        month_shap,
        expected_value,
        month_pred,
        top_n=10
    )
    st.plotly_chart(force_fig, use_container_width=True)

with tab3:
    contribution_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': month_features,
        'SHAP': month_shap,
        'Abs SHAP': np.abs(month_shap)
    }).sort_values('Abs SHAP', ascending=False)
    
    st.dataframe(
        contribution_df.style.format({
            'Value': '{:.4f}',
            'SHAP': '{:+.4f}',
            'Abs SHAP': '{:.4f}'
        }).background_gradient(subset=['SHAP'], cmap='RdYlGn'),
        use_container_width=True,
        height=500
    )

# =============================================================================
# FEATURE INTERACTIONS
# =============================================================================

st.markdown('<div class="section-header">🔗 Feature Interactions</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    feature1 = st.selectbox("Select first feature:", feature_names, key='feat1')

with col2:
    feature2 = st.selectbox("Select second feature:", feature_names, key='feat2')

if feature1 and feature2 and feature1 != feature2:
    interaction_fig = create_feature_interaction_plot(
        X_train_df,
        shap_values,
        feature1,
        feature2
    )
    st.plotly_chart(interaction_fig, use_container_width=True)

# =============================================================================
# EVENTS TIMELINE
# =============================================================================

st.markdown('<div class="section-header">📍 Events Timeline & Impact Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Add New Event")
    
    event_col1, event_col2, event_col3 = st.columns([2, 2, 1])
    
    with event_col1:
        event_date = st.date_input("Event date:", key='new_event_date')
    
    with event_col2:
        event_label = st.text_input("Event label:", key='new_event_label')
    
    with event_col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("➕ Add", use_container_width=True):
            if event_label:
                new_event = {
                    'date': event_date.strftime('%Y-%m-%d'),
                    'label': event_label
                }
                st.session_state.events.append(new_event)
                st.success(f"✅ Added: {event_label}")
                st.rerun()

with col2:
    st.markdown("#### Import/Export Events")
    
    # Import
    upload_file = st.file_uploader("Import JSON", type=['json'], key='import_events')
    if upload_file:
        try:
            events_data = json.load(upload_file)
            if isinstance(events_data, list):
                st.session_state.events = events_data
                st.success(f"✅ Imported {len(events_data)} events")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Import failed: {str(e)}")
    
    # Export
    if st.session_state.events:
        events_json = json.dumps(st.session_state.events, indent=2)
        st.download_button(
            "📥 Export JSON",
            events_json.encode(),
            "events.json",
            "application/json",
            use_container_width=True
        )

# Display events
if st.session_state.events:
    st.markdown("#### Current Events")
    
    events_df = pd.DataFrame(st.session_state.events)
    events_df['date'] = pd.to_datetime(events_df['date'])
    events_df = events_df.sort_values('date')
    
    # Display with delete buttons
    for idx, event in events_df.iterrows():
        col1, col2, col3 = st.columns([2, 4, 1])
        
        with col1:
            st.markdown(f"**{event['date'].strftime('%Y-%m-%d')}**")
        
        with col2:
            st.markdown(f"{event['label']}")
        
        with col3:
            if st.button("🗑️", key=f"del_{idx}", help="Delete event"):
                st.session_state.events = [e for e in st.session_state.events 
                                         if e['date'] != event['date'].strftime('%Y-%m-%d') 
                                         or e['label'] != event['label']]
                st.rerun()
    
    # Timeline visualization
    st.markdown("#### Timeline Visualization")
    
    fig_timeline = go.Figure()
    
    # Add target line
    fig_timeline.add_trace(go.Scatter(
        x=state.y_monthly.index,
        y=state.y_monthly.values,
        mode='lines',
        name='Target',
        line=dict(color='#1F2937', width=3)
    ))
    
    # Add model predictions if available
    if model_for_events != 'None' and model_for_events in state.bt_results:
        pred_series = state.bt_results[model_for_events]
        fig_timeline.add_trace(go.Scatter(
            x=pred_series.index,
            y=pred_series.values,
            mode='lines',
            name=model_for_events,
            line=dict(color='#3B82F6', width=2)
        ))
    
    # Add event markers
    for _, event in events_df.iterrows():
        event_date = event['date']
        
        # Find closest date in target
        if event_date in state.y_monthly.index:
            event_value = state.y_monthly.loc[event_date]
        else:
            closest_idx = state.y_monthly.index.get_indexer([event_date], method='nearest')[0]
            event_value = state.y_monthly.iloc[closest_idx]
        
        fig_timeline.add_vline(
            x=event_date,
            line_dash="dash",
            line_color="#EF4444",
            line_width=2
        )
        
        fig_timeline.add_annotation(
            x=event_date,
            y=event_value,
            text=event['label'],
            showarrow=True,
            arrowhead=2,
            arrowcolor="#EF4444",
            arrowsize=1,
            arrowwidth=2,
            ax=0,
            ay=-40,
            bgcolor="white",
            bordercolor="#EF4444",
            borderwidth=2,
            borderpad=4,
            font=dict(size=10)
        )
    
    fig_timeline.update_layout(
        title='Timeline with Events',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Event impact analysis
    if model_for_events != 'None' and model_for_events in state.bt_results:
        st.markdown("#### Event Impact Analysis")
        
        model_series = state.bt_results[model_for_events]
        impact_results = []
        
        for _, event in events_df.iterrows():
            event_date = event['date']
            
            # Pre-event window
            pre_start = event_date - pd.DateOffset(months=event_window_pre)
            pre_mask = (state.y_monthly.index > pre_start) & (state.y_monthly.index <= event_date)
            
            # Post-event window
            post_end = event_date + pd.DateOffset(months=event_window_post)
            post_mask = (state.y_monthly.index > event_date) & (state.y_monthly.index <= post_end)
            
            # Calculate metrics
            y_pre = state.y_monthly.loc[pre_mask]
            p_pre = model_series.loc[model_series.index.isin(y_pre.index)]
            
            y_post = state.y_monthly.loc[post_mask]
            p_post = model_series.loc[model_series.index.isin(y_post.index)]
            
            if len(y_pre) > 0 and len(p_pre) > 0:
                mae_pre = mean_absolute_error(y_pre, p_pre)
            else:
                mae_pre = np.nan
            
            if len(y_post) > 0 and len(p_post) > 0:
                mae_post = mean_absolute_error(y_post, p_post)
            else:
                mae_post = np.nan
            
            impact_results.append({
                'Event': event['label'],
                'Date': event_date.strftime('%Y-%m-%d'),
                f'MAE Pre ({event_window_pre}m)': mae_pre,
                f'MAE Post ({event_window_post}m)': mae_post,
                'ΔMAE': mae_post - mae_pre,
                'Pre Obs': len(y_pre),
                'Post Obs': len(y_post)
            })
        
        impact_df = pd.DataFrame(impact_results)
        
        st.dataframe(
            impact_df.style.format({
                f'MAE Pre ({event_window_pre}m)': '{:.4f}',
                f'MAE Post ({event_window_post}m)': '{:.4f}',
                'ΔMAE': '{:+.4f}'
            }).background_gradient(subset=['ΔMAE'], cmap='RdYlGn_r'),
            use_container_width=True
        )

else:
    st.info("📍 No events added yet. Add events to see timeline analysis.")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    shap_status = "✅ Active" if HAS_SHAP else "❌ Not installed"
    st.caption(f"SHAP: {shap_status}")

with col2:
    st.caption(f"Features: {len(feature_names)}")

with col3:
    st.caption(f"Observations: {len(y_train)}")

st.markdown("""
<div style='text-align: center; color: #94A3B8; font-size: 0.875rem; margin-top: 1rem;'>
    💡 <b>Tip:</b> SHAP values provide model-agnostic interpretability. 
    Install shap package for advanced features: <code>pip install shap</code>
</div>
""", unsafe_allow_html=True)
