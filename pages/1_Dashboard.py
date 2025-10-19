"""
📊 Executive Dashboard Pro v2.0
====================================
Professional analytics dashboard with KPIs, trends, and quality metrics.
Features: Real-time metrics, interactive charts, model performance, data quality.

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
from typing import Optional, Dict, List, Tuple, Any

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
            self.google_trends: Optional[pd.DataFrame] = None
            self.raw_daily: List[pd.DataFrame] = []
    
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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_percentage_change(series: pd.Series, periods: int) -> float:
    """Calculate percentage change over periods"""
    if series is None or len(series) <= periods:
        return np.nan
    
    series = series.dropna()
    if len(series) <= periods:
        return np.nan
    
    try:
        current = series.iloc[-1]
        previous = series.iloc[-1 - periods]
        if previous == 0:
            return np.nan
        return float(100 * (current - previous) / abs(previous))
    except:
        return np.nan

def calculate_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate moving average"""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    
    min_periods = max(1, window // 2)
    return series.rolling(window, min_periods=min_periods).mean()

def calculate_trend(series: pd.Series, periods: int = 12) -> str:
    """Determine trend direction"""
    if series is None or len(series) < periods:
        return "neutral"
    
    recent = series.tail(periods)
    if recent.empty:
        return "neutral"
    
    # Linear regression slope
    x = np.arange(len(recent))
    y = recent.values
    
    if len(x) < 2:
        return "neutral"
    
    slope = np.polyfit(x, y, 1)[0]
    
    if slope > 0.01:
        return "up"
    elif slope < -0.01:
        return "down"
    else:
        return "neutral"

def calculate_volatility(series: pd.Series, window: int = 12) -> float:
    """Calculate rolling volatility"""
    if series is None or series.empty:
        return np.nan
    
    returns = series.pct_change().dropna()
    if len(returns) < window:
        return np.nan
    
    return float(returns.tail(window).std() * np.sqrt(12) * 100)  # Annualized %

def calculate_model_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """Calculate error metrics"""
    y_actual, y_pred = actual.align(predicted, join='inner')
    
    if y_actual.empty:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan}
    
    errors = y_actual - y_pred
    
    mae = float(errors.abs().mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    mape = float((errors.abs() / (y_actual.abs() + 1e-10)).mean() * 100)
    
    # R-squared
    ss_res = ((errors) ** 2).sum()
    ss_tot = ((y_actual - y_actual.mean()) ** 2).sum()
    r2 = float(1 - (ss_res / (ss_tot + 1e-10)))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def analyze_data_quality(series: pd.Series) -> Dict[str, Any]:
    """Comprehensive data quality analysis"""
    if series is None or series.empty:
        return {
            'total_points': 0,
            'missing_points': 0,
            'duplicate_dates': 0,
            'outliers': pd.DataFrame(),
            'quality_score': 0
        }
    
    # Total expected points
    date_range = pd.date_range(
        series.index.min(),
        series.index.max(),
        freq='M'
    )
    total_expected = len(date_range)
    
    # Missing points
    missing_dates = date_range.difference(series.index)
    missing_count = len(missing_dates)
    
    # Duplicates
    duplicate_count = series.index.duplicated().sum()
    
    # Outliers using MAD
    values = series.dropna().values
    if len(values) > 0:
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad > 0:
            z_scores = 0.6745 * (values - median) / mad
            outlier_mask = np.abs(z_scores) > 3.5
            outliers = pd.DataFrame({
                'date': series.dropna().index[outlier_mask],
                'value': values[outlier_mask],
                'z_score': np.abs(z_scores[outlier_mask])
            })
        else:
            outliers = pd.DataFrame()
    else:
        outliers = pd.DataFrame()
    
    # Quality score (0-100)
    missing_penalty = (missing_count / max(1, total_expected)) * 30
    duplicate_penalty = min(duplicate_count * 10, 20)
    outlier_penalty = min(len(outliers) * 5, 20)
    
    quality_score = max(0, 100 - missing_penalty - duplicate_penalty - outlier_penalty)
    
    return {
        'total_points': total_expected,
        'actual_points': len(series),
        'missing_points': missing_count,
        'missing_dates': missing_dates,
        'duplicate_dates': duplicate_count,
        'outliers': outliers,
        'quality_score': quality_score
    }

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_sparkline(series: pd.Series, color: str = '#3B82F6') -> go.Figure:
    """Create small sparkline chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
    ))
    
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_main_chart(
    target: pd.Series,
    predictions: Dict[str, pd.Series],
    ma_windows: List[int],
    window_years: int
) -> go.Figure:
    """Create main interactive chart"""
    
    # Filter by time window
    cutoff_date = target.index.max() - pd.DateOffset(years=window_years)
    target_filtered = target[target.index >= cutoff_date]
    
    fig = go.Figure()
    
    # Target line
    fig.add_trace(go.Scatter(
        x=target_filtered.index,
        y=target_filtered.values,
        mode='lines+markers',
        name='Actual',
        line=dict(color='#1E40AF', width=3),
        marker=dict(size=6),
        hovertemplate='<b>Date</b>: %{x|%Y-%m}<br><b>Value</b>: %{y:.3f}<extra></extra>'
    ))
    
    # Moving averages
    colors_ma = ['#F59E0B', '#10B981']
    for i, window in enumerate(ma_windows):
        ma = calculate_moving_average(target, window)
        ma_filtered = ma[ma.index >= cutoff_date]
        
        if not ma_filtered.empty:
            fig.add_trace(go.Scatter(
                x=ma_filtered.index,
                y=ma_filtered.values,
                mode='lines',
                name=f'MA{window}',
                line=dict(color=colors_ma[i % len(colors_ma)], width=2, dash='dash'),
                hovertemplate=f'<b>MA{window}</b>: %{{y:.3f}}<extra></extra>'
            ))
    
    # Predictions
    pred_colors = ['#8B5CF6', '#EC4899', '#F97316', '#14B8A6']
    for i, (name, pred) in enumerate(predictions.items()):
        pred_filtered = pred[pred.index >= cutoff_date]
        
        if not pred_filtered.empty:
            fig.add_trace(go.Scatter(
                x=pred_filtered.index,
                y=pred_filtered.values,
                mode='lines',
                name=name,
                line=dict(color=pred_colors[i % len(pred_colors)], width=2),
                hovertemplate=f'<b>{name}</b>: %{{y:.3f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Target Timeline with Predictions & Moving Averages',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig

def create_correlation_chart(correlations: pd.Series, top_n: int = 15) -> go.Figure:
    """Create correlation bar chart"""
    
    # Get top positive and negative
    top_corr = pd.concat([
        correlations.nsmallest(top_n // 2),
        correlations.nlargest(top_n // 2)
    ]).sort_values()
    
    colors = ['#EF4444' if x < 0 else '#10B981' for x in top_corr.values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_corr.values,
        y=top_corr.index,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=1)
        ),
        text=[f'{x:.3f}' for x in top_corr.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Correlation: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Feature Correlations with Target',
        xaxis_title='Correlation Coefficient',
        yaxis_title='',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig

def create_quality_gauge(score: float) -> go.Figure:
    """Create quality score gauge"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Data Quality Score", 'font': {'size': 16}},
        number={'suffix': '%', 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1E40AF"},
            'steps': [
                {'range': [0, 50], 'color': "#FEE2E2"},
                {'range': [50, 75], 'color': "#FEF3C7"},
                {'range': [75, 100], 'color': "#D1FAE5"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# =============================================================================
# UI CONFIGURATION
# =============================================================================



# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #1e40af, #3b82f6, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-change {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .section-header {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .info-card {
        background: #f3f4f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Header
st.markdown('<h1 class="main-title">📊 Executive Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time analytics, trends, and performance metrics</p>', unsafe_allow_html=True)

# Check for data
if state.y_monthly is None or state.y_monthly.empty:
    st.warning("⚠️ **No target data loaded**")
    st.info("👉 Please go to **Data & Aggregation** page to upload your monthly target variable.")
    st.stop()

# Clean data
target = state.y_monthly.dropna().sort_index()

# =============================================================================
# KEY METRICS
# =============================================================================

st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)

# Calculate metrics
latest_value = target.iloc[-1]
latest_date = target.index[-1].strftime('%Y-%m')

mom_change = calculate_percentage_change(target, 1)
qoq_change = calculate_percentage_change(target, 3)
yoy_change = calculate_percentage_change(target, 12)
volatility = calculate_volatility(target)
trend = calculate_trend(target)

# Get trend emoji
trend_emoji = {
    'up': '📈',
    'down': '📉',
    'neutral': '➡️'
}[trend]

# Create metric cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Latest Value",
        value=f"{latest_value:,.2f}",
        help=f"As of {latest_date}"
    )
    # Mini sparkline
    sparkline = create_sparkline(target.tail(24), '#3B82F6')
    st.plotly_chart(sparkline, use_container_width=True, config={'displayModeBar': False})

with col2:
    delta_color = "normal" if np.isnan(mom_change) else ("inverse" if mom_change < 0 else "normal")
    st.metric(
        label="Month-over-Month",
        value=f"{mom_change:.2f}%" if not np.isnan(mom_change) else "N/A",
        delta=f"{abs(mom_change):.2f}%" if not np.isnan(mom_change) else None,
        delta_color=delta_color
    )

with col3:
    delta_color = "normal" if np.isnan(qoq_change) else ("inverse" if qoq_change < 0 else "normal")
    st.metric(
        label="Quarter-over-Quarter",
        value=f"{qoq_change:.2f}%" if not np.isnan(qoq_change) else "N/A",
        delta=f"{abs(qoq_change):.2f}%" if not np.isnan(qoq_change) else None,
        delta_color=delta_color
    )

with col4:
    delta_color = "normal" if np.isnan(yoy_change) else ("inverse" if yoy_change < 0 else "normal")
    st.metric(
        label="Year-over-Year",
        value=f"{yoy_change:.2f}%" if not np.isnan(yoy_change) else "N/A",
        delta=f"{abs(yoy_change):.2f}%" if not np.isnan(yoy_change) else None,
        delta_color=delta_color
    )

with col5:
    st.metric(
        label=f"Trend {trend_emoji}",
        value=trend.capitalize(),
        delta=f"Volatility: {volatility:.1f}%" if not np.isnan(volatility) else None,
        help="12-month trend direction and annualized volatility"
    )

# Additional info
st.markdown(f"**Data Range:** {target.index.min().strftime('%Y-%m')} to {target.index.max().strftime('%Y-%m')} ({len(target)} observations)")

# =============================================================================
# MAIN CHART
# =============================================================================

st.markdown('<div class="section-header">📊 Interactive Timeline</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    window_years = st.slider(
        "Time window (years)",
        min_value=1,
        max_value=max(1, min(10, len(target) // 12)),
        value=3
    )

with col2:
    ma_short = st.number_input("Short MA", 2, 24, 3)

with col3:
    ma_long = st.number_input("Long MA", 6, 48, 12)

with col4:
    show_predictions = st.checkbox("Show predictions", value=True)

# Prepare predictions
predictions_to_show = {}
if show_predictions and state.bt_results:
    selected_models = st.multiselect(
        "Select models to display:",
        options=list(state.bt_results.keys()),
        default=list(state.bt_results.keys())[:3]
    )
    predictions_to_show = {k: v for k, v in state.bt_results.items() if k in selected_models}

# Create chart
main_chart = create_main_chart(
    target,
    predictions_to_show,
    [ma_short, ma_long],
    window_years
)
st.plotly_chart(main_chart, use_container_width=True)

# =============================================================================
# MODEL PERFORMANCE
# =============================================================================

if state.bt_results and len(state.bt_results) > 0:
    st.markdown('<div class="section-header">🎯 Model Performance</div>', unsafe_allow_html=True)
    
    # Calculate metrics for all models
    model_metrics = []
    for model_name, predictions in state.bt_results.items():
        metrics = calculate_model_metrics(target, predictions)
        metrics['Model'] = model_name
        model_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(model_metrics)
    metrics_df = metrics_df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2']]
    metrics_df = metrics_df.sort_values('MAE')
    
    # Display table
    st.dataframe(
        metrics_df.style.format({
            'MAE': '{:.4f}',
            'RMSE': '{:.4f}',
            'MAPE': '{:.2f}%',
            'R2': '{:.4f}'
        }).background_gradient(subset=['R2'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Best model highlight
    if len(metrics_df) > 0:
        best_model = metrics_df.iloc[0]
        st.success(f"🏆 **Best Model:** {best_model['Model']} (MAE: {best_model['MAE']:.4f}, R²: {best_model['R2']:.4f})")

# =============================================================================
# FEATURE CORRELATIONS
# =============================================================================

if state.panel_monthly is not None and not state.panel_monthly.empty:
    st.markdown('<div class="section-header">🔗 Feature Correlations</div>', unsafe_handle_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        lookback_months = st.slider(
            "Analysis window (months)",
            min_value=12,
            max_value=min(120, len(target)),
            value=36,
            step=12
        )
        
        top_n = st.slider(
            "Number of features",
            min_value=10,
            max_value=30,
            value=15,
            step=5
        )
    
    with col2:
        # Align and calculate correlations
        y_aligned, X_aligned = target.align(
            state.panel_monthly.select_dtypes(include=[np.number]),
            join='inner'
        )
        
        # Take only recent data
        y_recent = y_aligned.tail(lookback_months)
        X_recent = X_aligned.loc[y_recent.index]
        
        if X_recent.shape[1] > 0:
            correlations = X_recent.corrwith(y_recent).dropna().sort_values(ascending=False)
            
            if len(correlations) > 0:
                # Show chart
                corr_chart = create_correlation_chart(correlations, top_n)
                st.plotly_chart(corr_chart, use_container_width=True)
            else:
                st.info("No valid correlations found")
        else:
            st.info("No numeric features in panel")

# =============================================================================
# DATA QUALITY
# =============================================================================

st.markdown('<div class="section-header">✅ Data Quality Analysis</div>', unsafe_allow_html=True)

quality = analyze_data_quality(target)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Points", quality['actual_points'])

with col2:
    st.metric("Missing Points", quality['missing_points'])

with col3:
    st.metric("Duplicates", quality['duplicate_dates'])

with col4:
    st.metric("Outliers", len(quality['outliers']))

# Quality score gauge
col1, col2 = st.columns([1, 2])

with col1:
    gauge = create_quality_gauge(quality['quality_score'])
    st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})

with col2:
    # Quality interpretation
    score = quality['quality_score']
    
    if score >= 90:
        st.success("🌟 **Excellent data quality!** Your data is ready for modeling.")
    elif score >= 75:
        st.info("✅ **Good data quality.** Minor issues detected but generally reliable.")
    elif score >= 50:
        st.warning("⚠️ **Fair data quality.** Some issues need attention before modeling.")
    else:
        st.error("❌ **Poor data quality.** Significant issues detected. Data cleaning recommended.")
    
    # Details
    with st.expander("📋 Quality Details", expanded=False):
        if quality['missing_points'] > 0:
            st.write(f"**Missing dates:** {quality['missing_points']}")
            if len(quality['missing_dates']) > 0:
                missing_df = pd.DataFrame({
                    'Missing Date': quality['missing_dates'].strftime('%Y-%m')
                })
                st.dataframe(missing_df.head(10), use_container_width=True)
        
        if len(quality['outliers']) > 0:
            st.write(f"**Outliers detected:** {len(quality['outliers'])}")
            st.dataframe(
                quality['outliers'].style.format({'value': '{:.4f}', 'z_score': '{:.2f}'}),
                use_container_width=True
            )

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

st.markdown('<div class="section-header">📊 Summary Statistics</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Descriptive Statistics")
    stats = target.describe()
    stats_df = pd.DataFrame(stats).T
    st.dataframe(
        stats_df.style.format("{:.4f}"),
        use_container_width=True
    )

with col2:
    st.markdown("### Distribution")
    
    hist_fig = go.Figure()
    
    hist_fig.add_trace(go.Histogram(
        x=target.values,
        nbinsx=30,
        marker=dict(color='#3B82F6', line=dict(color='white', width=1)),
        name='Distribution'
    ))
    
    hist_fig.update_layout(
        title='Value Distribution',
        xaxis_title='Value',
        yaxis_title='Frequency',
        template='plotly_white',
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(hist_fig, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.caption(f"💾 Data points: {len(target):,} observations")

with col3:
    st.caption("💻 Built with Streamlit Pro")

st.markdown("---")
st.info("💡 **Next steps:** Go to **Feature Engineering** to create features, then **Backtesting** to train models.")
