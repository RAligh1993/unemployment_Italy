"""
Dashboard Page - Real-time Overview and KPIs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# Apply consistent theme
st.markdown("""
<style>
    .dashboard-kpi {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(16, 185, 129, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .dashboard-kpi:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }
    
    .kpi-value {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .kpi-label {
        font-size: 14px;
        color: #9CA3AF;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .real-time-badge {
        display: inline-block;
        padding: 5px 15px;
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# Header with real-time indicator
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("""
    <h1 style="margin: 0;">📊 Executive Dashboard</h1>
    <p style="color: #9CA3AF; margin-top: 10px;">
        Real-time monitoring of Italy's unemployment indicators
    </p>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: right; padding-top: 20px;">
        <span class="real-time-badge">● REAL-TIME</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    current_time = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="text-align: right; padding-top: 25px; color: #9CA3AF;">
        Last Update: {current_time}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# KPI Section
st.markdown("### 🎯 Key Performance Indicators")

kpi_cols = st.columns(5)

kpis = [
    {"label": "Current Rate", "value": "7.8%", "delta": "-0.2%", "icon": "📉"},
    {"label": "Youth (15-24)", "value": "22.3%", "delta": "-1.1%", "icon": "👥"},
    {"label": "North Italy", "value": "5.2%", "delta": "-0.3%", "icon": "🗺️"},
    {"label": "South Italy", "value": "15.8%", "delta": "-0.5%", "icon": "📍"},
    {"label": "Forecast Accuracy", "value": "94.2%", "delta": "+1.3%", "icon": "🎯"}
]

for idx, (col, kpi) in enumerate(zip(kpi_cols, kpis)):
    with col:
        # Determine color based on delta
        delta_color = "#10B981" if kpi["delta"].startswith("-") or kpi["delta"].startswith("+1") else "#EF4444"
        
        st.markdown(f"""
        <div class="dashboard-kpi">
            <div style="font-size: 32px; margin-bottom: 10px;">{kpi["icon"]}</div>
            <div class="kpi-value">{kpi["value"]}</div>
            <div style="color: {delta_color}; font-size: 18px; font-weight: 600; margin-top: 5px;">
                {kpi["delta"]}
            </div>
            <div class="kpi-label">{kpi["label"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Main Charts Section
st.markdown("---")
st.markdown("### 📈 Trend Analysis & Forecasting")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Historical Trend", "🔮 Predictions", "🌍 Regional Analysis", "📰 Market Signals"])

with tab1:
    # Historical trend with multiple series
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Generate sample data
        dates = pd.date_range('2018-01-01', '2025-01-01', freq='M')
        unemployment = 10 + np.sin(np.arange(len(dates)) * 0.05) * 2 + np.random.normal(0, 0.3, len(dates))
        youth = unemployment * 2.8 + np.random.normal(0, 0.5, len(dates))
        
        fig = go.Figure()
        
        # Main unemployment rate
        fig.add_trace(go.Scatter(
            x=dates,
            y=unemployment,
            mode='lines',
            name='Total Unemployment',
            line=dict(color='#6366F1', width=3),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ))
        
        # Youth unemployment
        fig.add_trace(go.Scatter(
            x=dates,
            y=youth,
            mode='lines',
            name='Youth (15-24)',
            line=dict(color='#F59E0B', width=2)
        ))
        
        # Add events/annotations
        events = [
            {'date': '2020-03-01', 'text': 'COVID-19 Lockdown', 'y': 12},
            {'date': '2021-07-01', 'text': 'Recovery Plan', 'y': 11},
            {'date': '2024-01-01', 'text': 'New Policies', 'y': 8}
        ]
        
        for event in events:
            fig.add_vline(
                x=pd.Timestamp(event['date']),
                line_width=1,
                line_dash="dash",
                line_color="rgba(255, 255, 255, 0.3)"
            )
            fig.add_annotation(
                x=event['date'],
                y=event['y'],
                text=event['text'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(255, 255, 255, 0.5)",
                font=dict(size=12, color="white"),
                bgcolor="rgba(99, 102, 241, 0.8)",
                bordercolor="rgba(99, 102, 241, 1)",
                borderwidth=1
            )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            hovermode='x unified',
            title="Historical Unemployment Trends (2018-2025)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Quick Stats")
        
        stats = [
            ("Avg 2024", "7.9%"),
            ("Min 2024", "7.2%"),
            ("Max 2024", "8.4%"),
            ("Std Dev", "0.31%"),
            ("Trend", "↓ Declining")
        ]
        
        for label, value in stats:
            st.markdown(f"""
            <div style="
                background: rgba(30, 41, 59, 0.5);
                border-left: 3px solid #6366F1;
                padding: 12px;
                margin: 10px 0;
                border-radius: 8px;
            ">
                <div style="color: #9CA3AF; font-size: 12px;">{label}</div>
                <div style="color: white; font-size: 18px; font-weight: 600; margin-top: 5px;">
                    {value}
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    # Model predictions comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction fan chart
        dates_future = pd.date_range('2025-01-01', '2026-01-01', freq='M')
        base_forecast = 7.5 + np.sin(np.arange(len(dates_future)) * 0.1) * 0.5
        
        fig = go.Figure()
        
        # Add multiple confidence bands
        for pct, color, alpha in [(95, '#6366F1', 0.1), (80, '#8B5CF6', 0.2), (50, '#10B981', 0.3)]:
            margin = (100 - pct) / 20
            upper = base_forecast + margin
            lower = base_forecast - margin
            
            fig.add_trace(go.Scatter(
                x=dates_future,
                y=upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=dates_future,
                y=lower,
                mode='lines',
                line=dict(width=0),
                name=f'{pct}% CI',
                fill='tonexty',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})'
            ))
        
        # Central forecast
        fig.add_trace(go.Scatter(
            x=dates_future,
            y=base_forecast,
            mode='lines+markers',
            name='Central Forecast',
            line=dict(color='white', width=3),
            marker=dict(size=8, color='#10B981')
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            title="2025 Unemployment Forecast with Confidence Bands",
            yaxis_title="Unemployment Rate (%)",
            xaxis_title=""
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model performance comparison
        models_performance = pd.DataFrame({
            'Model': ['XGBoost', 'LSTM', 'Ridge ARX', 'Ensemble', 'Prophet'],
            'MAE': [0.18, 0.21, 0.23, 0.16, 0.24],
            'MAPE': [2.3, 2.7, 2.9, 2.1, 3.1]
        })
        
        fig = px.scatter(
            models_performance,
            x='MAE',
            y='MAPE',
            size=[100, 80, 70, 120, 60],
            color='Model',
            title='Model Performance Comparison',
            labels={'MAE': 'Mean Absolute Error', 'MAPE': 'MAPE (%)'},
            color_discrete_sequence=['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Regional analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Map placeholder (would use folium or similar in production)
        regions_data = pd.DataFrame({
            'Region': ['Lombardia', 'Lazio', 'Campania', 'Sicilia', 'Veneto', 'Piemonte', 'Emilia-Romagna', 'Toscana'],
            'Rate': [5.1, 7.8, 16.4, 18.2, 4.8, 6.9, 4.5, 6.2],
            'Change': [-0.3, -0.2, -0.8, -1.1, -0.2, -0.4, -0.1, -0.3]
        })
        
        fig = px.bar(
            regions_data.sort_values('Rate'),
            x='Rate',
            y='Region',
            orientation='h',
            color='Rate',
            color_continuous_scale='RdYlGn_r',
            title='Unemployment Rate by Region',
            text='Rate'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            xaxis_title="Unemployment Rate (%)",
            yaxis_title="",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🗺️ Regional Insights")
        
        # Regional summary cards
        for _, row in regions_data.nlargest(3, 'Rate').iterrows():
            color = "#EF4444" if row['Rate'] > 10 else "#F59E0B"
            st.markdown(f"""
            <div style="
                background: rgba(30, 41, 59, 0.5);
                border-left: 3px solid {color};
                padding: 15px;
                margin: 15px 0;
                border-radius: 10px;
            ">
                <div style="font-weight: 600; color: white; font-size: 16px;">
                    {row['Region']}
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                    <span style="color: {color}; font-size: 20px; font-weight: 700;">
                        {row['Rate']}%
                    </span>
                    <span style="color: #10B981; font-size: 14px;">
                        {row['Change']}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab4:
    # Market signals and indicators
    st.markdown("#### 📰 Real-time Market Signals")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Google Trends indicator
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 15px;
            padding: 20px;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 24px; margin-right: 10px;">🔍</span>
                <span style="font-weight: 600; color: white;">Google Trends</span>
            </div>
            <div style="font-size: 28px; font-weight: 700; color: #6366F1;">
                +12%
            </div>
            <div style="color: #9CA3AF; font-size: 12px; margin-top: 5px;">
                "Unemployment benefits" searches
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # News sentiment
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 15px;
            padding: 20px;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 24px; margin-right: 10px;">📰</span>
                <span style="font-weight: 600; color: white;">News Sentiment</span>
            </div>
            <div style="font-size: 28px; font-weight: 700; color: #10B981;">
                Positive
            </div>
            <div style="color: #9CA3AF; font-size: 12px; margin-top: 5px;">
                68% positive coverage
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Job postings
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(217, 119, 6, 0.1));
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 15px;
            padding: 20px;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 24px; margin-right: 10px;">💼</span>
                <span style="font-weight: 600; color: white;">Job Postings</span>
            </div>
            <div style="font-size: 28px; font-weight: 700; color: #F59E0B;">
                -5%
            </div>
            <div style="color: #9CA3AF; font-size: 12px; margin-top: 5px;">
                Monthly change
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Leading indicators chart
    st.markdown("#### 📊 Leading Indicators Timeline")
    
    dates = pd.date_range('2024-01-01', '2025-01-01', freq='W')
    
    # Create multiple indicator series
    indicators = pd.DataFrame({
        'date': dates,
        'Google Trends': np.cumsum(np.random.randn(len(dates))) + 100,
        'Job Postings': np.cumsum(np.random.randn(len(dates)) * 0.8) + 95,
        'News Sentiment': np.cumsum(np.random.randn(len(dates)) * 0.6) + 102,
        'Consumer Confidence': np.cumsum(np.random.randn(len(dates)) * 0.7) + 98
    })
    
    fig = go.Figure()
    
    colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444']
    
    for idx, col in enumerate(indicators.columns[1:]):
        fig.add_trace(go.Scatter(
            x=indicators['date'],
            y=indicators[col],
            mode='lines',
            name=col,
            line=dict(color=colors[idx], width=2)
        ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        hovermode='x unified',
        yaxis_title="Index (Base=100)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Auto-refresh indicator
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #9CA3AF;">
    <p style="font-size: 12px;">
        Dashboard auto-refreshes every 60 seconds | Next update in <span id="countdown">60</span>s
    </p>
</div>
""", unsafe_allow_html=True)

# Add refresh functionality
if st.checkbox("Enable auto-refresh", value=False):
    time.sleep(60)
    st.rerun()
