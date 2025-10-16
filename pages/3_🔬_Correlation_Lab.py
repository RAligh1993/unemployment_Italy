"""
Correlation Lab - Advanced correlation analysis with instant feedback
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from scipy import stats
import seaborn as sns

st.set_page_config(page_title="Correlation Lab", page_icon="🔬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .correlation-matrix-container {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(16, 185, 129, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(99, 102, 241, 0.3);
    }
    
    .corr-value-high {
        color: #10B981;
        font-size: 24px;
        font-weight: 700;
    }
    
    .corr-value-med {
        color: #F59E0B;
        font-size: 24px;
        font-weight: 700;
    }
    
    .corr-value-low {
        color: #EF4444;
        font-size: 24px;
        font-weight: 700;
    }
    
    .analysis-card {
        background: rgba(30, 41, 59, 0.8);
        border-left: 4px solid #6366F1;
        padding: 15px;
        margin: 15px 0;
        border-radius: 10px;
    }
    
    .lag-indicator {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid #6366F1;
        border-radius: 15px;
        font-size: 12px;
        font-weight: 600;
        color: #6366F1;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'correlation_data' not in st.session_state:
    st.session_state.correlation_data = None
    
# Header
st.markdown("""
<h1 style="margin: 0;">🔬 Correlation Laboratory</h1>
<p style="color: #9CA3AF; margin-top: 10px;">
    Advanced correlation analysis with real-time feature selection and lag detection
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Data upload section
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file with time series data"
    )
    
    if uploaded_file:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.correlation_data = df
        
        # Basic info
        st.markdown("#### 📊 Dataset Overview")
        st.metric("Total Records", len(df))
        st.metric("Total Features", len(df.columns))
        st.metric("Missing Values", df.isna().sum().sum())
        
        # Target selection
        st.markdown("#### 🎯 Select Target Variable")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        target = st.selectbox(
            "Target variable",
            numeric_cols,
            index=0 if numeric_cols else None
        )
        
        # Feature selection
        st.markdown("#### ✨ Feature Selection")
        
        if st.checkbox("Select All Features"):
            st.session_state.selected_features = [col for col in numeric_cols if col != target]
        else:
            st.session_state.selected_features = st.multiselect(
                "Choose features to analyze",
                [col for col in numeric_cols if col != target],
                default=st.session_state.selected_features
            )

with col2:
    if uploaded_file and target and st.session_state.selected_features:
        st.markdown("### 🎯 Instant Correlation Analysis")
        
        # Calculate correlations
        correlations = {}
        for feature in st.session_state.selected_features:
            corr_value = df[target].corr(df[feature])
            correlations[feature] = corr_value
        
        # Sort by absolute correlation
        sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        
        # Display top correlations as cards
        st.markdown("#### 🏆 Top Correlated Features")
        
        cols = st.columns(3)
        for idx, (feature, corr) in enumerate(list(sorted_corr.items())[:6]):
            col_idx = idx % 3
            
            abs_corr = abs(corr)
            if abs_corr >= 0.7:
                corr_class = "corr-value-high"
                strength = "Strong"
                icon = "🔥"
            elif abs_corr >= 0.4:
                corr_class = "corr-value-med"
                strength = "Moderate"
                icon = "⚡"
            else:
                corr_class = "corr-value-low"
                strength = "Weak"
                icon = "💫"
            
            with cols[col_idx]:
                st.markdown(f"""
                <div class="feature-card">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <span style="font-size: 28px;">{icon}</span>
                        <span class="{corr_class}">{corr:.3f}</span>
                    </div>
                    <div style="margin-top: 10px;">
                        <div style="font-weight: 600; color: white; font-size: 14px;">
                            {feature[:25]}{'...' if len(feature) > 25 else ''}
                        </div>
                        <div style="color: #9CA3AF; font-size: 12px; margin-top: 5px;">
                            {strength} {'positive' if corr > 0 else 'negative'} correlation
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Correlation heatmap
        st.markdown("#### 🗺️ Correlation Matrix Heatmap")
        
        # Create correlation matrix
        selected_cols = [target] + st.session_state.selected_features[:20]  # Limit to 20 for readability
        corr_matrix = df[selected_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif not uploaded_file:
        st.info("👈 Please upload a dataset to begin correlation analysis")
    elif not target:
        st.warning("👈 Please select a target variable")
    else:
        st.warning("👈 Please select at least one feature to analyze")

# Advanced Analysis Section
if uploaded_file and target and st.session_state.selected_features:
    st.markdown("---")
    st.markdown("### 🔍 Advanced Correlation Analysis")
    
    tabs = st.tabs(["⏱️ Lag Analysis", "📊 Partial Correlation", "📈 Rolling Correlation", "🎯 Feature Importance"])
    
    with tabs[0]:
        # Lag correlation analysis
        st.markdown("#### Time-Lagged Correlations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            max_lag = st.slider("Maximum lag (periods)", 1, 24, 12)
            selected_feature = st.selectbox(
                "Select feature for lag analysis",
                st.session_state.selected_features
            )
        
        with col2:
            if selected_feature:
                # Calculate lag correlations
                lag_correlations = []
                for lag in range(0, max_lag + 1):
                    if lag == 0:
                        corr = df[target].corr(df[selected_feature])
                    else:
                        corr = df[target].corr(df[selected_feature].shift(lag))
                    lag_correlations.append({'Lag': lag, 'Correlation': corr})
                
                lag_df = pd.DataFrame(lag_correlations)
                
                # Plot lag correlations
                fig = px.line(
                    lag_df,
                    x='Lag',
                    y='Correlation',
                    markers=True,
                    title=f'Lag Correlation: {selected_feature} → {target}'
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.5)
                fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.5)
                
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Best lag indicator
                best_lag = lag_df.loc[lag_df['Correlation'].abs().idxmax()]
                st.markdown(f"""
                <div class="analysis-card">
                    <h4 style="color: #6366F1; margin: 0;">🎯 Optimal Lag Detection</h4>
                    <p style="margin-top: 10px;">
                        Best correlation found at <span class="lag-indicator">Lag {int(best_lag['Lag'])}</span>
                        with correlation coefficient of <strong>{best_lag['Correlation']:.3f}</strong>
                    </p>
                    <p style="color: #9CA3AF; font-size: 14px; margin-top: 10px;">
                        This suggests that {selected_feature} has the strongest predictive power for {target} 
                        when lagged by {int(best_lag['Lag'])} period(s).
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    with tabs[1]:
        # Partial correlation
        st.markdown("#### Partial Correlation Analysis")
        
        st.info("Partial correlation measures the relationship between two variables while controlling for other variables.")
        
        if len(st.session_state.selected_features) >= 2:
            feature_x = st.selectbox("Feature X", st.session_state.selected_features, key='partial_x')
            feature_y = st.selectbox("Feature Y", [f for f in st.session_state.selected_features if f != feature_x], key='partial_y')
            control_vars = st.multiselect(
                "Control variables",
                [f for f in st.session_state.selected_features if f not in [feature_x, feature_y]]
            )
            
            if st.button("Calculate Partial Correlation"):
                # Simple correlation
                simple_corr = df[feature_x].corr(df[feature_y])
                
                # Partial correlation (simplified calculation)
                if control_vars:
                    # This is a simplified version - in production, use proper partial correlation
                    from sklearn.linear_model import LinearRegression
                    
                    # Residualize X
                    X_model = LinearRegression()
                    X_model.fit(df[control_vars], df[feature_x])
                    X_residuals = df[feature_x] - X_model.predict(df[control_vars])
                    
                    # Residualize Y
                    Y_model = LinearRegression()
                    Y_model.fit(df[control_vars], df[feature_y])
                    Y_residuals = df[feature_y] - Y_model.predict(df[control_vars])
                    
                    # Partial correlation
                    partial_corr = pd.Series(X_residuals).corr(pd.Series(Y_residuals))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Simple Correlation", f"{simple_corr:.3f}")
                    with col2:
                        st.metric("Partial Correlation", f"{partial_corr:.3f}")
                    
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4 style="color: #10B981;">📊 Interpretation</h4>
                        <p>After controlling for {', '.join(control_vars)}:</p>
                        <ul>
                            <li>The correlation changed by {abs(partial_corr - simple_corr):.3f}</li>
                            <li>This {'strengthened' if abs(partial_corr) > abs(simple_corr) else 'weakened'} the relationship</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Please select at least one control variable")
    
    with tabs[2]:
        # Rolling correlation
        st.markdown("#### Rolling Correlation Analysis")
        
        window_size = st.slider("Window size", 10, 100, 30)
        selected_feature_roll = st.selectbox(
            "Select feature",
            st.session_state.selected_features,
            key='rolling_feature'
        )
        
        if selected_feature_roll:
            # Calculate rolling correlation
            rolling_corr = df[target].rolling(window=window_size).corr(df[selected_feature_roll])
            
            # Create figure
            fig = go.Figure()
            
            # Add rolling correlation line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=rolling_corr,
                mode='lines',
                name='Rolling Correlation',
                line=dict(color='#6366F1', width=2)
            ))
            
            # Add stability bands
            mean_corr = rolling_corr.mean()
            std_corr = rolling_corr.std()
            
            fig.add_hline(y=mean_corr, line_dash="dash", line_color="white", opacity=0.5)
            fig.add_hrect(
                y0=mean_corr - std_corr,
                y1=mean_corr + std_corr,
                fillcolor="rgba(99, 102, 241, 0.1)",
                line_width=0
            )
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                title=f"Rolling Correlation (window={window_size}): {selected_feature_roll} vs {target}",
                yaxis_title="Correlation",
                xaxis_title="Index"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stability analysis
            stability = 1 - (std_corr / (abs(mean_corr) + 0.001))
            st.markdown(f"""
            <div class="analysis-card">
                <h4 style="color: #F59E0B;">📈 Stability Analysis</h4>
                <div style="display: flex; gap: 20px; margin-top: 15px;">
                    <div>
                        <span style="color: #9CA3AF;">Mean Correlation:</span>
                        <span style="font-weight: 600; color: white;"> {mean_corr:.3f}</span>
                    </div>
                    <div>
                        <span style="color: #9CA3AF;">Std Deviation:</span>
                        <span style="font-weight: 600; color: white;"> {std_corr:.3f}</span>
                    </div>
                    <div>
                        <span style="color: #9CA3AF;">Stability Score:</span>
                        <span style="font-weight: 600; color: {'#10B981' if stability > 0.7 else '#F59E0B'};"> {stability:.1%}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[3]:
        # Feature importance
        st.markdown("#### Feature Importance Ranking")
        
        # Calculate multiple importance metrics
        importance_data = []
        for feature in st.session_state.selected_features:
            # Correlation
            corr = abs(df[target].corr(df[feature]))
            
            # Mutual information (simplified)
            from sklearn.feature_selection import mutual_info_regression
            mi = mutual_info_regression(
                df[[feature]].fillna(0),
                df[target].fillna(0),
                random_state=42
            )[0]
            
            # Variance
            variance = df[feature].var()
            
            importance_data.append({
                'Feature': feature,
                'Correlation': corr,
                'Mutual Info': mi,
                'Variance': variance,
                'Combined Score': (corr * 0.5 + mi * 0.3 + min(variance, 1) * 0.2)
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('Combined Score', ascending=False)
        
        # Display as bar chart
        fig = px.bar(
            importance_df.head(15),
            x='Combined Score',
            y='Feature',
            orientation='h',
            color='Combined Score',
            color_continuous_scale='Viridis',
            title='Feature Importance Ranking'
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top features summary
        st.markdown("#### 🏆 Top 5 Most Important Features")
        
        for idx, row in importance_df.head(5).iterrows():
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, rgba(99, 102, 241, 0.2), transparent);
                padding: 10px;
                margin: 5px 0;
                border-radius: 10px;
                border-left: 3px solid #6366F1;
            ">
                <strong>{row['Feature']}</strong>
                <div style="display: flex; gap: 20px; margin-top: 5px; font-size: 12px; color: #9CA3AF;">
                    <span>Corr: {row['Correlation']:.3f}</span>
                    <span>MI: {row['Mutual Info']:.3f}</span>
                    <span>Score: {row['Combined Score']:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Export results
if st.session_state.correlation_data is not None:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Correlation Matrix"):
            corr_matrix = df[st.session_state.selected_features].corr()
            csv = corr_matrix.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Generate Report"):
            st.info("Report generation feature coming soon!")
    
    with col3:
        if st.button("🔄 Reset Analysis"):
            st.session_state.selected_features = []
            st.session_state.correlation_data = None
            st.rerun()
