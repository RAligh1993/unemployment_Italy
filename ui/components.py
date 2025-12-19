"""
Reusable UI Components for Streamlit
Professional dashboard elements
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go


class DashboardComponents:
    """
    Reusable dashboard components
    """
    
    @staticmethod
    def metric_card(label: str, 
                   value: float, 
                   delta: Optional[float] = None,
                   delta_label: str = "vs baseline",
                   format_str: str = ".4f",
                   help_text: Optional[str] = None) -> None:
        """
        Display metric card
        
        Args:
            label: Metric label
            value: Metric value
            delta: Change/comparison value (percentage)
            delta_label: Label for delta
            format_str: Format string
            help_text: Help tooltip
        """
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if np.isfinite(value):
                formatted_value = f"{value:{format_str}}"
            else:
                formatted_value = "N/A"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{formatted_value}</div>
                {f'<div class="metric-delta {"positive" if delta and delta < 0 else "negative"}">{delta:+.2f}% {delta_label}</div>' if delta is not None else ''}
            </div>
            """, unsafe_allow_html=True)
        
        if help_text:
            with col2:
                st.info(help_text)
    
    @staticmethod
    def comparison_table(metrics: Dict[str, Dict[str, float]],
                        highlight_best: bool = True,
                        metrics_to_show: Optional[List[str]] = None) -> None:
        """
        Display comparison table
        
        Args:
            metrics: Dict of {model_name: {metric: value}}
            highlight_best: Highlight best values
            metrics_to_show: List of metrics to display
        """
        if not metrics:
            st.warning("No metrics to display")
            return
        
        # Convert to dataframe
        df = pd.DataFrame(metrics).T
        df.index.name = 'Model'
        df.reset_index(inplace=True)
        
        # Filter metrics
        if metrics_to_show:
            cols = ['Model'] + [m for m in metrics_to_show if m in df.columns]
            df = df[cols]
        
        # Format numeric columns
        for col in df.columns:
            if col != 'Model' and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if np.isfinite(x) else "N/A")
        
        # Highlight best
        if highlight_best:
            def highlight_min(s):
                if s.name == 'Model':
                    return [''] * len(s)
                try:
                    numeric_vals = pd.to_numeric(s, errors='coerce')
                    is_min = numeric_vals == numeric_vals.min()
                    return ['background-color: #d4edda; font-weight: bold' if v else '' for v in is_min]
                except:
                    return [''] * len(s)
            
            styled_df = df.style.apply(highlight_min, axis=0)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def statistical_test_badge(test_name: str,
                               p_value: float,
                               statistic: float,
                               threshold: float = 0.05) -> None:
        """
        Display statistical test result as badge
        
        Args:
            test_name: Test name
            p_value: P-value
            statistic: Test statistic
            threshold: Significance threshold
        """
        is_significant = p_value < threshold
        
        if is_significant:
            badge_color = "#28A745"
            badge_text = "SIGNIFICANT"
            stars = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*"
        else:
            badge_color = "#DC3545"
            badge_text = "NOT SIGNIFICANT"
            stars = ""
        
        st.markdown(f"""
        <div style="
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid {badge_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #003366; font-size: 1.1rem;">{test_name}</strong><br>
                    <span style="color: #6c757d;">Statistic: {statistic:.4f}</span><br>
                    <span style="color: #6c757d;">P-value: {p_value:.4f} {stars}</span>
                </div>
                <div>
                    <span style="
                        background-color: {badge_color};
                        color: white;
                        padding: 0.5rem 1rem;
                        border-radius: 4px;
                        font-weight: bold;
                        font-size: 0.875rem;
                    ">{badge_text}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def progress_bar(label: str, 
                    value: float, 
                    max_value: float = 100,
                    color: str = "#003366") -> None:
        """
        Display custom progress bar
        
        Args:
            label: Progress label
            value: Current value
            max_value: Maximum value
            color: Bar color
        """
        percentage = min(100, (value / max_value) * 100)
        
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="font-weight: 500;">{label}</span>
                <span style="color: #6c757d;">{value:.1f} / {max_value:.1f}</span>
            </div>
            <div style="
                background-color: #E0E0E0;
                border-radius: 4px;
                height: 8px;
                overflow: hidden;
            ">
                <div style="
                    background-color: {color};
                    width: {percentage}%;
                    height: 100%;
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def info_box(title: str, 
                content: str, 
                box_type: str = "info") -> None:
        """
        Display styled info box
        
        Args:
            title: Box title
            content: Box content
            box_type: Type (info, success, warning, error)
        """
        colors = {
            'info': ('#17A2B8', '#d1ecf1'),
            'success': ('#28A745', '#d4edda'),
            'warning': ('#FFC107', '#fff3cd'),
            'error': ('#DC3545', '#f8d7da')
        }
        
        border_color, bg_color = colors.get(box_type, colors['info'])
        
        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            border-left: 4px solid {border_color};
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        ">
            <strong style="color: {border_color}; font-size: 1.1rem;">{title}</strong><br>
            <p style="margin: 0.5rem 0 0 0; color: #495057;">{content}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def accordion_section(title: str, content: str, expanded: bool = False) -> None:
        """
        Display accordion/expander section
        
        Args:
            title: Section title
            content: Section content
            expanded: Start expanded
        """
        with st.expander(title, expanded=expanded):
            st.markdown(content)
    
    @staticmethod
    def feature_pill(feature_name: str, 
                    importance: float,
                    color: Optional[str] = None) -> str:
        """
        Generate HTML for feature importance pill
        
        Args:
            feature_name: Feature name
            importance: Importance score
            color: Custom color
        
        Returns:
            HTML string
        """
        if color is None:
            color = "#003366" if importance > 0 else "#DC3545"
        
        return f"""
        <span style="
            display: inline-block;
            background-color: {color};
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            margin: 0.25rem;
            font-size: 0.875rem;
            font-weight: 500;
        ">
            {feature_name}: {importance:.3f}
        </span>
        """
    
    @staticmethod
    def collapsible_code(code: str, language: str = "python") -> None:
        """
        Display collapsible code block
        
        Args:
            code: Code string
            language: Programming language
        """
        with st.expander("📄 View Code", expanded=False):
            st.code(code, language=language)
    
    @staticmethod
    def data_quality_indicator(quality_score: float,
                               label: str = "Data Quality") -> None:
        """
        Display data quality indicator
        
        Args:
            quality_score: Quality score (0-100)
            label: Label text
        """
        if quality_score >= 80:
            color = "#28A745"
            status = "EXCELLENT"
        elif quality_score >= 60:
            color = "#FFC107"
            status = "GOOD"
        elif quality_score >= 40:
            color = "#FF6B35"
            status = "FAIR"
        else:
            color = "#DC3545"
            status = "POOR"
        
        st.markdown(f"""
        <div style="
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #6c757d; font-size: 0.875rem;">{label}</span><br>
                    <strong style="color: #003366; font-size: 1.5rem;">{quality_score:.1f}%</strong>
                </div>
                <div style="
                    background-color: {color};
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 4px;
                    font-weight: bold;
                ">{status}</div>
            </div>
            <div style="
                margin-top: 0.5rem;
                background-color: #E0E0E0;
                border-radius: 4px;
                height: 6px;
            ">
                <div style="
                    background-color: {color};
                    width: {quality_score}%;
                    height: 100%;
                    border-radius: 4px;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def timeline_step(step_number: int,
                     title: str,
                     description: str,
                     status: str = "pending") -> None:
        """
        Display timeline step
        
        Args:
            step_number: Step number
            title: Step title
            description: Step description
            status: Status (pending, active, completed)
        """
        status_colors = {
            'pending': '#6c757d',
            'active': '#FFC107',
            'completed': '#28A745'
        }
        
        color = status_colors.get(status, '#6c757d')
        icon = "✓" if status == "completed" else str(step_number)
        
        st.markdown(f"""
        <div style="
            display: flex;
            margin-bottom: 1.5rem;
        ">
            <div style="
                background-color: {color};
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-right: 1rem;
                flex-shrink: 0;
            ">{icon}</div>
            <div style="flex-grow: 1;">
                <strong style="color: #003366; font-size: 1.1rem;">{title}</strong><br>
                <span style="color: #6c757d;">{description}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def spinner_with_message(message: str = "Processing..."):
        """
        Display spinner with custom message
        
        Args:
            message: Loading message
        """
        return st.spinner(f"⏳ {message}")


class WelcomeScreen:
    """
    Welcome screen component
    """
    
    @staticmethod
    def render():
        """Render welcome screen"""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h1 style="color: white; margin: 0;">🚀 Welcome to Nowcasting Platform</h1>
            <p style="font-size: 1.2rem; margin: 1rem 0 0 0;">
                Professional forecasting tool for economists, data scientists, and policymakers
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="
                background-color: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            ">
                <div style="font-size: 3rem;">📊</div>
                <h3 style="color: #003366;">Auto-Detection</h3>
                <p style="color: #6c757d;">
                    Automatic frequency detection, target identification, and data validation
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="
                background-color: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            ">
                <div style="font-size: 3rem;">🤖</div>
                <h3 style="color: #003366;">Multiple Models</h3>
                <p style="color: #6c757d;">
                    Compare benchmarks, ML models, and specialized forecasting methods
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="
                background-color: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            ">
                <div style="font-size: 3rem;">📈</div>
                <h3 style="color: #003366;">Statistical Tests</h3>
                <p style="color: #6c757d;">
                    Rigorous evaluation with DM, CW tests, and bootstrap confidence intervals
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📋 How to Use:
        
        1. **Upload Data** → Upload your target variable (CSV/Excel)
        2. **Add Features** → Optionally add exogenous variables and alternative data
        3. **Configure** → Select models and evaluation settings
        4. **Run Analysis** → Execute nowcasting pipeline
        5. **Export Results** → Download comprehensive results package
        """)
        
        st.info("💡 **Tip:** The platform handles mixed frequencies automatically using MIDAS aggregation.")


class DataPreviewComponent:
    """
    Data preview and diagnostics component
    """
    
    @staticmethod
    def render(dataset_info: Any) -> None:
        """
        Render data preview
        
        Args:
            dataset_info: DatasetInfo object
        """
        st.subheader("📊 Data Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Observations", dataset_info.n_obs)
        
        with col2:
            st.metric("Frequency", dataset_info.frequency.upper())
        
        with col3:
            st.metric("Features", len(dataset_info.feature_cols))
        
        with col4:
            quality = "✅ Valid" if dataset_info.validation['valid'] else "❌ Invalid"
            st.metric("Status", quality)
        
        # Date range
        try:
            st.markdown(f"""
            **Date Range:** {dataset_info.date_range[0].strftime('%Y-%m-%d')} to {dataset_info.date_range[1].strftime('%Y-%m-%d')}
            """)
        except:
            st.markdown("**Date Range:** Not available")
        
        # Validation warnings
        if dataset_info.validation.get('warnings'):
            with st.expander("⚠️ Warnings", expanded=False):
                for warning in dataset_info.validation['warnings']:
                    st.warning(warning)
        
        # Validation errors
        if dataset_info.validation.get('errors'):
            with st.expander("❌ Errors", expanded=True):
                for error in dataset_info.validation['errors']:
                    st.error(error)
        
        # Data preview
        with st.expander("👁️ View Data", expanded=False):
            st.dataframe(dataset_info.df.head(20), use_container_width=True)


class ModelConfigurationComponent:
    """
    Model configuration component
    """
    
    @staticmethod
    def render() -> Dict:
        """
        Render model configuration UI
        
        Returns:
            Configuration dict
        """
        st.subheader("🤖 Model Configuration")
        
        config = {}
        
        # Benchmark models (always included)
        st.markdown("**📊 Benchmark Models** (Always Included)")
        st.info("Persistence, Historical Mean, AR(1), AR(2)")
        
        # ML models
        st.markdown("**🔬 Machine Learning Models**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            config['use_ridge'] = st.checkbox("Ridge Regression", value=True)
            if config['use_ridge']:
                config['ridge_alphas'] = st.multiselect(
                    "Ridge α values",
                    [1, 10, 50, 100],
                    default=[10, 50]
                )
            
            config['use_lasso'] = st.checkbox("Lasso Regression", value=True)
            if config['use_lasso']:
                config['lasso_alphas'] = st.multiselect(
                    "Lasso α values",
                    [0.01, 0.1, 1.0],
                    default=[0.1, 1.0]
                )
        
        with col2:
            config['use_elastic'] = st.checkbox("Elastic Net", value=False)
            
            config['use_delta_correction'] = st.checkbox("Delta Correction", value=True)
            if config['use_delta_correction']:
                config['blend_weights'] = st.multiselect(
                    "Blending weights (w)",
                    [0.5, 0.7, 0.9, 1.0],
                    default=[0.9, 1.0]
                )
        
        # MIDAS (if alternative data present)
        if st.checkbox("Alternative Data Available", value=False):
            config['use_midas'] = st.checkbox("MIDAS Aggregation", value=True)
            if config['use_midas']:
                config['midas_windows'] = st.multiselect(
                    "MIDAS windows",
                    [4, 8, 12],
                    default=[4, 8]
                )
                config['midas_lambdas'] = st.multiselect(
                    "Exponential λ",
                    [0.6, 0.8],
                    default=[0.6, 0.8]
                )
        
        return config


class ResultsDashboard:
    """
    Results dashboard component
    """
    
    @staticmethod
    def render(results: Dict) -> None:
        """
        Render results dashboard
        
        Args:
            results: Results dictionary
        """
        st.subheader("📈 Results Dashboard")
        
        # Best model identification
        metrics = results.get('metrics', {})
        if metrics:
            rmse_scores = {model: m.get('rmse', np.inf) for model, m in metrics.items()}
            best_model = min(rmse_scores, key=rmse_scores.get)
            
            st.success(f"🏆 **Best Model:** {best_model} (RMSE: {rmse_scores[best_model]:.4f})")
        
        # Key metrics
        st.markdown("### 📊 Key Metrics")
        
        if best_model and best_model in metrics:
            best_metrics = metrics[best_model]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                DashboardComponents.metric_card(
                    "RMSE",
                    best_metrics.get('rmse', np.nan),
                    format_str=".4f"
                )
            
            with col2:
                DashboardComponents.metric_card(
                    "MAE",
                    best_metrics.get('mae', np.nan),
                    format_str=".4f"
                )
            
            with col3:
                DashboardComponents.metric_card(
                    "Direction Accuracy",
                    best_metrics.get('direction_accuracy', np.nan),
                    format_str=".1f"
                )
            
            with col4:
                DashboardComponents.metric_card(
                    "Theil's U",
                    best_metrics.get('theil_u', np.nan),
                    format_str=".3f"
                )
        
        # Statistical tests
        test_results = results.get('statistical_tests', {})
        if test_results:
            st.markdown("### 📊 Statistical Tests")
            
            for test_name, test_result in test_results.items():
                DashboardComponents.statistical_test_badge(
                    test_name,
                    test_result.get('p_value', 1.0),
                    test_result.get('statistic', 0.0)
                )
