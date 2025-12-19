"""
Professional Chart Library
High-quality visualizations for nowcasting platform
Production-ready Plotly charts with themes
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG


class ChartTheme:
    """
    Professional color themes and styling
    """
    
    # ISTAT Theme
    ISTAT = {
        'primary': '#003366',      # Navy blue
        'secondary': '#FF6B35',    # Coral orange
        'success': '#28A745',      # Green
        'warning': '#FFC107',      # Amber
        'danger': '#DC3545',       # Red
        'info': '#17A2B8',         # Cyan
        'light': '#F8F9FA',        # Light gray
        'dark': '#343A40',         # Dark gray
        'background': '#FFFFFF',   # White
        'grid': '#E0E0E0',         # Light gray
        'text': '#2C3E50'          # Dark blue-gray
    }
    
    # Color scales
    DIVERGING = ['#DC3545', '#FF6B35', '#FFC107', '#28A745', '#17A2B8']
    SEQUENTIAL = ['#003366', '#0055A4', '#0077DD', '#4DA6FF', '#B3D9FF']
    CATEGORICAL = ['#003366', '#FF6B35', '#28A745', '#FFC107', '#17A2B8', 
                   '#DC3545', '#6C757D', '#9370DB']
    
    @staticmethod
    def get_layout_template() -> dict:
        """Get standard layout template"""
        return {
            'font': {
                'family': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                'size': 13,
                'color': ChartTheme.ISTAT['text']
            },
            'plot_bgcolor': ChartTheme.ISTAT['background'],
            'paper_bgcolor': ChartTheme.ISTAT['background'],
            'xaxis': {
                'gridcolor': ChartTheme.ISTAT['grid'],
                'linecolor': ChartTheme.ISTAT['dark'],
                'showgrid': True,
                'zeroline': False
            },
            'yaxis': {
                'gridcolor': ChartTheme.ISTAT['grid'],
                'linecolor': ChartTheme.ISTAT['dark'],
                'showgrid': True,
                'zeroline': True
            },
            'hovermode': 'x unified',
            'hoverlabel': {
                'bgcolor': 'white',
                'font_size': 12,
                'font_family': 'Inter'
            },
            'legend': {
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': ChartTheme.ISTAT['grid'],
                'borderwidth': 1
            }
        }


class NowcastingCharts:
    """
    Complete chart library for nowcasting platform
    """
    
    def __init__(self, theme: str = 'istat'):
        self.theme = ChartTheme()
        self.default_height = 500
        self.default_width = None  # Auto
    
    def plot_predictions_vs_actual(self,
                                   dates: pd.Series,
                                   actual: np.ndarray,
                                   predictions: Dict[str, np.ndarray],
                                   title: str = "Predictions vs Actual",
                                   highlight_test_start: Optional[pd.Timestamp] = None) -> go.Figure:
        """
        Time series plot: Actual vs multiple predictions
        
        Args:
            dates: Date series
            actual: Actual values
            predictions: Dict of {model_name: predictions}
            title: Chart title
            highlight_test_start: Date to mark train/test split
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2.5),
            marker=dict(size=6, symbol='circle'),
            hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.3f}<extra></extra>'
        ))
        
        # Predictions
        colors = self.theme.CATEGORICAL
        for idx, (model_name, pred) in enumerate(predictions.items()):
            color = colors[idx % len(colors)]
            
            # Determine line style
            if 'Persistence' in model_name or 'Mean' in model_name:
                dash = 'dot'
                width = 2
            else:
                dash = 'solid'
                width = 2.5
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=pred,
                mode='lines',
                name=model_name,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=f'<b>{model_name}</b><br>Date: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
            ))
        
        # Highlight test period
        if highlight_test_start is not None:
            fig.add_vline(
                x=highlight_test_start,
                line_dash="dash",
                line_color=self.theme.ISTAT['warning'],
                annotation_text="Test Start",
                annotation_position="top"
            )
        
        # Layout
        layout = self.theme.get_layout_template()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.theme.ISTAT['primary']}
            },
            'xaxis_title': 'Date',
            'yaxis_title': 'Value',
            'height': self.default_height,
            'showlegend': True,
            'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1}
        })
        
        fig.update_layout(**layout)
        
        return fig
    
    def plot_forecast_errors(self,
                            dates: pd.Series,
                            errors: Dict[str, np.ndarray],
                            title: str = "Forecast Errors") -> go.Figure:
        """
        Time series plot of forecast errors
        
        Args:
            dates: Date series
            errors: Dict of {model_name: errors}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = self.theme.CATEGORICAL
        
        for idx, (model_name, err) in enumerate(errors.items()):
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=err,
                mode='lines+markers',
                name=model_name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>{model_name}</b><br>Date: %{{x}}<br>Error: %{{y:.3f}}<extra></extra>'
            ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Layout
        layout = self.theme.get_layout_template()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.theme.ISTAT['primary']}
            },
            'xaxis_title': 'Date',
            'yaxis_title': 'Forecast Error',
            'height': self.default_height,
            'showlegend': True
        })
        
        fig.update_layout(**layout)
        
        return fig
    
    def plot_error_distribution(self,
                               errors: Dict[str, np.ndarray],
                               title: str = "Error Distribution") -> go.Figure:
        """
        Histogram + box plot of errors
        
        Args:
            errors: Dict of {model_name: errors}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Distribution", "Box Plot"),
            vertical_spacing=0.15
        )
        
        colors = self.theme.CATEGORICAL
        
        for idx, (model_name, err) in enumerate(errors.items()):
            color = colors[idx % len(colors)]
            err_clean = err[np.isfinite(err)]
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=err_clean,
                    name=model_name,
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=30,
                    hovertemplate='<b>%{fullData.name}</b><br>Error: %{x:.3f}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(
                    x=err_clean,
                    name=model_name,
                    marker_color=color,
                    boxmean='sd',
                    orientation='h',
                    hovertemplate='<b>%{fullData.name}</b><extra></extra>'
                ),
                row=2, col=1
            )
        
        # Layout
        layout = self.theme.get_layout_template()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.theme.ISTAT['primary']}
            },
            'height': 700,
            'showlegend': True,
            'barmode': 'overlay'
        })
        
        fig.update_xaxes(title_text="Forecast Error", row=1, col=1)
        fig.update_xaxes(title_text="Forecast Error", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_layout(**layout)
        
        return fig
    
    def plot_metrics_comparison(self,
                               metrics: Dict[str, Dict[str, float]],
                               metric_names: List[str] = ['rmse', 'mae'],
                               title: str = "Model Comparison") -> go.Figure:
        """
        Bar chart comparing metrics across models
        
        Args:
            metrics: Dict of {model_name: {metric: value}}
            metric_names: Metrics to display
            title: Chart title
        
        Returns:
            Plotly figure
        """
        model_names = list(metrics.keys())
        
        # Create subplots (one per metric)
        n_metrics = len(metric_names)
        fig = make_subplots(
            rows=1, cols=n_metrics,
            subplot_titles=[m.upper() for m in metric_names],
            horizontal_spacing=0.1
        )
        
        colors = self.theme.CATEGORICAL
        
        for col_idx, metric in enumerate(metric_names):
            values = [metrics[model].get(metric, np.nan) for model in model_names]
            
            # Color by performance (green=best, red=worst)
            if metric in ['rmse', 'mae', 'mape', 'mse']:
                # Lower is better
                min_val = np.nanmin(values)
                bar_colors = [self.theme.ISTAT['success'] if v == min_val else colors[i % len(colors)] 
                             for i, v in enumerate(values)]
            else:
                # Higher is better
                max_val = np.nanmax(values)
                bar_colors = [self.theme.ISTAT['success'] if v == max_val else colors[i % len(colors)] 
                             for i, v in enumerate(values)]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=metric.upper(),
                    marker_color=bar_colors,
                    text=[f'{v:.4f}' if np.isfinite(v) else 'N/A' for v in values],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>' + metric.upper() + ': %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=col_idx + 1
            )
        
        # Layout
        layout = self.theme.get_layout_template()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.theme.ISTAT['primary']}
            },
            'height': 500,
            'showlegend': False
        })
        
        for i in range(n_metrics):
            fig.update_xaxes(tickangle=-45, row=1, col=i+1)
        
        fig.update_layout(**layout)
        
        return fig
    
    def plot_rolling_performance(self,
                                backtest_results: List,
                                metric: str = 'rmse',
                                title: str = "Rolling Backtest Performance") -> go.Figure:
        """
        Line chart of performance across rolling splits
        
        Args:
            backtest_results: List of BacktestResult
            metric: Metric to plot
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Extract data
        splits = [r.split_id for r in backtest_results]
        values = [r.metrics.get(metric, np.nan) for r in backtest_results]
        
        # Line plot
        fig.add_trace(go.Scatter(
            x=splits,
            y=values,
            mode='lines+markers',
            name=metric.upper(),
            line=dict(color=self.theme.ISTAT['primary'], width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate=f'<b>Split %{{x}}</b><br>{metric.upper()}: %{{y:.4f}}<extra></extra>'
        ))
        
        # Mean line
        mean_val = np.nanmean(values)
        fig.add_hline(
            y=mean_val,
            line_dash="dash",
            line_color=self.theme.ISTAT['secondary'],
            annotation_text=f"Mean: {mean_val:.4f}",
            annotation_position="right"
        )
        
        # Layout
        layout = self.theme.get_layout_template()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.theme.ISTAT['primary']}
            },
            'xaxis_title': 'Split ID',
            'yaxis_title': metric.upper(),
            'height': self.default_height,
            'showlegend': False
        })
        
        fig.update_layout(**layout)
        
        return fig
    
    def plot_statistical_tests(self,
                              test_results: Dict[str, Dict],
                              title: str = "Statistical Test Results") -> go.Figure:
        """
        Visualization of statistical test results
        
        Args:
            test_results: Dict of {test_name: {'statistic': x, 'p_value': y, ...}}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        test_names = list(test_results.keys())
        statistics = [test_results[t].get('statistic', np.nan) for t in test_names]
        p_values = [test_results[t].get('p_value', np.nan) for t in test_names]
        is_sig = [test_results[t].get('is_significant', False) for t in test_names]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Test Statistics", "P-values"),
            horizontal_spacing=0.15
        )
        
        # Bar colors based on significance
        bar_colors_stat = [self.theme.ISTAT['success'] if sig else self.theme.ISTAT['danger'] 
                          for sig in is_sig]
        bar_colors_p = [self.theme.ISTAT['success'] if p < 0.05 else self.theme.ISTAT['danger'] 
                       for p in p_values]
        
        # Statistics
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=statistics,
                name='Statistic',
                marker_color=bar_colors_stat,
                text=[f'{s:.3f}' if np.isfinite(s) else 'N/A' for s in statistics],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Statistic: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # P-values
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=p_values,
                name='P-value',
                marker_color=bar_colors_p,
                text=[f'{p:.4f}' if np.isfinite(p) else 'N/A' for p in p_values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>P-value: %{y:.4f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Significance line
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=1, col=2,
                     annotation_text="α=0.05", annotation_position="right")
        
        # Layout
        layout = self.theme.get_layout_template()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.theme.ISTAT['primary']}
            },
            'height': 500,
            'showlegend': False
        })
        
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=1, col=2)
        fig.update_yaxes(title_text="Statistic", row=1, col=1)
        fig.update_yaxes(title_text="P-value", row=1, col=2)
        
        fig.update_layout(**layout)
        
        return fig
    
    def plot_feature_importance(self,
                               importance: Dict[str, float],
                               top_n: int = 20,
                               title: str = "Feature Importance") -> go.Figure:
        """
        Horizontal bar chart of feature importance
        
        Args:
            importance: Dict of {feature_name: importance_value}
            top_n: Show top N features
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Sort and select top N
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        
        # Color by sign
        colors = [self.theme.ISTAT['success'] if v > 0 else self.theme.ISTAT['danger'] 
                 for v in values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.3f}' for v in values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        # Layout
        layout = self.theme.get_layout_template()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.theme.ISTAT['primary']}
            },
            'xaxis_title': 'Importance',
            'yaxis_title': 'Feature',
            'height': max(500, top_n * 25),
            'showlegend': False
        })
        
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(**layout)
        
        return fig
    
    def plot_residual_diagnostics(self,
                                 residuals: np.ndarray,
                                 fitted_values: np.ndarray,
                                 title: str = "Residual Diagnostics") -> go.Figure:
        """
        Multi-panel residual diagnostic plots
        
        Args:
            residuals: Residuals
            fitted_values: Fitted values
            title: Chart title
        
        Returns:
            Plotly figure
        """
        residuals = np.asarray(residuals, float).ravel()
        fitted_values = np.asarray(fitted_values, float).ravel()
        
        mask = np.isfinite(residuals) & np.isfinite(fitted_values)
        residuals = residuals[mask]
        fitted_values = fitted_values[mask]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Residuals vs Fitted", "Q-Q Plot", 
                          "Scale-Location", "Residuals vs Order"),
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )
        
        # 1. Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=fitted_values,
                y=residuals,
                mode='markers',
                marker=dict(color=self.theme.ISTAT['primary'], size=6, opacity=0.6),
                hovertemplate='Fitted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Q-Q Plot
        from scipy import stats as sp_stats
        theoretical_quantiles = sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                marker=dict(color=self.theme.ISTAT['secondary'], size=6, opacity=0.6),
                hovertemplate='Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        # 45-degree line
        qq_min = min(theoretical_quantiles.min(), sample_quantiles.min())
        qq_max = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(
                x=[qq_min, qq_max],
                y=[qq_min, qq_max],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Scale-Location (sqrt standardized residuals vs fitted)
        std_residuals = residuals / np.std(residuals)
        sqrt_std_residuals = np.sqrt(np.abs(std_residuals))
        
        fig.add_trace(
            go.Scatter(
                x=fitted_values,
                y=sqrt_std_residuals,
                mode='markers',
                marker=dict(color=self.theme.ISTAT['info'], size=6, opacity=0.6),
                hovertemplate='Fitted: %{x:.3f}<br>√|Std Residual|: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Residuals vs Order (time)
        order = np.arange(len(residuals))
        fig.add_trace(
            go.Scatter(
                x=order,
                y=residuals,
                mode='markers',
                marker=dict(color=self.theme.ISTAT['warning'], size=6, opacity=0.6),
                hovertemplate='Order: %{x}<br>Residual: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        
        # Update axes labels
        fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        
        fig.update_xaxes(title_text="Fitted Values", row=2, col=1)
        fig.update_yaxes(title_text="√|Standardized Residuals|", row=2, col=1)
        
        fig.update_xaxes(title_text="Observation Order", row=2, col=2)
        fig.update_yaxes(title_text="Residuals", row=2, col=2)
        
        # Layout
        layout = self.theme.get_layout_template()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.theme.ISTAT['primary']}
            },
            'height': 800,
            'showlegend': False
        })
        
        fig.update_layout(**layout)
        
        return fig
    
    def plot_correlation_heatmap(self,
                                correlation_matrix: pd.DataFrame,
                                title: str = "Feature Correlation Matrix") -> go.Figure:
        """
        Heatmap of correlation matrix
        
        Args:
            correlation_matrix: Pandas correlation matrix
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="Correlation"),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        # Layout
        layout = self.theme.get_layout_template()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.theme.ISTAT['primary']}
            },
            'height': max(600, len(correlation_matrix) * 20),
            'width': max(600, len(correlation_matrix.columns) * 20),
            'xaxis': {'side': 'bottom', 'tickangle': -45},
            'yaxis': {'side': 'left'}
        })
        
        fig.update_layout(**layout)
        
        return fig
    
    def plot_gauge_chart(self,
                        value: float,
                        title: str,
                        min_val: float = 0,
                        max_val: float = 100,
                        thresholds: Optional[Dict] = None) -> go.Figure:
        """
        Gauge chart for single metric
        
        Args:
            value: Current value
            title: Chart title
            min_val: Minimum value
            max_val: Maximum value
            thresholds: Dict of {label: threshold}
        
        Returns:
            Plotly figure
        """
        if thresholds is None:
            thresholds = {'Poor': 33, 'Fair': 66, 'Good': 100}
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={'text': title, 'font': {'size': 20}},
            delta={'reference': (max_val + min_val) / 2},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': self.theme.ISTAT['primary']},
                'steps': [
                    {'range': [min_val, max_val * 0.33], 'color': self.theme.ISTAT['danger']},
                    {'range': [max_val * 0.33, max_val * 0.66], 'color': self.theme.ISTAT['warning']},
                    {'range': [max_val * 0.66, max_val], 'color': self.theme.ISTAT['success']}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font={'family': 'Inter', 'size': 14}
        )
        
        return fig
    
    def plot_waterfall(self,
                      categories: List[str],
                      values: List[float],
                      title: str = "Contribution Analysis") -> go.Figure:
        """
        Waterfall chart showing cumulative contributions
        
        Args:
            categories: Category names
            values: Contribution values
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Waterfall(
            name="",
            orientation="v",
            measure=["relative"] * (len(categories) - 1) + ["total"],
            x=categories,
            y=values,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": self.theme.ISTAT['danger']}},
            increasing={"marker": {"color": self.theme.ISTAT['success']}},
            totals={"marker": {"color": self.theme.ISTAT['primary']}},
            hovertemplate='<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>'
        ))
        
        # Layout
        layout = self.theme.get_layout_template()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 18, 'color': self.theme.ISTAT['primary']}
            },
            'xaxis_title': 'Component',
            'yaxis_title': 'Contribution',
            'height': 500,
            'showlegend': False
        })
        
        fig.update_xaxes(tickangle=-45)
        fig.update_layout(**layout)
        
        return fig


class InteractiveComponents:
    """
    Interactive dashboard components
    """
    
    @staticmethod
    def create_metric_card(value: float,
                          label: str,
                          delta: Optional[float] = None,
                          format_str: str = ".4f") -> Dict:
        """
        Create metric card data structure
        
        Args:
            value: Metric value
            label: Metric label
            delta: Change/comparison value
            format_str: Format string
        
        Returns:
            Dict with metric card data
        """
        card = {
            'value': value,
            'label': label,
            'formatted': f"{value:{format_str}}" if np.isfinite(value) else "N/A"
        }
        
        if delta is not None:
            card['delta'] = delta
            card['delta_formatted'] = f"{delta:+.2f}%"
            card['delta_color'] = 'green' if delta < 0 else 'red'  # For RMSE/MAE, lower is better
        
        return card
