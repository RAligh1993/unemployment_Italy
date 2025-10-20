"""
📊 Visualization Module
========================

Professional chart generation for unemployment nowcasting analysis.
Built on Plotly for interactive, publication-quality visualizations.

Chart Types:
    - Time series plots
    - Correlation heatmaps  
    - Coverage heatmaps
    - Error distribution plots
    - Forecast comparison charts
    - Regional maps
    - Statistical dashboards

Features:
    - Consistent styling
    - Interactive tooltips
    - Export-ready quality
    - Customizable themes
    - Responsive layouts

Author: ISTAT Nowcasting Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Configuration & Themes
# =============================================================================

class ColorTheme(Enum):
    """Pre-defined color themes"""
    DEFAULT = 'default'
    PROFESSIONAL = 'professional'
    COLORBLIND = 'colorblind'
    DARK = 'dark'
    ISTAT = 'istat'


@dataclass
class VisualizerConfig:
    """Configuration for visualizations"""
    theme: ColorTheme = ColorTheme.DEFAULT
    height: int = 500
    width: Optional[int] = None
    template: str = 'plotly_white'
    show_legend: bool = True
    font_family: str = 'Inter, Arial, sans-serif'
    font_size: int = 12
    line_width: int = 2
    marker_size: int = 6
    
    # Colors
    primary_color: str = '#3B82F6'
    secondary_color: str = '#10B981'
    warning_color: str = '#F59E0B'
    error_color: str = '#EF4444'
    neutral_color: str = '#6B7280'


class ChartThemes:
    """Predefined color schemes"""
    
    THEMES = {
        'default': {
            'colors': ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'],
            'template': 'plotly_white'
        },
        'professional': {
            'colors': ['#1E40AF', '#047857', '#B45309', '#991B1B', '#6D28D9', '#BE185D'],
            'template': 'plotly_white'
        },
        'colorblind': {
            'colors': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#949494'],
            'template': 'plotly_white'
        },
        'dark': {
            'colors': ['#60A5FA', '#34D399', '#FBBF24', '#F87171', '#A78BFA', '#F472B6'],
            'template': 'plotly_dark'
        },
        'istat': {
            'colors': ['#0F766E', '#14B8A6', '#06B6D4', '#0EA5E9', '#3B82F6', '#6366F1'],
            'template': 'plotly_white'
        }
    }


# =============================================================================
# Main Visualizer Class
# =============================================================================

class Visualizer:
    """
    Professional chart generator for unemployment nowcasting.
    
    This class provides a comprehensive set of visualization methods
    optimized for time series and nowcasting analysis.
    
    Usage:
        Basic:
            >>> viz = Visualizer()
            >>> fig = viz.plot_time_series(df, 'date', ['value'])
            >>> fig.show()
        
        With configuration:
            >>> config = VisualizerConfig(theme=ColorTheme.PROFESSIONAL, height=600)
            >>> viz = Visualizer(config)
            >>> fig = viz.plot_forecast_comparison(actual, predictions)
        
        Multiple charts:
            >>> viz = Visualizer()
            >>> fig1 = viz.plot_time_series(data, 'date', ['unemployment'])
            >>> fig2 = viz.plot_correlation_heatmap(features, target)
            >>> fig3 = viz.plot_coverage_heatmap(panel_data)
    
    Attributes:
        config: Visualization configuration
        colors: Current color scheme
    """
    
    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizerConfig()
        self.colors = self._get_color_scheme()
    
    def _get_color_scheme(self) -> List[str]:
        """Get colors for current theme"""
        theme_name = self.config.theme.value
        return ChartThemes.THEMES.get(theme_name, ChartThemes.THEMES['default'])['colors']
    
    def _apply_layout(self, 
                     fig: go.Figure, 
                     title: Optional[str] = None,
                     xaxis_title: Optional[str] = None,
                     yaxis_title: Optional[str] = None,
                     **kwargs) -> go.Figure:
        """Apply consistent layout styling"""
        
        layout_config = {
            'template': self.config.template,
            'height': kwargs.get('height', self.config.height),
            'font': {
                'family': self.config.font_family,
                'size': self.config.font_size
            },
            'hovermode': 'x unified',
            'showlegend': self.config.show_legend,
        }
        
        if self.config.width:
            layout_config['width'] = self.config.width
        
        if title:
            layout_config['title'] = {
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': self.config.font_size + 4, 'weight': 'bold'}
            }
        
        if xaxis_title:
            layout_config['xaxis_title'] = xaxis_title
        
        if yaxis_title:
            layout_config['yaxis_title'] = yaxis_title
        
        # Merge with custom kwargs
        layout_config.update(kwargs)
        
        fig.update_layout(**layout_config)
        
        return fig
    
    # =========================================================================
    # Time Series Plots
    # =========================================================================
    
    def plot_time_series(self,
                        df: pd.DataFrame,
                        date_col: str,
                        value_cols: List[str],
                        title: str = 'Time Series',
                        show_points: bool = False,
                        colors: Optional[List[str]] = None) -> go.Figure:
        """
        Plot one or more time series.
        
        Args:
            df: DataFrame with date and value columns
            date_col: Name of date column
            value_cols: List of value column names
            title: Chart title
            show_points: Whether to show markers
            colors: Custom color list (optional)
        
        Returns:
            go.Figure: Plotly figure
        
        Example:
            >>> fig = viz.plot_time_series(
            ...     df, 
            ...     'date', 
            ...     ['unemployment_rate'],
            ...     title='Italian Unemployment Rate'
            ... )
        """
        fig = go.Figure()
        
        colors = colors or self.colors
        
        for i, col in enumerate(value_cols):
            color = colors[i % len(colors)]
            
            mode = 'lines+markers' if show_points else 'lines'
            
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                mode=mode,
                name=col,
                line=dict(width=self.config.line_width, color=color),
                marker=dict(size=self.config.marker_size) if show_points else None,
                hovertemplate=f'<b>{col}</b><br>%{{x}}<br>%{{y:.3f}}<extra></extra>'
            ))
        
        self._apply_layout(
            fig,
            title=title,
            xaxis_title='Date',
            yaxis_title='Value'
        )
        
        return fig
    
    def plot_forecast_comparison(self,
                                actual: pd.Series,
                                forecasts: Dict[str, pd.Series],
                                title: str = 'Forecast Comparison',
                                top_k: Optional[int] = None) -> go.Figure:
        """
        Compare actual values with multiple forecasts.
        
        Args:
            actual: Actual time series
            forecasts: Dict of {model_name: forecast_series}
            title: Chart title
            top_k: Show only top K models (by date coverage)
        
        Returns:
            go.Figure: Plotly figure
        
        Example:
            >>> fig = viz.plot_forecast_comparison(
            ...     y_test,
            ...     {'Ridge': pred_ridge, 'ARIMAX': pred_arimax},
            ...     title='Model Comparison'
            ... )
        """
        fig = go.Figure()
        
        # Plot actual
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            mode='lines+markers',
            name='Actual',
            line=dict(width=self.config.line_width + 1, color='#1F2937'),
            marker=dict(size=self.config.marker_size + 2),
            hovertemplate='<b>Actual</b><br>%{x}<br>%{y:.3f}<extra></extra>'
        ))
        
        # Sort forecasts by coverage if top_k specified
        if top_k and len(forecasts) > top_k:
            sorted_forecasts = sorted(
                forecasts.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:top_k]
            forecasts = dict(sorted_forecasts)
        
        # Plot forecasts
        for i, (name, forecast) in enumerate(forecasts.items()):
            color = self.colors[i % len(self.colors)]
            
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines',
                name=name,
                line=dict(width=self.config.line_width, color=color),
                hovertemplate=f'<b>{name}</b><br>%{{x}}<br>%{{y:.3f}}<extra></extra>'
            ))
        
        self._apply_layout(
            fig,
            title=title,
            xaxis_title='Date',
            yaxis_title='Value'
        )
        
        return fig
    
    def plot_with_confidence_interval(self,
                                     dates: pd.Series,
                                     mean: pd.Series,
                                     lower: pd.Series,
                                     upper: pd.Series,
                                     title: str = 'Forecast with Confidence Interval',
                                     actual: Optional[pd.Series] = None) -> go.Figure:
        """
        Plot forecast with confidence interval.
        
        Args:
            dates: Date index
            mean: Point forecast
            lower: Lower bound
            upper: Upper bound
            title: Chart title
            actual: Actual values (optional)
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure()
        
        # Confidence interval (shaded area)
        fig.add_trace(go.Scatter(
            x=dates,
            y=upper,
            mode='lines',
            name='Upper 95%',
            line=dict(width=0),
            showlegend=False,
            hovertemplate='Upper: %{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=lower,
            mode='lines',
            name='Lower 95%',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.2)',
            showlegend=True,
            hovertemplate='Lower: %{y:.3f}<extra></extra>'
        ))
        
        # Mean forecast
        fig.add_trace(go.Scatter(
            x=dates,
            y=mean,
            mode='lines',
            name='Forecast',
            line=dict(width=self.config.line_width, color=self.config.primary_color),
            hovertemplate='Forecast: %{y:.3f}<extra></extra>'
        ))
        
        # Actual (if provided)
        if actual is not None:
            fig.add_trace(go.Scatter(
                x=actual.index,
                y=actual.values,
                mode='lines+markers',
                name='Actual',
                line=dict(width=self.config.line_width, color='#1F2937'),
                marker=dict(size=self.config.marker_size),
                hovertemplate='Actual: %{y:.3f}<extra></extra>'
            ))
        
        self._apply_layout(fig, title=title, xaxis_title='Date', yaxis_title='Value')
        
        return fig
    
    # =========================================================================
    # Error Analysis Plots
    # =========================================================================
    
    def plot_rolling_error(self,
                          actual: pd.Series,
                          forecasts: Dict[str, pd.Series],
                          window: int = 12,
                          error_type: str = 'mae',
                          title: Optional[str] = None) -> go.Figure:
        """
        Plot rolling error metrics.
        
        Args:
            actual: Actual values
            forecasts: Model predictions
            window: Rolling window size
            error_type: 'mae', 'rmse', or 'mape'
            title: Chart title
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure()
        
        if title is None:
            title = f'Rolling {error_type.upper()} (window={window})'
        
        for i, (name, forecast) in enumerate(forecasts.items()):
            # Align series
            y_al, f_al = actual.align(forecast, join='inner')
            
            # Calculate errors
            if error_type.lower() == 'mae':
                errors = (y_al - f_al).abs()
            elif error_type.lower() == 'rmse':
                errors = (y_al - f_al) ** 2
            elif error_type.lower() == 'mape':
                errors = ((y_al - f_al).abs() / (y_al.abs() + 1e-10)) * 100
            else:
                raise ValueError(f"Unknown error_type: {error_type}")
            
            # Rolling
            rolling = errors.rolling(window, min_periods=max(1, window // 2))
            
            if error_type.lower() == 'rmse':
                rolling_metric = rolling.mean().apply(np.sqrt)
            else:
                rolling_metric = rolling.mean()
            
            color = self.colors[i % len(self.colors)]
            
            fig.add_trace(go.Scatter(
                x=rolling_metric.index,
                y=rolling_metric.values,
                mode='lines',
                name=name,
                line=dict(width=self.config.line_width, color=color),
                hovertemplate=f'<b>{name}</b><br>%{{x}}<br>%{{y:.3f}}<extra></extra>'
            ))
        
        self._apply_layout(
            fig,
            title=title,
            xaxis_title='Date',
            yaxis_title=error_type.upper()
        )
        
        return fig
    
    def plot_error_distribution(self,
                               actual: pd.Series,
                               forecast: pd.Series,
                               model_name: str = 'Model',
                               bins: int = 30) -> go.Figure:
        """
        Plot error distribution histogram.
        
        Args:
            actual: Actual values
            forecast: Predicted values
            model_name: Model name for title
            bins: Number of histogram bins
        
        Returns:
            go.Figure: Plotly figure
        """
        # Calculate errors
        y_al, f_al = actual.align(forecast, join='inner')
        errors = (y_al - f_al).dropna()
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=bins,
            name='Errors',
            marker=dict(
                color=self.config.primary_color,
                line=dict(color='white', width=1)
            ),
            hovertemplate='Error: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add mean line
        mean_error = errors.mean()
        fig.add_vline(
            x=mean_error,
            line_dash="dash",
            line_color=self.config.error_color,
            annotation_text=f'Mean: {mean_error:.3f}',
            annotation_position="top"
        )
        
        self._apply_layout(
            fig,
            title=f'Error Distribution - {model_name}',
            xaxis_title='Forecast Error',
            yaxis_title='Frequency'
        )
        
        return fig
    
    def plot_actual_vs_predicted(self,
                                actual: pd.Series,
                                predicted: pd.Series,
                                model_name: str = 'Model') -> go.Figure:
        """
        Scatter plot: actual vs predicted with regression line.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            model_name: Model name
        
        Returns:
            go.Figure: Plotly figure
        """
        # Align
        y_al, p_al = actual.align(predicted, join='inner')
        
        # Calculate R²
        ss_res = ((y_al - p_al) ** 2).sum()
        ss_tot = ((y_al - y_al.mean()) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Create scatter
        fig = go.Figure()
        
        # Scatter points
        fig.add_trace(go.Scatter(
            x=p_al,
            y=y_al,
            mode='markers',
            name='Observations',
            marker=dict(
                size=self.config.marker_size,
                color=self.config.primary_color,
                opacity=0.6
            ),
            hovertemplate='Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>'
        ))
        
        # Perfect prediction line
        min_val = min(y_al.min(), p_al.min())
        max_val = max(y_al.max(), p_al.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color=self.config.neutral_color, width=2),
            showlegend=True
        ))
        
        self._apply_layout(
            fig,
            title=f'Actual vs Predicted - {model_name} (R² = {r2:.3f})',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        
        # Equal aspect ratio
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        
        return fig
    
    # =========================================================================
    # Heatmaps
    # =========================================================================
    
    def plot_correlation_heatmap(self,
                                features: pd.DataFrame,
                                target: Optional[pd.Series] = None,
                                top_n: int = 20,
                                title: str = 'Feature Correlations') -> go.Figure:
        """
        Correlation heatmap for features (and optionally target).
        
        Args:
            features: Feature DataFrame
            target: Target series (optional)
            top_n: Number of top features to show
            title: Chart title
        
        Returns:
            go.Figure: Plotly figure
        """
        # Calculate correlations
        if target is not None:
            # Correlations with target
            y_al, X_al = target.align(features, join='inner')
            correlations = X_al.corrwith(y_al).abs().sort_values(ascending=False)
            
            # Select top features
            top_features = correlations.head(top_n).index.tolist()
            corr_matrix = X_al[top_features].corr()
            
        else:
            # Feature correlation matrix
            if len(features.columns) > top_n:
                # Select top features by variance
                variances = features.var().sort_values(ascending=False)
                top_features = variances.head(top_n).index.tolist()
                corr_matrix = features[top_features].corr()
            else:
                corr_matrix = features.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title='Correlation'),
            hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        self._apply_layout(
            fig,
            title=title,
            height=max(500, len(corr_matrix) * 25)
        )
        
        return fig
    
    def plot_coverage_heatmap(self,
                             df: pd.DataFrame,
                             max_rows: int = 100,
                             max_cols: int = 50,
                             title: str = 'Data Coverage') -> go.Figure:
        """
        Heatmap showing data presence/absence.
        
        Args:
            df: Input DataFrame
            max_rows: Maximum rows to display
            max_cols: Maximum columns to display
            title: Chart title
        
        Returns:
            go.Figure: Plotly figure
        """
        # Limit size
        df_subset = df.tail(max_rows).iloc[:, :max_cols]
        
        # Create presence matrix (1 = data, 0 = missing)
        presence = df_subset.notna().astype(int).T
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=presence.values,
            x=presence.columns,
            y=presence.index,
            colorscale=[[0, '#EF4444'], [1, '#10B981']],
            showscale=False,
            hovertemplate='Row: %{x}<br>Column: %{y}<br>Present: %{z}<extra></extra>'
        ))
        
        self._apply_layout(
            fig,
            title=title,
            xaxis_title='Row Index',
            yaxis_title='Column',
            height=max(400, len(presence) * 15)
        )
        
        return fig
    
    # =========================================================================
    # Statistical Charts
    # =========================================================================
    
    def plot_feature_importance(self,
                               importance: pd.Series,
                               top_n: int = 20,
                               title: str = 'Feature Importance') -> go.Figure:
        """
        Horizontal bar chart for feature importance.
        
        Args:
            importance: Series with feature names as index and importance as values
            top_n: Number of top features
            title: Chart title
        
        Returns:
            go.Figure: Plotly figure
        """
        # Sort and take top N
        importance_sorted = importance.abs().sort_values(ascending=True).tail(top_n)
        
        # Color by positive/negative
        colors = [self.config.secondary_color if x >= 0 else self.config.error_color 
                 for x in importance_sorted.values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_sorted.values,
            y=importance_sorted.index,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{x:.4f}' for x in importance_sorted.values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        self._apply_layout(
            fig,
            title=title,
            xaxis_title='Importance',
            yaxis_title='',
            height=max(400, top_n * 25),
            showlegend=False
        )
        
        return fig
    
    def plot_box_comparison(self,
                           data_dict: Dict[str, pd.Series],
                           title: str = 'Distribution Comparison') -> go.Figure:
        """
        Box plots for comparing distributions.
        
        Args:
            data_dict: {name: Series} dictionary
            title: Chart title
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure()
        
        for i, (name, data) in enumerate(data_dict.items()):
            color = self.colors[i % len(self.colors)]
            
            fig.add_trace(go.Box(
                y=data,
                name=name,
                marker_color=color,
                boxmean='sd',  # Show mean and std dev
                hovertemplate='<b>%{fullData.name}</b><br>Value: %{y:.3f}<extra></extra>'
            ))
        
        self._apply_layout(
            fig,
            title=title,
            yaxis_title='Value'
        )
        
        return fig
    
    # =========================================================================
    # Specialized Charts
    # =========================================================================
    
    def plot_italian_regions_bar(self,
                                 df: pd.DataFrame,
                                 region_col: str,
                                 value_col: str,
                                 title: str = 'Regional Unemployment') -> go.Figure:
        """
        Bar chart for Italian regional data.
        
        Args:
            df: DataFrame with regional data
            region_col: Column with region names
            value_col: Column with values
            title: Chart title
        
        Returns:
            go.Figure: Plotly figure
        """
        # Sort by value
        df_sorted = df.sort_values(value_col, ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_sorted[value_col],
            y=df_sorted[region_col],
            orientation='h',
            marker=dict(
                color=df_sorted[value_col],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title='Rate (%)')
            ),
            text=[f'{v:.2f}%' for v in df_sorted[value_col]],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Rate: %{x:.2f}%<extra></extra>'
        ))
        
        self._apply_layout(
            fig,
            title=title,
            xaxis_title='Unemployment Rate (%)',
            yaxis_title='',
            height=max(500, len(df) * 25),
            showlegend=False
        )
        
        return fig
    
    def plot_gauge(self,
                  value: float,
                  title: str = 'Current Value',
                  min_val: float = 0,
                  max_val: float = 100,
                  thresholds: Optional[Dict[str, float]] = None) -> go.Figure:
        """
        Gauge chart for single metric.
        
        Args:
            value: Current value
            title: Chart title
            min_val: Minimum value
            max_val: Maximum value
            thresholds: Dict of {label: threshold_value}
        
        Returns:
            go.Figure: Plotly figure
        """
        # Default thresholds
        if thresholds is None:
            thresholds = {
                'Low': max_val * 0.33,
                'Medium': max_val * 0.67,
                'High': max_val
            }
        
        # Steps for gauge
        steps = []
        colors = ['#10B981', '#FBBF24', '#EF4444']
        
        prev_threshold = min_val
        for i, (label, threshold) in enumerate(thresholds.items()):
            steps.append({
                'range': [prev_threshold, threshold],
                'color': colors[i % len(colors)]
            })
            prev_threshold = threshold
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20}},
            delta={'reference': max_val * 0.5},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': self.config.primary_color},
                'steps': steps,
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font={'family': self.config.font_family}
        )
        
        return fig


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_plot(series: pd.Series, title: Optional[str] = None) -> go.Figure:
    """
    Quick one-liner time series plot.
    
    Args:
        series: Time series with datetime index
        title: Chart title
    
    Returns:
        go.Figure: Plotly figure
    
    Example:
        >>> fig = quick_plot(unemployment_rate, 'Italian Unemployment')
        >>> fig.show()
    """
    viz = Visualizer()
    
    df = series.to_frame('value').reset_index()
    df.columns = ['date', 'value']
    
    return viz.plot_time_series(df, 'date', ['value'], title=title or series.name)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing visualizer module...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=24, freq='M')
    actual = pd.Series(np.random.rand(24) * 5 + 10, index=dates, name='actual')
    forecast1 = actual + np.random.randn(24) * 0.5
    forecast2 = actual + np.random.randn(24) * 0.7
    
    viz = Visualizer()
    
    # Test time series
    df = pd.DataFrame({'date': dates, 'value': actual.values})
    fig1 = viz.plot_time_series(df, 'date', ['value'], title='Test Series')
    print("✓ Time series plot created")
    
    # Test forecast comparison
    fig2 = viz.plot_forecast_comparison(
        actual,
        {'Model 1': forecast1, 'Model 2': forecast2},
        title='Forecast Test'
    )
    print("✓ Forecast comparison created")
    
    # Test error distribution
    fig3 = viz.plot_error_distribution(actual, forecast1, 'Model 1')
    print("✓ Error distribution created")
    
    print("\n✅ All tests passed!")
