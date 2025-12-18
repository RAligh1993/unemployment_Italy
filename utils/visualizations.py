"""
Visualization Module
Beautiful, professional Plotly charts for nowcasting

Institution: ISTAT
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


class Visualizer:
    """Professional visualization suite with Plotly"""
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize visualizer
        
        Args:
            theme: Plotly template ('plotly', 'plotly_white', 'plotly_dark')
        """
        self.theme = theme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'gray': '#7f7f7f'
        }
        
    def plot_time_series_interactive(self,
                                    df: pd.DataFrame,
                                    date_col: str,
                                    value_col: str,
                                    title: str = "Time Series",
                                    ylabel: str = "Value") -> go.Figure:
        """
        Create interactive time series plot
        
        Args:
            df: DataFrame with data
            date_col: Date column name
            value_col: Value column name
            title: Plot title
            ylabel: Y-axis label
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Main line
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[value_col],
            mode='lines+markers',
            name='Actual',
            line=dict(color=self.colors['primary'], width=2.5),
            marker=dict(size=6, symbol='circle'),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="Zero",
            annotation_position="right"
        )
        
        # Layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'weight': 'bold'}
            },
            xaxis_title="Date",
            yaxis_title=ylabel,
            template=self.theme,
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        return fig
    
    def plot_forecast_comparison(self,
                                dates: pd.Series,
                                actual: np.ndarray,
                                predicted: np.ndarray,
                                baseline: np.ndarray,
                                title: str = "Forecast Comparison",
                                ci_lower: Optional[np.ndarray] = None,
                                ci_upper: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create beautiful forecast comparison plot
        
        Args:
            dates: Date series
            actual: Actual values
            predicted: Model predictions
            baseline: Baseline predictions
            title: Plot title
            ci_lower: Lower confidence interval (optional)
            ci_upper: Upper confidence interval (optional)
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Confidence interval (if provided)
        if ci_lower is not None and ci_upper is not None:
            fig.add_trace(go.Scatter(
                x=pd.concat([dates, dates[::-1]]),
                y=np.concatenate([ci_upper, ci_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                hoverinfo='skip',
                showlegend=True
            ))
        
        # Baseline
        fig.add_trace(go.Scatter(
            x=dates,
            y=baseline,
            mode='lines',
            name='Baseline (Historical Mean)',
            line=dict(color=self.colors['gray'], width=2, dash='dash'),
            hovertemplate='<b>Baseline:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Model predictions
        fig.add_trace(go.Scatter(
            x=dates,
            y=predicted,
            mode='lines+markers',
            name='MIDAS Forecast',
            line=dict(color=self.colors['danger'], width=2.5),
            marker=dict(size=7, symbol='square'),
            hovertemplate='<b>MIDAS:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2.5),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='<b>Actual:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        # Layout
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
            xaxis_title="Date",
            yaxis_title="Unemployment Change (pp)",
            template=self.theme,
            hovermode='x unified',
            height=550,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_model_comparison(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Create horizontal bar chart comparing models
        
        Args:
            results_df: DataFrame with Model, RMSE, Improvement_pct columns
            
        Returns:
            Plotly figure
        """
        # Sort by RMSE
        df_sorted = results_df.sort_values('RMSE', ascending=True)
        
        # Color based on improvement
        colors = []
        for improvement in df_sorted.get('Improvement_pct', [0]*len(df_sorted)):
            if improvement > 5:
                colors.append(self.colors['success'])
            elif improvement > 0:
                colors.append(self.colors['info'])
            else:
                colors.append(self.colors['danger'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df_sorted['Model'],
            x=df_sorted['RMSE'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=df_sorted['RMSE'].apply(lambda x: f'{x:.4f}'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>RMSE: %{x:.4f}<extra></extra>'
        ))
        
        # Baseline line (if available)
        if 'Baseline_RMSE' in df_sorted.columns:
            baseline_rmse = df_sorted['Baseline_RMSE'].iloc[0]
            fig.add_vline(
                x=baseline_rmse,
                line_dash="dash",
                line_color="red",
                annotation_text="Baseline",
                annotation_position="top"
            )
        
        fig.update_layout(
            title={
                'text': "Model Performance Comparison",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="RMSE (Lower is Better)",
            yaxis_title="",
            template=self.theme,
            height=max(400, len(df_sorted) * 40),
            showlegend=False
        )
        
        return fig
    
    def plot_period_performance(self, period_df: pd.DataFrame) -> go.Figure:
        """
        Create period-wise performance comparison
        
        Args:
            period_df: DataFrame with Period, Baseline_RMSE, Model_RMSE, Improvement_pct
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("RMSE by Period", "Improvement (%)"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # RMSE comparison
        fig.add_trace(
            go.Bar(
                name='Baseline',
                x=period_df['Period'],
                y=period_df['Baseline_RMSE'],
                marker_color=self.colors['gray'],
                text=period_df['Baseline_RMSE'].apply(lambda x: f'{x:.3f}'),
                textposition='auto',
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='MIDAS',
                x=period_df['Period'],
                y=period_df['Model_RMSE'],
                marker_color=self.colors['primary'],
                text=period_df['Model_RMSE'].apply(lambda x: f'{x:.3f}'),
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # Improvement bars
        improvement_colors = [
            self.colors['success'] if x > 0 else self.colors['danger']
            for x in period_df['Improvement_pct']
        ]
        
        fig.add_trace(
            go.Bar(
                name='Improvement',
                x=period_df['Period'],
                y=period_df['Improvement_pct'],
                marker_color=improvement_colors,
                text=period_df['Improvement_pct'].apply(lambda x: f'{x:+.1f}%'),
                textposition='outside',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Zero line for improvement
        fig.add_hline(
            y=0, line_dash="dash", line_color="gray",
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Period", row=1, col=1)
        fig.update_xaxes(title_text="Period", row=1, col=2)
        fig.update_yaxes(title_text="RMSE", row=1, col=1)
        fig.update_yaxes(title_text="Improvement (%)", row=1, col=2)
        
        fig.update_layout(
            title_text="Period-wise Performance Analysis",
            template=self.theme,
            height=500,
            showlegend=True,
            barmode='group'
        )
        
        return fig
    
    def plot_error_distribution(self,
                               errors_model: np.ndarray,
                               errors_baseline: np.ndarray,
                               model_name: str = "MIDAS") -> go.Figure:
        """
        Create error distribution comparison
        
        Args:
            errors_model: Model forecast errors
            errors_baseline: Baseline forecast errors
            model_name: Name of model
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Baseline histogram
        fig.add_trace(go.Histogram(
            x=errors_baseline,
            name='Baseline',
            opacity=0.6,
            marker_color=self.colors['gray'],
            nbinsx=20,
            histnorm='probability'
        ))
        
        # Model histogram
        fig.add_trace(go.Histogram(
            x=errors_model,
            name=model_name,
            opacity=0.6,
            marker_color=self.colors['primary'],
            nbinsx=20,
            histnorm='probability'
        ))
        
        # Zero line
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title="Forecast Error Distribution",
            xaxis_title="Forecast Error",
            yaxis_title="Probability",
            template=self.theme,
            barmode='overlay',
            height=450,
            showlegend=True
        )
        
        return fig
    
    def plot_scatter_actual_predicted(self,
                                     actual: np.ndarray,
                                     predicted: np.ndarray,
                                     model_name: str = "Model") -> go.Figure:
        """
        Create scatter plot of actual vs predicted
        
        Args:
            actual: Actual values
            predicted: Predicted values
            model_name: Name of model
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Scatter points
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name=model_name,
            marker=dict(
                size=10,
                color=self.colors['primary'],
                opacity=0.6,
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>Actual:</b> %{x:.3f}<br><b>Predicted:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Perfect prediction line (45-degree)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Forecast',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=True
        ))
        
        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(actual, predicted)
        
        fig.add_annotation(
            text=f'R² = {r2:.3f}',
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title="Actual vs Predicted",
            xaxis_title="Actual Unemployment Change (pp)",
            yaxis_title="Predicted Unemployment Change (pp)",
            template=self.theme,
            height=500,
            showlegend=True
        )
        
        # Equal aspect ratio
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        
        return fig
    
    def plot_gauge_chart(self,
                        value: float,
                        title: str = "Nowcast Indicator",
                        range_vals: List[float] = [-2, 2]) -> go.Figure:
        """
        Create gauge chart for nowcast
        
        Args:
            value: Current value
            title: Chart title
            range_vals: [min, max] range
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20}},
            delta={'reference': 0, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': range_vals, 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': self.colors['primary']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [range_vals[0], range_vals[0]/2], 'color': '#d4edda'},
                    {'range': [range_vals[0]/2, 0], 'color': '#c3e6cb'},
                    {'range': [0, range_vals[1]/2], 'color': '#fff3cd'},
                    {'range': [range_vals[1]/2, range_vals[1]], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(
            template=self.theme,
            height=400,
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        return fig
