"""
Helper Functions Module
Utility functions for formatting, validation, and common operations


Institution: ISTAT
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Any
from datetime import datetime, timedelta
import re


def format_number(value: float, 
                 decimals: int = 2, 
                 percentage: bool = False,
                 signed: bool = True) -> str:
    """
    Format number for display
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        percentage: Add percentage sign
        signed: Show +/- sign
        
    Returns:
        Formatted string
    """
    if np.isnan(value):
        return "N/A"
    
    sign = "+" if value >= 0 and signed else ""
    formatted = f"{sign}{value:.{decimals}f}"
    
    if percentage:
        formatted += "%"
    
    return formatted


def format_date(date: Union[str, datetime, pd.Timestamp],
               format_str: str = "%Y-%m-%d") -> str:
    """
    Format date for display
    
    Args:
        date: Date to format
        format_str: Output format string
        
    Returns:
        Formatted date string
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    return date.strftime(format_str)


def format_metric_card(title: str,
                      value: str,
                      delta: Optional[str] = None,
                      color: str = "blue") -> str:
    """
    Create HTML for metric card
    
    Args:
        title: Metric title
        value: Main value
        delta: Change value (optional)
        color: Card accent color
        
    Returns:
        HTML string
    """
    delta_html = f'<p style="color: gray; margin: 0;">{delta}</p>' if delta else ""
    
    html = f"""
    <div style="
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
    ">
        <h5 style="margin: 0 0 0.5rem 0; color: #333;">{title}</h5>
        <h3 style="margin: 0; color: {color};">{value}</h3>
        {delta_html}
    </div>
    """
    
    return html


def get_signal_status(gt_signals: Dict) -> Dict:
    """
    Interpret GT signal status
    
    Args:
        gt_signals: Dictionary from forecaster
        
    Returns:
        Dictionary with level and message
    """
    intensity = gt_signals.get('overall_intensity', 50)
    status = gt_signals.get('status', 'normal')
    
    if status == 'high':
        level = "🔴 High"
        message = "Elevated search activity detected"
        color = "#d62728"
    elif status == 'low':
        level = "🟢 Low"
        message = "Below-average search activity"
        color = "#2ca02c"
    else:
        level = "🟡 Normal"
        message = "Normal search activity"
        color = "#ff9800"
    
    return {
        'level': level,
        'message': message,
        'color': color,
        'intensity': intensity
    }


def validate_data_quality(df: pd.DataFrame,
                         required_cols: List[str]) -> Dict:
    """
    Validate data quality
    
    Args:
        df: DataFrame to validate
        required_cols: List of required columns
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []
    
    # Check required columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for missing values
    missing_counts = df[required_cols].isna().sum()
    for col, count in missing_counts.items():
        if count > 0:
            pct = (count / len(df)) * 100
            if pct > 10:
                issues.append(f"Column '{col}' has {count} missing values ({pct:.1f}%)")
            else:
                warnings.append(f"Column '{col}' has {count} missing values ({pct:.1f}%)")
    
    # Check for duplicates
    if df.duplicated().any():
        n_dups = df.duplicated().sum()
        warnings.append(f"Found {n_dups} duplicate rows")
    
    # Check date ordering (if date column exists)
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        if not dates.is_monotonic_increasing:
            warnings.append("Dates are not in chronological order")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'n_rows': len(df),
        'n_cols': len(df.columns)
    }


def calculate_summary_stats(series: pd.Series) -> Dict:
    """
    Calculate summary statistics
    
    Args:
        series: Pandas Series
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
        'skewness': series.skew(),
        'kurtosis': series.kurt()
    }


def detect_outliers(series: pd.Series,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in series
    
    Args:
        series: Data series
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        
        outliers = (series < lower) | (series > upper)
        
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers.values


def create_lag_matrix(series: pd.Series,
                     n_lags: int) -> pd.DataFrame:
    """
    Create lag matrix from time series
    
    Args:
        series: Time series
        n_lags: Number of lags
        
    Returns:
        DataFrame with lag columns
    """
    df = pd.DataFrame()
    
    for lag in range(n_lags):
        df[f'lag_{lag}'] = series.shift(lag)
    
    return df


def split_train_test_temporal(df: pd.DataFrame,
                              test_size: Union[int, float],
                              date_col: str = 'date') -> tuple:
    """
    Split data temporally (respecting time order)
    
    Args:
        df: DataFrame to split
        test_size: Number of rows (int) or fraction (float) for test
        date_col: Date column name
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by date
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    
    if isinstance(test_size, float):
        test_size = int(len(df_sorted) * test_size)
    
    split_idx = len(df_sorted) - test_size
    
    train = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test = df_sorted.iloc[split_idx:].reset_index(drop=True)
    
    return train, test


def compute_correlation_matrix(df: pd.DataFrame,
                               target_col: Optional[str] = None,
                               method: str = 'pearson') -> pd.DataFrame:
    """
    Compute correlation matrix
    
    Args:
        df: DataFrame
        target_col: If provided, sort by correlation with this column
        method: 'pearson', 'spearman', or 'kendall'
        
    Returns:
        Correlation matrix
    """
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    corr_matrix = df[numeric_cols].corr(method=method)
    
    if target_col and target_col in corr_matrix.columns:
        # Sort by absolute correlation with target
        target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
        corr_matrix = corr_matrix.loc[target_corr.index, target_corr.index]
    
    return corr_matrix


def create_alert_html(alert: Dict) -> str:
    """
    Create HTML for alert display
    
    Args:
        alert: Alert dictionary
        
    Returns:
        HTML string
    """
    severity_colors = {
        'high': '#f8d7da',
        'medium': '#fff3cd',
        'low': '#d1ecf1'
    }
    
    severity_icons = {
        'high': '🔴',
        'medium': '🟡',
        'low': '🔵'
    }
    
    color = severity_colors.get(alert['severity'], '#e2e3e5')
    icon = severity_icons.get(alert['severity'], '⚪')
    
    html = f"""
    <div style="
        background-color: {color};
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid {color.replace('d7da', '0000').replace('f3cd', 'a733').replace('ecf1', '7ab8')};
    ">
        <h5 style="margin: 0 0 0.5rem 0;">{icon} {alert['type'].replace('_', ' ').title()}</h5>
        <p style="margin: 0 0 0.5rem 0;"><strong>{alert['message']}</strong></p>
        <p style="margin: 0; font-size: 0.9em; color: #666;">
            <em>Action: {alert['action']}</em>
        </p>
    </div>
    """
    
    return html


def save_results_to_excel(results: Dict[str, pd.DataFrame],
                          filename: str = "nowcasting_results.xlsx"):
    """
    Save multiple DataFrames to Excel with separate sheets
    
    Args:
        results: Dictionary of {sheet_name: dataframe}
        filename: Output filename
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return filename


def generate_report_summary(results_df: pd.DataFrame,
                           best_model_name: str,
                           test_period: tuple) -> str:
    """
    Generate text summary of results
    
    Args:
        results_df: Results DataFrame
        best_model_name: Name of best model
        test_period: (start_date, end_date) tuple
        
    Returns:
        Markdown formatted summary
    """
    best_row = results_df[results_df['Model'] == best_model_name].iloc[0]
    
    summary = f"""
# Unemployment Nowcasting Results Summary

## Test Period
- **Start:** {test_period[0]}
- **End:** {test_period[1]}
- **Duration:** {(pd.to_datetime(test_period[1]) - pd.to_datetime(test_period[0])).days} days

## Best Model: {best_model_name}

### Performance Metrics
- **RMSE:** {best_row['RMSE']:.4f}
- **MAE:** {best_row.get('MAE', 'N/A')}
- **Improvement:** {best_row.get('Improvement_pct', 0):+.2f}%

### Statistical Significance
- **p-value:** {best_row.get('p_value', 1.0):.4f}
- **Significant at 5%:** {'✅ Yes' if best_row.get('p_value', 1.0) < 0.05 else '❌ No'}

## All Models Tested
{results_df[['Model', 'RMSE', 'Improvement_pct']].to_markdown(index=False)}

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return summary


def export_to_csv(df: pd.DataFrame, 
                 filename: str,
                 include_timestamp: bool = True) -> str:
    """
    Export DataFrame to CSV with optional timestamp
    
    Args:
        df: DataFrame to export
        filename: Base filename
        include_timestamp: Add timestamp to filename
        
    Returns:
        Final filename
    """
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base, ext = filename.rsplit('.', 1)
        filename = f"{base}_{timestamp}.{ext}"
    
    df.to_csv(filename, index=False)
    
    return filename


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names (lowercase, remove spaces, etc.)
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df_clean = df.copy()
    
    # Replace spaces with underscores, lowercase
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Remove special characters (except underscore)
    df_clean.columns = df_clean.columns.str.replace(r'[^a-z0-9_]', '', regex=True)
    
    return df_clean


def interpolate_missing_values(df: pd.DataFrame,
                               method: str = 'linear',
                               limit: Optional[int] = 3) -> pd.DataFrame:
    """
    Interpolate missing values in DataFrame
    
    Args:
        df: DataFrame with missing values
        method: Interpolation method ('linear', 'polynomial', 'spline')
        limit: Maximum number of consecutive NaNs to fill
        
    Returns:
        DataFrame with interpolated values
    """
    df_filled = df.copy()
    
    # Only interpolate numeric columns
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df_filled[col] = df_filled[col].interpolate(method=method, limit=limit)
    
    return df_filled
