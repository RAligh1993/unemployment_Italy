"""
Real-Time Forecaster Module
Generates live nowcasts with confidence intervals and alerts

Author: Ali Ghanbari
Institution: ISTAT
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RealTimeForecaster:
    """Real-time unemployment nowcasting engine"""
    
    def __init__(self, 
                 trained_model,
                 train_stats: Dict,
                 gt_keywords: Optional[List[str]] = None):
        """
        Initialize forecaster
        
        Args:
            trained_model: Fitted model object
            train_stats: Training data statistics (mean, std)
            gt_keywords: List of GT keywords for monitoring
        """
        self.model = trained_model
        self.train_stats = train_stats
        self.gt_keywords = gt_keywords or []
        
    def generate_nowcast(self,
                        latest_data: Optional[pd.DataFrame] = None,
                        confidence_level: float = 0.95) -> Dict:
        """
        Generate nowcast for current month
        
        Args:
            latest_data: Latest available data (if None, uses synthetic)
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with nowcast and metadata
        """
        if latest_data is None:
            # Generate synthetic latest data for demo
            latest_data = self._generate_synthetic_latest()
        
        # Generate prediction
        try:
            prediction = self.model.predict(latest_data)[0]
        except:
            prediction = 0.0
        
        # Confidence intervals (±z * std)
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        train_std = self.train_stats.get('std', 0.5)
        
        ci_lower = prediction - z_score * train_std
        ci_upper = prediction + z_score * train_std
        
        # Historical comparison
        train_mean = self.train_stats.get('mean', 0.0)
        vs_historical = prediction - train_mean
        
        # Generate GT signals
        gt_signals = self._analyze_gt_signals(latest_data)
        
        return {
            'prediction': prediction,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
            'vs_historical': vs_historical,
            'gt_signals': gt_signals,
            'timestamp': datetime.now(),
            'reference_month': self._get_reference_month()
        }
    
    def _generate_synthetic_latest(self) -> pd.DataFrame:
        """Generate synthetic latest data for demo"""
        # Create dummy row with reasonable values
        data = {
            'date': [pd.Timestamp.now()],
            'unemp_lag1': [6.5],
            'unemp_lag2': [6.6],
            'unemp_youth_lag1': [12.0],
            'CCI_lag1': [105.0],
            'HICP_lag1': [0.2]
        }
        
        # Add GT keywords if available
        for kw in self.gt_keywords:
            data[kw] = [50.0 + np.random.randn() * 5]
        
        return pd.DataFrame(data)
    
    def _analyze_gt_signals(self, latest_data: pd.DataFrame) -> Dict:
        """
        Analyze Google Trends signals for early warnings
        
        Args:
            latest_data: Latest data with GT features
            
        Returns:
            Dictionary with signal analysis
        """
        signals = {
            'status': 'normal',
            'elevated_keywords': [],
            'declined_keywords': [],
            'overall_intensity': 50.0
        }
        
        if len(self.gt_keywords) == 0:
            return signals
        
        # Analyze each keyword
        gt_values = []
        
        for kw in self.gt_keywords:
            if kw in latest_data.columns:
                value = latest_data[kw].iloc[0]
                gt_values.append(value)
                
                # Flag if significantly elevated or declined
                if value > 65:  # Threshold for elevated
                    signals['elevated_keywords'].append({
                        'keyword': kw,
                        'value': value,
                        'status': 'elevated'
                    })
                elif value < 35:  # Threshold for declined
                    signals['declined_keywords'].append({
                        'keyword': kw,
                        'value': value,
                        'status': 'declined'
                    })
        
        # Overall intensity
        if len(gt_values) > 0:
            signals['overall_intensity'] = np.mean(gt_values)
            
            # Determine status
            if signals['overall_intensity'] > 60:
                signals['status'] = 'high'
            elif signals['overall_intensity'] < 40:
                signals['status'] = 'low'
        
        return signals
    
    def _get_reference_month(self) -> str:
        """Get reference month for nowcast"""
        # Typically, nowcasting for previous or current month
        now = datetime.now()
        ref_month = now - timedelta(days=15)  # Mid-month reference
        return ref_month.strftime('%B %Y')
    
    def generate_forecast_path(self,
                              n_steps: int = 6,
                              scenario: str = 'base') -> pd.DataFrame:
        """
        Generate multi-step forecast path
        
        Args:
            n_steps: Number of months to forecast
            scenario: 'base', 'optimistic', 'pessimistic'
            
        Returns:
            DataFrame with forecast path
        """
        dates = pd.date_range(
            start=datetime.now(),
            periods=n_steps,
            freq='MS'
        )
        
        # Base forecast (simple extrapolation)
        base_value = 0.0
        trend = -0.05 if scenario == 'optimistic' else 0.05 if scenario == 'pessimistic' else 0.0
        volatility = 0.1
        
        forecasts = []
        
        for i in range(n_steps):
            forecast = base_value + trend * i + np.random.randn() * volatility
            forecasts.append(forecast)
        
        df = pd.DataFrame({
            'date': dates,
            'forecast': forecasts,
            'scenario': scenario
        })
        
        return df
    
    def detect_anomalies(self,
                        recent_data: pd.DataFrame,
                        threshold: float = 2.5) -> Dict:
        """
        Detect anomalies in recent data
        
        Args:
            recent_data: Recent unemployment data
            threshold: Number of standard deviations for anomaly
            
        Returns:
            Dictionary with anomaly information
        """
        if 'target' not in recent_data.columns:
            return {'anomalies_detected': False}
        
        values = recent_data['target'].values
        mean = np.mean(values)
        std = np.std(values)
        
        # Z-scores
        z_scores = np.abs((values - mean) / std) if std > 0 else np.zeros(len(values))
        
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        return {
            'anomalies_detected': len(anomaly_indices) > 0,
            'n_anomalies': len(anomaly_indices),
            'anomaly_dates': recent_data.iloc[anomaly_indices]['date'].tolist() if len(anomaly_indices) > 0 else [],
            'anomaly_values': values[anomaly_indices].tolist() if len(anomaly_indices) > 0 else [],
            'threshold': threshold
        }


class AlertSystem:
    """Early warning alert system"""
    
    def __init__(self, 
                 alert_threshold: float = 0.5,
                 gt_threshold: float = 70):
        """
        Initialize alert system
        
        Args:
            alert_threshold: Unemployment change threshold for alerts (pp)
            gt_threshold: GT intensity threshold for warnings
        """
        self.alert_threshold = alert_threshold
        self.gt_threshold = gt_threshold
        
    def generate_alerts(self,
                       nowcast: Dict,
                       recent_history: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Generate alerts based on nowcast
        
        Args:
            nowcast: Nowcast dictionary from forecaster
            recent_history: Recent historical data
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Alert 1: Large unemployment change
        prediction = nowcast['prediction']
        
        if abs(prediction) > self.alert_threshold:
            alerts.append({
                'type': 'unemployment_change',
                'severity': 'high' if abs(prediction) > 1.0 else 'medium',
                'message': f"Significant unemployment change predicted: {prediction:+.2f} pp",
                'action': "Review labor market indicators and policy responses"
            })
        
        # Alert 2: High GT signal intensity
        gt_signals = nowcast.get('gt_signals', {})
        
        if gt_signals.get('overall_intensity', 50) > self.gt_threshold:
            alerts.append({
                'type': 'gt_elevated',
                'severity': 'medium',
                'message': f"Google Trends search intensity elevated: {gt_signals['overall_intensity']:.0f}",
                'action': "Monitor for potential labor market stress"
            })
        
        # Alert 3: Wide confidence interval (high uncertainty)
        ci_width = nowcast['ci_upper'] - nowcast['ci_lower']
        
        if ci_width > 1.5:
            alerts.append({
                'type': 'high_uncertainty',
                'severity': 'low',
                'message': f"Wide prediction interval: ±{ci_width/2:.2f} pp",
                'action': "Interpret nowcast with caution; consider additional data sources"
            })
        
        # Alert 4: Trend reversal (if history available)
        if recent_history is not None and 'target' in recent_history.columns:
            recent_mean = recent_history['target'].tail(3).mean()
            
            if (recent_mean > 0 and prediction < -0.3) or (recent_mean < 0 and prediction > 0.3):
                alerts.append({
                    'type': 'trend_reversal',
                    'severity': 'medium',
                    'message': f"Potential trend reversal: recent avg {recent_mean:+.2f}, nowcast {prediction:+.2f}",
                    'action': "Investigate underlying causes of directional change"
                })
        
        return alerts
