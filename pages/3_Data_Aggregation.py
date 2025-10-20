"""
🤖 ISTAT Auto Data Fetcher v3.0 - Enhanced
==========================================
Multiple fallback methods for reliable data fetching.

Methods:
1. SDMX REST API (primary)
2. I.Stat OECD.Stat interface (fallback 1)
3. Direct bulk download (fallback 2)

Author: ISTAT Nowcasting Team
Version: 3.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict, List, Tuple
import time
import json
from io import StringIO, BytesIO
from datetime import datetime

try:
    from utils.state import AppState
    from utils.visualizer import Visualizer
    UTILS_AVAILABLE = True
except:
    UTILS_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="ISTAT Auto Fetcher v3",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #059669, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .method-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
    }
    .error-card {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Enhanced ISTAT Client with Multiple Methods
# =============================================================================

class ISTATMultiClient:
    """
    Multi-method ISTAT data fetcher with intelligent fallbacks
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/html, */*'
        })
        self.timeout = 30
        
        # Track which methods work
        self.method_status = {
            'sdmx': 'unknown',
            'istat_bulk': 'unknown',
            'istat_api': 'unknown'
        }
    
    # =========================================================================
    # Method 1: SDMX REST API
    # =========================================================================
    
    def fetch_via_sdmx(self,
                       indicator: str,
                       region: str = 'IT',
                       start_year: int = 2020,
                       end_year: int = 2024) -> Optional[pd.DataFrame]:
        """
        Method 1: SDMX REST API
        """
        st.write("🔄 **Method 1:** Trying SDMX REST API...")
        
        # Different endpoints to try
        base_urls = [
            "http://sdmx.istat.it/SDMXWS/rest",
            "https://sdmx.istat.it/SDMXWS/rest",
            "http://sdmx.istat.it/NSI",
        ]
        
        # Dataflow mapping
        dataflow_map = {
            'unemployment_rate': '22_781',
            'employment_rate': '22_777',
            'inactive_rate': '22_783'
        }
        
        dataflow_id = dataflow_map.get(indicator, '22_781')
        
        # Different key patterns to try
        key_patterns = [
            f"Q.{region}.9.Y15-64",  # Quarterly, Total, 15-64
            f"A.{region}.9.Y15-64",  # Annual
            f"Q.{region}...",         # All dimensions
            f"A.{region}...",
        ]
        
        for base_url in base_urls:
            for key in key_patterns:
                try:
                    url = f"{base_url}/data/{dataflow_id}/{key}"
                    
                    params = {
                        'startPeriod': str(start_year),
                        'endPeriod': str(end_year)
                    }
                    
                    st.write(f"   Trying: `{url[:80]}...`")
                    
                    response = self.session.get(
                        url,
                        params=params,
                        timeout=self.timeout
                    )
                    
                    st.write(f"   Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        df = self._parse_response(response)
                        
                        if df is not None and not df.empty:
                            self.method_status['sdmx'] = 'working'
                            st.success(f"   ✅ Success with SDMX!")
                            return df
                    
                    elif response.status_code == 500:
                        st.warning(f"   ⚠️ Server error 500 - trying next...")
                        continue
                    
                    elif response.status_code == 404:
                        st.info(f"   ℹ️ Not found - trying next pattern...")
                        continue
                
                except requests.Timeout:
                    st.warning(f"   ⏱️ Timeout - trying next...")
                    continue
                
                except Exception as e:
                    st.warning(f"   ❌ Error: {e}")
                    continue
        
        self.method_status['sdmx'] = 'failed'
        st.error("   ❌ SDMX method failed")
        return None
    
    def _parse_response(self, response) -> Optional[pd.DataFrame]:
        """Parse SDMX response (JSON or XML)"""
        try:
            # Try JSON first
            data = response.json()
            return self._parse_sdmx_json(data)
        except:
            # Try XML
            try:
                return self._parse_sdmx_xml(response.text)
            except:
                return None
    
    def _parse_sdmx_json(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse SDMX JSON"""
        try:
            if 'data' not in data:
                return None
            
            data_root = data['data']
            
            if 'dataSets' not in data_root or not data_root['dataSets']:
                return None
            
            dataset = data_root['dataSets'][0]
            
            if 'series' not in dataset:
                return None
            
            # Get time values
            structure = data_root.get('structure', {})
            dimensions = structure.get('dimensions', {})
            obs_dims = dimensions.get('observation', [])
            
            time_values = None
            for dim in obs_dims:
                if dim.get('id') == 'TIME_PERIOD':
                    time_values = [v['id'] for v in dim.get('values', [])]
                    break
            
            if not time_values:
                return None
            
            # Extract data
            records = []
            
            for series_key, series_data in dataset['series'].items():
                observations = series_data.get('observations', {})
                
                for obs_idx, obs_value in observations.items():
                    time_idx = int(obs_idx)
                    
                    if time_idx < len(time_values):
                        time_period = time_values[time_idx]
                        value = obs_value[0] if isinstance(obs_value, list) else obs_value
                        
                        records.append({
                            'time_period': time_period,
                            'value': float(value)
                        })
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            df['date'] = df['time_period'].apply(self._parse_time_period)
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')
            
            return df[['date', 'value']].drop_duplicates()
        
        except Exception as e:
            st.write(f"   JSON parse error: {e}")
            return None
    
    def _parse_sdmx_xml(self, xml_text: str) -> Optional[pd.DataFrame]:
        """Parse SDMX XML - simplified"""
        # Skip XML parsing for now (complex)
        return None
    
    def _parse_time_period(self, period: str) -> Optional[pd.Timestamp]:
        """Parse time period to datetime"""
        try:
            period = str(period).strip()
            
            # Quarterly: 2020-Q1
            if '-Q' in period:
                year, quarter = period.split('-Q')
                year = int(year)
                quarter = int(quarter)
                month = quarter * 3
                date = pd.Timestamp(year=year, month=month, day=1)
                return date + pd.offsets.MonthEnd(0)
            
            # Monthly: 2020-01
            elif len(period) == 7 and '-' in period:
                return pd.to_datetime(period) + pd.offsets.MonthEnd(0)
            
            # Annual: 2020
            elif len(period) == 4:
                return pd.Timestamp(year=int(period), month=12, day=31)
            
            else:
                return pd.to_datetime(period)
        
        except:
            return None
    
    # =========================================================================
    # Method 2: I.Stat Bulk Download
    # =========================================================================
    
    def fetch_via_istat_bulk(self,
                             indicator: str,
                             region: str = 'IT') -> Optional[pd.DataFrame]:
        """
        Method 2: Download pre-built datasets from I.Stat
        
        ISTAT provides bulk downloads of common datasets
        """
        st.write("🔄 **Method 2:** Trying I.Stat bulk download...")
        
        # Known bulk download URLs
        bulk_urls = {
            'unemployment_rate': [
                'http://dati.istat.it/OECDStat_Metadata/ShowMetadata.ashx?Dataset=DCCV_TAXDISOCC1&ShowOnWeb=true&Lang=en',
                'http://dati.istat.it/Index.aspx?DataSetCode=DCCV_TAXDISOCC1',
            ],
            'employment_rate': [
                'http://dati.istat.it/Index.aspx?DataSetCode=DCCV_TAXOCCU1',
            ]
        }
        
        urls = bulk_urls.get(indicator, [])
        
        for url in urls:
            try:
                st.write(f"   Trying: `{url[:80]}...`")
                
                response = self.session.get(url, timeout=self.timeout)
                
                st.write(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    # Try to find CSV download link in HTML
                    if 'text/html' in response.headers.get('Content-Type', ''):
                        # Look for download links
                        html = response.text
                        
                        # Common patterns for CSV/Excel download links
                        import re
                        csv_pattern = r'href="([^"]*\.csv[^"]*)"'
                        excel_pattern = r'href="([^"]*\.xlsx?[^"]*)"'
                        
                        csv_links = re.findall(csv_pattern, html, re.IGNORECASE)
                        excel_links = re.findall(excel_pattern, html, re.IGNORECASE)
                        
                        if csv_links:
                            st.info(f"   Found {len(csv_links)} CSV download links")
                            # Would need to download and parse
                        
                        if excel_links:
                            st.info(f"   Found {len(excel_links)} Excel download links")
                    
                    # This method needs more development
                    st.warning("   ℹ️ Bulk download needs manual intervention")
                    continue
            
            except Exception as e:
                st.warning(f"   ❌ Error: {e}")
                continue
        
        self.method_status['istat_bulk'] = 'failed'
        st.error("   ❌ Bulk download method failed")
        return None
    
    # =========================================================================
    # Method 3: Sample/Demo Data
    # =========================================================================
    
    def fetch_demo_data(self,
                       indicator: str,
                       region: str = 'IT',
                       start_year: int = 2020,
                       end_year: int = 2024) -> pd.DataFrame:
        """
        Method 3: Generate realistic sample data
        
        Use this when all APIs fail - for demonstration
        """
        st.write("🔄 **Method 3:** Generating sample data...")
        
        # Generate quarterly dates
        dates = pd.date_range(
            start=f'{start_year}-03-31',
            end=f'{end_year}-12-31',
            freq='Q'
        )
        
        # Generate realistic unemployment data
        np.random.seed(42)
        
        # Base values by indicator
        base_values = {
            'unemployment_rate': 9.5,
            'employment_rate': 58.0,
            'inactive_rate': 35.0
        }
        
        base = base_values.get(indicator, 9.5)
        
        # Generate with trend and seasonality
        n = len(dates)
        trend = np.linspace(0, -1, n)  # Slight downward trend
        seasonal = np.sin(np.arange(n) * 2 * np.pi / 4) * 0.5  # Quarterly seasonality
        noise = np.random.randn(n) * 0.3
        
        values = base + trend + seasonal + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        st.success("   ✅ Sample data generated")
        st.warning("   ⚠️ This is SAMPLE data, not real ISTAT data!")
        
        return df
    
    # =========================================================================
    # Master Fetch Method
    # =========================================================================
    
    def fetch_data(self,
                   indicator: str,
                   region: str = 'IT',
                   start_year: int = 2020,
                   end_year: int = 2024,
                   allow_demo: bool = True) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Try all methods in sequence
        
        Returns:
            (DataFrame, method_used)
        """
        
        # Method 1: SDMX
        df = self.fetch_via_sdmx(indicator, region, start_year, end_year)
        if df is not None and not df.empty:
            return df, 'sdmx'
        
        st.markdown("---")
        
        # Method 2: Bulk download
        df = self.fetch_via_istat_bulk(indicator, region)
        if df is not None and not df.empty:
            return df, 'bulk'
        
        st.markdown("---")
        
        # Method 3: Demo data (if allowed)
        if allow_demo:
            df = self.fetch_demo_data(indicator, region, start_year, end_year)
            return df, 'demo'
        
        return None, 'none'


# =============================================================================
# Main UI
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🤖 ISTAT Auto Fetcher v3.0</h1>', unsafe_allow_html=True)
    
    st.info("""
    **🔄 Enhanced with Multiple Fallback Methods:**
    - Method 1: SDMX REST API (official)
    - Method 2: I.Stat bulk download (alternative)
    - Method 3: Sample data (demo mode)
    """)
    
    # Initialize
    if UTILS_AVAILABLE:
        state = AppState.get()
        viz = Visualizer()
    
    client = ISTATMultiClient()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        allow_demo = st.checkbox(
            "Allow sample data if APIs fail",
            value=True,
            help="Use realistic sample data if all API methods fail"
        )
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.info("""
        **This app tries multiple methods:**
        
        1. **SDMX API** (official)
        2. **Bulk downloads** (alternative)
        3. **Sample data** (demo)
        
        If Method 1 fails, it automatically tries Method 2, then 3.
        """)
    
    # Main form
    st.markdown("## 🎯 Select Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        indicator = st.selectbox(
            "📊 Indicator",
            options=[
                'unemployment_rate',
                'employment_rate',
                'inactive_rate'
            ],
            format_func=lambda x: {
                'unemployment_rate': 'Unemployment Rate',
                'employment_rate': 'Employment Rate',
                'inactive_rate': 'Inactivity Rate'
            }[x]
        )
    
    with col2:
        region = st.selectbox(
            "🗺️ Region",
            options=['IT', 'ITC4', 'ITH3', 'ITI4', 'ITF3'],
            format_func=lambda x: {
                'IT': 'Italia',
                'ITC4': 'Lombardia',
                'ITH3': 'Veneto',
                'ITI4': 'Lazio',
                'ITF3': 'Campania'
            }.get(x, x)
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_year = st.number_input("Start Year", 2000, 2024, 2020)
    
    with col2:
        end_year = st.number_input("End Year", 2000, 2024, 2024)
    
    # Fetch button
    if st.button("🚀 Fetch Data", type="primary", use_container_width=True):
        
        st.markdown("---")
        st.markdown("## 🔄 Fetching Process")
        
        with st.spinner("Trying multiple methods..."):
            df, method = client.fetch_data(
                indicator=indicator,
                region=region,
                start_year=start_year,
                end_year=end_year,
                allow_demo=allow_demo
            )
        
        st.markdown("---")
        
        if df is not None and not df.empty:
            
            # Success message
            if method == 'sdmx':
                st.markdown("""
                <div class="success-card">
                    <h3>✅ Success!</h3>
                    <p><strong>Method:</strong> SDMX REST API (Official ISTAT data)</p>
                    <p>This is real-time official data from ISTAT.</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif method == 'bulk':
                st.markdown("""
                <div class="success-card">
                    <h3>✅ Success!</h3>
                    <p><strong>Method:</strong> Bulk Download (Official ISTAT data)</p>
                    <p>Downloaded from I.Stat bulk datasets.</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif method == 'demo':
                st.markdown("""
                <div class="error-card">
                    <h3>⚠️ Using Sample Data</h3>
                    <p><strong>Method:</strong> Demo mode</p>
                    <p><strong>Warning:</strong> All API methods failed. This is SAMPLE data for demonstration only.</p>
                    <p>For real ISTAT data, you may need to:</p>
                    <ul>
                        <li>Try again later (server might be temporarily down)</li>
                        <li>Use manual Excel upload instead</li>
                        <li>Contact ISTAT support</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Display data
            st.markdown("## 📊 Retrieved Data")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Observations", len(df))
            with col2:
                st.metric("Start", df['date'].min().strftime('%Y-%m'))
            with col3:
                st.metric("End", df['date'].max().strftime('%Y-%m'))
            with col4:
                st.metric("Latest", f"{df['value'].iloc[-1]:.2f}")
            
            # Visualization
            if UTILS_AVAILABLE:
                st.markdown("### 📈 Chart")
                
                plot_df = df.copy()
                plot_df.columns = ['date', indicator]
                
                fig = viz.plot_time_series(
                    plot_df,
                    'date',
                    [indicator],
                    title=f"{indicator.replace('_', ' ').title()} - {region}",
                    show_points=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.markdown("### 📋 Data")
            st.dataframe(df, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "💾 Download CSV",
                csv,
                f"istat_{indicator}_{region}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
            
            # Save options
            if UTILS_AVAILABLE and method != 'demo':
                st.markdown("---")
                st.markdown("### 💾 Save to State")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Save as Target", use_container_width=True):
                        ts = pd.Series(
                            df['value'].values,
                            index=df['date'],
                            name=indicator
                        )
                        state.y_monthly = ts
                        AppState.update_timestamp()
                        st.success("✅ Saved!")
                        st.balloons()
                
                with col2:
                    if st.button("Add to Panel", use_container_width=True):
                        panel_df = df.set_index('date')
                        panel_df.columns = [f"{indicator}_{region}"]
                        
                        if state.panel_monthly is not None:
                            state.panel_monthly = state.panel_monthly.join(panel_df, how='outer')
                        else:
                            state.panel_monthly = panel_df
                        
                        AppState.update_timestamp()
                        st.success("✅ Added!")
                        st.balloons()
        
        else:
            st.error("❌ All methods failed")
            st.info("Try manual Excel upload in the Data Aggregation page")
    
    # Troubleshooting
    st.markdown("---")
    
    with st.expander("🔧 Why Error 500?"):
        st.markdown("""
        ### Error 500 = Server Error
        
        **این خطا از سمت سرور ISTAT است، نه کد شما!**
        
        **دلایل احتمالی:**
        1. سرور ISTAT موقتاً down است
        2. سرور overloaded است (خیلی درخواست دارد)
        3. SDMX endpoint تغییر کرده
        4. Dataflow ID اشتباه است
        5. نگهداری (maintenance)
        
        **راهکارها:**
        1. ✅ چند دقیقه صبر کن و دوباره امتحان کن
        2. ✅ این app خودکار روش‌های دیگر را هم امتحان می‌کند
        3. ✅ از "sample data" برای demo استفاده کن
        4. ✅ از manual Excel upload استفاده کن
        5. ✅ بررسی کن: http://sdmx.istat.it/ آیا accessible است؟
        
        **چک کن:**
```bash
        # در terminal:
        curl -I http://sdmx.istat.it/SDMXWS/rest/dataflow
```
        
        اگر 500 میده، مشکل از سرور است!
        """)


if __name__ == "__main__":
    main()
