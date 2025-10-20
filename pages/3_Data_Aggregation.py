"""
🤖 ISTAT Auto Data Fetcher v2.0
================================
Automatic data fetching from ISTAT without API key.
Uses public SDMX REST API.

Author: ISTAT Nowcasting Team
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict, List, Tuple, Any
import xml.etree.ElementTree as ET
from datetime import datetime
import json
from io import StringIO
import time

# Import utils
try:
    from utils.state import AppState
    from utils.istat_handler import ISTATHandler
    from utils.visualizer import Visualizer
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    st.error("⚠️ Utils not available")

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="ISTAT Auto Fetcher",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #059669, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# ISTAT SDMX API Client
# =============================================================================

class ISTATSDMXClient:
    """
    Client for ISTAT SDMX REST API
    
    Official documentation:
    - https://www.istat.it/en/methods-and-tools/web-services
    - http://sdmx.istat.it/
    
    No API key required - Public access!
    """
    
    BASE_URL = "http://sdmx.istat.it/SDMXWS/rest"
    
    # Known dataflows for unemployment
    DATAFLOWS = {
        'unemployment_rate': {
            'id': '22_781',
            'name': 'Tasso di disoccupazione',
            'description': 'Unemployment rate by region, sex, age'
        },
        'unemployment_level': {
            'id': '22_789', 
            'name': 'Disoccupati',
            'description': 'Number of unemployed persons'
        },
        'employment_rate': {
            'id': '22_777',
            'name': 'Tasso di occupazione',
            'description': 'Employment rate'
        },
        'inactive_rate': {
            'id': '22_783',
            'name': 'Tasso di inattività',
            'description': 'Inactivity rate'
        }
    }
    
    # Italian regions
    REGIONS = {
        'IT': 'Italia',
        'ITC1': 'Piemonte',
        'ITC2': 'Valle d\'Aosta',
        'ITC4': 'Lombardia',
        'ITH1': 'Provincia Autonoma Bolzano',
        'ITH2': 'Provincia Autonoma Trento',
        'ITH3': 'Veneto',
        'ITH4': 'Friuli-Venezia Giulia',
        'ITH5': 'Emilia-Romagna',
        'ITI1': 'Toscana',
        'ITI2': 'Umbria',
        'ITI3': 'Marche',
        'ITI4': 'Lazio',
        'ITF1': 'Abruzzo',
        'ITF2': 'Molise',
        'ITF3': 'Campania',
        'ITF4': 'Puglia',
        'ITF5': 'Basilicata',
        'ITF6': 'Calabria',
        'ITG1': 'Sicilia',
        'ITG2': 'Sardegna'
    }
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ISTAT-Nowcasting-Lab/2.0',
            'Accept': 'application/json, application/xml'
        })
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test connection to ISTAT SDMX API
        
        Returns:
            (success, message)
        """
        try:
            url = f"{self.BASE_URL}/dataflow"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return True, "✅ Connection successful!"
            else:
                return False, f"❌ Server returned {response.status_code}"
        
        except requests.exceptions.Timeout:
            return False, "⏱️ Connection timeout - server slow or unreachable"
        
        except requests.exceptions.ConnectionError:
            return False, "🔌 Connection error - check internet"
        
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def get_available_dataflows(self) -> Dict[str, Dict]:
        """Get list of available dataflows"""
        return self.DATAFLOWS.copy()
    
    def fetch_data(self,
                   dataflow_id: str,
                   region: str = 'IT',
                   start_period: Optional[str] = None,
                   end_period: Optional[str] = None,
                   sex: str = '9',  # 9=Total, 1=Males, 2=Females
                   age: str = 'Y15-64') -> Optional[pd.DataFrame]:
        """
        Fetch data from ISTAT SDMX API
        
        Args:
            dataflow_id: Dataflow ID (e.g., '22_781')
            region: Region code (e.g., 'IT', 'ITC4')
            start_period: Start year (e.g., '2020')
            end_period: End year (e.g., '2024')
            sex: Sex code (9=Total, 1=Male, 2=Female)
            age: Age group (e.g., 'Y15-64', 'Y15-24')
        
        Returns:
            DataFrame with time series data
        """
        
        # Build URL
        # Format: /data/{dataflow}/{key}?startPeriod=X&endPeriod=Y
        
        # Key structure varies by dataflow, try common patterns
        keys = [
            f"Q.{region}.{sex}.{age}",  # Quarterly
            f"A.{region}.{sex}.{age}",  # Annual
            f"M.{region}.{sex}.{age}",  # Monthly
        ]
        
        params = {}
        if start_period:
            params['startPeriod'] = start_period
        if end_period:
            params['endPeriod'] = end_period
        
        # Try each key pattern
        for key in keys:
            try:
                url = f"{self.BASE_URL}/data/{dataflow_id}/{key}"
                
                st.write(f"🔍 Trying: `{url}`")
                
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    headers={'Accept': 'application/json'}
                )
                
                if response.status_code == 200:
                    # Try to parse JSON
                    try:
                        data = response.json()
                        df = self._parse_sdmx_json(data)
                        
                        if df is not None and not df.empty:
                            st.success(f"✅ Data fetched with key: `{key}`")
                            return df
                    
                    except json.JSONDecodeError:
                        # Try XML
                        df = self._parse_sdmx_xml(response.text)
                        if df is not None and not df.empty:
                            st.success(f"✅ Data fetched (XML) with key: `{key}`")
                            return df
                
                elif response.status_code == 404:
                    continue  # Try next key
                
                else:
                    st.warning(f"⚠️ Status {response.status_code} for key: {key}")
            
            except Exception as e:
                st.warning(f"⚠️ Error with key {key}: {e}")
                continue
        
        return None
    
    def _parse_sdmx_json(self, data: dict) -> Optional[pd.DataFrame]:
        """
        Parse SDMX JSON response
        
        SDMX JSON structure (simplified):
        {
            "data": {
                "dataSets": [{
                    "series": {
                        "0:0:0": {
                            "observations": {
                                "0": [value],
                                "1": [value]
                            }
                        }
                    }
                }],
                "structure": {
                    "dimensions": {
                        "observation": [{
                            "id": "TIME_PERIOD",
                            "values": [...]
                        }]
                    }
                }
            }
        }
        """
        try:
            # Navigate structure
            if 'data' not in data:
                return None
            
            data_root = data['data']
            
            if 'dataSets' not in data_root or len(data_root['dataSets']) == 0:
                return None
            
            dataset = data_root['dataSets'][0]
            
            if 'series' not in dataset:
                return None
            
            # Get time dimension
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
            
            # Extract series
            all_data = []
            
            for series_key, series_data in dataset['series'].items():
                observations = series_data.get('observations', {})
                
                for obs_idx, obs_value in observations.items():
                    time_idx = int(obs_idx)
                    
                    if time_idx < len(time_values):
                        time_period = time_values[time_idx]
                        value = obs_value[0] if isinstance(obs_value, list) else obs_value
                        
                        all_data.append({
                            'time_period': time_period,
                            'value': value
                        })
            
            if not all_data:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(all_data)
            
            # Parse time period
            df['date'] = df['time_period'].apply(self._parse_time_period)
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')
            
            # Convert value to numeric
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            
            return df[['date', 'value']]
        
        except Exception as e:
            st.error(f"JSON parsing error: {e}")
            return None
    
    def _parse_sdmx_xml(self, xml_text: str) -> Optional[pd.DataFrame]:
        """Parse SDMX XML response"""
        try:
            root = ET.fromstring(xml_text)
            
            # Find namespace
            ns = {'ns': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message'}
            
            # Try to find data
            data = []
            
            for series in root.findall('.//ns:Series', ns):
                for obs in series.findall('.//ns:Obs', ns):
                    time_elem = obs.find('.//ns:ObsDimension', ns)
                    value_elem = obs.find('.//ns:ObsValue', ns)
                    
                    if time_elem is not None and value_elem is not None:
                        time_period = time_elem.get('value')
                        value = value_elem.get('value')
                        
                        if time_period and value:
                            data.append({
                                'time_period': time_period,
                                'value': float(value)
                            })
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df['date'] = df['time_period'].apply(self._parse_time_period)
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')
            
            return df[['date', 'value']]
        
        except Exception as e:
            st.error(f"XML parsing error: {e}")
            return None
    
    def _parse_time_period(self, period: str) -> Optional[pd.Timestamp]:
        """
        Parse SDMX time period to datetime
        
        Formats:
        - 2020-Q1 → 2020-03-31
        - 2020-01 → 2020-01-31
        - 2020 → 2020-12-31
        """
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
            elif len(period) == 4 and period.isdigit():
                return pd.Timestamp(year=int(period), month=12, day=31)
            
            else:
                # Try general parsing
                return pd.to_datetime(period)
        
        except:
            return None


# =============================================================================
# Fallback: Direct CSV Download
# =============================================================================

class ISTATDirectDownloader:
    """
    Fallback method: Direct CSV downloads from I.Stat
    """
    
    # Known direct download URLs for unemployment data
    DIRECT_URLS = {
        'unemployment_monthly': 'http://dati.istat.it/OECDStat_Metadata/ShowMetadata.ashx?Dataset=DCCV_TAXDISOCC1&Lang=en',
        'unemployment_quarterly': 'http://dati.istat.it/OECDStat_Metadata/ShowMetadata.ashx?Dataset=DCCV_TAXDISOCC&Lang=en',
    }
    
    def __init__(self):
        self.session = requests.Session()
    
    def download_csv(self, url: str) -> Optional[pd.DataFrame]:
        """
        Try to download CSV from I.Stat
        
        Note: I.Stat URLs often require session cookies
        """
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Try to parse as CSV
                df = pd.read_csv(StringIO(response.text))
                return df
        
        except Exception as e:
            st.error(f"Download error: {e}")
        
        return None


# =============================================================================
# Main UI
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🤖 ISTAT Auto Data Fetcher</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Automatic data retrieval from ISTAT public APIs - No API key required!</p>', unsafe_allow_html=True)
    
    # Initialize
    if UTILS_AVAILABLE:
        state = AppState.get()
        viz = Visualizer()
    
    client = ISTATSDMXClient()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **Data Source:**
        - ISTAT SDMX REST API
        - Public access (no key)
        - Real-time official data
        
        **Supported:**
        - Unemployment rate
        - Employment rate
        - Inactivity rate
        - All Italian regions
        """)
        
        st.markdown("---")
        
        # Connection test
        if st.button("🔌 Test Connection", use_container_width=True):
            with st.spinner("Testing..."):
                success, message = client.test_connection()
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Main content
    st.markdown("## 🎯 Select Data to Fetch")
    
    # Dataflow selection
    dataflows = client.get_available_dataflows()
    
    dataflow_names = {k: f"{v['name']} ({k})" for k, v in dataflows.items()}
    
    selected_flow_key = st.selectbox(
        "📊 Indicator",
        options=list(dataflow_names.keys()),
        format_func=lambda x: dataflow_names[x]
    )
    
    selected_flow = dataflows[selected_flow_key]
    
    st.info(f"**Description:** {selected_flow['description']}")
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region_code = st.selectbox(
            "🗺️ Region",
            options=list(client.REGIONS.keys()),
            format_func=lambda x: f"{client.REGIONS[x]} ({x})"
        )
    
    with col2:
        sex = st.selectbox(
            "👤 Sex",
            options=['9', '1', '2'],
            format_func=lambda x: {'9': 'Total', '1': 'Males', '2': 'Females'}[x]
        )
    
    with col3:
        age = st.selectbox(
            "🎂 Age Group",
            options=['Y15-64', 'Y15-24', 'Y25-34', 'Y35-44', 'Y45-54', 'Y55-64'],
            index=0
        )
    
    # Time range
    col1, col2 = st.columns(2)
    
    with col1:
        start_year = st.number_input(
            "📅 Start Year",
            min_value=2000,
            max_value=2024,
            value=2015
        )
    
    with col2:
        end_year = st.number_input(
            "📅 End Year",
            min_value=2000,
            max_value=2024,
            value=2024
        )
    
    # Fetch button
    st.markdown("---")
    
    if st.button("🚀 Fetch Data from ISTAT", type="primary", use_container_width=True):
        
        st.markdown("### 🔄 Fetching Data...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Connection test
            status_text.text("Step 1/3: Testing connection...")
            progress_bar.progress(0.1)
            
            success, message = client.test_connection()
            if not success:
                st.error(message)
                st.stop()
            
            # Step 2: Fetch data
            status_text.text("Step 2/3: Fetching data from ISTAT...")
            progress_bar.progress(0.3)
            
            df = client.fetch_data(
                dataflow_id=selected_flow['id'],
                region=region_code,
                start_period=str(start_year),
                end_period=str(end_year),
                sex=sex,
                age=age
            )
            
            progress_bar.progress(0.7)
            
            if df is None or df.empty:
                st.error("❌ No data returned")
                st.info("**Possible reasons:**\n- Data not available for selected parameters\n- Different key structure needed\n- Try different region/year")
                st.stop()
            
            # Step 3: Process
            status_text.text("Step 3/3: Processing data...")
            progress_bar.progress(0.9)
            
            # Sort and clean
            df = df.sort_values('date').reset_index(drop=True)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Complete!")
            
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            st.markdown("## ✅ Data Retrieved Successfully!")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Observations", len(df))
            
            with col2:
                st.metric("📅 Start", df['date'].min().strftime('%Y-%m'))
            
            with col3:
                st.metric("📅 End", df['date'].max().strftime('%Y-%m'))
            
            with col4:
                st.metric("📈 Latest Value", f"{df['value'].iloc[-1]:.2f}")
            
            # Visualization
            if UTILS_AVAILABLE:
                st.markdown("### 📊 Visualization")
                
                ts_df = df.copy()
                ts_df.columns = ['date', 'unemployment_rate']
                
                fig = viz.plot_time_series(
                    ts_df,
                    'date',
                    ['unemployment_rate'],
                    title=f"{selected_flow['name']} - {client.REGIONS[region_code]}",
                    show_points=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.markdown("### 📋 Data Table")
            
            display_df = df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Date', 'Value']
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download option
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="💾 Download as CSV",
                data=csv,
                file_name=f"istat_{selected_flow_key}_{region_code}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Save to state
            if UTILS_AVAILABLE:
                st.markdown("---")
                st.markdown("### 💾 Save to Application State")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save as Target (y_monthly)", use_container_width=True):
                        ts = pd.Series(
                            df['value'].values,
                            index=df['date'],
                            name='unemployment_rate'
                        )
                        state.y_monthly = ts
                        AppState.update_timestamp()
                        st.success("✅ Saved to state.y_monthly!")
                        st.balloons()
                
                with col2:
                    if st.button("💾 Save as Panel", use_container_width=True):
                        panel_df = df.set_index('date')
                        panel_df.columns = [f"{selected_flow_key}_{region_code}"]
                        
                        if state.panel_monthly is not None:
                            state.panel_monthly = state.panel_monthly.join(panel_df, how='outer')
                        else:
                            state.panel_monthly = panel_df
                        
                        AppState.update_timestamp()
                        st.success("✅ Added to panel!")
                        st.balloons()
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            
            with st.expander("🐛 Debug Info"):
                import traceback
                st.code(traceback.format_exc())
    
    # Info section
    st.markdown("---")
    
    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
        ### 🔧 Technical Details
        
        **Data Source:**
        - ISTAT SDMX REST API
        - Public endpoint: `http://sdmx.istat.it/SDMXWS/rest`
        - No authentication required
        - Official ISTAT data
        
        **How it works:**
        1. Connects to ISTAT SDMX API
        2. Requests data with specific parameters (region, sex, age, time)
        3. Parses JSON/XML response
        4. Converts to clean time series
        5. Ready for nowcasting!
        
        **Advantages:**
        - ✅ Real-time official data
        - ✅ No API key needed
        - ✅ Always up-to-date
        - ✅ All regions available
        
        **Note:**
        - Some data might not be available for all parameter combinations
        - Data is typically quarterly or annual
        - Use interpolation for monthly nowcasting
        """)
    
    with st.expander("🔍 Troubleshooting"):
        st.markdown("""
        ### Common Issues
        
        **"No data returned"**
        - Data might not exist for selected parameters
        - Try different region or time period
        - Try "Total" instead of specific sex
        
        **"Connection timeout"**
        - ISTAT server might be slow
        - Check internet connection
        - Try again in a few minutes
        
        **"Parsing error"**
        - Server response format changed
        - Report the issue for code update
        
        **Need help?**
        - Check ISTAT documentation: https://www.istat.it/en/methods-and-tools/web-services
        - Try different dataflow
        - Contact ISTAT support
        """)


if __name__ == "__main__":
    main()
