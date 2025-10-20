"""
🎯 Auto Unemployment Fetcher Pro
=================================
Working version with reliable data sources.

Sources (in order):
1. Eurostat Bulk API (most reliable)
2. ISTAT SDMX (with correct flow IDs)
3. Demo fallback (only if all fail)

Author: ISTAT Nowcasting Team
Version: 5.0.0 (Production Ready)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import time

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Auto Fetcher Pro",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.5rem;
    }
    .source-eurostat {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
    }
    .source-istat {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    .source-demo {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HTTP Session with Retries
# =============================================================================

@st.cache_resource
def get_http_session():
    """Create HTTP session with retry logic"""
    session = requests.Session()
    
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, application/vnd.sdmx.data+json, text/csv, */*',
        'Accept-Encoding': 'gzip, deflate, br',
    })
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

HTTP = get_http_session()

# =============================================================================
# Eurostat Bulk Download (Most Reliable!)
# =============================================================================

class EurostatBulkFetcher:
    """
    Eurostat Bulk Download - این خیلی پایداره!
    
    استفاده از TSV bulk files به جای API
    """
    
    # Bulk download URLs for unemployment
    BULK_URLS = {
        'monthly': 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/une_rt_m',
        'quarterly': 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/une_rt_q',
    }
    
    def __init__(self):
        self.session = HTTP
    
    def fetch(self,
             geo: str = 'IT',
             sex: str = 'T',
             age: str = 'Y15-74',
             s_adj: str = 'SA',
             start_year: int = 2015,
             end_year: int = 2024) -> Optional[pd.DataFrame]:
        """
        Fetch from Eurostat Bulk API
        
        Args:
            geo: Country code (IT, DE, FR, etc.)
            sex: T (Total), M (Male), F (Female)
            age: Y15-74, Y15-24, Y25-74, TOTAL
            s_adj: SA, NSA, TC
            start_year: Start year
            end_year: End year
        
        Returns:
            DataFrame with columns: date, value
        """
        
        st.write("🔄 **Trying Eurostat Bulk SDMX...**")
        
        # Build SDMX query
        # Format: FREQ.S_ADJ.AGE.UNIT.SEX.GEO
        key = f"M.{s_adj}.{age}.PC_ACT.{sex}.{geo}"
        
        url = f"{self.BULK_URLS['monthly']}/{key}"
        
        params = {
            'startPeriod': start_year,
            'endPeriod': end_year,
            'format': 'jsonstat'  # Easier to parse
        }
        
        try:
            st.write(f"   URL: `{url}`")
            
            response = self.session.get(url, params=params, timeout=30)
            
            st.write(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                
                # Try to parse
                try:
                    data = response.json()
                    df = self._parse_jsonstat(data)
                    
                    if df is not None and not df.empty:
                        st.success(f"   ✅ Got {len(df)} observations from Eurostat!")
                        return df
                    else:
                        st.warning("   ⚠️ Empty data")
                
                except Exception as e:
                    st.error(f"   ❌ Parse error: {e}")
                    
                    # Try alternative parsing
                    try:
                        df = self._parse_sdmx_json(response.json())
                        if df is not None and not df.empty:
                            st.success(f"   ✅ Got {len(df)} with alternative parser!")
                            return df
                    except:
                        pass
            
            else:
                st.warning(f"   ⚠️ HTTP {response.status_code}")
                if response.status_code == 404:
                    st.info("   💡 Try different age/sex combination")
        
        except Exception as e:
            st.error(f"   ❌ Request failed: {e}")
        
        return None
    
    def _parse_jsonstat(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse JSON-stat format"""
        try:
            # JSON-stat structure
            if 'dimension' not in data:
                # Try to find dataset
                if isinstance(data, dict) and 'dataset' in data:
                    data = data['dataset']
                elif isinstance(data, dict):
                    # Sometimes nested
                    for key, val in data.items():
                        if isinstance(val, dict) and 'dimension' in val:
                            data = val
                            break
            
            if 'dimension' not in data:
                return None
            
            dimension = data['dimension']
            value_array = data.get('value', [])
            
            # Get time dimension
            time_cat = dimension.get('time', {}).get('category', {})
            time_index = time_cat.get('index', {})
            
            if not time_index:
                return None
            
            # Get time codes in order
            if isinstance(time_index, dict):
                time_codes = [k for k, v in sorted(time_index.items(), key=lambda x: x[1])]
            elif isinstance(time_index, list):
                time_codes = time_index
            else:
                return None
            
            # Build records
            records = []
            
            if isinstance(value_array, dict):
                # Sparse format
                for idx_str, val in value_array.items():
                    idx = int(idx_str)
                    if idx < len(time_codes) and val is not None:
                        time_code = time_codes[idx]
                        records.append({
                            'time': time_code,
                            'value': float(val)
                        })
            elif isinstance(value_array, list):
                # Dense format
                for idx, val in enumerate(value_array):
                    if idx < len(time_codes) and val is not None:
                        time_code = time_codes[idx]
                        records.append({
                            'time': time_code,
                            'value': float(val)
                        })
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            df['date'] = df['time'].apply(self._parse_time)
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')
            
            return df[['date', 'value']]
        
        except Exception as e:
            st.write(f"   Parse error: {e}")
            return None
    
    def _parse_sdmx_json(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse SDMX-JSON format (alternative)"""
        try:
            # Navigate to data
            if 'data' in data:
                data_root = data['data']
            else:
                data_root = data
            
            if 'dataSets' not in data_root:
                return None
            
            datasets = data_root['dataSets']
            if not datasets:
                return None
            
            dataset = datasets[0]
            
            # Get series
            series = dataset.get('series', {})
            
            if not series:
                # Try observations directly
                observations = dataset.get('observations', {})
                if observations:
                    series = {'0': {'observations': observations}}
            
            # Get time dimension
            structure = data_root.get('structure', {})
            dimensions = structure.get('dimensions', {})
            obs_dims = dimensions.get('observation', [])
            
            time_values = None
            for dim in obs_dims:
                if dim.get('id') in ['TIME_PERIOD', 'time', 'TIME']:
                    values = dim.get('values', [])
                    time_values = [v.get('id') or v.get('name') for v in values]
                    break
            
            if not time_values:
                return None
            
            # Extract data
            records = []
            
            for series_key, series_data in series.items():
                observations = series_data.get('observations', {})
                
                for obs_idx, obs_value in observations.items():
                    time_idx = int(obs_idx)
                    
                    if time_idx < len(time_values):
                        time_period = time_values[time_idx]
                        
                        if isinstance(obs_value, list):
                            value = obs_value[0]
                        else:
                            value = obs_value
                        
                        if value is not None:
                            records.append({
                                'time': time_period,
                                'value': float(value)
                            })
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            df['date'] = df['time'].apply(self._parse_time)
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')
            
            return df[['date', 'value']]
        
        except Exception as e:
            st.write(f"   Alternative parse error: {e}")
            return None
    
    def _parse_time(self, time_str: str) -> Optional[pd.Timestamp]:
        """Parse time period to datetime"""
        try:
            time_str = str(time_str).strip()
            
            # Monthly: 2024-01
            if len(time_str) == 7 and '-' in time_str:
                return pd.to_datetime(time_str + '-01') + pd.offsets.MonthEnd(0)
            
            # Quarterly: 2024-Q1
            if '-Q' in time_str:
                year, quarter = time_str.split('-Q')
                year = int(year)
                quarter = int(quarter)
                month = quarter * 3
                return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            
            # Year: 2024
            if len(time_str) == 4 and time_str.isdigit():
                return pd.Timestamp(year=int(time_str), month=12, day=31)
            
            # Try pandas
            return pd.to_datetime(time_str)
        
        except:
            return None


# =============================================================================
# ISTAT SDMX (Backup)
# =============================================================================

class ISTATFetcher:
    """
    ISTAT SDMX fetcher با flow IDs درست
    """
    
    # Known working endpoints
    ENDPOINTS = [
        'http://sdmx.istat.it/SDMXWS/rest',
        'https://sdmx.istat.it/SDMXWS/rest',
    ]
    
    # Known dataflows for unemployment
    # این flow IDs واقعی ISTAT هستند
    FLOWS = {
        'unemployment_monthly': 'DCCV_TAXDISOCC1',
        'unemployment_quarterly': 'DCCV_TAXDISOCC',
    }
    
    def __init__(self):
        self.session = HTTP
    
    def fetch(self,
             geo: str = 'IT',
             sex: str = '9',
             age: str = 'Y15-74',
             start_year: int = 2015,
             end_year: int = 2024) -> Optional[pd.DataFrame]:
        """Fetch from ISTAT"""
        
        st.write("🔄 **Trying ISTAT SDMX...**")
        
        # Try monthly first
        for flow_id in self.FLOWS.values():
            for endpoint in self.ENDPOINTS:
                
                # Different key structures to try
                keys = [
                    f"M.{geo}.{sex}.{age}",
                    f"{geo}.{sex}.{age}",
                    f"M.{geo}..{sex}.{age}",
                ]
                
                for key in keys:
                    url = f"{endpoint}/data/{flow_id}/{key}"
                    
                    params = {
                        'startPeriod': start_year,
                        'endPeriod': end_year
                    }
                    
                    try:
                        st.write(f"   Trying: `{url[:100]}...`")
                        
                        response = self.session.get(url, params=params, timeout=20)
                        
                        st.write(f"   Status: {response.status_code}")
                        
                        if response.status_code == 200:
                            try:
                                data = response.json()
                                df = self._parse_sdmx(data)
                                
                                if df is not None and not df.empty:
                                    st.success(f"   ✅ Got {len(df)} from ISTAT!")
                                    return df
                            
                            except Exception as e:
                                st.write(f"   Parse error: {e}")
                        
                        elif response.status_code == 404:
                            st.write("   Not found, trying next...")
                            continue
                    
                    except Exception as e:
                        st.write(f"   Error: {e}")
                        continue
        
        st.warning("   ⚠️ ISTAT method exhausted")
        return None
    
    def _parse_sdmx(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse SDMX response"""
        try:
            # Similar to Eurostat parser
            data_root = data.get('data', data)
            datasets = data_root.get('dataSets', [])
            
            if not datasets:
                return None
            
            dataset = datasets[0]
            series = dataset.get('series', {})
            
            # Get time dimension
            structure = data_root.get('structure', {})
            dimensions = structure.get('dimensions', {})
            obs_dims = dimensions.get('observation', [])
            
            time_values = None
            for dim in obs_dims:
                if dim.get('id') in ['TIME_PERIOD', 'time']:
                    time_values = [v['id'] for v in dim.get('values', [])]
                    break
            
            if not time_values:
                return None
            
            # Extract
            records = []
            for s in series.values():
                for idx, val in (s.get('observations') or {}).items():
                    t_idx = int(idx)
                    if t_idx < len(time_values):
                        time_period = time_values[t_idx]
                        value = val[0] if isinstance(val, list) else val
                        
                        if value is not None:
                            records.append({
                                'time': time_period,
                                'value': float(value)
                            })
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            df['date'] = df['time'].apply(self._parse_time)
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')
            
            return df[['date', 'value']]
        
        except:
            return None
    
    def _parse_time(self, time_str: str) -> Optional[pd.Timestamp]:
        """Parse time"""
        try:
            time_str = str(time_str).strip()
            
            if len(time_str) == 7 and '-' in time_str:
                return pd.to_datetime(time_str + '-01') + pd.offsets.MonthEnd(0)
            
            if '-Q' in time_str:
                year, quarter = time_str.split('-Q')
                month = int(quarter) * 3
                return pd.Timestamp(int(year), month, 1) + pd.offsets.MonthEnd(0)
            
            return pd.to_datetime(time_str)
        except:
            return None


# =============================================================================
# Demo Data (Last Resort)
# =============================================================================

def generate_demo_data(start_year: int, end_year: int, base: float = 9.5) -> pd.DataFrame:
    """Generate realistic demo data"""
    
    dates = pd.date_range(
        start=f'{start_year}-01-01',
        end=f'{end_year}-12-31',
        freq='M'
    )
    
    n = len(dates)
    np.random.seed(42)
    
    # Seasonal + trend + noise
    t = np.arange(n)
    seasonal = np.sin(t * 2 * np.pi / 12) * 0.3
    trend = -np.linspace(0, 1.2, n)
    noise = np.random.randn(n) * 0.2
    
    values = base + seasonal + trend + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    })


# =============================================================================
# Main Fetcher
# =============================================================================

@dataclass
class FetchResult:
    """Result of fetch operation"""
    df: pd.DataFrame
    source: str  # 'EUROSTAT', 'ISTAT', or 'DEMO'
    success: bool
    message: str


class UnemploymentFetcher:
    """Main fetcher with fallback logic"""
    
    def __init__(self):
        self.eurostat = EurostatBulkFetcher()
        self.istat = ISTATFetcher()
    
    def fetch(self,
             geo: str = 'IT',
             sex: str = 'Total',
             age: str = 'Y15-74',
             s_adj: str = 'SA',
             start_year: int = 2015,
             end_year: int = 2024,
             try_demo: bool = True) -> FetchResult:
        """
        Fetch unemployment data with automatic fallback
        
        Priority:
        1. Eurostat (most reliable)
        2. ISTAT (for Italy only)
        3. Demo (if allowed)
        """
        
        # Map sex
        sex_map = {'Total': 'T', 'Male': 'M', 'Female': 'F'}
        sex_code = sex_map.get(sex, 'T')
        
        # Map for ISTAT
        istat_sex = {'Total': '9', 'Male': '1', 'Female': '2'}
        istat_sex_code = istat_sex.get(sex, '9')
        
        st.markdown("---")
        st.markdown("## 🔄 Fetching Process")
        
        # Method 1: Eurostat
        st.markdown("### Method 1: Eurostat")
        
        df = self.eurostat.fetch(
            geo=geo,
            sex=sex_code,
            age=age,
            s_adj=s_adj,
            start_year=start_year,
            end_year=end_year
        )
        
        if df is not None and not df.empty:
            return FetchResult(
                df=df,
                source='EUROSTAT',
                success=True,
                message='Successfully fetched from Eurostat'
            )
        
        st.markdown("---")
        
        # Method 2: ISTAT (only for Italy)
        if geo == 'IT':
            st.markdown("### Method 2: ISTAT")
            
            df = self.istat.fetch(
                geo=geo,
                sex=istat_sex_code,
                age=age,
                start_year=start_year,
                end_year=end_year
            )
            
            if df is not None and not df.empty:
                return FetchResult(
                    df=df,
                    source='ISTAT',
                    success=True,
                    message='Successfully fetched from ISTAT'
                )
            
            st.markdown("---")
        
        # Method 3: Demo
        if try_demo:
            st.markdown("### Method 3: Demo Data")
            st.warning("⚠️ All APIs failed. Generating demo data...")
            
            df = generate_demo_data(start_year, end_year)
            
            return FetchResult(
                df=df,
                source='DEMO',
                success=False,
                message='All APIs failed - using demo data'
            )
        
        # Complete failure
        return FetchResult(
            df=pd.DataFrame(),
            source='NONE',
            success=False,
            message='All methods failed and demo disabled'
        )


# =============================================================================
# UI
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🎯 Auto Unemployment Fetcher Pro</h1>', unsafe_allow_html=True)
    
    st.info("""
    **🚀 Automatic data fetching with smart fallbacks:**
    - **Method 1:** Eurostat Bulk API (most reliable)
    - **Method 2:** ISTAT SDMX (Italy only, backup)
    - **Method 3:** Demo data (last resort)
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        geo = st.selectbox(
            "🗺️ Country",
            options=['IT', 'DE', 'FR', 'ES', 'EU27_2020', 'PT', 'NL', 'BE'],
            format_func=lambda x: {
                'IT': '🇮🇹 Italy',
                'DE': '🇩🇪 Germany',
                'FR': '🇫🇷 France',
                'ES': '🇪🇸 Spain',
                'EU27_2020': '🇪🇺 EU27',
                'PT': '🇵🇹 Portugal',
                'NL': '🇳🇱 Netherlands',
                'BE': '🇧🇪 Belgium'
            }.get(x, x)
        )
        
        sex = st.selectbox(
            "👤 Sex",
            options=['Total', 'Male', 'Female']
        )
        
        age = st.selectbox(
            "🎂 Age Group",
            options=['Y15-74', 'Y15-24', 'Y25-74', 'TOTAL'],
            format_func=lambda x: {
                'Y15-74': '15-74 years',
                'Y15-24': '15-24 years (youth)',
                'Y25-74': '25-74 years',
                'TOTAL': 'Total (all ages)'
            }[x]
        )
        
        s_adj = st.selectbox(
            "📊 Adjustment",
            options=['SA', 'NSA'],
            format_func=lambda x: {
                'SA': 'Seasonally adjusted',
                'NSA': 'Not seasonally adjusted'
            }[x]
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_year = st.number_input("Start Year", 2000, 2024, 2015)
        
        with col2:
            end_year = st.number_input("End Year", 2000, 2025, 2024)
        
        st.markdown("---")
        
        try_demo = st.checkbox("Allow demo data if APIs fail", value=True)
    
    # Fetch button
    if st.button("🚀 Fetch Data", type="primary", use_container_width=True):
        
        fetcher = UnemploymentFetcher()
        
        with st.spinner("Fetching unemployment data..."):
            result = fetcher.fetch(
                geo=geo,
                sex=sex,
                age=age,
                s_adj=s_adj,
                start_year=start_year,
                end_year=end_year,
                try_demo=try_demo
            )
        
        st.markdown("---")
        
        # Display result
        if result.success and result.source != 'DEMO':
            # Real data
            st.markdown(f"""
            <div class="source-badge source-{result.source.lower()}">
                ✅ Source: {result.source}
            </div>
            """, unsafe_allow_html=True)
            
            st.success(result.message)
        
        elif result.source == 'DEMO':
            # Demo data
            st.markdown("""
            <div class="source-badge source-demo">
                ⚠️ Source: DEMO DATA
            </div>
            """, unsafe_allow_html=True)
            
            st.warning(result.message)
            st.error("""
            **This is sample data, not real statistics!**
            
            **Possible solutions:**
            - Wait a few minutes and try again (servers may be busy)
            - Try different country (e.g., Germany, France)
            - Use manual Excel upload instead
            - Check your internet connection
            """)
        
        else:
            st.error("❌ All methods failed")
            st.stop()
        
        # Show data
        df = result.df
        
        if df.empty:
            st.error("No data returned")
            st.stop()
        
        st.markdown("## 📊 Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Observations", len(df))
        
        with col2:
            st.metric("📅 Start", df['date'].min().strftime('%Y-%m'))
        
        with col3:
            st.metric("📅 End", df['date'].max().strftime('%Y-%m'))
        
        with col4:
            st.metric("📌 Latest", f"{df['value'].iloc[-1]:.2f}%")
        
        # Chart
        st.markdown("### 📈 Time Series")
        
        df_plot = df.set_index('date')
        st.line_chart(df_plot['value'])
        
        # Table
        st.markdown("### 📋 Data Table")
        
        display_df = df.copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['value'] = display_df['value'].round(2)
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download
        st.markdown("### 💾 Download")
        
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"unemployment_{result.source}_{geo}_{start_year}_{end_year}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Save to state (if utils available)
        try:
            from utils.state import AppState
            
            st.markdown("---")
            st.markdown("### 💾 Save to Application")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save as Target", use_container_width=True):
                    state = AppState.get()
                    ts = pd.Series(
                        df['value'].values,
                        index=df['date'],
                        name='unemployment'
                    )
                    state.y_monthly = ts
                    AppState.update_timestamp()
                    st.success("✅ Saved to state.y_monthly!")
                    st.balloons()
            
            with col2:
                if st.button("💾 Add to Panel", use_container_width=True):
                    state = AppState.get()
                    
                    panel_df = df.set_index('date')
                    panel_df.columns = [f'unemployment_{geo}']
                    
                    if state.panel_monthly is not None:
                        state.panel_monthly = state.panel_monthly.join(panel_df, how='outer')
                    else:
                        state.panel_monthly = panel_df
                    
                    AppState.update_timestamp()
                    st.success("✅ Added to panel!")
                    st.balloons()
        
        except ImportError:
            pass
    
    # Info section
    st.markdown("---")
    
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        ### 🎯 How It Works
        
        This tool automatically tries multiple data sources in order:
        
        1. **Eurostat Bulk API** (Primary)
           - Most reliable and stable
           - Covers all EU countries
           - Monthly unemployment data
           - Official statistics
        
        2. **ISTAT SDMX** (Backup - Italy only)
           - Direct from Italian statistics
           - Used as fallback for Italy
           - May have different update schedule
        
        3. **Demo Data** (Last Resort)
           - Realistic synthetic data
           - Only if all APIs fail
           - **Clearly marked as demo**
        
        ### 📊 Data Quality
        
        - **Eurostat/ISTAT**: Official government statistics
        - **Demo**: Sample data for testing only
        
        ### 🔧 Troubleshooting
        
        If you get demo data:
        - Try different country (Germany/France very reliable)
        - Wait a few minutes (servers may be busy)
        - Check internet connection
        - Use manual upload as alternative
        """)
    
    with st.expander("🔍 Technical Details"):
        st.markdown("""
        ### API Endpoints Used
        
        **Eurostat:**
        - `https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/une_rt_m`
        - Format: SDMX 2.1 / JSON-stat
        - Dataset: une_rt_m (monthly unemployment rate)
        
        **ISTAT:**
        - `http://sdmx.istat.it/SDMXWS/rest`
        - Dataflow: DCCV_TAXDISOCC1 (monthly)
        - Format: SDMX JSON
        
        ### Error Handling
        
        - Automatic retries (3x)
        - Multiple endpoint fallbacks
        - Smart parsing (JSON-stat + SDMX)
        - Clear error messages
        """)


if __name__ == "__main__":
    main()
