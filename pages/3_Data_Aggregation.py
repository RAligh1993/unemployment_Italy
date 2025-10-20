"""
🎯 Professional Data Fetcher
============================
Individual indicator selection with complete filters.
Like official Eurostat/ISTAT websites!

Features:
- Individual indicator selection
- Complete dimension filters (sex, age, adjustment, etc.)
- Real ISTAT integration
- Multi-source support
- Professional UI

Author: ISTAT Nowcasting Team
Version: 8.0.0 (Professional)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import time

# =============================================================================
# Config
# =============================================================================

st.set_page_config(
    page_title="Data Fetcher Pro",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(120deg, #0891b2, #06b6d4, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .filter-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        margin: 1rem 0;
    }
    .dimension-label {
        font-weight: 600;
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    .indicator-box {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        margin: 0.75rem 0;
        transition: all 0.2s;
        cursor: pointer;
    }
    .indicator-box:hover {
        border-color: #06b6d4;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.15);
        transform: translateY(-2px);
    }
    .indicator-selected {
        border-color: #06b6d4;
        background: #ecfeff;
    }
    .source-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .tag-eurostat { background: #dbeafe; color: #1e40af; }
    .tag-istat { background: #d1fae5; color: #065f46; }
    .tag-monthly { background: #fef3c7; color: #92400e; }
    .tag-quarterly { background: #e0e7ff; color: #3730a3; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HTTP Session
# =============================================================================

@st.cache_resource
def get_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

HTTP = get_session()

# =============================================================================
# Indicator Catalog
# =============================================================================

@dataclass
class Indicator:
    """Complete indicator specification"""
    id: str
    name: str
    description: str
    source: str  # 'eurostat' or 'istat'
    dataset_code: str
    frequency: str  # 'M', 'Q', 'A'
    category: str
    
    # Available dimensions
    has_sex: bool = True
    has_age: bool = True
    has_adjustment: bool = True
    has_unit: bool = False
    
    # Default filters
    default_sex: str = 'T'
    default_age: str = 'Y15-74'
    default_adjustment: str = 'SA'
    default_unit: Optional[str] = None
    
    # Dimension options
    sex_options: List[str] = field(default_factory=lambda: ['T', 'M', 'F'])
    age_options: List[str] = field(default_factory=lambda: ['Y15-74', 'Y15-24', 'Y25-74', 'TOTAL'])
    adjustment_options: List[str] = field(default_factory=lambda: ['SA', 'NSA'])
    unit_options: List[str] = field(default_factory=list)

# Complete catalog
INDICATORS = {
    # =============================================================================
    # EUROSTAT - Labour Market
    # =============================================================================
    
    'unemp_rate_m': Indicator(
        'unemp_rate_m',
        'Unemployment Rate (Monthly)',
        'Monthly unemployment rate by sex, age - Eurostat',
        'eurostat',
        'une_rt_m',
        'M',
        'Labour Market',
        has_sex=True,
        has_age=True,
        has_adjustment=True,
        age_options=['Y15-74', 'Y15-24', 'Y25-74', 'TOTAL']
    ),
    
    'unemp_rate_q': Indicator(
        'unemp_rate_q',
        'Unemployment Rate (Quarterly)',
        'Quarterly unemployment rate - Eurostat',
        'eurostat',
        'une_rt_q',
        'Q',
        'Labour Market',
        has_sex=True,
        has_age=True,
        has_adjustment=True,
    ),
    
    'emp_rate_q': Indicator(
        'emp_rate_q',
        'Employment Rate (Quarterly)',
        'Employment rate by sex, age - Eurostat',
        'eurostat',
        'lfsi_emp_q',
        'Q',
        'Labour Market',
        has_sex=True,
        has_age=True,
        has_adjustment=False,
        age_options=['Y15-64', 'Y15-24', 'Y25-54', 'Y55-64']
    ),
    
    'job_vacancy_q': Indicator(
        'job_vacancy_q',
        'Job Vacancy Rate (Quarterly)',
        'Job vacancy rate by NACE sector - Eurostat',
        'eurostat',
        'jvs_q_nace2',
        'Q',
        'Labour Market',
        has_sex=False,
        has_age=False,
        has_adjustment=True,
    ),
    
    'labour_force_q': Indicator(
        'labour_force_q',
        'Labour Force Participation (Quarterly)',
        'Activity rate - Eurostat',
        'eurostat',
        'lfsi_act_q',
        'Q',
        'Labour Market',
        has_sex=True,
        has_age=True,
        has_adjustment=False,
    ),
    
    # =============================================================================
    # EUROSTAT - Economic Activity
    # =============================================================================
    
    'gdp_q': Indicator(
        'gdp_q',
        'GDP (Quarterly)',
        'Gross Domestic Product - Eurostat',
        'eurostat',
        'namq_10_gdp',
        'Q',
        'Economic Activity',
        has_sex=False,
        has_age=False,
        has_adjustment=True,
        has_unit=True,
        default_unit='CLV10_MNAC',
        unit_options=['CLV10_MNAC', 'CP_MNAC', 'PC_GDP']
    ),
    
    'ind_prod_m': Indicator(
        'ind_prod_m',
        'Industrial Production (Monthly)',
        'Industrial production index - Eurostat',
        'eurostat',
        'sts_inpr_m',
        'M',
        'Economic Activity',
        has_sex=False,
        has_age=False,
        has_adjustment=True,
        has_unit=True,
        default_unit='I15',
        unit_options=['I15', 'PCH_SM']
    ),
    
    'retail_m': Indicator(
        'retail_m',
        'Retail Sales (Monthly)',
        'Retail trade volume - Eurostat',
        'eurostat',
        'sts_trtu_m',
        'M',
        'Economic Activity',
        has_sex=False,
        has_age=False,
        has_adjustment=True,
        has_unit=True,
        default_unit='I15',
        unit_options=['I15', 'PCH_SM']
    ),
    
    'construction_m': Indicator(
        'construction_m',
        'Construction Production (Monthly)',
        'Construction production index - Eurostat',
        'eurostat',
        'sts_copr_m',
        'M',
        'Economic Activity',
        has_sex=False,
        has_age=False,
        has_adjustment=True,
    ),
    
    # =============================================================================
    # EUROSTAT - Prices
    # =============================================================================
    
    'hicp_m': Indicator(
        'hicp_m',
        'Inflation - HICP (Monthly)',
        'Harmonised Index of Consumer Prices - Eurostat',
        'eurostat',
        'prc_hicp_midx',
        'M',
        'Prices',
        has_sex=False,
        has_age=False,
        has_adjustment=False,
        has_unit=True,
        default_unit='I15',
        unit_options=['I15', 'RCH_A']
    ),
    
    'ppi_m': Indicator(
        'ppi_m',
        'Producer Prices (Monthly)',
        'Producer price index - Eurostat',
        'eurostat',
        'sts_inpp_m',
        'M',
        'Prices',
        has_sex=False,
        has_age=False,
        has_adjustment=False,
    ),
    
    # =============================================================================
    # EUROSTAT - Confidence
    # =============================================================================
    
    'consumer_conf_m': Indicator(
        'consumer_conf_m',
        'Consumer Confidence (Monthly)',
        'Consumer confidence indicator - Eurostat',
        'eurostat',
        'ei_bsco_m',
        'M',
        'Confidence',
        has_sex=False,
        has_age=False,
        has_adjustment=True,
    ),
    
    'business_conf_m': Indicator(
        'business_conf_m',
        'Business Confidence (Monthly)',
        'Business climate indicator - Eurostat',
        'eurostat',
        'ei_bsin_m',
        'M',
        'Confidence',
        has_sex=False,
        has_age=False,
        has_adjustment=True,
    ),
    
    # =============================================================================
    # ISTAT - Italy Specific
    # =============================================================================
    
    'istat_unemp_m': Indicator(
        'istat_unemp_m',
        'Unemployment Rate (Monthly) - ISTAT',
        'Tasso di disoccupazione mensile - ISTAT',
        'istat',
        'DCCV_TAXDISOCC1',
        'M',
        'Labour Market',
        has_sex=True,
        has_age=True,
        has_adjustment=True,
        sex_options=['9', '1', '2'],  # 9=Total, 1=Male, 2=Female
        age_options=['Y15-74', 'Y15-24', 'Y25-34', 'Y15-64']
    ),
    
    'istat_unemp_q': Indicator(
        'istat_unemp_q',
        'Unemployment Rate (Quarterly) - ISTAT',
        'Tasso di disoccupazione trimestrale - ISTAT',
        'istat',
        'DCCV_TAXDISOCC',
        'Q',
        'Labour Market',
        has_sex=True,
        has_age=True,
        has_adjustment=True,
        sex_options=['9', '1', '2'],
        age_options=['Y15-74', 'Y15-24', 'Y25-34']
    ),
    
    'istat_emp_m': Indicator(
        'istat_emp_m',
        'Employment Rate (Monthly) - ISTAT',
        'Tasso di occupazione - ISTAT',
        'istat',
        'DCCV_TAXOCCU1',
        'M',
        'Labour Market',
        has_sex=True,
        has_age=True,
        has_adjustment=True,
        sex_options=['9', '1', '2'],
        age_options=['Y15-64', 'Y15-24', 'Y25-34']
    ),
}

# =============================================================================
# Eurostat Fetcher
# =============================================================================

class EurostatFetcher:
    """Eurostat data fetcher with complete filters"""
    
    BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    
    def __init__(self):
        self.session = HTTP
    
    def fetch(self,
             indicator: Indicator,
             geo: str,
             start_year: int,
             end_year: int,
             filters: Dict[str, str]) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Fetch from Eurostat with complete filters
        
        Returns:
            (DataFrame, debug_message)
        """
        
        url = f"{self.BASE_URL}/{indicator.dataset_code}"
        
        # Build params
        params = {
            'lang': 'en',
            'geo': geo,
            **filters
        }
        
        debug = f"Fetching from Eurostat:\n"
        debug += f"URL: {url}\n"
        debug += f"Params: {params}\n"
        
        # Try different Accept headers
        accept_headers = [
            'application/json',
            'application/vnd.sdmx.data+json',
            '*/*'
        ]
        
        for accept in accept_headers:
            try:
                headers = {
                    'Accept': accept,
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=30
                )
                
                debug += f"Accept: {accept}\n"
                debug += f"Status: {response.status_code}\n"
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Try parsers
                        df = self._parse_jsonstat(data)
                        
                        if df is None or df.empty:
                            df = self._parse_sdmx(data)
                        
                        if df is not None and not df.empty:
                            # Filter by year
                            df = df[
                                (df['date'].dt.year >= start_year) &
                                (df['date'].dt.year <= end_year)
                            ]
                            
                            if not df.empty:
                                debug += f"✅ Success: {len(df)} observations\n"
                                return df, debug
                            else:
                                debug += "⚠️ Empty after date filter\n"
                        else:
                            debug += "⚠️ Parsers returned empty\n"
                    
                    except Exception as e:
                        debug += f"❌ Parse error: {e}\n"
                
                elif response.status_code == 406:
                    debug += "⚠️ 406 - trying next Accept\n"
                    continue
                
                else:
                    debug += f"❌ HTTP {response.status_code}\n"
            
            except Exception as e:
                debug += f"❌ Request error: {e}\n"
        
        debug += "❌ All attempts failed\n"
        return None, debug
    
    def _parse_jsonstat(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse JSON-stat"""
        try:
            # Navigate to data
            if 'dimension' not in data:
                for val in data.values():
                    if isinstance(val, dict) and 'dimension' in val:
                        data = val
                        break
            
            if 'dimension' not in data:
                return None
            
            dimension = data['dimension']
            value_array = data.get('value', {})
            
            # Find time
            time_key = None
            for key in ['time', 'TIME', 'TIME_PERIOD']:
                if key in dimension:
                    time_key = key
                    break
            
            if not time_key:
                return None
            
            time_cat = dimension[time_key].get('category', {})
            time_index = time_cat.get('index', {})
            
            if isinstance(time_index, dict):
                time_codes = [k for k, v in sorted(time_index.items(), key=lambda x: x[1])]
            elif isinstance(time_index, list):
                time_codes = time_index
            else:
                return None
            
            # Extract
            records = []
            
            if isinstance(value_array, dict):
                for idx_str, val in value_array.items():
                    idx = int(idx_str)
                    if idx < len(time_codes) and val is not None:
                        records.append({
                            'time': time_codes[idx],
                            'value': float(val)
                        })
            elif isinstance(value_array, list):
                for idx, val in enumerate(value_array):
                    if idx < len(time_codes) and val is not None:
                        records.append({
                            'time': time_codes[idx],
                            'value': float(val)
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
    
    def _parse_sdmx(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse SDMX"""
        try:
            if 'data' in data:
                data = data['data']
            
            dataSets = data.get('dataSets', [])
            if not dataSets:
                return None
            
            dataset = dataSets[0]
            
            structure = data.get('structure', {})
            dimensions = structure.get('dimensions', {})
            obs_dims = dimensions.get('observation', [])
            
            time_values = None
            for dim in obs_dims:
                if dim.get('id', '').upper() in ['TIME_PERIOD', 'TIME']:
                    time_values = [v.get('id', v.get('name')) for v in dim.get('values', [])]
                    break
            
            if not time_values:
                return None
            
            records = []
            series = dataset.get('series', {})
            
            if series:
                for s in series.values():
                    obs = s.get('observations', {})
                    for idx_str, val in obs.items():
                        idx = int(idx_str)
                        if idx < len(time_values):
                            value = val[0] if isinstance(val, list) else val
                            if value is not None:
                                records.append({
                                    'time': time_values[idx],
                                    'value': float(value)
                                })
            else:
                obs = dataset.get('observations', {})
                for idx_str, val in obs.items():
                    idx = int(idx_str)
                    if idx < len(time_values):
                        value = val[0] if isinstance(val, list) else val
                        if value is not None:
                            records.append({
                                'time': time_values[idx],
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
        """Parse time string"""
        try:
            time_str = str(time_str).strip()
            
            # Monthly
            if len(time_str) == 7:
                if '-' in time_str:
                    return pd.to_datetime(time_str + '-01') + pd.offsets.MonthEnd(0)
                elif 'M' in time_str:
                    year, month = time_str.split('M')
                    return pd.Timestamp(int(year), int(month), 1) + pd.offsets.MonthEnd(0)
            
            # Quarterly
            if '-Q' in time_str or 'Q' in time_str:
                if '-Q' in time_str:
                    year, quarter = time_str.split('-Q')
                else:
                    year = time_str[:4]
                    quarter = time_str[-1]
                
                month = int(quarter) * 3
                return pd.Timestamp(int(year), month, 1) + pd.offsets.MonthEnd(0)
            
            # Annual
            if len(time_str) == 4 and time_str.isdigit():
                return pd.Timestamp(int(time_str), 12, 31)
            
            return pd.to_datetime(time_str)
        
        except:
            return None

# =============================================================================
# ISTAT Fetcher (Fixed!)
# =============================================================================

class ISTATFetcher:
    """
    ISTAT fetcher with working endpoints
    """
    
    # Multiple endpoints to try
    ENDPOINTS = [
        'http://sdmx.istat.it/SDMXWS/rest',
        'https://sdmx.istat.it/SDMXWS/rest',
    ]
    
    def __init__(self):
        self.session = HTTP
    
    def fetch(self,
             indicator: Indicator,
             start_year: int,
             end_year: int,
             filters: Dict[str, str]) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Fetch from ISTAT
        
        Returns:
            (DataFrame, debug_message)
        """
        
        debug = f"Fetching from ISTAT:\n"
        debug += f"Dataset: {indicator.dataset_code}\n"
        debug += f"Filters: {filters}\n"
        
        # Build key from filters
        # ISTAT format: FREQ.GEO.SEX.AGE or variations
        
        # Get values
        freq = 'M' if indicator.frequency == 'M' else 'Q'
        geo = 'IT'
        sex = filters.get('sex', '9')
        age = filters.get('age', 'Y15-74')
        adj = filters.get('s_adj', 'Y')  # Y=SA, N=NSA
        
        # Try different key structures
        key_patterns = [
            f"{freq}.{geo}.{sex}.{age}",
            f"{freq}.{geo}.{adj}.{sex}.{age}",
            f"{geo}.{sex}.{age}",
            "all",  # Wildcard
        ]
        
        for endpoint in self.ENDPOINTS:
            for key in key_patterns:
                
                url = f"{endpoint}/data/{indicator.dataset_code}/{key}"
                
                params = {
                    'startPeriod': start_year,
                    'endPeriod': end_year,
                    'format': 'jsondata'
                }
                
                debug += f"\nTrying: {url}\n"
                
                try:
                    response = self.session.get(url, params=params, timeout=20)
                    
                    debug += f"Status: {response.status_code}\n"
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            df = self._parse_istat(data)
                            
                            if df is not None and not df.empty:
                                # Filter by year
                                df = df[
                                    (df['date'].dt.year >= start_year) &
                                    (df['date'].dt.year <= end_year)
                                ]
                                
                                if not df.empty:
                                    debug += f"✅ Success: {len(df)} obs\n"
                                    return df, debug
                        
                        except Exception as e:
                            debug += f"Parse error: {e}\n"
                    
                    elif response.status_code == 404:
                        debug += "404 - trying next\n"
                        continue
                    
                    elif response.status_code == 500:
                        debug += "500 - server error\n"
                        continue
                
                except Exception as e:
                    debug += f"Request error: {e}\n"
                    continue
        
        debug += "❌ All ISTAT attempts failed\n"
        return None, debug
    
    def _parse_istat(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse ISTAT JSON"""
        try:
            # Check structure
            if 'dataSets' in data:
                dataSets = data['dataSets']
            elif 'data' in data and 'dataSets' in data['data']:
                dataSets = data['data']['dataSets']
            else:
                return None
            
            if not dataSets:
                return None
            
            dataset = dataSets[0]
            
            # Get structure
            if 'structure' in data:
                structure = data['structure']
            elif 'data' in data and 'structure' in data['data']:
                structure = data['data']['structure']
            else:
                return None
            
            dimensions = structure.get('dimensions', {})
            obs_dims = dimensions.get('observation', [])
            
            # Find time dimension
            time_values = None
            for dim in obs_dims:
                dim_id = dim.get('id', '').upper()
                if 'TIME' in dim_id or 'PERIOD' in dim_id:
                    time_values = [v['id'] for v in dim.get('values', [])]
                    break
            
            if not time_values:
                return None
            
            # Extract observations
            records = []
            
            # Try series first
            series = dataset.get('series', {})
            if series:
                for s in series.values():
                    obs = s.get('observations', {})
                    for idx_str, val in obs.items():
                        idx = int(idx_str)
                        if idx < len(time_values):
                            value = val[0] if isinstance(val, list) else val
                            if value is not None:
                                records.append({
                                    'time': time_values[idx],
                                    'value': float(value)
                                })
            else:
                # Try direct observations
                obs = dataset.get('observations', {})
                for idx_str, val in obs.items():
                    idx = int(idx_str)
                    if idx < len(time_values):
                        value = val[0] if isinstance(val, list) else val
                        if value is not None:
                            records.append({
                                'time': time_values[idx],
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
            return None
    
    def _parse_time(self, time_str: str) -> Optional[pd.Timestamp]:
        """Parse time"""
        try:
            time_str = str(time_str).strip()
            
            # Monthly: 2024-01 or 2024M01
            if len(time_str) == 7:
                if '-' in time_str:
                    return pd.to_datetime(time_str + '-01') + pd.offsets.MonthEnd(0)
                elif 'M' in time_str:
                    year, month = time_str.split('M')
                    return pd.Timestamp(int(year), int(month), 1) + pd.offsets.MonthEnd(0)
            
            # Quarterly: 2024-Q1
            if '-Q' in time_str or 'Q' in time_str:
                if '-Q' in time_str:
                    year, quarter = time_str.split('-Q')
                else:
                    year = time_str[:4]
                    quarter = time_str[-1]
                
                month = int(quarter) * 3
                return pd.Timestamp(int(year), month, 1) + pd.offsets.MonthEnd(0)
            
            # Annual
            if len(time_str) == 4 and time_str.isdigit():
                return pd.Timestamp(int(time_str), 12, 31)
            
            return pd.to_datetime(time_str)
        
        except:
            return None

# =============================================================================
# UI
# =============================================================================

def main():
    st.markdown('<h1 class="main-title">🎯 Professional Data Fetcher</h1>', unsafe_allow_html=True)
    
    st.info("""
    **🎯 Individual indicator selection with complete filters**
    
    Select indicators one by one and customize all dimensions:
    - Sex (Total, Male, Female)
    - Age groups (15-74, 15-24, 25-74, etc.)
    - Seasonal adjustment (SA, NSA)
    - Units (for GDP, production indices)
    
    Like official Eurostat/ISTAT websites!
    """)
    
    # Initialize state
    if 'selected_indicators' not in st.session_state:
        st.session_state.selected_indicators = []
    
    if 'fetched_data' not in st.session_state:
        st.session_state.fetched_data = {}
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Global Settings")
        
        geo = st.selectbox(
            "🗺️ Country",
            options=['IT', 'DE', 'FR', 'ES', 'PT', 'NL', 'BE'],
            format_func=lambda x: {
                'IT': '🇮🇹 Italy',
                'DE': '🇩🇪 Germany',
                'FR': '🇫🇷 France',
                'ES': '🇪🇸 Spain',
                'PT': '🇵🇹 Portugal',
                'NL': '🇳🇱 Netherlands',
                'BE': '🇧🇪 Belgium'
            }[x]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_year = st.number_input("Start", 2000, 2024, 2020)
        
        with col2:
            end_year = st.number_input("End", 2000, 2024, 2024)
        
        st.markdown("---")
        
        show_debug = st.checkbox("Show debug info", value=False)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.selected_indicators = []
            st.session_state.fetched_data = {}
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Select Indicators", "🚀 Fetch Data", "💾 View Results"])
    
    # =============================================================================
    # TAB 1: Select
    # =============================================================================
    
    with tab1:
        st.markdown("## 📊 Select Indicators")
        
        # Group by category
        categories = {}
        for ind_id, ind in INDICATORS.items():
            if ind.category not in categories:
                categories[ind.category] = []
            categories[ind.category].append((ind_id, ind))
        
        # Filter by source
        source_filter = st.radio(
            "Filter by source:",
            ['All', 'Eurostat only', 'ISTAT only'],
            horizontal=True
        )
        
        for category, indicators in categories.items():
            st.markdown(f"### {category}")
            
            cols = st.columns(2)
            
            for idx, (ind_id, ind) in enumerate(indicators):
                
                # Apply source filter
                if source_filter == 'Eurostat only' and ind.source != 'eurostat':
                    continue
                if source_filter == 'ISTAT only' and ind.source != 'istat':
                    continue
                
                # ISTAT only for Italy
                if ind.source == 'istat' and geo != 'IT':
                    continue
                
                col = cols[idx % 2]
                
                with col:
                    is_selected = ind_id in st.session_state.selected_indicators
                    
                    box_class = "indicator-box indicator-selected" if is_selected else "indicator-box"
                    
                    st.markdown(f"""
                    <div class="{box_class}">
                        <h4>{ind.name}</h4>
                        <p style="font-size:0.9rem; color:#64748b;">{ind.description}</p>
                        <span class="source-tag tag-{ind.source}">{ind.source.upper()}</span>
                        <span class="source-tag tag-{ind.frequency.lower()}">{ind.frequency}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    button_label = "✅ Selected" if is_selected else "➕ Select"
                    button_type = "secondary" if is_selected else "primary"
                    
                    if st.button(
                        button_label,
                        key=f"select_{ind_id}",
                        use_container_width=True,
                        type=button_type
                    ):
                        if is_selected:
                            st.session_state.selected_indicators.remove(ind_id)
                        else:
                            st.session_state.selected_indicators.append(ind_id)
                        st.rerun()
        
        # Show selected
        if st.session_state.selected_indicators:
            st.markdown("---")
            st.markdown(f"### ✅ Selected: {len(st.session_state.selected_indicators)} indicators")
            
            for ind_id in st.session_state.selected_indicators:
                ind = INDICATORS[ind_id]
                st.markdown(f"- **{ind.name}** ({ind.source})")
    
    # =============================================================================
    # TAB 2: Fetch
    # =============================================================================
    
    with tab2:
        st.markdown("## 🚀 Fetch Data")
        
        if not st.session_state.selected_indicators:
            st.warning("⚠️ No indicators selected. Go to 'Select Indicators' tab first.")
            st.stop()
        
        st.success(f"✅ {len(st.session_state.selected_indicators)} indicators selected")
        
        # Show filters for each
        st.markdown("### 🎛️ Configure Filters")
        
        configs = {}
        
        for ind_id in st.session_state.selected_indicators:
            ind = INDICATORS[ind_id]
            
            with st.expander(f"⚙️ {ind.name}", expanded=True):
                
                filters = {}
                
                cols = st.columns(4)
                
                col_idx = 0
                
                # Sex
                if ind.has_sex:
                    with cols[col_idx]:
                        st.markdown('<p class="dimension-label">Sex</p>', unsafe_allow_html=True)
                        
                        sex_labels = {
                            'T': 'Total', 'M': 'Male', 'F': 'Female',
                            '9': 'Total', '1': 'Male', '2': 'Female'
                        }
                        
                        sex = st.selectbox(
                            "Sex",
                            options=ind.sex_options,
                            format_func=lambda x: sex_labels.get(x, x),
                            key=f"sex_{ind_id}",
                            label_visibility="collapsed"
                        )
                        filters['sex'] = sex
                    
                    col_idx += 1
                
                # Age
                if ind.has_age:
                    with cols[col_idx % 4]:
                        st.markdown('<p class="dimension-label">Age Group</p>', unsafe_allow_html=True)
                        
                        age = st.selectbox(
                            "Age",
                            options=ind.age_options,
                            key=f"age_{ind_id}",
                            label_visibility="collapsed"
                        )
                        filters['age'] = age
                    
                    col_idx += 1
                
                # Adjustment
                if ind.has_adjustment:
                    with cols[col_idx % 4]:
                        st.markdown('<p class="dimension-label">Adjustment</p>', unsafe_allow_html=True)
                        
                        adj_labels = {'SA': 'Seasonally adjusted', 'NSA': 'Not adjusted', 'Y': 'Adjusted', 'N': 'Not adjusted'}
                        
                        adj = st.selectbox(
                            "Adjustment",
                            options=ind.adjustment_options,
                            format_func=lambda x: adj_labels.get(x, x),
                            key=f"adj_{ind_id}",
                            label_visibility="collapsed"
                        )
                        filters['s_adj'] = adj
                    
                    col_idx += 1
                
                # Unit
                if ind.has_unit:
                    with cols[col_idx % 4]:
                        st.markdown('<p class="dimension-label">Unit</p>', unsafe_allow_html=True)
                        
                        unit = st.selectbox(
                            "Unit",
                            options=ind.unit_options,
                            key=f"unit_{ind_id}",
                            label_visibility="collapsed"
                        )
                        filters['unit'] = unit
                
                configs[ind_id] = filters
        
        # Fetch button
        st.markdown("---")
        
        if st.button("🚀 Fetch All Selected Indicators", type="primary", use_container_width=True):
            
            eurostat_fetcher = EurostatFetcher()
            istat_fetcher = ISTATFetcher()
            
            progress_bar = st.progress(0)
            status = st.empty()
            
            total = len(st.session_state.selected_indicators)
            
            for idx, ind_id in enumerate(st.session_state.selected_indicators):
                
                ind = INDICATORS[ind_id]
                filters = configs[ind_id]
                
                status.text(f"Fetching {idx+1}/{total}: {ind.name}...")
                
                # Fetch based on source
                if ind.source == 'eurostat':
                    df, debug = eurostat_fetcher.fetch(ind, geo, start_year, end_year, filters)
                else:  # istat
                    df, debug = istat_fetcher.fetch(ind, start_year, end_year, filters)
                
                if df is not None and not df.empty:
                    st.session_state.fetched_data[ind_id] = {
                        'data': df,
                        'indicator': ind,
                        'filters': filters,
                        'debug': debug
                    }
                    st.success(f"✅ {ind.name}: {len(df)} observations")
                else:
                    st.warning(f"⚠️ {ind.name}: No data")
                    
                    if show_debug:
                        with st.expander(f"Debug: {ind.name}"):
                            st.code(debug)
                
                progress_bar.progress((idx + 1) / total)
            
            progress_bar.empty()
            status.empty()
            
            st.success(f"✅ Fetched {len(st.session_state.fetched_data)}/{total} indicators")
            
            if len(st.session_state.fetched_data) > 0:
                st.info("💡 Go to 'View Results' tab to see your data!")
    
    # =============================================================================
    # TAB 3: Results
    # =============================================================================
    
    with tab3:
        st.markdown("## 💾 Fetched Data")
        
        if not st.session_state.fetched_data:
            st.info("No data fetched yet. Go to 'Fetch Data' tab.")
            st.stop()
        
        st.success(f"✅ {len(st.session_state.fetched_data)} indicators available")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_obs = sum(len(item['data']) for item in st.session_state.fetched_data.values())
            st.metric("📈 Total Observations", total_obs)
        
        with col2:
            # Date range
            all_dates = []
            for item in st.session_state.fetched_data.values():
                all_dates.extend(item['data']['date'].tolist())
            
            if all_dates:
                min_date = min(all_dates)
                max_date = max(all_dates)
                st.metric("📅 Date Range", f"{min_date.year}-{max_date.year}")
        
        with col3:
            eurostat_count = sum(1 for item in st.session_state.fetched_data.values() 
                               if item['indicator'].source == 'eurostat')
            istat_count = len(st.session_state.fetched_data) - eurostat_count
            st.metric("🌍 Sources", f"E:{eurostat_count} I:{istat_count}")
        
        st.markdown("---")
        
        # Show each
        for ind_id, item in st.session_state.fetched_data.items():
            
            ind = item['indicator']
            df = item['data']
            filters = item['filters']
            
            with st.expander(f"📊 {ind.name} ({len(df)} obs)", expanded=False):
                
                # Info
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Source:** {ind.source.upper()}")
                    st.markdown(f"**Filters:** {filters}")
                
                with col2:
                    st.metric("Start", df['date'].min().strftime('%Y-%m'))
                    st.metric("End", df['date'].max().strftime('%Y-%m'))
                    st.metric("Latest", f"{df['value'].iloc[-1]:.2f}")
                
                # Chart
                st.line_chart(df.set_index('date')['value'])
                
                # Table
                st.dataframe(df.head(20), use_container_width=True)
                
                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    "💾 Download CSV",
                    csv,
                    f"{ind_id}_{geo}_{start_year}_{end_year}.csv",
                    "text/csv",
                    key=f"download_{ind_id}"
                )
        
        # Save to state
        st.markdown("---")
        st.markdown("### 💾 Save to Application State")
        
        try:
            from utils.state import AppState
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Select which to save as target
                target_options = list(st.session_state.fetched_data.keys())
                target_ind = st.selectbox(
                    "Select indicator for target (y_monthly):",
                    options=target_options,
                    format_func=lambda x: INDICATORS[x].name
                )
                
                if st.button("💾 Save as Target", use_container_width=True):
                    state = AppState.get()
                    
                    df = st.session_state.fetched_data[target_ind]['data']
                    
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
                if st.button("💾 Save All as Panel", use_container_width=True):
                    state = AppState.get()
                    
                    # Combine all
                    panel_parts = []
                    
                    for ind_id, item in st.session_state.fetched_data.items():
                        df = item['data'].set_index('date')
                        df.columns = [ind_id]
                        panel_parts.append(df)
                    
                    if panel_parts:
                        panel = pd.concat(panel_parts, axis=1)
                        state.panel_monthly = panel
                        AppState.update_timestamp()
                        
                        st.success(f"✅ Saved {len(panel_parts)} indicators to panel!")
                        st.balloons()
        
        except ImportError:
            st.info("Utils not available - can only download")

if __name__ == "__main__":
    main()
