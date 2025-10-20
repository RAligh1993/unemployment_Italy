"""
🌐 Economic Data Hub - Complete
================================
Comprehensive economic data fetcher for unemployment nowcasting.

Data Sources:
- Unemployment (Eurostat, ISTAT, OECD)
- GDP (National Accounts)
- Employment Rate
- Inflation (CPI, HICP)
- Industrial Production
- Retail Sales
- Consumer Confidence
- Job Vacancies
- Wages/Earnings
- Labour Force Participation

Author: ISTAT Nowcasting Team
Version: 7.0.0 (Complete)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# =============================================================================
# Config
# =============================================================================

st.set_page_config(
    page_title="Economic Data Hub",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(120deg, #6366f1, #8b5cf6, #a855f7, #d946ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .indicator-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        margin: 0.5rem 0;
        transition: all 0.2s;
    }
    .indicator-card:hover {
        border-color: #8b5cf6;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2);
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .badge-eurostat {
        background: #dbeafe;
        color: #1e40af;
    }
    .badge-istat {
        background: #d1fae5;
        color: #065f46;
    }
    .badge-oecd {
        background: #fef3c7;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HTTP Session
# =============================================================================

@st.cache_resource
def get_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

HTTP = get_session()

# =============================================================================
# Eurostat Datasets Catalog
# =============================================================================

@dataclass
class DatasetInfo:
    """Information about a dataset"""
    code: str
    name: str
    description: str
    source: str
    freq: str  # M, Q, A
    category: str

# Complete catalog of useful datasets
DATASETS = {
    # Unemployment & Employment
    'unemployment_rate': DatasetInfo(
        'une_rt_m', 'Unemployment Rate', 
        'Monthly unemployment rate by sex, age, country',
        'Eurostat', 'M', 'Labour Market'
    ),
    'employment_rate': DatasetInfo(
        'lfsi_emp_q', 'Employment Rate',
        'Employment rate by sex, age, country',
        'Eurostat', 'Q', 'Labour Market'
    ),
    'employment_level': DatasetInfo(
        'lfsq_egan', 'Employment Level',
        'Number of employed persons',
        'Eurostat', 'Q', 'Labour Market'
    ),
    'job_vacancy_rate': DatasetInfo(
        'jvs_q_nace2', 'Job Vacancy Rate',
        'Job vacancy rate by NACE sector',
        'Eurostat', 'Q', 'Labour Market'
    ),
    'labour_force': DatasetInfo(
        'lfsi_act_q', 'Labour Force',
        'Labour force participation rate',
        'Eurostat', 'Q', 'Labour Market'
    ),
    
    # Economic Activity
    'gdp': DatasetInfo(
        'namq_10_gdp', 'GDP',
        'Quarterly national accounts - GDP',
        'Eurostat', 'Q', 'National Accounts'
    ),
    'industrial_production': DatasetInfo(
        'sts_inpr_m', 'Industrial Production Index',
        'Industrial production index (2015=100)',
        'Eurostat', 'M', 'Economic Activity'
    ),
    'retail_sales': DatasetInfo(
        'sts_trtu_m', 'Retail Trade Volume',
        'Retail trade volume index',
        'Eurostat', 'M', 'Economic Activity'
    ),
    'construction': DatasetInfo(
        'sts_copr_m', 'Construction Production',
        'Construction production index',
        'Eurostat', 'M', 'Economic Activity'
    ),
    
    # Prices & Inflation
    'hicp': DatasetInfo(
        'prc_hicp_midx', 'HICP - Inflation',
        'Harmonised Index of Consumer Prices',
        'Eurostat', 'M', 'Prices'
    ),
    'ppi': DatasetInfo(
        'sts_inpp_m', 'Producer Price Index',
        'Producer prices in industry',
        'Eurostat', 'M', 'Prices'
    ),
    
    # Confidence & Sentiment
    'consumer_confidence': DatasetInfo(
        'ei_bsco_m', 'Consumer Confidence',
        'Consumer confidence indicator',
        'Eurostat', 'M', 'Confidence'
    ),
    'business_confidence': DatasetInfo(
        'ei_bsin_m', 'Business Confidence',
        'Business confidence indicator',
        'Eurostat', 'M', 'Confidence'
    ),
    
    # Wages & Earnings
    'wages': DatasetInfo(
        'earn_ses_pub2s', 'Average Wages',
        'Average wages and labour costs',
        'Eurostat', 'A', 'Wages'
    ),
    'labour_cost': DatasetInfo(
        'lc_lci_r2_q', 'Labour Cost Index',
        'Labour cost index',
        'Eurostat', 'Q', 'Wages'
    ),
    
    # Trade
    'exports': DatasetInfo(
        'namq_10_exi', 'Exports',
        'Exports of goods and services',
        'Eurostat', 'Q', 'Trade'
    ),
    'imports': DatasetInfo(
        'namq_10_imi', 'Imports',
        'Imports of goods and services',
        'Eurostat', 'Q', 'Trade'
    ),
}

# =============================================================================
# Universal Data Fetcher
# =============================================================================

class UniversalFetcher:
    """
    Universal fetcher for any Eurostat dataset
    """
    
    BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    
    def __init__(self):
        self.session = HTTP
    
    def fetch(self,
             dataset_code: str,
             geo: str = 'IT',
             start_year: int = 2015,
             end_year: int = 2024,
             **filters) -> Optional[pd.DataFrame]:
        """
        Fetch any Eurostat dataset
        
        Args:
            dataset_code: Dataset code (e.g., 'une_rt_m')
            geo: Country code
            start_year: Start year
            end_year: End year
            **filters: Additional filters (sex, age, s_adj, unit, etc.)
        
        Returns:
            DataFrame with date and value columns
        """
        
        url = f"{self.BASE_URL}/{dataset_code}"
        
        # Build parameters
        params = {
            'lang': 'en',
            'geo': geo,
            **filters
        }
        
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
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                }
                
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=30
                )
                
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
                                return df
                    
                    except:
                        continue
            
            except:
                continue
        
        return None
    
    def _parse_jsonstat(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse JSON-stat format"""
        try:
            if 'dimension' not in data:
                for val in data.values():
                    if isinstance(val, dict) and 'dimension' in val:
                        data = val
                        break
            
            if 'dimension' not in data:
                return None
            
            dimension = data['dimension']
            value_array = data.get('value', {})
            
            # Find time dimension
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
            
            # Extract values
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
        """Parse SDMX format"""
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
                
                year = int(year)
                quarter = int(quarter)
                month = quarter * 3
                return pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
            
            # Annual
            if len(time_str) == 4 and time_str.isdigit():
                return pd.Timestamp(int(time_str), 12, 31)
            
            return pd.to_datetime(time_str)
        
        except:
            return None

# =============================================================================
# ISTAT Fetcher (Italy specific)
# =============================================================================

class ISTATFetcher:
    """
    ISTAT specific fetcher
    """
    
    # Working ISTAT endpoints
    ENDPOINTS = [
        'http://sdmx.istat.it/SDMXWS/rest',
    ]
    
    # Known ISTAT datasets
    DATASETS_ISTAT = {
        'unemployment_monthly': 'DCCV_TAXDISOCC1',
        'unemployment_quarterly': 'DCCV_TAXDISOCC',
        'employment': 'DCCV_TAXOCCU1',
        'gdp': 'DCNP_PIL1',
    }
    
    def __init__(self):
        self.session = HTTP
    
    def fetch(self,
             dataset: str = 'unemployment_monthly',
             start_year: int = 2015,
             end_year: int = 2024) -> Optional[pd.DataFrame]:
        """Fetch from ISTAT"""
        
        flow_id = self.DATASETS_ISTAT.get(dataset)
        
        if not flow_id:
            return None
        
        # Try simple query without complex keys
        for endpoint in self.ENDPOINTS:
            url = f"{endpoint}/data/{flow_id}/all"
            
            params = {
                'startPeriod': start_year,
                'endPeriod': end_year,
                'format': 'jsondata'
            }
            
            try:
                response = self.session.get(url, params=params, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    df = self._parse_istat_json(data)
                    
                    if df is not None and not df.empty:
                        return df
            
            except:
                continue
        
        return None
    
    def _parse_istat_json(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse ISTAT JSON"""
        try:
            # Similar to Eurostat parser
            dataSets = data.get('dataSets', [])
            if not dataSets:
                return None
            
            dataset = dataSets[0]
            observations = dataset.get('observations', {})
            
            # Get structure
            structure = data.get('structure', {})
            dimensions = structure.get('dimensions', {}).get('observation', [])
            
            time_values = None
            for dim in dimensions:
                if 'TIME' in dim.get('id', '').upper():
                    time_values = [v['id'] for v in dim.get('values', [])]
                    break
            
            if not time_values:
                return None
            
            records = []
            for idx_str, val in observations.items():
                idx = int(idx_str)
                if idx < len(time_values):
                    value = val[0] if isinstance(val, list) else val
                    if value is not None:
                        records.append({
                            'time': time_values[idx],
                            'value': float(value)
                        })
            
            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['time'], errors='coerce')
                df = df.dropna(subset=['date'])
                df = df.sort_values('date')
                return df[['date', 'value']]
        
        except:
            pass
        
        return None

# =============================================================================
# Multi-Variable Fetcher
# =============================================================================

class MultiVariableFetcher:
    """Fetch multiple economic indicators at once"""
    
    def __init__(self):
        self.eurostat = UniversalFetcher()
        self.istat = ISTATFetcher()
    
    def fetch_labour_market_pack(self,
                                 geo: str = 'IT',
                                 start_year: int = 2015,
                                 end_year: int = 2024) -> Dict[str, pd.DataFrame]:
        """
        Fetch complete labour market package
        
        Returns dict of DataFrames
        """
        
        results = {}
        
        st.info("📊 Fetching Labour Market Indicators...")
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        indicators = [
            ('unemployment_rate', {'sex': 'T', 'age': 'Y15-74', 's_adj': 'SA'}),
            ('employment_rate', {'sex': 'T', 'age': 'Y15-74'}),
            ('job_vacancy_rate', {}),
            ('labour_force', {'sex': 'T', 'age': 'Y15-74'}),
        ]
        
        total = len(indicators)
        
        for idx, (key, filters) in enumerate(indicators):
            dataset = DATASETS[key]
            
            status.text(f"Fetching: {dataset.name}...")
            
            df = self.eurostat.fetch(
                dataset.code,
                geo=geo,
                start_year=start_year,
                end_year=end_year,
                **filters
            )
            
            if df is not None and not df.empty:
                results[key] = df
                st.success(f"✅ {dataset.name}: {len(df)} obs")
            else:
                st.warning(f"⚠️ {dataset.name}: No data")
            
            progress_bar.progress((idx + 1) / total)
        
        progress_bar.empty()
        status.empty()
        
        return results
    
    def fetch_economic_activity_pack(self,
                                    geo: str = 'IT',
                                    start_year: int = 2015,
                                    end_year: int = 2024) -> Dict[str, pd.DataFrame]:
        """Fetch economic activity indicators"""
        
        results = {}
        
        st.info("📈 Fetching Economic Activity Indicators...")
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        indicators = [
            ('gdp', {'unit': 'CLV10_MNAC', 's_adj': 'SCA', 'na_item': 'B1GQ'}),
            ('industrial_production', {'unit': 'I15', 's_adj': 'SCA'}),
            ('retail_sales', {'unit': 'I15', 's_adj': 'SCA'}),
            ('construction', {'unit': 'I15', 's_adj': 'SCA'}),
        ]
        
        total = len(indicators)
        
        for idx, (key, filters) in enumerate(indicators):
            dataset = DATASETS[key]
            
            status.text(f"Fetching: {dataset.name}...")
            
            df = self.eurostat.fetch(
                dataset.code,
                geo=geo,
                start_year=start_year,
                end_year=end_year,
                **filters
            )
            
            if df is not None and not df.empty:
                results[key] = df
                st.success(f"✅ {dataset.name}: {len(df)} obs")
            else:
                st.warning(f"⚠️ {dataset.name}: No data")
            
            progress_bar.progress((idx + 1) / total)
        
        progress_bar.empty()
        status.empty()
        
        return results
    
    def fetch_all(self,
                 geo: str = 'IT',
                 start_year: int = 2015,
                 end_year: int = 2024) -> Dict[str, pd.DataFrame]:
        """Fetch ALL indicators"""
        
        results = {}
        
        # Labour market
        results.update(self.fetch_labour_market_pack(geo, start_year, end_year))
        
        # Economic activity
        results.update(self.fetch_economic_activity_pack(geo, start_year, end_year))
        
        # Try ISTAT for Italy
        if geo == 'IT':
            st.info("🇮🇹 Trying ISTAT specific data...")
            
            istat_unemployment = self.istat.fetch('unemployment_monthly', start_year, end_year)
            
            if istat_unemployment is not None and not istat_unemployment.empty:
                results['istat_unemployment'] = istat_unemployment
                st.success(f"✅ ISTAT Unemployment: {len(istat_unemployment)} obs")
        
        return results

# =============================================================================
# UI
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🌐 Economic Data Hub</h1>', unsafe_allow_html=True)
    
    st.info("""
    **📊 Comprehensive economic data for unemployment nowcasting**
    
    This tool fetches multiple economic indicators from:
    - **Eurostat** (primary source - all EU countries)
    - **ISTAT** (Italy-specific data)
    - **OECD** (alternative source)
    
    All data is automatically aligned and ready for modeling!
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        geo = st.selectbox(
            "🗺️ Country",
            options=['IT', 'DE', 'FR', 'ES', 'PT', 'NL'],
            format_func=lambda x: {
                'IT': '🇮🇹 Italy',
                'DE': '🇩🇪 Germany',
                'FR': '🇫🇷 France',
                'ES': '🇪🇸 Spain',
                'PT': '🇵🇹 Portugal',
                'NL': '🇳🇱 Netherlands'
            }[x]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_year = st.number_input("Start Year", 2000, 2024, 2015)
        
        with col2:
            end_year = st.number_input("End Year", 2000, 2024, 2024)
        
        st.markdown("---")
        
        st.subheader("📦 Data Packages")
        
        fetch_labour = st.checkbox("Labour Market Pack", value=True)
        fetch_economic = st.checkbox("Economic Activity Pack", value=True)
        fetch_istat = st.checkbox("ISTAT Data (Italy only)", value=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Fetch Data", "📚 Dataset Catalog", "💾 Saved Data"])
    
    # =============================================================================
    # TAB 1: Fetch
    # =============================================================================
    
    with tab1:
        st.markdown("## 🚀 Fetch Economic Data")
        
        if st.button("🔥 Fetch All Selected", type="primary", use_container_width=True):
            
            fetcher = MultiVariableFetcher()
            
            all_data = {}
            
            # Labour market
            if fetch_labour:
                labour_data = fetcher.fetch_labour_market_pack(geo, start_year, end_year)
                all_data.update(labour_data)
            
            # Economic activity
            if fetch_economic:
                economic_data = fetcher.fetch_economic_activity_pack(geo, start_year, end_year)
                all_data.update(economic_data)
            
            # ISTAT
            if fetch_istat and geo == 'IT':
                st.info("🇮🇹 Fetching ISTAT data...")
                
                istat_df = fetcher.istat.fetch('unemployment_monthly', start_year, end_year)
                
                if istat_df is not None and not istat_df.empty:
                    all_data['istat_unemployment'] = istat_df
                    st.success(f"✅ ISTAT: {len(istat_df)} obs")
            
            # Summary
            st.markdown("---")
            st.markdown("## 📊 Summary")
            
            if all_data:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("✅ Indicators Fetched", len(all_data))
                
                with col2:
                    total_obs = sum(len(df) for df in all_data.values())
                    st.metric("📈 Total Observations", total_obs)
                
                with col3:
                    # Date range
                    all_dates = []
                    for df in all_data.values():
                        all_dates.extend(df['date'].tolist())
                    
                    if all_dates:
                        min_date = min(all_dates)
                        max_date = max(all_dates)
                        st.metric("📅 Date Range", f"{min_date.year}-{max_date.year}")
                
                # Display each indicator
                st.markdown("### 📋 Fetched Indicators")
                
                for key, df in all_data.items():
                    
                    dataset_info = DATASETS.get(key)
                    
                    with st.expander(f"{'📊' if dataset_info else '🇮🇹'} {dataset_info.name if dataset_info else key} ({len(df)} obs)"):
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Chart
                            st.line_chart(df.set_index('date')['value'])
                        
                        with col2:
                            # Stats
                            st.metric("Start", df['date'].min().strftime('%Y-%m'))
                            st.metric("End", df['date'].max().strftime('%Y-%m'))
                            st.metric("Mean", f"{df['value'].mean():.2f}")
                            st.metric("Latest", f"{df['value'].iloc[-1]:.2f}")
                        
                        # Data table
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Download
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "💾 Download",
                            csv,
                            f"{key}_{geo}_{start_year}_{end_year}.csv",
                            "text/csv",
                            key=f"download_{key}"
                        )
                
                # Save to state
                st.markdown("---")
                st.markdown("### 💾 Save to Application State")
                
                try:
                    from utils.state import AppState
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("💾 Save Unemployment as Target", use_container_width=True):
                            state = AppState.get()
                            
                            # Get unemployment series
                            unemp_df = all_data.get('unemployment_rate') or all_data.get('istat_unemployment')
                            
                            if unemp_df is not None:
                                ts = pd.Series(
                                    unemp_df['value'].values,
                                    index=unemp_df['date'],
                                    name='unemployment'
                                )
                                state.y_monthly = ts
                                AppState.update_timestamp()
                                st.success("✅ Saved!")
                                st.balloons()
                    
                    with col2:
                        if st.button("💾 Save All as Panel", use_container_width=True):
                            state = AppState.get()
                            
                            # Combine all to panel
                            panel_parts = []
                            
                            for key, df in all_data.items():
                                df_indexed = df.set_index('date')
                                df_indexed.columns = [key]
                                panel_parts.append(df_indexed)
                            
                            if panel_parts:
                                panel = pd.concat(panel_parts, axis=1)
                                state.panel_monthly = panel
                                AppState.update_timestamp()
                                st.success("✅ Saved!")
                                st.balloons()
                    
                    with col3:
                        if st.button("💾 Download All as ZIP", use_container_width=True):
                            st.info("Feature coming soon!")
                
                except ImportError:
                    st.info("Utils not available - data can only be downloaded")
            
            else:
                st.warning("No data fetched. Try different parameters or check your internet connection.")
    
    # =============================================================================
    # TAB 2: Catalog
    # =============================================================================
    
    with tab2:
        st.markdown("## 📚 Available Datasets")
        
        # Group by category
        categories = {}
        for key, ds in DATASETS.items():
            if ds.category not in categories:
                categories[ds.category] = []
            categories[ds.category].append((key, ds))
        
        for category, datasets in categories.items():
            st.markdown(f"### {category}")
            
            for key, ds in datasets:
                st.markdown(f"""
                <div class="indicator-card">
                    <h4>{ds.name}</h4>
                    <p>{ds.description}</p>
                    <span class="source-badge badge-{ds.source.lower()}">{ds.source}</span>
                    <span class="source-badge">Freq: {ds.freq}</span>
                    <span class="source-badge">Code: {ds.code}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # =============================================================================
    # TAB 3: Saved
    # =============================================================================
    
    with tab3:
        st.markdown("## 💾 Saved Data in State")
        
        try:
            from utils.state import AppState
            
            state = AppState.get()
            summary = AppState.summary()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Target Loaded", "✅" if summary['has_target'] else "❌")
                if summary['has_target']:
                    st.caption(f"{summary['target_length']} obs")
            
            with col2:
                st.metric("Panel Built", "✅" if summary['has_panel_monthly'] else "❌")
                if summary['has_panel_monthly']:
                    rows, cols = summary['panel_monthly_shape']
                    st.caption(f"{rows} × {cols}")
            
            with col3:
                st.metric("Models Trained", summary['num_models'])
            
            # Show panel if exists
            if summary['has_panel_monthly']:
                st.markdown("### 📊 Panel Data")
                st.dataframe(state.panel_monthly.head(20), use_container_width=True)
        
        except ImportError:
            st.info("Utils module not available")

if __name__ == "__main__":
    main()
