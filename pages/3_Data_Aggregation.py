"""
🎯 Smart Data Fetcher - Ultimate Edition
=========================================
Multiple reliable data sources + Easy manual upload

Sources:
1. Eurostat API (stable, includes Italy data)
2. Manual Excel upload with smart processing
3. Direct I.Stat guide

Author: ISTAT Nowcasting Team
Version: 4.0.0 (Ultimate)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict, List, Tuple
from io import BytesIO, StringIO
from datetime import datetime
import time

try:
    from utils.state import AppState
    from utils.istat_handler import ISTATHandler
    from utils.excel_processor import ExcelProcessor
    from utils.data_detector import DataDetector
    from utils.visualizer import Visualizer
    UTILS_AVAILABLE = True
except:
    UTILS_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Smart Data Fetcher",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #7c3aed, #a78bfa, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .method-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .success-card {
        background: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .guide-card {
        background: #dbeafe;
        border-left: 5px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Eurostat API Client (More Reliable!)
# =============================================================================

class EurostatClient:
    """
    Eurostat API client - خیلی پایدارتر از ISTAT!
    
    Eurostat has Italy data and is much more stable.
    """
    
    BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    
    # Unemployment datasets
    DATASETS = {
        'unemployment_rate': 'une_rt_q',  # Quarterly unemployment rate
        'unemployment_monthly': 'une_rt_m',  # Monthly unemployment rate
        'youth_unemployment': 'yth_empl_090',  # Youth unemployment
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        })
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test Eurostat API"""
        try:
            # Simple test query
            url = f"{self.BASE_URL}/une_rt_q"
            params = {
                'format': 'JSON',
                'lang': 'EN',
                'geo': 'IT',
                'time': '2024'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                return True, "✅ Eurostat API working!"
            else:
                return False, f"Status: {response.status_code}"
        
        except Exception as e:
            return False, f"Error: {e}"
    
    def fetch_unemployment(self,
                          country: str = 'IT',
                          start_year: int = 2015,
                          end_year: int = 2024,
                          freq: str = 'Q') -> Optional[pd.DataFrame]:
        """
        Fetch unemployment data from Eurostat
        
        Args:
            country: Country code (IT, DE, FR, etc.)
            start_year: Start year
            end_year: End year
            freq: Frequency ('Q' or 'M')
        
        Returns:
            DataFrame with date and value
        """
        
        # Select dataset
        dataset = self.DATASETS['unemployment_monthly'] if freq == 'M' else self.DATASETS['unemployment_rate']
        
        url = f"{self.BASE_URL}/{dataset}"
        
        # Build parameters
        params = {
            'format': 'JSON',
            'lang': 'EN',
            'geo': country,
            'sex': 'T',  # Total
            'age': 'Y15-74',
            's_adj': 'SA',  # Seasonally adjusted
        }
        
        try:
            st.write(f"🔍 Fetching from Eurostat: `{dataset}`...")
            
            response = self.session.get(url, params=params, timeout=30)
            
            st.write(f"   Status: {response.status_code}")
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            # Parse Eurostat JSON
            df = self._parse_eurostat_json(data, start_year, end_year)
            
            if df is not None and not df.empty:
                st.success(f"   ✅ Got {len(df)} observations from Eurostat!")
                return df
            
            return None
        
        except Exception as e:
            st.error(f"   ❌ Eurostat error: {e}")
            return None
    
    def _parse_eurostat_json(self, 
                            data: dict,
                            start_year: int,
                            end_year: int) -> Optional[pd.DataFrame]:
        """Parse Eurostat JSON response"""
        try:
            # Eurostat JSON structure:
            # data -> value -> {index: value}
            # data -> dimension -> time -> category -> index -> {index: time_code}
            
            if 'value' not in data:
                return None
            
            values = data['value']
            
            # Get time dimension
            if 'dimension' not in data or 'time' not in data['dimension']:
                return None
            
            time_info = data['dimension']['time']['category']['index']
            
            # Build DataFrame
            records = []
            
            for idx, time_code in time_info.items():
                if idx in values:
                    value = values[idx]
                    
                    # Parse time code
                    date = self._parse_eurostat_time(time_code)
                    
                    if date is not None:
                        year = date.year
                        
                        if start_year <= year <= end_year:
                            records.append({
                                'date': date,
                                'value': float(value)
                            })
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
        
        except Exception as e:
            st.write(f"   Parse error: {e}")
            return None
    
    def _parse_eurostat_time(self, time_code: str) -> Optional[pd.Timestamp]:
        """
        Parse Eurostat time codes
        
        Formats:
        - 2024Q1 → 2024-03-31
        - 2024M01 → 2024-01-31
        - 2024 → 2024-12-31
        """
        try:
            time_code = str(time_code).strip()
            
            # Quarterly: 2024Q1
            if 'Q' in time_code:
                year = int(time_code[:4])
                quarter = int(time_code[-1])
                month = quarter * 3
                date = pd.Timestamp(year=year, month=month, day=1)
                return date + pd.offsets.MonthEnd(0)
            
            # Monthly: 2024M01
            elif 'M' in time_code:
                year = int(time_code[:4])
                month = int(time_code[5:7])
                date = pd.Timestamp(year=year, month=month, day=1)
                return date + pd.offsets.MonthEnd(0)
            
            # Annual: 2024
            elif len(time_code) == 4 and time_code.isdigit():
                year = int(time_code)
                return pd.Timestamp(year=year, month=12, day=31)
            
            return None
        
        except:
            return None


# =============================================================================
# Excel Upload Helper
# =============================================================================

class SmartExcelUploader:
    """Smart Excel uploader with auto-processing"""
    
    def __init__(self):
        if UTILS_AVAILABLE:
            self.excel_processor = ExcelProcessor()
            self.istat_handler = ISTATHandler()
            self.detector = DataDetector()
    
    def process_upload(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Process uploaded Excel/CSV file"""
        
        try:
            file_name = uploaded_file.name
            file_ext = file_name.split('.')[-1].lower()
            
            st.info(f"📄 Processing: **{file_name}**")
            
            # Read file
            if file_ext == 'csv':
                # Try multiple encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"✅ Read with {encoding}")
                        break
                    except:
                        continue
            
            else:
                # Excel
                if UTILS_AVAILABLE:
                    uploaded_file.seek(0)
                    sheets = self.excel_processor.read_file(uploaded_file)
                    
                    if len(sheets) > 1:
                        sheet_name = st.selectbox("Select sheet:", list(sheets.keys()))
                    else:
                        sheet_name = list(sheets.keys())[0]
                    
                    df = sheets[sheet_name]
                else:
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file)
            
            if df is None or df.empty:
                st.error("❌ Could not read file")
                return None
            
            st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Auto-detect
            if UTILS_AVAILABLE:
                
                # Check if ISTAT format
                if self.istat_handler.is_istat_format(df):
                    st.info("🇮🇹 ISTAT format detected!")
                    return self._process_istat_format(df)
                
                # Check if time series
                analysis = self.detector.analyze_dataset(df)
                
                if analysis.is_time_series:
                    st.info(f"📊 Time series detected! Date: {analysis.date_column}")
                    return self._process_time_series(df, analysis)
            
            # Show preview
            st.dataframe(df.head(), use_container_width=True)
            
            return df
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None
    
    def _process_istat_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process ISTAT long format"""
        
        cols = self.istat_handler.detect_columns(df)
        options = self.istat_handler.get_filter_options(df, cols)
        
        st.markdown("#### 🎯 Filter ISTAT Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            territory = st.selectbox("Territory", options.get('territorys', ['Italy']))
        with col2:
            sex = st.selectbox("Sex", options.get('sexs', ['Total']))
        with col3:
            age = st.selectbox("Age", options.get('ages', ['Y15-64']))
        
        filtered = self.istat_handler.filter_data(df, cols, territory, sex, age)
        ts = self.istat_handler.convert_to_timeseries(filtered, cols)
        
        result = ts.to_frame('unemployment_rate').reset_index()
        result.columns = ['date', 'value']
        
        return result
    
    def _process_time_series(self, df: pd.DataFrame, analysis) -> pd.DataFrame:
        """Process regular time series"""
        
        date_col = analysis.date_column
        
        # Select value column
        value_cols = analysis.value_columns
        
        if len(value_cols) == 1:
            value_col = value_cols[0]
        else:
            value_col = st.selectbox("Select value column:", value_cols)
        
        result = df[[date_col, value_col]].copy()
        result.columns = ['date', 'value']
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date')
        
        return result


# =============================================================================
# Main UI
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🎯 Smart Data Fetcher</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="method-card">
        <h3>🚀 3 Reliable Methods</h3>
        <p><strong>✅ Method 1:</strong> Eurostat API (Italy data, very stable)</p>
        <p><strong>✅ Method 2:</strong> Manual Excel/CSV upload (100% reliable)</p>
        <p><strong>✅ Method 3:</strong> Direct I.Stat download guide</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize
    if UTILS_AVAILABLE:
        state = AppState.get()
        viz = Visualizer()
    
    eurostat = EurostatClient()
    uploader = SmartExcelUploader() if UTILS_AVAILABLE else None
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🌍 Eurostat API",
        "📁 Manual Upload",
        "📖 I.Stat Guide"
    ])
    
    # =========================================================================
    # TAB 1: Eurostat API
    # =========================================================================
    
    with tab1:
        st.markdown("### 🌍 Eurostat API (Recommended)")
        
        st.info("""
        **Why Eurostat?**
        - ✅ Much more stable than ISTAT API
        - ✅ Includes all Italy data
        - ✅ Well-documented and reliable
        - ✅ Used by researchers worldwide
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            country = st.selectbox(
                "Country",
                ['IT', 'DE', 'FR', 'ES', 'NL'],
                format_func=lambda x: {
                    'IT': '🇮🇹 Italy',
                    'DE': '🇩🇪 Germany',
                    'FR': '🇫🇷 France',
                    'ES': '🇪🇸 Spain',
                    'NL': '🇳🇱 Netherlands'
                }[x]
            )
        
        with col2:
            start_year = st.number_input("Start Year", 2000, 2024, 2015)
        
        with col3:
            end_year = st.number_input("End Year", 2000, 2024, 2024)
        
        freq = st.radio("Frequency", ['Q', 'M'], format_func=lambda x: 'Quarterly' if x == 'Q' else 'Monthly', horizontal=True)
        
        if st.button("🚀 Fetch from Eurostat", type="primary", use_container_width=True):
            
            with st.spinner("Fetching from Eurostat..."):
                df = eurostat.fetch_unemployment(country, start_year, end_year, freq)
            
            if df is not None and not df.empty:
                
                st.markdown("""
                <div class="success-card">
                    <h3>✅ Success!</h3>
                    <p>Data retrieved from Eurostat (official EU statistics)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Observations", len(df))
                with col2:
                    st.metric("Start", df['date'].min().strftime('%Y-%m'))
                with col3:
                    st.metric("End", df['date'].max().strftime('%Y-%m'))
                with col4:
                    st.metric("Latest", f"{df['value'].iloc[-1]:.2f}%")
                
                # Visualization
                if UTILS_AVAILABLE:
                    plot_df = df.copy()
                    plot_df.columns = ['date', 'unemployment_rate']
                    
                    fig = viz.plot_time_series(
                        plot_df,
                        'date',
                        ['unemployment_rate'],
                        title=f"Unemployment Rate - {country}",
                        show_points=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.dataframe(df, use_container_width=True, height=300)
                
                # Save options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "💾 Download CSV",
                        csv,
                        f"eurostat_unemployment_{country}_{datetime.now().strftime('%Y%m%d')}.csv",
                        use_container_width=True
                    )
                
                if UTILS_AVAILABLE:
                    with col2:
                        if st.button("💾 Save as Target", use_container_width=True):
                            ts = pd.Series(df['value'].values, index=df['date'], name='unemployment')
                            state.y_monthly = ts
                            AppState.update_timestamp()
                            st.success("✅ Saved!")
                            st.balloons()
                    
                    with col3:
                        if st.button("💾 Add to Panel", use_container_width=True):
                            panel_df = df.set_index('date')
                            panel_df.columns = [f'unemp_{country}']
                            
                            if state.panel_monthly is not None:
                                state.panel_monthly = state.panel_monthly.join(panel_df, how='outer')
                            else:
                                state.panel_monthly = panel_df
                            
                            AppState.update_timestamp()
                            st.success("✅ Added!")
                            st.balloons()
            
            else:
                st.error("❌ Could not fetch data")
                st.info("Try manual upload instead!")
    
    # =========================================================================
    # TAB 2: Manual Upload
    # =========================================================================
    
    with tab2:
        st.markdown("### 📁 Manual Excel/CSV Upload")
        
        st.info("""
        **Upload your data file:**
        - Excel: .xlsx, .xls, .xlsm
        - CSV: any encoding
        - Auto-detects ISTAT format
        - Auto-detects time series
        """)
        
        uploaded_file = st.file_uploader(
            "Choose file",
            type=['csv', 'xlsx', 'xls', 'xlsm'],
            key='manual_upload'
        )
        
        if uploaded_file and uploader:
            
            df = uploader.process_upload(uploaded_file)
            
            if df is not None and not df.empty:
                
                st.success("✅ File processed successfully!")
                
                # Show data
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(df, use_container_width=True, height=400)
                
                with col2:
                    st.metric("Rows", len(df))
                    st.metric("Columns", len(df.columns))
                    
                    if 'date' in df.columns and 'value' in df.columns:
                        st.metric("Latest", f"{df['value'].iloc[-1]:.2f}")
                
                # Visualization
                if UTILS_AVAILABLE and 'date' in df.columns and 'value' in df.columns:
                    plot_df = df.copy()
                    
                    fig = viz.plot_time_series(
                        plot_df,
                        'date',
                        ['value'],
                        title="Uploaded Data",
                        show_points=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Save options
                if UTILS_AVAILABLE and 'date' in df.columns and 'value' in df.columns:
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💾 Save as Target", key='save_target_upload', use_container_width=True):
                            ts = pd.Series(df['value'].values, index=pd.to_datetime(df['date']), name='unemployment')
                            state.y_monthly = ts
                            AppState.update_timestamp()
                            st.success("✅ Saved!")
                            st.balloons()
                    
                    with col2:
                        if st.button("💾 Add to Panel", key='save_panel_upload', use_container_width=True):
                            panel_df = df.set_index(pd.to_datetime(df['date']))
                            panel_df.columns = ['unemployment_manual']
                            
                            if state.panel_monthly is not None:
                                state.panel_monthly = state.panel_monthly.join(panel_df, how='outer')
                            else:
                                state.panel_monthly = panel_df
                            
                            AppState.update_timestamp()
                            st.success("✅ Added!")
                            st.balloons()
    
    # =========================================================================
    # TAB 3: I.Stat Guide
    # =========================================================================
    
    with tab3:
        st.markdown("### 📖 How to Download from I.Stat")
        
        st.markdown("""
        <div class="guide-card">
            <h4>📝 Step-by-Step Guide</h4>
            <p>Follow these steps to manually download data from I.Stat:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        #### 1️⃣ Go to I.Stat Website
        
        🔗 **Link:** [http://dati.istat.it/](http://dati.istat.it/)
        
        ---
        
        #### 2️⃣ Navigate to Unemployment Data
        
        - Click on **"Lavoro e retribuzioni"** (Work and wages)
        - Then **"Disoccupazione"** (Unemployment)
        - Select **"Tasso di disoccupazione"** (Unemployment rate)
        
        ---
        
        #### 3️⃣ Customize Your Query
        
        - **Territory:** Select Italy or specific region
        - **Sex:** Total, Males, Females
        - **Age:** 15-64, 15-24, etc.
        - **Time period:** Select your date range
        
        ---
        
        #### 4️⃣ Download the Data
        
        - Click **"Estrazione dati"** (Extract data)
        - Choose format: **Excel** or **CSV**
        - Download the file
        
        ---
        
        #### 5️⃣ Upload Here
        
        - Go to the **"Manual Upload"** tab
        - Upload your downloaded file
        - The app will automatically detect and process it!
        
        ---
        
        ### 🎥 Alternative: Direct Links
        
        **Unemployment Rate (Monthly):**
        - [http://dati.istat.it/Index.aspx?DataSetCode=DCCV_TAXDISOCC1](http://dati.istat.it/Index.aspx?DataSetCode=DCCV_TAXDISOCC1)
        
        **Unemployment Rate (Quarterly):**
        - [http://dati.istat.it/Index.aspx?DataSetCode=DCCV_TAXDISOCC](http://dati.istat.it/Index.aspx?DataSetCode=DCCV_TAXDISOCC)
        
        ---
        
        ### 💡 Pro Tips
        
        1. ✅ **Use Eurostat tab** - faster and more reliable
        2. ✅ **Download Excel format** - easier to process
        3. ✅ **Select "Seasonally adjusted" data** if available
        4. ✅ **Include headers** in your download
        """)
        
        st.markdown("""
        <div class="guide-card">
            <h4>❓ Need Help?</h4>
            <p>If you're having trouble:</p>
            <ul>
                <li>Try the <strong>Eurostat tab</strong> first (easiest!)</li>
                <li>Download Excel manually and use <strong>Manual Upload</strong></li>
                <li>Check I.Stat documentation: <a href="https://www.istat.it/en/methods-and-tools">ISTAT Methods & Tools</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
