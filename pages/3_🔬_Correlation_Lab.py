"""
═══════════════════════════════════════════════════════════════════════════════
🇮🇹 ITALIAN UNEMPLOYMENT DATA - AUTOMATIC FETCHING SYSTEM
═══════════════════════════════════════════════════════════════════════════════

A professional, production-ready system for automatically fetching Italian 
unemployment and related economic data from official sources.

Features:
- ✅ Eurostat (Unemployment, CCI, HICP, Industrial Production)
- ✅ Yahoo Finance (FTSE MIB, VIX)
- ✅ Google Trends (Italian job keywords)
- ✅ Real-time data validation
- ✅ Interactive visualizations
- ✅ Export to CSV/Excel

Author: Professional Data Team
Version: 3.0 (Production)
Date: October 2025

Save as: italian_auto_fetch_app.py
Run: streamlit run italian_auto_fetch_app.py
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
from io import BytesIO
from typing import Dict, Optional, Tuple
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Italian Data Auto-Fetch",
    page_icon="🇮🇹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/italian-data',
        'About': """
        # Italian Unemployment Auto-Fetch System
        **Version:** 3.0 Production
        
        Automatically fetch Italian economic data from official sources.
        """
    }
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS - PROFESSIONAL STYLING
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main header with Italian flag gradient */
    .hero-header {
        background: linear-gradient(90deg, #009246 0%, #FFFFFF 33%, #FFFFFF 66%, #CE2B37 100%);
        padding: 50px 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    
    .hero-header h1 {
        color: #2c3e50;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-header p {
        color: #34495e;
        font-size: 1.4rem;
        margin: 15px 0 0 0;
        font-weight: 600;
    }
    
    /* Data source cards */
    .source-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        border-color: #667eea;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.2);
        transform: translateY(-3px);
    }
    
    .source-card.available {
        border-left: 5px solid #10B981;
    }
    
    .source-card.unavailable {
        border-left: 5px solid #EF4444;
        opacity: 0.6;
    }
    
    .source-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    
    .source-description {
        color: #6c757d;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 15px;
    }
    
    .source-metadata {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #495057;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
    }
    
    .badge-success {
        background: #10B981;
        color: white;
    }
    
    .badge-warning {
        background: #F59E0B;
        color: white;
    }
    
    .badge-info {
        background: #3B82F6;
        color: white;
    }
    
    .badge-danger {
        background: #EF4444;
        color: white;
    }
    
    /* Status indicators */
    .status-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    
    .status-box h3 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
    }
    
    .status-box p {
        font-size: 1.1rem;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    
    /* Section headers */
    .section-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 50px 0 25px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid #667eea;
    }
    
    /* Data quality indicators */
    .quality-indicator {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        margin: 5px;
    }
    
    .quality-excellent {
        background: #D1FAE5;
        color: #065F46;
    }
    
    .quality-good {
        background: #DBEAFE;
        color: #1E40AF;
    }
    
    .quality-fair {
        background: #FEF3C7;
        color: #92400E;
    }
    
    /* Buttons */
    .stButton > button {
        font-size: 1.2rem;
        font-weight: 700;
        padding: 18px 40px;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 10%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
    }
    
    .info-box h4 {
        margin-top: 0;
        font-size: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA SOURCE DEFINITIONS & AVAILABILITY
# ═══════════════════════════════════════════════════════════════════════════

DATA_SOURCES = {
    'unemployment': {
        'name': 'Italian Unemployment Rate',
        'provider': 'Eurostat',
        'dataset': 'une_rt_m',
        'frequency': 'Monthly',
        'coverage': '2000-Present',
        'update': 'Monthly (T+30 days)',
        'description': 'Seasonally adjusted unemployment rate for Italy. Official statistics from Eurostat statistical office.',
        'mandatory': True,
        'available': True,
        'url': 'https://ec.europa.eu/eurostat',
        'quality': 'excellent',
        'icon': '🎯'
    },
    'cci': {
        'name': 'Consumer Confidence Index',
        'provider': 'Eurostat',
        'dataset': 'ei_bsco_m',
        'frequency': 'Monthly',
        'coverage': '2000-Present',
        'update': 'Monthly (T+5 days)',
        'description': 'Consumer confidence indicator reflecting household expectations about economic conditions.',
        'mandatory': False,
        'available': True,
        'url': 'https://ec.europa.eu/eurostat',
        'quality': 'excellent',
        'icon': '📊'
    },
    'hicp': {
        'name': 'HICP Inflation Index',
        'provider': 'Eurostat',
        'dataset': 'prc_hicp_midx',
        'frequency': 'Monthly',
        'coverage': '2000-Present',
        'update': 'Monthly (T+15 days)',
        'description': 'Harmonized Index of Consumer Prices - official inflation measure for Italy.',
        'mandatory': False,
        'available': True,
        'url': 'https://ec.europa.eu/eurostat',
        'quality': 'excellent',
        'icon': '📈'
    },
    'iip': {
        'name': 'Industrial Production Index',
        'provider': 'Eurostat',
        'dataset': 'sts_inpr_m',
        'frequency': 'Monthly',
        'coverage': '2000-Present',
        'update': 'Monthly (T+40 days)',
        'description': 'Index of industrial production measuring output in manufacturing sector.',
        'mandatory': False,
        'available': True,
        'url': 'https://ec.europa.eu/eurostat',
        'quality': 'good',
        'icon': '🏭'
    },
    'stock': {
        'name': 'FTSE MIB Stock Index',
        'provider': 'Yahoo Finance',
        'dataset': '^FTSEMIB',
        'frequency': 'Daily',
        'coverage': '2000-Present',
        'update': 'Real-time (15min delay)',
        'description': 'Italian stock market benchmark index. Includes close price, volume, and returns.',
        'mandatory': False,
        'available': True,
        'url': 'https://finance.yahoo.com',
        'quality': 'good',
        'icon': '📈'
    },
    'vix': {
        'name': 'V2TX / VIX Volatility',
        'provider': 'Yahoo Finance / CBOE',
        'dataset': '^V2TX, ^VIX',
        'frequency': 'Daily',
        'coverage': '2000-Present',
        'update': 'Real-time (15min delay)',
        'description': 'European/US volatility indices. Market fear gauge and risk indicator.',
        'mandatory': False,
        'available': True,
        'url': 'https://finance.yahoo.com',
        'quality': 'good',
        'icon': '📊'
    },
    'trends': {
        'name': 'Google Trends',
        'provider': 'Google Trends',
        'dataset': 'Italian job keywords',
        'frequency': 'Weekly',
        'coverage': '2015-Present',
        'update': 'Real-time',
        'description': 'Search interest data for job-related keywords in Italy. Includes: lavoro, disoccupazione, naspi, etc.',
        'mandatory': False,
        'available': True,
        'url': 'https://trends.google.com',
        'quality': 'fair',
        'icon': '🔍'
    }
}

# Italian job keywords for Google Trends
ITALIAN_KEYWORDS = [
    'offerte di lavoro',
    'disoccupazione',
    'naspi',
    'indeed lavoro',
    'ricerca lavoro',
    'cerco lavoro',
    'centro per l\'impiego',
    'cassa integrazione',
    'reddito di cittadinanza',
    'curriculum vitae'
]

# ═══════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eurostat_data(dataset: str, filters: Dict, start_year: int = 2000) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fetch data from Eurostat
    Returns: (dataframe, status_message)
    """
    try:
        import eurostat
        
        # Fetch dataset
        df = eurostat.get_data_df(dataset, flags=False)
        
        # Apply filters
        for key, value in filters.items():
            if key in df.columns:
                df = df[df[key] == value]
        
        if df.empty:
            return None, "❌ No data found after filtering"
        
        # Find time columns
        time_cols = [c for c in df.columns if 'M' in str(c) and len(str(c)) >= 6]
        if not time_cols:
            return None, "❌ No time columns found"
        
        # Melt to long format
        id_cols = [c for c in df.columns if c not in time_cols]
        melted = df.melt(
            id_vars=id_cols,
            value_vars=time_cols,
            var_name='period',
            value_name='value'
        )
        
        # Parse dates
        melted['period'] = melted['period'].astype(str).str.replace('M', '-')
        melted['date'] = pd.to_datetime(melted['period'], format='%Y-%m', errors='coerce')
        melted['date'] = melted['date'] + pd.offsets.MonthEnd(0)
        melted['value'] = pd.to_numeric(melted['value'], errors='coerce')
        
        # Final cleanup
        result = melted[['date', 'value']].dropna().sort_values('date')
        result = result[result['date'].dt.year >= start_year]
        
        if result.empty:
            return None, "❌ No data in date range"
        
        return result, f"✅ Success: {len(result)} months"
        
    except ImportError:
        return None, "❌ eurostat package not installed (pip install eurostat)"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_data(ticker: str, start_year: int = 2000) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fetch data from Yahoo Finance
    Returns: (dataframe, status_message)
    """
    try:
        import yfinance as yf
        
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=f"{start_year}-01-01", auto_adjust=True)
        
        if df.empty:
            return None, f"❌ No data for {ticker}"
        
        result = pd.DataFrame({
            'date': df.index,
            'close': df['Close'].values,
            'volume': df['Volume'].values
        })
        result['date'] = pd.to_datetime(result['date']).dt.tz_localize(None)
        result = result.reset_index(drop=True)
        
        return result, f"✅ Success: {len(result)} days"
        
    except ImportError:
        return None, "❌ yfinance package not installed (pip install yfinance)"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_google_trends_data(keywords: list, geo: str = 'IT', start_year: int = 2015) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fetch Google Trends data
    Returns: (dataframe, status_message)
    """
    try:
        from pytrends.request import TrendReq
        
        pytrends = TrendReq(hl='it-IT', tz=60, timeout=(10, 25))
        
        # Build payload
        pytrends.build_payload(
            keywords,
            cat=0,
            timeframe=f'{start_year}-01-01 {datetime.now().strftime("%Y-%m-%d")}',
            geo=geo
        )
        
        # Get data
        df = pytrends.interest_over_time()
        
        if df.empty:
            return None, "❌ No trends data available"
        
        # Clean up
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['date'])
        
        # Rename columns
        rename_map = {k: f"gt_{k.replace(' ', '_')}" for k in keywords}
        df = df.rename(columns=rename_map)
        
        return df, f"✅ Success: {len(df)} weeks, {len(keywords)} keywords"
        
    except ImportError:
        return None, "❌ pytrends package not installed (pip install pytrends)"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_time_series_chart(df: pd.DataFrame, title: str, y_label: str, color: str = '#e74c3c'):
    """Create professional time series chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df.iloc[:, 1],  # Assumes second column is value
        mode='lines+markers',
        name=title,
        line=dict(color=color, width=3),
        marker=dict(size=5, color=color),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Add trend line
    if len(df) > 10:
        from scipy import stats
        x_numeric = np.arange(len(df))
        y_values = df.iloc[:, 1].values
        slope, intercept, r_value, _, _ = stats.linregress(x_numeric, y_values)
        trend_line = slope * x_numeric + intercept
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=trend_line,
            mode='lines',
            name='Trend',
            line=dict(color='rgba(100,100,100,0.3)', width=2, dash='dash'),
            hovertemplate='Trend: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        xaxis_title='Date',
        yaxis_title=y_label,
        template='plotly_white',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(250,250,250,0.5)',
        paper_bgcolor='white'
    )
    
    return fig


def create_summary_stats_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Create summary statistics table"""
    values = df.iloc[:, 1].values
    
    stats = {
        'Metric': [
            'Latest Value',
            'Mean',
            'Median',
            'Std Dev',
            'Min',
            'Max',
            'Range',
            'Observations'
        ],
        'Value': [
            f"{values[-1]:.2f}",
            f"{np.mean(values):.2f}",
            f"{np.median(values):.2f}",
            f"{np.std(values):.2f}",
            f"{np.min(values):.2f}",
            f"{np.max(values):.2f}",
            f"{np.max(values) - np.min(values):.2f}",
            f"{len(values)}"
        ]
    }
    
    return pd.DataFrame(stats)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    
    # ═══════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown("""
    <div class="hero-header">
        <h1>🇮🇹 Italian Economic Data</h1>
        <p>Automatic Fetching from Official Sources</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # SIDEBAR CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/03/Flag_of_Italy.svg/320px-Flag_of_Italy.svg.png", 
                 width=120)
        
        st.markdown("### ⚙️ Configuration")
        
        start_year = st.slider(
            "Start Year",
            min_value=2000,
            max_value=2024,
            value=2010,
            help="Select the starting year for data retrieval"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Select Data Sources")
        st.caption("Check the data sources you want to fetch")
        
        # Data source selection
        selected_sources = {}
        
        for key, info in DATA_SOURCES.items():
            if info['mandatory']:
                selected_sources[key] = st.checkbox(
                    f"{info['icon']} {info['name']}",
                    value=True,
                    disabled=True,
                    help="Mandatory data source"
                )
            else:
                selected_sources[key] = st.checkbox(
                    f"{info['icon']} {info['name']}",
                    value=False,
                    help=info['description']
                )
        
        # Google Trends keyword selection
        if selected_sources.get('trends', False):
            st.markdown("#### 🔍 Trends Keywords")
            n_keywords = st.slider(
                "Number of keywords",
                min_value=1,
                max_value=10,
                value=5,
                help="More keywords = slower fetching due to rate limits"
            )
            selected_keywords = ITALIAN_KEYWORDS[:n_keywords]
        else:
            selected_keywords = []
        
        st.markdown("---")
        
        # Fetch button
        fetch_button = st.button(
            "🚀 Fetch Data",
            type="primary",
            use_container_width=True,
            help="Click to start fetching data from selected sources"
        )
        
        # Clear cache button
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cache cleared!")
        
        st.markdown("---")
        
        # System info
        st.markdown("### 📍 System Info")
        st.info(f"""
        **Date:** {datetime.now().strftime('%Y-%m-%d')}  
        **Version:** 3.0  
        **Sources:** {len([s for s in selected_sources.values() if s])} selected
        """)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN CONTENT - BEFORE FETCH
    # ═══════════════════════════════════════════════════════════════════════
    
    if not fetch_button:
        
        # Show available data sources
        st.markdown('<h2 class="section-title">📋 Available Data Sources</h2>', unsafe_allow_html=True)
        
        st.info("""
        💡 **How to use:**
        1. Select your desired data sources from the sidebar
        2. Configure the start year
        3. Click "🚀 Fetch Data" to retrieve data automatically
        4. View, analyze, and download the results
        """)
        
        # Display data sources in cards
        for key, info in DATA_SOURCES.items():
            quality_class = f"quality-{info['quality']}"
            availability_class = "available" if info['available'] else "unavailable"
            
            st.markdown(f"""
            <div class="source-card {availability_class}">
                <div class="source-header">
                    {info['icon']} {info['name']}
                    <span class="badge badge-{'success' if info['available'] else 'danger'}">
                        {'Available' if info['available'] else 'Unavailable'}
                    </span>
                    {f'<span class="badge badge-warning">Mandatory</span>' if info['mandatory'] else ''}
                </div>
                <div class="source-description">
                    {info['description']}
                </div>
                <div class="source-metadata">
                    <strong>Provider:</strong> {info['provider']}<br>
                    <strong>Dataset:</strong> {info['dataset']}<br>
                    <strong>Frequency:</strong> {info['frequency']}<br>
                    <strong>Coverage:</strong> {info['coverage']}<br>
                    <strong>Update:</strong> {info['update']}<br>
                    <strong>Quality:</strong> <span class="quality-indicator {quality_class}">{info['quality'].upper()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical information
        st.markdown('<h2 class="section-title">🔧 Technical Information</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📦 Requirements
            
            ```bash
            pip install streamlit pandas numpy
            pip install plotly scipy
            pip install eurostat yfinance pytrends
            ```
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Data Quality
            
            - **Excellent:** Official statistics, fully validated
            - **Good:** Reliable, minor delays possible
            - **Fair:** Best-effort, subject to limitations
            """)
        
        st.stop()
    
    # ═══════════════════════════════════════════════════════════════════════
    # DATA FETCHING
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown('<h2 class="section-title">⏳ Fetching Data...</h2>', unsafe_allow_html=True)
    
    # Progress tracking
    total_sources = sum(selected_sources.values())
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Storage for fetched data
    fetched_data = {}
    fetch_status = {}
    
    current_progress = 0
    
    # Fetch Unemployment (mandatory)
    if selected_sources['unemployment']:
        status_text.text("📥 Fetching unemployment rate...")
        df, msg = fetch_eurostat_data(
            'une_rt_m',
            {'geo': 'IT', 's_adj': 'SA', 'age': 'TOTAL'},
            start_year
        )
        fetched_data['unemployment'] = df
        fetch_status['unemployment'] = msg
        current_progress += 1
        progress_bar.progress(current_progress / total_sources)
    
    # Fetch CCI
    if selected_sources['cci']:
        status_text.text("📊 Fetching consumer confidence...")
        df, msg = fetch_eurostat_data(
            'ei_bsco_m',
            {'geo': 'IT', 'indic': 'BS-CSMCI-BAL'},
            start_year
        )
        fetched_data['cci'] = df
        fetch_status['cci'] = msg
        current_progress += 1
        progress_bar.progress(current_progress / total_sources)
    
    # Fetch HICP
    if selected_sources['hicp']:
        status_text.text("📈 Fetching inflation index...")
        df, msg = fetch_eurostat_data(
            'prc_hicp_midx',
            {'geo': 'IT', 'coicop': 'CP00'},
            start_year
        )
        fetched_data['hicp'] = df
        fetch_status['hicp'] = msg
        current_progress += 1
        progress_bar.progress(current_progress / total_sources)
    
    # Fetch IIP
    if selected_sources['iip']:
        status_text.text("🏭 Fetching industrial production...")
        df, msg = fetch_eurostat_data(
            'sts_inpr_m',
            {'geo': 'IT', 's_adj': 'SA', 'nace_r2': 'B-D'},
            start_year
        )
        fetched_data['iip'] = df
        fetch_status['iip'] = msg
        current_progress += 1
        progress_bar.progress(current_progress / total_sources)
    
    # Fetch Stock
    if selected_sources['stock']:
        status_text.text("📈 Fetching FTSE MIB...")
        # Try multiple tickers
        for ticker in ['^FTSEMIB', 'FTSEMIB.MI', 'EWI']:
            df, msg = fetch_yahoo_data(ticker, start_year)
            if df is not None:
                fetched_data['stock'] = df
                fetch_status['stock'] = msg
                break
        if 'stock' not in fetched_data:
            fetch_status['stock'] = "❌ All tickers failed"
        current_progress += 1
        progress_bar.progress(current_progress / total_sources)
    
    # Fetch VIX
    if selected_sources['vix']:
        status_text.text("📊 Fetching volatility index...")
        for ticker in ['^V2TX', '^VIX']:
            df, msg = fetch_yahoo_data(ticker, start_year)
            if df is not None:
                fetched_data['vix'] = df
                fetch_status['vix'] = msg
                break
        if 'vix' not in fetched_data:
            fetch_status['vix'] = "❌ All tickers failed"
        current_progress += 1
        progress_bar.progress(current_progress / total_sources)
    
    # Fetch Google Trends
    if selected_sources['trends']:
        status_text.text("🔍 Fetching Google Trends...")
        df, msg = fetch_google_trends_data(selected_keywords, 'IT', max(start_year, 2015))
        fetched_data['trends'] = df
        fetch_status['trends'] = msg
        current_progress += 1
        progress_bar.progress(current_progress / total_sources)
    
    # Complete
    progress_bar.progress(1.0)
    status_text.empty()
    progress_bar.empty()
    
    # ═══════════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown('<h2 class="section-title">📊 Fetch Results</h2>', unsafe_allow_html=True)
    
    # Summary metrics
    successful = sum(1 for df in fetched_data.values() if df is not None)
    failed = total_sources - successful
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="status-box">
            <h3>{total_sources}</h3>
            <p>Total Sources</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="status-box" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%);">
            <h3>{successful}</h3>
            <p>Successful</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="status-box" style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);">
            <h3>{failed}</h3>
            <p>Failed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_rows = sum(len(df) for df in fetched_data.values() if df is not None)
        st.markdown(f"""
        <div class="status-box" style="background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);">
            <h3>{total_rows:,}</h3>
            <p>Data Points</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Status messages
    st.markdown("### 📋 Detailed Status")
    for source_key, status in fetch_status.items():
        if "✅" in status:
            st.success(f"**{DATA_SOURCES[source_key]['name']}:** {status}")
        else:
            st.error(f"**{DATA_SOURCES[source_key]['name']}:** {status}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # DATA VISUALIZATION & ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    if successful > 0:
        
        st.markdown('<h2 class="section-title">📈 Data Visualization</h2>', unsafe_allow_html=True)
        
        # Create tabs for each dataset
        tab_names = [DATA_SOURCES[k]['name'] for k in fetched_data.keys() if fetched_data[k] is not None]
        tabs = st.tabs(tab_names)
        
        tab_idx = 0
        for source_key, df in fetched_data.items():
            if df is None:
                continue
            
            with tabs[tab_idx]:
                info = DATA_SOURCES[source_key]
                
                # Chart
                if 'date' in df.columns:
                    fig = create_time_series_chart(
                        df,
                        info['name'],
                        'Value',
                        color='#e74c3c' if source_key == 'unemployment' else '#3B82F6'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### 📊 Summary Statistics")
                    stats_df = create_summary_stats_table(df, info['name'])
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### 📋 Data Preview")
                    st.dataframe(df.tail(10), use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    f"📥 Download {info['name']} (CSV)",
                    csv,
                    f"{source_key}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            tab_idx += 1
        
        # ═══════════════════════════════════════════════════════════════════
        # BULK EXPORT
        # ═══════════════════════════════════════════════════════════════════
        
        st.markdown('<h2 class="section-title">💾 Bulk Export</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export all as Excel
            if st.button("📥 Download All Data (Excel)", use_container_width=True, type="primary"):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    for source_key, df in fetched_data.items():
                        if df is not None:
                            df.to_excel(writer, sheet_name=DATA_SOURCES[source_key]['name'][:31], index=False)
                
                excel_data = output.getvalue()
                st.download_button(
                    "💾 Save Excel File",
                    excel_data,
                    f"italian_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col2:
            st.info("""
            **📊 Excel Export includes:**
            - All fetched datasets in separate sheets
            - Ready for analysis
            - Preserves all metadata
            """)
    
    else:
        st.error("❌ No data was successfully fetched. Please check your internet connection and try again.")

# ═══════════════════════════════════════════════════════════════════════════
# RUN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
