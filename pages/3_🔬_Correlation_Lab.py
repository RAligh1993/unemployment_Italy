"""
═══════════════════════════════════════════════════════════════════════════════
🇮🇹 ITALIAN UNEMPLOYMENT NOWCASTING SYSTEM
═══════════════════════════════════════════════════════════════════════════════

A comprehensive time series forecasting system for Italian unemployment rate
using multiple data sources and advanced econometric models.

Author: AI Research Team
Version: 2.0 (Production)
Date: October 2025

Features:
- Multi-source data integration (Eurostat, Yahoo Finance, Google Trends)
- Manual CSV upload & Automatic data fetching
- Advanced forecasting models (Ridge, MIDAS, ETS, Ensembles)
- Interactive visualizations
- Walk-forward validation
- Publication-ready outputs

═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Italian Unemployment Nowcasting",
    page_icon="🇮🇹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/italian-unemployment',
        'Report a bug': 'https://github.com/yourusername/italian-unemployment/issues',
        'About': """
        # Italian Unemployment Nowcasting System
        
        **Version:** 2.0  
        **Author:** AI Research Team  
        **License:** MIT
        
        A comprehensive forecasting system for Italian unemployment rate.
        """
    }
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Italian flag gradient header */
    .italian-header {
        text-align: center;
        background: linear-gradient(90deg, #009246 0%, #FFFFFF 33%, #FFFFFF 66%, #CE2B37 100%);
        padding: 60px 20px;
        border-radius: 20px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .italian-header h1 {
        color: #2c3e50;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .italian-header p {
        color: #34495e;
        font-size: 1.4rem;
        margin: 15px 0 0 0;
        font-weight: 500;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        height: 100%;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }
    
    .feature-card h2 {
        font-size: 2rem;
        margin: 20px 0;
        font-weight: 700;
    }
    
    .feature-card p {
        font-size: 1.1rem;
        line-height: 1.6;
        opacity: 0.95;
    }
    
    .feature-icon {
        font-size: 5rem;
        margin-bottom: 20px;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Auto-fetch card gradient */
    .auto-fetch-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Modeling card gradient */
    .modeling-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Stats boxes */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stat-box h3 {
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
    }
    
    .stat-box p {
        font-size: 1.2rem;
        margin: 10px 0 0 0;
        opacity: 0.9;
    }
    
    /* Section headers */
    .section-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 60px 0 30px 0;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        display: block;
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 15px auto 0;
        border-radius: 2px;
    }
    
    /* Info cards */
    .info-card {
        background: #f8f9fa;
        padding: 30px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 20px 0;
    }
    
    .info-card h4 {
        color: #2c3e50;
        font-size: 1.5rem;
        margin-top: 0;
    }
    
    /* Buttons */
    .stButton > button {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 15px 30px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px 20px;
        background: #f8f9fa;
        border-radius: 15px;
        margin-top: 60px;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="italian-header">
    <h1>🇮🇹 Italian Unemployment Nowcasting</h1>
    <p>Advanced Time Series Forecasting System</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# QUICK STATS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-header">📊 System Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-box">
        <h3>3</h3>
        <p>Data Input Methods</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <h3>10+</h3>
        <p>Forecasting Models</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <h3>5</h3>
        <p>Data Sources</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <h3>25+</h3>
        <p>Years Coverage</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-header">🚀 Choose Your Workflow</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📤</div>
        <h2>Manual Upload</h2>
        <p>Upload your own CSV files for custom analysis and panel building</p>
        <br>
        <p><strong>✨ Perfect for:</strong></p>
        <ul style="text-align: left; padding-left: 30px; font-size: 1rem;">
            <li>Custom datasets</li>
            <li>Proprietary data</li>
            <li>Historical archives</li>
            <li>Offline analysis</li>
            <li>Multi-file aggregation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    if st.button("📤 Go to Manual Upload", use_container_width=True, type="primary", key="btn_manual"):
        st.switch_page("pages/1_📤_Manual_Upload.py")

with col2:
    st.markdown("""
    <div class="feature-card auto-fetch-card">
        <div class="feature-icon">🤖</div>
        <h2>Auto-Fetch</h2>
        <p>Automatically fetch latest data from official sources with one click</p>
        <br>
        <p><strong>✨ Perfect for:</strong></p>
        <ul style="text-align: left; padding-left: 30px; font-size: 1rem;">
            <li>Real-time updates</li>
            <li>Official statistics</li>
            <li>Quick prototyping</li>
            <li>Reproducible research</li>
            <li>Live dashboards</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    if st.button("🤖 Go to Auto-Fetch", use_container_width=True, type="primary", key="btn_auto"):
        st.switch_page("pages/2_🤖_Auto_Fetch.py")

with col3:
    st.markdown("""
    <div class="feature-card modeling-card">
        <div class="feature-icon">🔮</div>
        <h2>Modeling</h2>
        <p>Train forecasting models and generate predictions with advanced algorithms</p>
        <br>
        <p><strong>✨ Perfect for:</strong></p>
        <ul style="text-align: left; padding-left: 30px; font-size: 1rem;">
            <li>Forecast generation</li>
            <li>Model comparison</li>
            <li>Walk-forward validation</li>
            <li>Ensemble methods</li>
            <li>Performance metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    if st.button("🔮 Go to Modeling", use_container_width=True, type="primary", key="btn_model"):
        st.switch_page("pages/3_🔮_Modeling.py")

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM FEATURES
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-header">⭐ Key Features</h2>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Sources", "🔧 Processing", "🔮 Models", "📈 Outputs"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🌍 Official Sources</h4>
            <ul>
                <li><strong>Eurostat:</strong> Monthly unemployment, CCI, HICP inflation</li>
                <li><strong>Yahoo Finance:</strong> FTSE MIB stock index, daily prices</li>
                <li><strong>CBOE/Yahoo:</strong> V2TX/VIX volatility indices</li>
                <li><strong>Google Trends:</strong> 41 Italian job-market keywords</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>📅 Coverage & Frequency</h4>
            <ul>
                <li><strong>Period:</strong> 2000-2025 (25+ years)</li>
                <li><strong>Monthly:</strong> Unemployment, CCI, HICP</li>
                <li><strong>Daily:</strong> Stock prices, volatility</li>
                <li><strong>Weekly:</strong> Google Trends data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔄 Data Processing
        - ✅ Automatic date parsing & alignment
        - ✅ Multiple frequency aggregation
        - ✅ Missing data imputation
        - ✅ Outlier detection & handling
        - ✅ Vintage control (day 15 cutoff)
        - ✅ Business days filtering
        """)
    
    with col2:
        st.markdown("""
        ### 🧹 Data Cleaning
        - ✅ Constant column removal
        - ✅ High correlation filtering
        - ✅ Coverage analysis
        - ✅ Quality validation
        - ✅ Panel balancing
        - ✅ Feature selection
        """)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Benchmark Models
        - **NAIVE:** Random walk baseline
        - **MA3:** 3-month moving average
        - **MA12:** 12-month moving average
        - **ETS:** Exponential smoothing
        """)
        
        st.markdown("""
        ### 🔥 Ridge Regression
        - **AR_Ridge_BASE:** Autoregressive + controls
        - **AR_Ridge_FIN:** + Financial features
        - **GT_Ridge_PCA:** + Google Trends PCA
        - **Combined:** Full feature set
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Advanced Methods
        - **MIDAS:** Mixed Data Sampling with Almon weights
        - **MIDAS_AR:** MIDAS + AR lags
        - **Ensemble_Simple:** Mean of all models
        - **Ensemble_Trim:** Trimmed mean ensemble
        """)
        
        st.markdown("""
        ### 📐 Validation
        - Walk-forward one-step-ahead
        - Zero data snooping guarantee
        - Cross-validated hyperparameters
        - Multiple error metrics
        """)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 Visualizations
        - Interactive time series plots
        - Forecast vs actual overlays
        - Error distribution charts
        - Correlation heatmaps
        - Coverage diagnostics
        - Model comparison bars
        """)
    
    with col2:
        st.markdown("""
        ### 💾 Export Formats
        - CSV (comma-separated)
        - Excel (multi-sheet)
        - Publication-ready figures (PNG, 200 DPI)
        - Summary statistics tables
        - Model performance reports
        - Configuration metadata
        """)

# ═══════════════════════════════════════════════════════════════════════════
# METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-header">📚 Methodology</h2>', unsafe_allow_html=True)

with st.expander("🔬 Forecasting Pipeline", expanded=False):
    st.markdown("""
    ### End-to-End Workflow
    
    **1. Data Collection** 🔽
    ```
    Multiple sources → Automated fetching → Quality validation
    ```
    
    **2. Feature Engineering** ⚙️
    ```
    Raw data → AR lags → MTD aggregation → Financial features → Google Trends PCA → COVID dummies
    ```
    
    **3. Panel Building** 🔨
    ```
    Daily data → Monthly aggregation → Alignment → Cleaning → Balanced panel
    ```
    
    **4. Model Training** 🎓
    ```
    Walk-forward CV → Hyperparameter tuning → One-step-ahead forecasts → Ensemble
    ```
    
    **5. Evaluation** 📊
    ```
    Multiple metrics → Visual diagnostics → Model comparison → Best model selection
    ```
    
    **6. Deployment** 🚀
    ```
    Real-time forecasts → Export results → Publication-ready outputs
    ```
    """)

with st.expander("📊 Evaluation Metrics", expanded=False):
    st.markdown("""
    ### Performance Measures
    
    | Metric | Formula | Interpretation |
    |--------|---------|----------------|
    | **MAE** | `mean(|actual - forecast|)` | Average absolute error in pp |
    | **RMSE** | `sqrt(mean((actual - forecast)²))` | Root mean squared error |
    | **MASE** | `MAE / mean(|y[t] - y[t-1]|)` | Scaled vs naive benchmark |
    | **MAPE** | `mean(|actual - forecast| / |actual|) × 100` | Percentage error |
    
    **MASE Interpretation:**
    - MASE < 1.0: Better than naive
    - MASE = 1.0: Equal to naive
    - MASE > 1.0: Worse than naive
    """)

with st.expander("🎯 COVID-19 Adjustments", expanded=False):
    st.markdown("""
    ### Italian Lockdown Periods
    
    **Lockdown 1:**
    - Start: March 9, 2020
    - End: May 18, 2020
    - Duration: 70 days
    
    **Lockdown 2:**
    - Start: November 6, 2020
    - End: December 3, 2020
    - Duration: 27 days
    
    **Lockdown 3:**
    - Start: March 15, 2021
    - End: April 26, 2021
    - Duration: 42 days
    
    **COVID Era Dummy:**
    - All periods from March 2020 onwards
    
    Models can include these dummies to capture structural breaks.
    """)

# ═══════════════════════════════════════════════════════════════════════════
# QUICK START GUIDE
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-header">🚀 Quick Start Guide</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ Get Data
    
    **Option A: Auto-Fetch** ⚡
    1. Click "Auto-Fetch"
    2. Select data sources
    3. Click "Fetch Data"
    4. Data ready instantly!
    
    **Option B: Upload** 📤
    1. Click "Manual Upload"
    2. Upload CSV files
    3. Configure settings
    4. Build panel
    """)

with col2:
    st.markdown("""
    ### 2️⃣ Prepare Data
    
    1. Review data quality
    2. Check coverage
    3. Handle missing values
    4. Aggregate frequencies
    5. Clean outliers
    6. Export panel
    
    All automated! ✨
    """)

with col3:
    st.markdown("""
    ### 3️⃣ Forecast
    
    1. Click "Modeling"
    2. Choose scenarios
    3. Select models
    4. Run backtesting
    5. View results
    6. Export forecasts
    
    Production-ready! 🎯
    """)

# ═══════════════════════════════════════════════════════════════════════════
# TECHNICAL INFORMATION
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-header">🔧 Technical Details</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    with st.expander("📦 Requirements"):
        st.code("""
# Core
streamlit >= 1.28.0
pandas >= 2.0.0
numpy >= 1.24.0

# Visualization
plotly >= 5.17.0

# Modeling
scikit-learn >= 1.3.0
statsmodels >= 0.14.0

# Data Sources
yfinance >= 0.2.28
eurostat >= 1.0.0
pytrends >= 4.9.2

# Utils
xlsxwriter >= 3.1.2
openpyxl >= 3.1.2
requests >= 2.31.0
scipy >= 1.11.0
        """, language="text")

with col2:
    with st.expander("🏗️ Project Structure"):
        st.code("""
italian_unemployment_project/
│
├── app.py                      # Home page
│
├── pages/
│   ├── 1_📤_Manual_Upload.py  # Upload interface
│   ├── 2_🤖_Auto_Fetch.py     # Auto-fetch interface
│   └── 3_🔮_Modeling.py        # Forecasting models
│
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
        """, language="text")

# Installation guide
with st.expander("⚙️ Installation & Setup"):
    st.markdown("""
    ### Step-by-Step Installation
    
    **1. Install Python 3.9+**
    
    **2. Create project directory**
    ```bash
    mkdir italian_unemployment_project
    cd italian_unemployment_project
    ```
    
    **3. Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    
    **4. Run application**
    ```bash
    streamlit run app.py
    ```
    
    **5. Access in browser**
    ```
    http://localhost:8501
    ```
    
    ### Troubleshooting
    
    - **Port already in use:** `streamlit run app.py --server.port 8502`
    - **Cache issues:** Click "Clear Cache" in sidebar
    - **Data fetch errors:** Check internet connection
    """)

# ═══════════════════════════════════════════════════════════════════════════
# ABOUT & CONTACT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-header">ℹ️ About</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📖 Documentation
    
    - [User Guide](https://github.com)
    - [API Reference](https://github.com)
    - [Tutorial Videos](https://youtube.com)
    - [Research Paper](https://arxiv.org)
    """)

with col2:
    st.markdown("""
    ### 🤝 Contribute
    
    - [GitHub Repository](https://github.com)
    - [Issue Tracker](https://github.com)
    - [Pull Requests](https://github.com)
    - [Discussions](https://github.com)
    """)

with col3:
    st.markdown("""
    ### 📧 Contact
    
    - Email: [research@example.com](mailto:research@example.com)
    - Twitter: [@ItalyForecast](https://twitter.com)
    - LinkedIn: [Company Page](https://linkedin.com)
    - Website: [example.com](https://example.com)
    """)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    <h3>🇮🇹 Italian Unemployment Nowcasting System</h3>
    <p style="font-size: 1.1rem; margin: 10px 0;">
        <strong>Version 2.0</strong> | Production Release | October 2025
    </p>
    <p style="margin: 20px 0;">
        Built with ❤️ using Streamlit • Data from Eurostat, Yahoo Finance, Google Trends
    </p>
    <p style="color: #adb5bd; margin: 10px 0;">
        For research and educational purposes • MIT License
    </p>
    <p style="margin-top: 20px;">
        <a href="https://github.com" style="margin: 0 10px;">GitHub</a> •
        <a href="https://docs.example.com" style="margin: 0 10px;">Documentation</a> •
        <a href="https://example.com" style="margin: 0 10px;">Website</a>
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR INFO
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/03/Flag_of_Italy.svg/320px-Flag_of_Italy.svg.png", width=100)
    
    st.markdown("### 📍 System Status")
    st.success("✅ All systems operational")
    
    st.markdown("### 📊 Quick Stats")
    st.info(f"""
    **Current Date:** {datetime.now().strftime('%Y-%m-%d')}  
    **System Version:** 2.0  
    **Pages:** 4  
    **Models:** 10+
    """)
    
    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [📤 Manual Upload](pages/1_📤_Manual_Upload.py)
    - [🤖 Auto-Fetch](pages/2_🤖_Auto_Fetch.py)
    - [🔮 Modeling](pages/3_🔮_Modeling.py)
    """)
    
    st.markdown("---")
    st.caption("© 2025 Italian Unemployment Nowcasting System")
