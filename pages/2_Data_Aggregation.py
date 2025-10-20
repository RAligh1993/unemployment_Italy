"""
📊 Data Aggregation Pro v5.0
==============================
Professional data intake with ISTAT support and smart processing.

Features:
    ✅ Standard CSV/Excel upload
    ✅ ISTAT format processor (quarterly → monthly)
    ✅ Multi-sheet Excel support
    ✅ Auto-detection (columns, formats, frequency)
    ✅ Interactive visualizations
    ✅ State management integration

Author: ISTAT Nowcasting Team
Version: 5.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
from typing import Optional, Dict, List
from io import BytesIO

# Import from utils
try:
    from utils.state import AppState
    from utils.istat_handler import ISTATHandler
    from utils.excel_processor import ExcelProcessor
    from utils.data_detector import DataDetector
    from utils.visualizer import Visualizer
    UTILS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Utils import failed: {e}")
    st.info("Make sure utils/ folder exists with all required modules")
    UTILS_AVAILABLE = False
    st.stop()

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Data Aggregation Pro",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #1e3a8a, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .section-header {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.2s;
    }
    .feature-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Initialize Components
# =============================================================================

# State
state = AppState.get()

# Handlers
istat_handler = ISTATHandler()
excel_processor = ExcelProcessor()
detector = DataDetector()
visualizer = Visualizer()

# =============================================================================
# Header
# =============================================================================

st.markdown('<h1 class="main-title">📊 Data Aggregation Pro v5.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🇮🇹 Advanced Excel & Multi-format Data Processing for Italian Unemployment Nowcasting</p>', unsafe_allow_html=True)

# Feature highlights
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>📁 Excel & CSV</h4>
        <p>Multi-sheet support<br/>All formats (.xlsx, .xls, .xlsm)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>🇮🇹 ISTAT Format</h4>
        <p>Auto-detection<br/>Quarterly → Monthly</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🤖 Smart Detection</h4>
        <p>Auto columns<br/>Italian regions</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Visualization</h4>
        <p>Interactive charts<br/>Quality metrics</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📊 Output")
    align_month_end = st.checkbox("Align to month-end", value=True)
    
    st.markdown("---")
    st.subheader("💾 Current State")
    
    # Show state summary
    summary = AppState.summary()
    
    st.metric("Target Loaded", "✅" if summary['has_target'] else "❌")
    if summary['has_target']:
        st.caption(f"{summary['target_length']} observations")
    
    st.metric("Panel Built", "✅" if summary['has_panel_monthly'] else "❌")
    if summary['has_panel_monthly']:
        rows, cols = summary['panel_monthly_shape']
        st.caption(f"{rows} × {cols}")
    
    st.metric("Models Trained", summary['num_models'])
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All Data", use_container_width=True):
        AppState.clear()
        st.success("✅ Cleared!")
        st.rerun()

# =============================================================================
# Main Content - Tabs
# =============================================================================

tab1, tab2 = st.tabs(["📁 Standard Upload", "🇮🇹 ISTAT Format Processor"])

# =============================================================================
# TAB 1: Standard Upload
# =============================================================================

with tab1:
    st.markdown('<div class="section-header">📤 Upload Data Files</div>', unsafe_allow_html=True)
    
    st.info("""
    **📋 Supported Formats:**
    - Excel: .xlsx, .xls, .xlsm, .xlsb (multi-sheet)
    - CSV: .csv (all encodings)
    
    **🎯 Detection:**
    - Automatic header detection
    - Date column identification
    - Numeric column conversion
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Choose file",
        type=['csv', 'xlsx', 'xls', 'xlsm', 'xlsb'],
        key='standard_upload'
    )
    
    if uploaded_file:
        
        try:
            file_name = uploaded_file.name
            file_ext = file_name.split('.')[-1].lower()
            
            with st.expander(f"📄 {file_name}", expanded=True):
                
                # Read file
                if file_ext == 'csv':
                    st.info("📄 Reading CSV...")
                    
                    # Try multiple encodings
                    df = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding=encoding)
                            st.success(f"✅ Read with {encoding} encoding")
                            break
                        except:
                            continue
                    
                    if df is None:
                        st.error("❌ Could not read CSV with any encoding")
                        st.stop()
                
                else:
                    st.info("📊 Reading Excel...")
                    
                    # Read Excel
                    uploaded_file.seek(0)
                    sheets = excel_processor.read_file(uploaded_file)
                    
                    if not sheets:
                        st.error("❌ No sheets could be read")
                        st.stop()
                    
                    st.success(f"✅ Found {len(sheets)} sheet(s)")
                    
                    # Select sheet
                    if len(sheets) > 1:
                        sheet_name = st.selectbox(
                            "Select sheet:",
                            options=list(sheets.keys())
                        )
                    else:
                        sheet_name = list(sheets.keys())[0]
                    
                    df = sheets[sheet_name]
                    st.info(f"Using sheet: **{sheet_name}**")
                
                # Auto-detect
                st.markdown("### 🔍 Auto-Detection")
                
                with st.spinner("Analyzing..."):
                    analysis = detector.analyze_dataset(df)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Quality Score", f"{analysis.quality_score:.0f}/100")
                with col4:
                    st.metric("Time Series", "✅" if analysis.is_time_series else "❌")
                
                # Detection results
                if analysis.date_column:
                    st.success(f"📅 Date column: **{analysis.date_column}**")
                    st.info(f"⏱️ Frequency: **{analysis.frequency}**")
                
                if analysis.detected_formats:
                    st.success(f"🎯 Detected formats: {', '.join(analysis.detected_formats)}")
                
                # Show data preview
                st.markdown("### 👀 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Visualize if time series
                if analysis.is_time_series and analysis.date_column and analysis.value_columns:
                    st.markdown("### 📊 Visualization")
                    
                    # Select columns to plot
                    cols_to_plot = st.multiselect(
                        "Select columns to visualize:",
                        options=analysis.value_columns,
                        default=analysis.value_columns[:min(3, len(analysis.value_columns))]
                    )
                    
                    if cols_to_plot:
                        try:
                            fig = visualizer.plot_time_series(
                                df,
                                analysis.date_column,
                                cols_to_plot,
                                title=f"{file_name} - Time Series"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create chart: {e}")
                
                # Save options
                st.markdown("### 💾 Save Data")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save as Panel", use_container_width=True):
                        state.panel_monthly = df
                        AppState.update_timestamp()
                        st.success("✅ Saved to state.panel_monthly!")
                        st.balloons()
                
                with col2:
                    if analysis.is_time_series and len(analysis.value_columns) > 0:
                        if st.button("🎯 Save as Target", use_container_width=True, type="primary"):
                            # Use first value column as target
                            target_col = analysis.value_columns[0]
                            
                            ts = pd.Series(
                                df[target_col].values,
                                index=pd.to_datetime(df[analysis.date_column]),
                                name='unemployment_rate'
                            )
                            
                            state.y_monthly = ts
                            AppState.update_timestamp()
                            st.success(f"✅ Saved {target_col} as target!")
                            st.balloons()
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            with st.expander("🐛 Debug Info"):
                st.code(traceback.format_exc())

# =============================================================================
# TAB 2: ISTAT Format Processor
# =============================================================================

with tab2:
    st.markdown('<div class="section-header">🇮🇹 ISTAT Format Processor</div>', unsafe_allow_html=True)
    
    st.info("""
    **🎯 This processor handles ISTAT long-format data:**
    
    **Input Format:**
    - Territory: Italy, Lombardia, Veneto, ...
    - Sex: Males, Females, Total
    - Age: Y15-74, Y15-24, Y25-64, ...
    - Time: 2020-Q1, 2020-Q2, ... (Quarterly)
    - Value: Unemployment rate
    
    **Output:**
    - Clean time series (monthly or quarterly)
    - Ready for nowcasting models
    """)
    
    # File upload
    istat_file = st.file_uploader(
        "📁 Upload ISTAT file (Excel or CSV)",
        type=['xlsx', 'xls', 'csv'],
        key='istat_upload'
    )
    
    if istat_file:
        
        try:
            file_name = istat_file.name
            
            # Read file
            with st.spinner("📖 Reading file..."):
                if file_name.endswith('.csv'):
                    df = pd.read_csv(istat_file)
                else:
                    istat_file.seek(0)
                    sheets = excel_processor.read_file(istat_file)
                    
                    if len(sheets) > 1:
                        sheet_name = st.selectbox("Select sheet:", list(sheets.keys()), key='istat_sheet')
                    else:
                        sheet_name = list(sheets.keys())[0]
                    
                    df = sheets[sheet_name]
            
            st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Preview
            with st.expander("👀 Preview Data", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)
            
            # Check ISTAT format
            if istat_handler.is_istat_format(df):
                st.success("✅ ISTAT format confirmed!")
                
                # Detect columns
                cols = istat_handler.detect_columns(df)
                
                with st.expander("🔍 Detected Columns"):
                    for key, value in cols.to_dict().items():
                        st.write(f"- **{key}**: `{value}`")
                
                # Get filter options
                options = istat_handler.get_filter_options(df, cols)
                
                # Coverage analysis
                coverage = istat_handler.analyze_coverage(df, cols)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", coverage['total_rows'])
                with col2:
                    if coverage['time_range']:
                        st.metric("Date Range", f"{coverage['time_range'][0]} → {coverage['time_range'][1]}")
                with col3:
                    st.metric("Missing %", f"{coverage['missing_pct']:.1%}")
                
                # Filter interface
                st.markdown("---")
                st.markdown("### 🎯 Select Target Series")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'territorys' in options:
                        territory = st.selectbox(
                            "🗺️ Territory",
                            options=options['territorys']
                        )
                    else:
                        territory = None
                        st.warning("⚠️ Territory column not found")
                
                with col2:
                    if 'sexs' in options:
                        sex = st.selectbox(
                            "👤 Sex",
                            options=options['sexs']
                        )
                    else:
                        sex = None
                        st.warning("⚠️ Sex column not found")
                
                with col3:
                    if 'ages' in options:
                        age = st.selectbox(
                            "🎂 Age Group",
                            options=options['ages']
                        )
                    else:
                        age = None
                        st.warning("⚠️ Age column not found")
                
                # Filter data
                filtered = istat_handler.filter_data(df, cols, territory, sex, age)
                
                st.metric("📊 Filtered Rows", len(filtered))
                
                if len(filtered) == 0:
                    st.warning("⚠️ No data after filtering. Try different criteria.")
                
                elif len(filtered) > 0:
                    
                    # Convert to time series
                    if st.button("🔄 Convert to Time Series", type="primary", use_container_width=True):
                        
                        with st.spinner("Converting..."):
                            ts = istat_handler.convert_to_timeseries(filtered, cols)
                        
                        st.success(f"✅ Created time series: {len(ts)} observations")
                        
                        # Display
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Chart
                            fig = visualizer.plot_time_series(
                                ts.to_frame('unemployment_rate').reset_index(),
                                'index',
                                ['unemployment_rate'],
                                title=f'{territory} - {sex} - {age}'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### 📈 Statistics")
                            
                            st.metric("Start", ts.index.min().strftime('%Y-%m'))
                            st.metric("End", ts.index.max().strftime('%Y-%m'))
                            st.metric("Count", len(ts))
                            st.metric("Mean", f"{ts.mean():.2f}")
                            st.metric("Std", f"{ts.std():.2f}")
                            st.metric("Min", f"{ts.min():.2f}")
                            st.metric("Max", f"{ts.max():.2f}")
                        
                        # Frequency conversion
                        st.markdown("---")
                        st.markdown("### 🔄 Frequency Conversion")
                        
                        freq_detected = istat_handler.detect_frequency(ts)
                        st.info(f"Detected frequency: **{freq_detected}**")
                        
                        if freq_detected == 'quarterly':
                            
                            convert_monthly = st.checkbox(
                                "📅 Convert Quarterly → Monthly",
                                value=True,
                                help="Interpolate quarterly data to monthly"
                            )
                            
                            if convert_monthly:
                                
                                interp_method = st.radio(
                                    "Interpolation method:",
                                    ['linear', 'cubic', 'quadratic'],
                                    horizontal=True,
                                    help="Linear = simple, Cubic = smooth"
                                )
                                
                                with st.spinner("Interpolating..."):
                                    monthly = istat_handler.quarterly_to_monthly(ts, method=interp_method)
                                
                                st.success(f"✅ Monthly series: {len(monthly)} months")
                                
                                # Comparison chart
                                fig_compare = visualizer.plot_forecast_comparison(
                                    ts,
                                    {'Monthly (interpolated)': monthly},
                                    title='Quarterly → Monthly Conversion'
                                )
                                st.plotly_chart(fig_compare, use_container_width=True)
                                
                                # Save options
                                st.markdown("---")
                                st.markdown("### 💾 Save Data")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if st.button("💾 Save Quarterly", use_container_width=True):
                                        state.y_monthly = ts
                                        AppState.update_timestamp()
                                        st.success("✅ Saved quarterly data!")
                                        st.balloons()
                                
                                with col2:
                                    if st.button("💾 Save Monthly", use_container_width=True, type="primary"):
                                        state.y_monthly = monthly
                                        AppState.update_timestamp()
                                        st.success("✅ Saved monthly data!")
                                        st.balloons()
                            
                            else:
                                # Save quarterly only
                                if st.button("💾 Save as Target", type="primary", use_container_width=True):
                                    state.y_monthly = ts
                                    AppState.update_timestamp()
                                    st.success("✅ Saved!")
                                    st.balloons()
                        
                        else:
                            # Not quarterly
                            if st.button("💾 Save as Target", type="primary", use_container_width=True):
                                state.y_monthly = ts
                                AppState.update_timestamp()
                                st.success("✅ Saved!")
                                st.balloons()
            
            else:
                st.warning("⚠️ This doesn't look like ISTAT format")
                st.info("Try using the **Standard Upload** tab instead")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            
            with st.expander("🐛 Debug Info"):
                st.code(traceback.format_exc())

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("📊 Data Aggregation Pro v5.0")

with col2:
    st.caption("🇮🇹 ISTAT Unemployment Nowcasting")

with col3:
    st.caption("💻 Built with Streamlit + Utils")
```

---

## 🚀 **نحوه استفاده:**

### 1️⃣ ساختار نهایی:
```
unemployment_Italy/
├── utils/
│   ├── __init__.py              ✅
│   ├── state.py                 ✅
│   ├── istat_handler.py         ✅
│   ├── excel_processor.py       ✅
│   ├── data_detector.py         ✅
│   └── visualizer.py            ✅
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Data_Aggregation.py   ✅ (جدید)
│   └── ...
├── app.py
└── requirements.txt
