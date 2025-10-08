import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🇮🇹 Italian Unemployment Nowcaster",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #1e40af; text-align: center;}
</style>
""", unsafe_allow_html=True)

# ===== FUNCTIONS =====

def load_excel_sheet(uploaded_file, sheet_name):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        return df
    except Exception as e:
        st.error(f"❌ Error loading sheet '{sheet_name}': {str(e)}")
        return None

def clean_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def parse_dates(series):
    return pd.to_datetime(series, errors='coerce', dayfirst=True)

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {}
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    scale = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    mase = mae / scale if scale > 0 else np.nan
    
    return {'MAE': mae, 'RMSE': rmse, 'MASE': mase, 'N': len(y_true)}

# ===== MAIN APP =====

st.title("🇮🇹 Italian Unemployment Nowcasting System")
st.markdown("### Real-time forecasting for Italian labor market")

# Sidebar
with st.sidebar:
    st.image("https://flagcdn.com/w160/it.png", width=100)
    st.header("📂 Upload Data")
    
    uploaded_file = st.file_uploader(
        "Choose Excel file",
        type=['xlsx', 'xls'],
        help="Upload economic_data1.xlsx"
    )
    
    st.markdown("---")
    
    if uploaded_file is not None:
        st.success("✅ File uploaded!")
        
        st.subheader("⚙️ Settings")
        min_train = st.slider("Min training months", 24, 60, 48)
        
        st.markdown("---")
        
        st.subheader("🤖 Models")
        run_naive = st.checkbox("NAIVE", value=True)
        run_ma3 = st.checkbox("MA3", value=True)
        run_ridge = st.checkbox("Ridge", value=True)
        
        st.markdown("---")
        
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    else:
        st.info("👆 Upload Excel file")
        run_button = False

# Main content
if uploaded_file is None:
    st.info("📁 Please upload your Excel file from the sidebar")
    
    with st.expander("📋 Required Excel Structure"):
        st.markdown("""
        **Sheet: monthly**
        - Columns: `date`, `unemp`, `cci`, `prc_hicp`, `iip`
        
        **Sheet: daily_stock** (Optional)
        - Columns: `date`, `close`, `volume`
        
        **Sheet: VIX** (Optional)
        - Columns: `date`, `vix`
        
        **Sheet: google** (Optional)
        - Columns: `date`, keywords...
        """)
    
    st.stop()

if run_button:
    with st.spinner("⏳ Processing..."):
        
        monthly_df = load_excel_sheet(uploaded_file, 'monthly')
        
        if monthly_df is None:
            st.error("❌ Could not load 'monthly' sheet")
            st.stop()
        
        with st.expander("🔍 Raw Data Preview"):
            st.dataframe(monthly_df.head(10))
        
        monthly_df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') 
                              for c in monthly_df.columns]
        
        if 'date' not in monthly_df.columns:
            st.error("❌ 'date' column not found")
            st.stop()
        
        monthly_df['date'] = parse_dates(monthly_df['date'])
        monthly_df = monthly_df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        
        unemp_col = None
        for col in ['unemp', 'unemployment', 'unemployment_rate']:
            if col in monthly_df.columns:
                unemp_col = col
                break
        
        if unemp_col is None:
            st.error("❌ Unemployment column not found")
            st.stop()
        
        monthly_df[unemp_col] = clean_numeric(monthly_df[unemp_col])
        monthly_df = monthly_df.dropna(subset=[unemp_col])
        
        st.success(f"✅ Loaded {len(monthly_df)} observations")
        st.info(f"📊 Range: {monthly_df['date'].min().date()} → {monthly_df['date'].max().date()}")
        
        # Display data
        st.subheader("📊 Loaded Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Observations", len(monthly_df))
        with col2:
            st.metric("Start", monthly_df['date'].min().strftime('%Y-%m'))
        with col3:
            st.metric("End", monthly_df['date'].max().strftime('%Y-%m'))
        
        # Time series
        st.subheader("📈 Unemployment Rate")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly_df['date'], monthly_df[unemp_col], 
                linewidth=2.5, color='#2196F3', marker='o', markersize=4)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
        ax.set_title('Italian Unemployment Rate', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Modeling
        st.markdown("---")
        st.header("🤖 Forecasts")
        
        results = []
        all_forecasts = []
        
        # NAIVE
        if run_naive:
            st.subheader("1️⃣ NAIVE")
            
            naive_forecasts = []
            for t in range(1, len(monthly_df)):
                naive_forecasts.append({
                    'date': monthly_df['date'].iloc[t],
                    'actual': monthly_df[unemp_col].iloc[t],
                    'forecast': monthly_df[unemp_col].iloc[t-1],
                    'model': 'NAIVE'
                })
            
            naive_df = pd.DataFrame(naive_forecasts)
            metrics = compute_metrics(naive_df['actual'], naive_df['forecast'])
            
            results.append({'Model': 'NAIVE', **metrics})
            all_forecasts.append(naive_df)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['MAE']:.4f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
            col3.metric("MASE", f"{metrics['MASE']:.3f}")
        
        # MA3
        if run_ma3:
            st.subheader("2️⃣ MA3")
            
            ma3_forecasts = []
            for t in range(3, len(monthly_df)):
                ma3_forecasts.append({
                    'date': monthly_df['date'].iloc[t],
                    'actual': monthly_df[unemp_col].iloc[t],
                    'forecast': monthly_df[unemp_col].iloc[t-3:t].mean(),
                    'model': 'MA3'
                })
            
            ma3_df = pd.DataFrame(ma3_forecasts)
            metrics = compute_metrics(ma3_df['actual'], ma3_df['forecast'])
