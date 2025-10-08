import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import requests
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ============= CONFIG =============
st.set_page_config(
    page_title="🇮🇹 Italian Unemployment AI Nowcaster",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CUSTOM CSS =============
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    .stApp {background-color: #f8f9fa;}
    h1 {
        color: #1e40af;
        text-align: center;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    .ai-response {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #0284c7;
        margin: 15px 0;
    }
    .news-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .trend-up {color: #16a34a; font-weight: bold;}
    .trend-down {color: #dc2626; font-weight: bold;}
    .trend-neutral {color: #ca8a04; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ============= CLAUDE API INTEGRATION =============

def call_claude_api(prompt, context_data=None):
    """
    Call Claude API for AI insights
    """
    try:
        # استفاده از API که در document دیدیم
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['content'][0]['text']
        else:
            return "⚠️ AI Assistant temporarily unavailable"
    
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ============= AI ANALYSIS FUNCTIONS =============

def analyze_trend_with_ai(df, current_rate, forecast_rate):
    """تحلیل هوشمند روند با Claude"""
    
    # آماده‌سازی context
    recent_data = df.tail(6)[['date', 'unemployment']].to_dict('records')
    
    trend = "declining" if forecast_rate < current_rate else "rising" if forecast_rate > current_rate else "stable"
    change = abs(forecast_rate - current_rate)
    
    prompt = f"""
    You are an expert labor market economist analyzing Italian unemployment data.
    
    Current Context:
    - Current unemployment rate: {current_rate:.1f}%
    - Forecasted next month: {forecast_rate:.1f}%
    - Trend: {trend} (change of {change:.2f} percentage points)
    - Recent 6-month history: {recent_data}
    
    Please provide:
    1. A brief assessment (2-3 sentences) of the current labor market situation
    2. Key factors that might be influencing this trend
    3. Short-term outlook (positive/negative/neutral)
    4. One actionable policy recommendation
    
    Keep response concise and practical. Write in a professional but accessible tone.
    """
    
    with st.spinner("🤖 AI analyzing trends..."):
        response = call_claude_api(prompt)
    
    return response

def get_unemployment_news():
    """دریافت اخبار بیکاری ایتالیا"""
    
    # Simulated news (در واقعیت از API خبری می‌گیری)
    news_items = [
        {
            "title": "Italian Youth Unemployment Shows Signs of Improvement",
            "source": "Reuters",
            "date": "2025-10-07",
            "sentiment": "positive",
            "summary": "Youth unemployment in Italy dropped to 18.5% in September..."
        },
        {
            "title": "Manufacturing Sector Adds 15,000 Jobs in Q3",
            "source": "Il Sole 24 Ore",
            "date": "2025-10-06",
            "sentiment": "positive",
            "summary": "Italian manufacturing sector continues recovery with significant job additions..."
        },
        {
            "title": "Concerns Over Seasonal Employment Decline",
            "source": "Financial Times",
            "date": "2025-10-05",
            "sentiment": "negative",
            "summary": "Tourism sector employment shows seasonal decline in autumn months..."
        }
    ]
    
    return news_items

def chat_with_ai(user_message, conversation_history):
    """Chatbot برای پاسخ به سوالات کاربر"""
    
    # ساخت context از تاریخچه مکالمه
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    
    prompt = f"""
    You are an AI assistant specialized in Italian labor market analysis and unemployment forecasting.
    
    Previous conversation:
    {context}
    
    User: {user_message}
    
    Provide a helpful, accurate response about:
    - Italian unemployment trends
    - Economic indicators
    - Forecasting methods
    - Policy implications
    
    Keep responses concise (2-4 sentences) and data-driven when possible.
    """
    
    response = call_claude_api(prompt)
    return response

# ============= UTILITY FUNCTIONS =============

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

# ============= MAIN APP =============

st.title("🤖 Italian Unemployment AI Nowcaster")
st.markdown("### Advanced forecasting with AI-powered insights")

# ============= SIDEBAR =============
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
        
        # Tabs for settings
        tab1, tab2 = st.tabs(["⚙️ Models", "🤖 AI Settings"])
        
        with tab1:
            st.subheader("Select Models")
            run_naive = st.checkbox("NAIVE", value=True)
            run_ma3 = st.checkbox("MA3", value=True)
            run_ridge = st.checkbox("Ridge", value=True)
            
            min_train = st.slider("Min training months", 24, 60, 48)
        
        with tab2:
            st.subheader("AI Features")
            enable_ai = st.checkbox("Enable AI Analysis", value=True)
            enable_news = st.checkbox("Show News Feed", value=True)
            enable_chat = st.checkbox("Enable AI Chat", value=True)
        
        st.markdown("---")
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    else:
        st.info("👆 Upload Excel file")
        run_button = False
        enable_ai = False
        enable_news = False
        enable_chat = False

# ============= MAIN CONTENT =============

if uploaded_file is None:
    st.info("📁 Please upload your Excel file from the sidebar")
    
    # Welcome section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI-Powered</h3>
            <p>Advanced Claude AI for trend analysis and insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📰 Real-time News</h3>
            <p>Latest unemployment news with sentiment analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💬 Interactive Chat</h3>
            <p>Ask questions about unemployment trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# ============= DATA PROCESSING =============

if run_button:
    with st.spinner("⏳ Processing data..."):
        
        monthly_df = load_excel_sheet(uploaded_file, 'monthly')
        
        if monthly_df is None:
            st.error("❌ Could not load 'monthly' sheet")
            st.stop()
        
        # Clean data
        monthly_df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') 
                              for c in monthly_df.columns]
        
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
        
        # Rename for consistency
        monthly_df = monthly_df.rename(columns={unemp_col: 'unemployment'})
        
        st.success(f"✅ Loaded {len(monthly_df)} observations")
        
        # ============= KEY METRICS =============
        
        current_rate = monthly_df['unemployment'].iloc[-1]
        prev_rate = monthly_df['unemployment'].iloc[-2]
        change = current_rate - prev_rate
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Rate",
                f"{current_rate:.1f}%",
                f"{change:+.2f}%"
            )
        
        with col2:
            trend_6m = monthly_df['unemployment'].iloc[-6:].mean()
            st.metric(
                "6-Month Avg",
                f"{trend_6m:.1f}%"
            )
        
        with col3:
            volatility = monthly_df['unemployment'].iloc[-12:].std()
            st.metric(
                "Volatility (12m)",
                f"{volatility:.2f}pp"
            )
        
        with col4:
            yoy_change = current_rate - monthly_df['unemployment'].iloc[-13] if len(monthly_df) >= 13 else 0
            st.metric(
                "YoY Change",
                f"{yoy_change:+.2f}%"
            )
        
        # ============= AI INSIGHTS =============
        
        if enable_ai:
            st.markdown("---")
            st.header("🤖 AI-Powered Insights")
            
            # Simple forecast for demo
            forecast_rate = monthly_df['unemployment'].iloc[-1]  # Will be replaced by model
            
            # Get AI analysis
            ai_response = analyze_trend_with_ai(
                monthly_df, 
                current_rate, 
                forecast_rate
            )
            
            st.markdown(f"""
            <div class="ai-response">
                <h4>🧠 AI Analysis</h4>
                {ai_response}
            </div>
            """, unsafe_allow_html=True)
            
            # Trend indicator
            if forecast_rate < current_rate:
                trend_class = "trend-down"
                trend_icon = "📉"
                trend_text = "Improving (Declining Unemployment)"
            elif forecast_rate > current_rate:
                trend_class = "trend-up"
                trend_icon = "📈"
                trend_text = "Worsening (Rising Unemployment)"
            else:
                trend_class = "trend-neutral"
                trend_icon = "➡️"
                trend_text = "Stable"
            
            st.markdown(f"""
            <div style="text-align: center; font-size: 1.5rem; margin: 20px 0;">
                {trend_icon} <span class="{trend_class}">Trend: {trend_text}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # ============= NEWS FEED =============
        
        if enable_news:
            st.markdown("---")
            st.header("📰 Latest News & Sentiment")
            
            news_items = get_unemployment_news()
            
            for news in news_items:
                sentiment_color = {
                    'positive': '#16a34a',
                    'negative': '#dc2626',
                    'neutral': '#ca8a04'
                }[news['sentiment']]
                
                sentiment_emoji = {
                    'positive': '😊',
                    'negative': '😟',
                    'neutral': '😐'
                }[news['sentiment']]
                
                st.markdown(f"""
                <div class="news-card">
                    <h4 style="margin: 0 0 10px 0;">{news['title']}</h4>
                    <p style="color: #666; font-size: 0.9rem; margin: 5px 0;">
                        <strong>{news['source']}</strong> • {news['date']} • 
                        <span style="color: {sentiment_color};">{sentiment_emoji} {news['sentiment'].title()}</span>
                    </p>
                    <p style="margin: 10px 0 0 0;">{news['summary']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # ============= TIME SERIES CHART =============
        
        st.markdown("---")
        st.header("📈 Unemployment Trend Analysis")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot main line
        ax.plot(monthly_df['date'], monthly_df['unemployment'], 
                linewidth=3, color='#2563eb', marker='o', markersize=6,
                label='Actual Rate')
        
        # Add trend line
        z = np.polyfit(range(len(monthly_df)), monthly_df['unemployment'], 1)
        p = np.poly1d(z)
        ax.plot(monthly_df['date'], p(range(len(monthly_df))), 
                "--", color='#dc2626', linewidth=2, alpha=0.7,
                label='Trend Line')
        
        # Highlight last 12 months
        last_12 = monthly_df.tail(12)
        ax.fill_between(last_12['date'], last_12['unemployment'], 
                         alpha=0.3, color='#fbbf24',
                         label='Last 12 Months')
        
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax.set_ylabel('Unemployment Rate (%)', fontsize=13, fontweight='bold')
        ax.set_title('Italian Unemployment Rate - Historical Trend', 
                     fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        
        # ============= MODELING =============
        
        st.markdown("---")
        st.header("🤖 Forecast Models")
        
        results = []
        all_forecasts = []
        
        # NAIVE
        if run_naive:
            with st.expander("1️⃣ NAIVE Model", expanded=True):
                naive_forecasts = []
                for t in range(1, len(monthly_df)):
                    naive_forecasts.append({
                        'date': monthly_df['date'].iloc[t],
                        'actual': monthly_df['unemployment'].iloc[t],
                        'forecast': monthly_df['unemployment'].iloc[t-1],
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
            with st.expander("2️⃣ MA3 Model"):
                ma3_forecasts = []
                for t in range(3, len(monthly_df)):
                    ma3_forecasts.append({
                        'date': monthly_df['date'].iloc[t],
                        'actual': monthly_df['unemployment'].iloc[t],
                        'forecast': monthly_df['unemployment'].iloc[t-3:t].mean(),
                        'model': 'MA3'
                    })
                
                ma3_df = pd.DataFrame(ma3_forecasts)
                metrics = compute_metrics(ma3_df['actual'], ma3_df['forecast'])
                
                results.append({'Model': 'MA3', **metrics})
                all_forecasts.append(ma3_df)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{metrics['MAE']:.4f}")
                col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
                col3.metric("MASE", f"{metrics['MASE']:.3f}")
        
        # Ridge
        if run_ridge and len(monthly_df) >= min_train + 12:
            with st.expander("3️⃣ Ridge Regression"):
                for lag in [1, 2, 3, 12]:
                    monthly_df[f'lag{lag}'] = monthly_df['unemployment'].shift(lag)
                
                ridge_forecasts = []
                
                for t in range(min_train, len(monthly_df)):
                    train = monthly_df.iloc[:t].copy()
                    test = monthly_df.iloc[[t]].copy()
                    
                    features = ['lag1', 'lag2', 'lag3', 'lag12']
                    train = train.dropna(subset=['unemployment'] + features)
                    
                    if len(train) < min_train:
                        continue
                    
                    X_train = train[features].values
                    y_train = train['unemployment'].values
                    X_test = test[features].values
                    
                    pipe = Pipeline([
                        ('impute', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler()),
                        ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 20)))
                    ])
                    
                    try:
                        pipe.fit(X_train, y_train)
                        yhat = pipe.predict(X_test)[0]
                        
                        ridge_forecasts.append({
                            'date': test['date'].values[0],
                            'actual': test['unemployment'].values[0],
                            'forecast': yhat,
                            'model': 'Ridge'
                        })
                    except:
                        continue
                
                if ridge_forecasts:
                    ridge_df = pd.DataFrame(ridge_forecasts)
                    metrics = compute_metrics(ridge_df['actual'], ridge_df['forecast'])
                    
                    results.append({'Model': 'Ridge', **metrics})
                    all_forecasts.append(ridge_df)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MAE", f"{metrics['MAE']:.4f}")
                    col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
                    col3.metric("MASE", f"{metrics['MASE']:.3f}")
        
        # ============= RESULTS COMPARISON =============
        
        if results:
            st.markdown("---")
            st.header("📊 Model Comparison")
            
            results_df = pd.DataFrame(results).sort_values('MASE')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    results_df.style.format({
                        'MAE': '{:.4f}',
                        'RMSE': '{:.4f}',
                        'MASE': '{:.3f}',
                        'N': '{:.0f}'
                    }).background_gradient(subset=['MASE'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
            
            with col2:
                best_model = results_df.iloc[0]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏆 Best Model</h3>
                    <h2 style="color: #2563eb; margin: 10px 0;">{best_model['Model']}</h2>
                    <p style="font-size: 1.2rem;">MASE: {best_model['MASE']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance chart
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            
            colors = ['#16a34a' if x < 1.0 else '#dc2626' if x > 1.5 else '#ca8a04'
                     for x in results_df['MASE']]
            
            bars = ax2.barh(results_df['Model'], results_df['MASE'], 
                           color=colors, edgecolor='black', alpha=0.8, height=0.6)
            ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, label='NAIVE baseline', zorder=0)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, results_df['MASE'])):
                ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}',
                        va='center', fontweight='bold', fontsize=10)
            
            ax2.set_xlabel('MASE (Lower = Better)', fontsize=12, fontweight='bold')
            ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
            ax2.legend(fontsize=10)
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            
            # ============= FORECAST VISUALIZATION =============
            
            if all_forecasts:
                st.markdown("---")
                st.header("🔮 Forecast Comparison (Last 24 Months)")
                
                fig3, ax3 = plt.subplots(figsize=(14, 6))
                
                colors_map = {'NAIVE': '#dc2626', 'MA3': '#ca8a04', 'Ridge': '#16a34a'}
                
                for fc_df in all_forecasts:
                    model_name = fc_df['model'].iloc[0]
                    color = colors_map.get(model_name, '#3b82f6')
                    
                    fc_recent = fc_df.tail(24)
                    
                    ax3.plot(fc_recent['date'], fc_recent['forecast'], 
                            label=f'{model_name} Forecast',
                            linewidth=2.5, marker='s', markersize=5, alpha=0.7, color=color)
                
                # Actual
                actual_recent = monthly_df.tail(24)
                ax3.plot(actual_recent['date'], actual_recent['unemployment'],
                        label='Actual', linewidth=3.5, color='black', marker='o', markersize=6, zorder=10)
                
                ax3.set_xlabel('Date', fontsize=13, fontweight='bold')
                ax3.set_ylabel('Unemployment Rate (%)', fontsize=13, fontweight='bold')
                ax3.set_title('Model Forecasts vs Actual - Last 24 Months', 
                             fontsize=15, fontweight='bold', pad=20)
                ax3.legend(loc='best', fontsize=11, framealpha=0.9)
                ax3.grid(alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()
            
            # ============= DOWNLOAD SECTION =============
            
            st.markdown("---")
            st.header("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Summary (CSV)",
                    csv,
                    "model_comparison.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                if all_forecasts:
                    all_fc_df = pd.concat(all_forecasts, ignore_index=True)
                    fc_csv = all_fc_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Forecasts (CSV)",
                        fc_csv,
                        "all_forecasts.csv",
                        "text/csv",
                        use_container_width=True
                    )

# ============= AI CHATBOT =============

if enable_chat and uploaded_file is not None:
    st.markdown("---")
    st.header("💬 Ask AI Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about unemployment trends..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = chat_with_ai(prompt, st.session_state.messages)
            st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

# ============= FOOTER =============

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='font-size: 1.1rem;'>🎓 <strong>ISTAT Internship Project 2025</strong></p>
    <p>🤖 Powered by Claude AI | 📊 Italian Unemployment Nowcasting</p>
    <p style='font-size: 0.9rem; color: #999;'>Advanced ML Models • Real-time Analysis • AI Insights</p>
</div>
""", unsafe_allow_html=True)
