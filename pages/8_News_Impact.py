"""
📰 News Impact Analyzer Pro v4.0
===================================
Clean, robust RSS-based news analysis WITHOUT any API keys.
Analyzes Italian, European and International economic news impact on unemployment.

Author: AI Assistant
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import hashlib

# Optional dependencies with graceful fallback
try:
    import feedparser
    HAS_FEEDPARSER = True
except:
    HAS_FEEDPARSER = False
    st.error("⚠️ feedparser not installed. Run: pip install feedparser")
    st.stop()

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except:
    HAS_VADER = False

# =============================================================================
# CONFIGURATION
# =============================================================================

RSS_SOURCES = {
    "🇮🇹 ITALY": {
        "ANSA Economy": "https://www.ansa.it/sito/notizie/economia/economia_rss.xml",
        "Il Sole 24 Ore": "https://www.ilsole24ore.com/rss/economia--lavoro.xml",
        "La Repubblica": "https://www.repubblica.it/rss/economia/rss2.0.xml",
        "Corriere Economia": "https://xml2.corriereobjects.it/rss/economia.xml",
    },
    "🇪🇺 EUROPE": {
        "Euronews Business": "https://www.euronews.com/rss?level=theme&name=news&theme=business",
        "Financial Times": "https://www.ft.com/rss/home/europe",
        "DW Business": "https://rss.dw.com/rdf/rss-en-bus",
    },
    "🌍 INTERNATIONAL": {
        "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "Reuters Business": "https://www.reuters.com/business",
        "AP Business": "https://apnews.com/apf-business",
        "Bloomberg": "https://www.bloomberg.com/feed/podcast/bloomberg-businessweek.xml",
    }
}

# Economic keywords with sentiment scores
KEYWORDS = {
    "negative": {
        "disoccupazione": -0.9, "licenziamenti": -1.0, "cassa integrazione": -0.8,
        "crisi": -0.7, "recessione": -0.9, "fallimento": -0.8,
        "unemployment": -0.9, "layoffs": -1.0, "jobless": -0.8,
        "recession": -0.9, "crisis": -0.7, "bankruptcy": -0.8,
    },
    "positive": {
        "assunzioni": 0.9, "occupazione": 0.7, "posti di lavoro": 0.8,
        "crescita": 0.6, "ripresa": 0.7, "sviluppo": 0.6,
        "hiring": 0.9, "employment": 0.7, "job growth": 0.9,
        "recovery": 0.7, "expansion": 0.6, "growth": 0.6,
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if 'news_data' not in st.session_state:
        st.session_state.news_data = pd.DataFrame()
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'vader_analyzer' not in st.session_state and HAS_VADER:
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        st.session_state.vader_analyzer = SentimentIntensityAnalyzer()

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_rss_feed(url: str, source_name: str, region: str) -> pd.DataFrame:
    """Fetch and parse RSS feed with robust error handling"""
    articles = []
    
    try:
        feed = feedparser.parse(url)
        
        if not feed.entries:
            return pd.DataFrame()
        
        for entry in feed.entries:
            # Extract date with multiple fallbacks
            pub_date = None
            for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
                if hasattr(entry, date_field) and getattr(entry, date_field):
                    try:
                        pub_date = datetime(*getattr(entry, date_field)[:6])
                        break
                    except:
                        continue
            
            if not pub_date:
                pub_date = datetime.now()
            
            # Extract content
            title = entry.get('title', '').strip()
            summary = entry.get('summary', entry.get('description', '')).strip()
            link = entry.get('link', '')
            
            if title:  # Only add if we have a title
                articles.append({
                    'date': pub_date,
                    'title': title,
                    'content': summary,
                    'url': link,
                    'source': source_name,
                    'region': region,
                })
        
        return pd.DataFrame(articles)
    
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch {source_name}: {str(e)[:50]}")
        return pd.DataFrame()

def calculate_keyword_score(text: str) -> float:
    """Calculate sentiment score based on economic keywords"""
    if not text:
        return 0.0
    
    text_lower = text.lower()
    score = 0.0
    
    for word, weight in KEYWORDS['negative'].items():
        if word in text_lower:
            score += weight
    
    for word, weight in KEYWORDS['positive'].items():
        if word in text_lower:
            score += weight
    
    # Normalize to [-1, 1]
    return max(-1.0, min(1.0, score / 2))

def calculate_vader_score(text: str) -> float:
    """Calculate VADER sentiment score"""
    if not HAS_VADER or 'vader_analyzer' not in st.session_state:
        return 0.0
    
    try:
        analyzer = st.session_state.vader_analyzer
        scores = analyzer.polarity_scores(text)
        return scores['compound']
    except:
        return 0.0

def score_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment scores to articles"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Combine title and content for analysis
    df['full_text'] = df['title'] + ' ' + df['content']
    
    # Calculate scores
    df['keyword_score'] = df['full_text'].apply(calculate_keyword_score)
    
    if HAS_VADER:
        df['vader_score'] = df['full_text'].apply(calculate_vader_score)
    else:
        df['vader_score'] = 0.0
    
    # Combined score
    df['combined_score'] = (df['keyword_score'] + df['vader_score']) / 2
    
    # Clean up
    df = df.drop('full_text', axis=1)
    
    return df

def create_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate articles to daily metrics by region"""
    if df.empty:
        return pd.DataFrame()
    
    df['date_only'] = df['date'].dt.date
    
    daily = df.groupby(['date_only', 'region']).agg({
        'title': 'count',
        'keyword_score': 'mean',
        'vader_score': 'mean',
        'combined_score': 'mean',
    }).reset_index()
    
    daily.columns = ['date', 'region', 'article_count', 'keyword_score', 'vader_score', 'combined_score']
    daily['date'] = pd.to_datetime(daily['date'])
    
    return daily

def create_monthly_metrics(daily_df: pd.DataFrame, smooth_window: int = 3) -> pd.DataFrame:
    """Create monthly aggregated metrics with optional smoothing"""
    if daily_df.empty:
        return pd.DataFrame()
    
    monthly_dfs = []
    
    for region in daily_df['region'].unique():
        region_data = daily_df[daily_df['region'] == region].set_index('date')
        
        monthly = pd.DataFrame({
            f'{region}_count': region_data['article_count'].resample('M').sum(),
            f'{region}_keyword': region_data['keyword_score'].resample('M').mean(),
            f'{region}_vader': region_data['vader_score'].resample('M').mean(),
            f'{region}_combined': region_data['combined_score'].resample('M').mean(),
        })
        
        # Apply smoothing
        if smooth_window > 1:
            for col in monthly.columns:
                monthly[f'{col}_ma{smooth_window}'] = monthly[col].rolling(
                    smooth_window, min_periods=1
                ).mean()
        
        monthly_dfs.append(monthly)
    
    return pd.concat(monthly_dfs, axis=1)

# =============================================================================
# UI CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="News Impact Analyzer Pro",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(120deg, #2c3e50, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MAIN APP
# =============================================================================

init_session_state()

# Header
st.markdown('<h1 class="main-header">📰 News Impact Analyzer Pro</h1>', unsafe_allow_html=True)
st.caption("🚀 **RSS-Based Economic News Analysis** • No API Keys Required • Real-time Updates")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Date range
    st.subheader("📅 Date Range")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)
    
    date_range = st.date_input(
        "Select period:",
        value=(start_date, end_date),
        max_value=end_date
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range if not isinstance(date_range, tuple) else date_range[0]
    
    # Region selection
    st.subheader("🌍 Regions")
    selected_regions = st.multiselect(
        "Select regions to analyze:",
        options=list(RSS_SOURCES.keys()),
        default=list(RSS_SOURCES.keys())
    )
    
    # Analysis options
    st.subheader("📊 Analysis Options")
    smooth_window = st.slider("Smoothing window (months)", 1, 12, 3)
    
    use_vader = st.checkbox(
        "Use VADER sentiment", 
        value=HAS_VADER,
        disabled=not HAS_VADER,
        help="Advanced sentiment analysis" if HAS_VADER else "Install NLTK to enable"
    )
    
    # Cache control
    st.subheader("🔄 Data Management")
    cache_duration = st.select_slider(
        "Cache duration (minutes)",
        options=[5, 15, 30, 60, 120],
        value=30
    )
    
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.news_data = pd.DataFrame()
        st.success("Cache cleared!")
        st.rerun()

# Main content
st.markdown("---")

# Fetch button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fetch_button = st.button("🚀 Fetch & Analyze News", use_container_width=True, type="primary")

if fetch_button and selected_regions:
    all_articles = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_sources = sum(len(RSS_SOURCES[region]) for region in selected_regions)
    current = 0
    
    # Fetch from all selected sources
    for region in selected_regions:
        for source_name, url in RSS_SOURCES[region].items():
            current += 1
            status_text.text(f"📡 Fetching {source_name}... ({current}/{total_sources})")
            
            df = fetch_rss_feed(url, source_name, region)
            
            if not df.empty:
                # Filter by date range
                df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
                all_articles.append(df)
            
            progress_bar.progress(current / total_sources)
    
    progress_bar.empty()
    status_text.empty()
    
    # Combine and process
    if all_articles:
        combined_df = pd.concat(all_articles, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['title', 'date'], keep='first')
        
        # Score articles
        with st.spinner("🧠 Analyzing sentiment..."):
            combined_df = score_articles(combined_df)
        
        # Save to session state
        st.session_state.news_data = combined_df.sort_values('date', ascending=False)
        st.session_state.last_update = datetime.now()
        
        st.success(f"✅ Fetched {len(combined_df):,} articles from {len(all_articles)} sources!")
    else:
        st.warning("⚠️ No articles found for the selected period. Try a wider date range.")

# Display results
if not st.session_state.news_data.empty:
    df = st.session_state.news_data
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📰 Total Articles", f"{len(df):,}")
    
    with col2:
        avg_sentiment = df['combined_score'].mean()
        st.metric(
            "📈 Avg Sentiment", 
            f"{avg_sentiment:.3f}",
            delta=f"{'Positive' if avg_sentiment > 0 else 'Negative'}"
        )
    
    with col3:
        days_coverage = (df['date'].max() - df['date'].min()).days
        st.metric("📅 Days Coverage", f"{days_coverage}")
    
    with col4:
        regions_count = df['region'].nunique()
        st.metric("🌍 Regions", f"{regions_count}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📰 Latest News", "📊 Daily Trends", "📈 Monthly Analysis", "💾 Export"])
    
    with tab1:
        st.subheader("Latest 50 Articles")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            filter_region = st.multiselect(
                "Filter by region:",
                options=df['region'].unique(),
                default=df['region'].unique()
            )
        with col2:
            filter_source = st.multiselect(
                "Filter by source:",
                options=df['source'].unique(),
                default=[]
            )
        
        filtered_df = df[df['region'].isin(filter_region)]
        if filter_source:
            filtered_df = filtered_df[filtered_df['source'].isin(filter_source)]
        
        # Display table
        display_df = filtered_df[['date', 'title', 'source', 'region', 'combined_score']].head(50)
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['combined_score'] = display_df['combined_score'].round(3)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "combined_score": st.column_config.ProgressColumn(
                    "Sentiment",
                    min_value=-1,
                    max_value=1,
                    format="%.3f"
                )
            }
        )
    
    with tab2:
        st.subheader("Daily Trends by Region")
        
        daily_metrics = create_daily_metrics(df)
        
        if not daily_metrics.empty:
            # Article count over time
            fig1 = px.bar(
                daily_metrics,
                x='date',
                y='article_count',
                color='region',
                title="📰 Daily Article Count by Region",
                labels={'article_count': 'Number of Articles', 'date': 'Date'}
            )
            fig1.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Sentiment over time
            fig2 = go.Figure()
            
            for region in daily_metrics['region'].unique():
                region_data = daily_metrics[daily_metrics['region'] == region]
                fig2.add_trace(go.Scatter(
                    x=region_data['date'],
                    y=region_data['combined_score'],
                    mode='lines+markers',
                    name=region,
                    line=dict(width=2)
                ))
            
            fig2.update_layout(
                title="📈 Daily Sentiment Score by Region",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                template='plotly_white',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Monthly Aggregated Analysis")
        
        monthly_metrics = create_monthly_metrics(daily_metrics, smooth_window)
        
        if not monthly_metrics.empty:
            st.dataframe(
                monthly_metrics.round(3),
                use_container_width=True
            )
            
            # Monthly sentiment comparison
            fig3 = go.Figure()
            
            for col in monthly_metrics.columns:
                if '_combined' in col and '_ma' not in col:
                    region = col.split('_')[0]
                    fig3.add_trace(go.Scatter(
                        x=monthly_metrics.index,
                        y=monthly_metrics[col],
                        mode='lines+markers',
                        name=region,
                        line=dict(width=3)
                    ))
            
            fig3.update_layout(
                title="📊 Monthly Combined Sentiment Score",
                xaxis_title="Month",
                yaxis_title="Sentiment Score",
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab4:
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_articles = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Articles (CSV)",
                csv_articles,
                "news_articles.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            daily_csv = create_daily_metrics(df).to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Daily Metrics (CSV)",
                daily_csv,
                "daily_metrics.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col3:
            monthly_csv = create_monthly_metrics(create_daily_metrics(df)).to_csv().encode('utf-8')
            st.download_button(
                "📥 Download Monthly Metrics (CSV)",
                monthly_csv,
                "monthly_metrics.csv",
                "text/csv",
                use_container_width=True
            )
        
        st.info("💡 **Tip:** Use the exported data for further analysis in Excel, Python, or R!")

else:
    # Welcome screen
    st.info("👆 **Click 'Fetch & Analyze News' to start!**")
    
    st.markdown("### 🌟 Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔒 No API Keys**")
        st.write("Uses RSS feeds only - completely free and unlimited")
    
    with col2:
        st.markdown("**🌍 Multi-Region**")
        st.write("Analyzes Italian, European, and International news sources")
    
    with col3:
        st.markdown("**🧠 Smart Analysis**")
        st.write("Keyword-based + VADER sentiment analysis")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.last_update:
        st.caption(f"🕐 Last update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.caption("📖 **Sentiment Range:** -1 (Very Negative) to +1 (Very Positive)")

with col3:
    st.caption("💻 Built with Streamlit • 🚀 Powered by RSS")
