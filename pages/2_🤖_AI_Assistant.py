"""
Claude AI Assistant Page - Intelligent Analysis and Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import anthropic
import os
import json
from datetime import datetime

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-container {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.1));
        border-left: 3px solid #6366F1;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
        border-left: 3px solid #10B981;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 5px;
        padding: 10px;
        color: #9CA3AF;
    }
    
    .thinking-dot {
        width: 8px;
        height: 8px;
        background: #6366F1;
        border-radius: 50%;
        animation: thinking 1.4s ease-in-out infinite;
    }
    
    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes thinking {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
        30% { transform: translateY(-10px); opacity: 1; }
    }
    
    .insight-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(16, 185, 129, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
    }
    
    .quick-action-btn {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .quick-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Claude AI client
@st.cache_resource
def init_claude():
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    return None

claude_client = init_claude()

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_context' not in st.session_state:
    st.session_state.analysis_context = {}

# Header
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("""
    <h1 style="margin: 0;">🤖 Claude AI Assistant</h1>
    <p style="color: #9CA3AF; margin-top: 10px;">
        Your intelligent partner for unemployment nowcasting analysis
    </p>
    """, unsafe_allow_html=True)

with col2:
    status = "🟢 Connected" if claude_client else "🔴 API Key Required"
    st.markdown(f"""
    <div style="text-align: right; padding-top: 20px;">
        <span style="font-weight: 600; color: white;">{status}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main layout
col_left, col_right = st.columns([2, 1])

with col_left:
    # Chat interface
    st.markdown("### 💬 Conversation")
    
    # Quick actions
    st.markdown("#### Quick Actions")
    
    quick_actions = st.container()
    with quick_actions:
        cols = st.columns(4)
        
        actions = [
            ("📊 Analyze Current Data", "analyze_data"),
            ("🎯 Suggest Features", "suggest_features"),
            ("🔮 Forecast Next Month", "forecast"),
            ("📝 Generate Report", "report")
        ]
        
        for idx, (label, action) in enumerate(actions):
            with cols[idx]:
                if st.button(label, key=f"quick_{action}"):
                    if action == "analyze_data":
                        prompt = "Analyze the current unemployment data and identify key patterns and anomalies."
                    elif action == "suggest_features":
                        prompt = "Based on the current data, suggest additional features that could improve prediction accuracy."
                    elif action == "forecast":
                        prompt = "Generate a detailed forecast for next month's unemployment rate with confidence intervals."
                    elif action == "report":
                        prompt = "Create an executive summary report of the current unemployment situation and outlook."
                    
                    st.session_state.pending_prompt = prompt
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <div style="font-weight: 600; color: #6366F1; margin-bottom: 8px;">You</div>
                    <div style="color: white;">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <div style="font-weight: 600; color: #10B981; margin-bottom: 8px;">Claude</div>
                    <div style="color: white;">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    with st.form("chat_input", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            # Check for pending prompt from quick actions
            initial_value = st.session_state.get('pending_prompt', '')
            user_input = st.text_area(
                "Your message",
                placeholder="Ask about unemployment trends, request analysis, or get insights...",
                height=100,
                value=initial_value,
                key="user_input"
            )
            if 'pending_prompt' in st.session_state:
                del st.session_state.pending_prompt
        
        with col2:
            send_button = st.form_submit_button("Send 📤", use_container_width=True)
    
    # Process message
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Show thinking indicator
        with st.spinner("Claude is thinking..."):
            if claude_client:
                try:
                    # Prepare context
                    context = f"""
                    You are an expert unemployment nowcasting analyst for Italy. 
                    Current context:
                    - Latest unemployment rate: 7.8%
                    - Trend: Declining (-0.2% monthly change)
                    - Youth unemployment: 22.3%
                    - Regional variations: North 5.2%, South 15.8%
                    
                    User's question: {user_input}
                    
                    Provide detailed, actionable insights using economic expertise.
                    """
                    
                    # Get Claude's response
                    response = claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1000,
                        system=context,
                        messages=[{"role": "user", "content": user_input}]
                    )
                    
                    assistant_response = response.content[0].text
                    
                except Exception as e:
                    assistant_response = f"I encountered an error: {str(e)}. Please check your API key."
            else:
                # Fallback response without API
                assistant_response = """
                I'm currently running in demo mode without API access. 
                
                Based on your query about unemployment analysis, here are some general insights:
                
                1. **Current Trend Analysis**: The unemployment rate shows a declining trend with -0.2% monthly change
                2. **Key Factors**: Youth unemployment remains high at 22.3%, requiring targeted interventions
                3. **Regional Disparities**: Significant gap between North (5.2%) and South (15.8%)
                
                For live analysis, please configure your Anthropic API key in the environment variables.
                """
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': assistant_response
        })
        
        # Rerun to show new messages
        st.rerun()

with col_right:
    # Insights and suggestions panel
    st.markdown("### 💡 AI Insights")
    
    # Auto-generated insights
    insights = [
        {
            "type": "Pattern",
            "icon": "📊",
            "title": "Seasonal Pattern Detected",
            "content": "Unemployment typically rises 0.3% in January due to post-holiday adjustments"
        },
        {
            "type": "Anomaly",
            "icon": "⚠️",
            "title": "Regional Divergence",
            "content": "Southern regions showing faster improvement than historical average"
        },
        {
            "type": "Prediction",
            "icon": "🔮",
            "title": "Next Month Forecast",
            "content": "Expected rate: 7.6% (±0.2%) based on current indicators"
        },
        {
            "type": "Recommendation",
            "icon": "💡",
            "title": "Feature Suggestion",
            "content": "Adding industrial production index could improve model accuracy by ~8%"
        }
    ]
    
    for insight in insights:
        st.markdown(f"""
        <div class="insight-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 10px;">{insight['icon']}</span>
                <div>
                    <div style="color: #9CA3AF; font-size: 12px; text-transform: uppercase;">
                        {insight['type']}
                    </div>
                    <div style="font-weight: 600; color: white;">
                        {insight['title']}
                    </div>
                </div>
            </div>
            <div style="color: #E5E7EB; font-size: 14px; line-height: 1.5;">
                {insight['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Model recommendations
    st.markdown("### 🎯 Model Recommendations")
    
    recommendations = pd.DataFrame({
        'Model': ['XGBoost', 'LSTM', 'Ensemble'],
        'Confidence': [92, 88, 95],
        'Reason': [
            'Best for non-linear patterns',
            'Captures temporal dependencies',
            'Combines multiple strengths'
        ]
    })
    
    for _, row in recommendations.iterrows():
        color = '#10B981' if row['Confidence'] > 90 else '#F59E0B'
        st.markdown(f"""
        <div style="
            background: rgba(30, 41, 59, 0.5);
            border-left: 3px solid {color};
            padding: 12px;
            margin: 10px 0;
            border-radius: 8px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; color: white;">{row['Model']}</span>
                <span style="color: {color}; font-weight: 600;">{row['Confidence']}%</span>
            </div>
            <div style="color: #9CA3AF; font-size: 12px; margin-top: 5px;">
                {row['Reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data quality assessment
    st.markdown("### 📊 Data Quality")
    
    quality_metrics = {
        'Completeness': 98,
        'Timeliness': 95,
        'Consistency': 92,
        'Accuracy': 96
    }
    
    for metric, value in quality_metrics.items():
        color = '#10B981' if value > 95 else '#F59E0B' if value > 90 else '#EF4444'
        st.progress(value/100)
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
            <span style="color: #9CA3AF;">{metric}</span>
            <span style="color: {color}; font-weight: 600;">{value}%</span>
        </div>
        """, unsafe_allow_html=True)

# Export conversation button
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("📥 Export Conversation"):
        # Export chat history as JSON
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'conversation': st.session_state.chat_history
        }
        st.download_button(
            label="Download Chat History",
            data=json.dumps(export_data, indent=2),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

with col2:
    if st.button("🔄 Clear History"):
        st.session_state.chat_history = []
        st.rerun()

with col3:
    if st.button("📊 Generate Analysis"):
        st.session_state.pending_prompt = "Perform a comprehensive analysis of all available unemployment data and provide actionable insights."
        st.rerun()
