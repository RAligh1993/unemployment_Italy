"""
🤖 AI Assistant Pro v2.0
======================================
Multi-provider AI chatbot with RAG and project context.
Features: Floating chat widget, 4 providers, document retrieval, smart context injection.

Author: AI Assistant
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

# Optional dependencies
try:
    import openai
    HAS_OPENAI = True
except:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except:
    HAS_GEMINI = False

try:
    import requests
    HAS_REQUESTS = True
except:
    HAS_REQUESTS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

try:
    from utils.state import AppState
except Exception:
    class _State:
        def __init__(self):
            self.y_monthly: Optional[pd.Series] = None
            self.panel_monthly: Optional[pd.DataFrame] = None
            self.bt_results: Dict[str, pd.Series] = {}
            self.bt_metrics: Optional[pd.DataFrame] = None
    
    class AppState:
        @staticmethod
        def init():
            if "_app" not in st.session_state:
                st.session_state["_app"] = _State()
            return st.session_state["_app"]
        
        @staticmethod
        def get():
            return AppState.init()

state = AppState.init()

# Initialize chat histories
for panel in ['main', 'float']:
    if f'chat_{panel}' not in st.session_state:
        st.session_state[f'chat_{panel}'] = []

# Initialize RAG store
if 'rag_docs' not in st.session_state:
    st.session_state.rag_docs = []

if 'show_float_chat' not in st.session_state:
    st.session_state.show_float_chat = False

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with floating chat
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .chat-bubble {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
        transition: all 0.3s;
        z-index: 9999;
    }
    
    .chat-bubble:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 30px rgba(59, 130, 246, 0.6);
    }
    
    .chat-bubble-icon {
        color: white;
        font-size: 28px;
    }
    
    .float-chat-container {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 380px;
        height: 550px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        z-index: 9998;
        display: flex;
        flex-direction: column;
    }
    
    .message-user {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        max-width: 80%;
        align-self: flex-end;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .message-assistant {
        background: #F3F4F6;
        color: #1F2937;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .section-header {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .provider-card {
        background: white;
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    .provider-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .provider-card-selected {
        border-color: #3b82f6;
        background: linear-gradient(135deg, #EFF6FF, #F5F3FF);
    }
    
    .chat-container {
        background: white;
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.5rem;
        height: 600px;
        overflow-y: auto;
    }
    
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #FFF7ED;
        border-left: 4px solid #F97316;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONTEXT BUILDER
# =============================================================================

def build_project_context() -> str:
    """Build context from current project state"""
    lines = ["📊 ISTAT Unemployment Nowcasting Lab Context:"]
    
    # Target info
    if hasattr(state, 'y_monthly') and state.y_monthly is not None and not state.y_monthly.empty:
        y = state.y_monthly
        lines.append(f"\n🎯 Target Data:")
        lines.append(f"  - Range: {y.index.min().strftime('%Y-%m')} to {y.index.max().strftime('%Y-%m')}")
        lines.append(f"  - Observations: {len(y)}")
        lines.append(f"  - Latest value: {y.iloc[-1]:.3f}")
        lines.append(f"  - Mean: {y.mean():.3f}, Std: {y.std():.3f}")
    
    # Panel info
    if hasattr(state, 'panel_monthly') and state.panel_monthly is not None and not state.panel_monthly.empty:
        lines.append(f"\n📊 Feature Panel:")
        lines.append(f"  - Features: {state.panel_monthly.shape[1]}")
        lines.append(f"  - Months: {state.panel_monthly.shape[0]}")
    
    # Model performance
    if hasattr(state, 'bt_metrics') and state.bt_metrics is not None and not state.bt_metrics.empty:
        best = state.bt_metrics.nsmallest(1, 'MAE').iloc[0]
        lines.append(f"\n🏆 Best Model:")
        lines.append(f"  - Name: {best['model']}")
        lines.append(f"  - MAE: {best['MAE']:.4f}")
        lines.append(f"  - RMSE: {best['RMSE']:.4f}")
    
    # Top correlations
    try:
        if (hasattr(state, 'y_monthly') and state.y_monthly is not None and 
            hasattr(state, 'panel_monthly') and state.panel_monthly is not None):
            y_al, X_al = state.y_monthly.align(
                state.panel_monthly.select_dtypes(include=[np.number]), 
                join='inner'
            )
            if not X_al.empty:
                corr = X_al.corrwith(y_al).abs().sort_values(ascending=False).head(5)
                lines.append(f"\n🔗 Top Correlations:")
                for feat, val in corr.items():
                    lines.append(f"  - {feat}: {val:.3f}")
    except:
        pass
    
    return "\n".join(lines)

# =============================================================================
# RAG FUNCTIONS
# =============================================================================

@dataclass
class RagDocument:
    name: str
    content: str
    timestamp: datetime

def read_uploaded_file(file) -> str:
    """Read uploaded file content"""
    try:
        file_type = file.name.split('.')[-1].lower()
        
        if file_type in ['txt', 'md']:
            return file.read().decode('utf-8', errors='ignore')
        
        elif file_type == 'csv':
            df = pd.read_csv(file)
            return df.to_string()
        
        elif file_type == 'json':
            data = json.load(file)
            return json.dumps(data, indent=2)
        
        elif file_type == 'pdf':
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages[:20]:  # Limit to 20 pages
                    text.append(page.extract_text())
                return '\n\n'.join(text)
            except:
                return "[PDF parsing not available - install PyPDF2]"
        
        else:
            return "[Unsupported file type]"
    
    except Exception as e:
        return f"[Error reading file: {str(e)}]"

def build_rag_index(documents: List[RagDocument]):
    """Build TF-IDF index for RAG"""
    if not HAS_SKLEARN or not documents:
        return None
    
    texts = [doc.content for doc in documents]
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    return {
        'vectorizer': vectorizer,
        'matrix': tfidf_matrix,
        'documents': documents
    }

def retrieve_relevant_docs(query: str, rag_index: Optional[Dict], top_k: int = 3) -> List[Tuple[str, float, str]]:
    """Retrieve relevant documents for query"""
    if not rag_index or not HAS_SKLEARN:
        return []
    
    try:
        vectorizer = rag_index['vectorizer']
        tfidf_matrix = rag_index['matrix']
        documents = rag_index['documents']
        
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = documents[idx]
            score = similarities[idx]
            snippet = doc.content[:500] + '...' if len(doc.content) > 500 else doc.content
            results.append((doc.name, score, snippet))
        
        return results
    
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

# =============================================================================
# LLM PROVIDERS
# =============================================================================

def call_openai(messages: List[Dict], model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    """Call OpenAI API"""
    if not HAS_OPENAI:
        return "❌ OpenAI library not installed. Install: pip install openai"
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "❌ OPENAI_API_KEY not set in environment"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1500
        )
        return response.choices[0].message.content
    
    except Exception as e:
        return f"❌ OpenAI Error: {str(e)}"

def call_anthropic(messages: List[Dict], model: str = "claude-3-5-sonnet-latest", temperature: float = 0.7) -> str:
    """Call Anthropic Claude API"""
    if not HAS_ANTHROPIC:
        return "❌ Anthropic library not installed. Install: pip install anthropic"
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return "❌ ANTHROPIC_API_KEY not set in environment"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Extract system message
        system_msg = ""
        user_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_msg += msg['content'] + "\n"
            else:
                user_messages.append(msg)
        
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=temperature,
            system=system_msg.strip(),
            messages=user_messages
        )
        
        return response.content[0].text
    
    except Exception as e:
        return f"❌ Anthropic Error: {str(e)}"

def call_gemini(messages: List[Dict], model: str = "gemini-1.5-flash", temperature: float = 0.7) -> str:
    """Call Google Gemini API"""
    if not HAS_GEMINI:
        return "❌ Google GenAI library not installed. Install: pip install google-generativeai"
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return "❌ GOOGLE_API_KEY not set in environment"
    
    try:
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        
        # Combine all messages
        prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        response = model_obj.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

def call_local(messages: List[Dict], model: str = "llama2", endpoint: str = None) -> str:
    """Call local Ollama endpoint"""
    if not HAS_REQUESTS:
        return "❌ Requests library not installed. Install: pip install requests"
    
    if endpoint is None:
        endpoint = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434/api/chat')
    
    try:
        payload = {
            'model': model,
            'messages': messages,
            'stream': False
        }
        
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return data.get('message', {}).get('content', str(data))
    
    except Exception as e:
        return f"❌ Local Model Error: {str(e)}"

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Header
st.markdown('<h1 class="main-title">🤖 AI Assistant Pro</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #64748B; font-size: 1.1rem;">Multi-provider chatbot with RAG and project context</p>', unsafe_allow_html=True)

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Provider selection
    st.subheader("🤖 AI Provider")
    
    providers = {
        'OpenAI': ('gpt-4o-mini', HAS_OPENAI),
        'Claude': ('claude-3-5-sonnet-latest', HAS_ANTHROPIC),
        'Gemini': ('gemini-1.5-flash', HAS_GEMINI),
        'Local/Ollama': ('llama2', HAS_REQUESTS)
    }
    
    provider = st.selectbox(
        "Select provider:",
        options=list(providers.keys())
    )
    
    default_model, is_available = providers[provider]
    
    if not is_available:
        st.warning(f"⚠️ {provider} not available. Install required package.")
    
    model = st.text_input("Model name:", value=default_model, key='model_name')
    
    # Provider-specific settings
    if provider == 'Local/Ollama':
        st.markdown("##### 🔗 Local Endpoint")
        local_endpoint = st.text_input(
            "API Endpoint:",
            value=os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434/api/chat'),
            help="Default Ollama endpoint or custom local API",
            key='local_endpoint'
        )
        st.caption("💡 Start Ollama: `ollama serve`")
    else:
        local_endpoint = None
    
    # Settings
    st.markdown("---")
    st.subheader("🎛️ Settings")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    inject_context = st.checkbox(
        "Include project context",
        value=True,
        help="Inject current project state into system message"
    )
    
    use_rag = st.checkbox(
        "Use RAG retrieval",
        value=False,
        help="Retrieve relevant documents before answering"
    )
    
    if use_rag:
        rag_top_k = st.slider("Documents to retrieve", 1, 5, 3)
    else:
        rag_top_k = 0
    
    st.markdown("---")
    
    # API Keys Management
    st.subheader("🔑 API Keys")
    
    with st.expander("Configure API Keys", expanded=False):
        st.markdown("**Current Status:**")
        
        api_keys = {
            'OpenAI': 'OPENAI_API_KEY',
            'Anthropic': 'ANTHROPIC_API_KEY',
            'Google': 'GOOGLE_API_KEY'
        }
        
        for name, key in api_keys.items():
            has_key = bool(os.getenv(key))
            status = "✅" if has_key else "❌"
            st.markdown(f"{status} **{name}**")
        
        st.markdown("---")
        st.markdown("**Set via environment variables:**")
        st.code("""
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export OLLAMA_ENDPOINT="http://localhost:11434/api/chat"
        """, language='bash')
        
        st.caption("⚠️ Restart app after setting keys")
    
    # Quick status
    st.markdown("**Quick Status:**")
    for name, key in api_keys.items():
        status = "✅" if os.getenv(key) else "❌"
        st.markdown(f"{status} {name}")

# =============================================================================
# RAG DOCUMENT UPLOAD
# =============================================================================

st.markdown('<div class="section-header">📚 Document Library (RAG)</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload documents for context:",
        type=['txt', 'md', 'csv', 'json', 'pdf'],
        accept_multiple_files=True,
        key='rag_upload'
    )

with col2:
    if st.button("🗑️ Clear All Docs", use_container_width=True):
        st.session_state.rag_docs = []
        st.session_state.rag_index = None
        st.success("Cleared!")
        st.rerun()

if uploaded_files:
    new_docs = []
    for file in uploaded_files:
        content = read_uploaded_file(file)
        new_docs.append(RagDocument(
            name=file.name,
            content=content,
            timestamp=datetime.now()
        ))
    
    st.session_state.rag_docs = new_docs
    st.session_state.rag_index = build_rag_index(new_docs)
    
    st.success(f"✅ Indexed {len(new_docs)} documents")

# Display current documents
if st.session_state.rag_docs:
    with st.expander(f"📄 Current Documents ({len(st.session_state.rag_docs)})", expanded=False):
        for doc in st.session_state.rag_docs:
            st.markdown(f"**{doc.name}** - {len(doc.content)} characters")

# =============================================================================
# PROJECT CONTEXT
# =============================================================================

st.markdown('<div class="section-header">📊 Project Context</div>', unsafe_allow_html=True)

with st.expander("View current project context", expanded=False):
    context = build_project_context()
    st.code(context, language='markdown')

# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

st.markdown('<div class="section-header">💬 Chat Interface</div>', unsafe_allow_html=True)

# Chat history display
chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_main:
        if msg['role'] == 'user':
            st.markdown(f'<div class="message-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

# Input area
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_area(
        "Your message:",
        height=100,
        placeholder="Ask me anything about your nowcasting project...",
        key='main_input'
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    
    send_button = st.button("📤 Send", use_container_width=True, type="primary")
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.session_state.chat_main = []
    st.rerun()

if send_button and user_input.strip():
    with st.spinner("🤔 Thinking..."):
        # Build messages
        messages = []
        
        # System message
        system_content = "You are a helpful AI assistant specialized in time series forecasting and econometrics."
        
        if inject_context:
            system_content += "\n\n" + build_project_context()
        
        # RAG retrieval
        if use_rag and st.session_state.get('rag_index'):
            relevant_docs = retrieve_relevant_docs(
                user_input,
                st.session_state.rag_index,
                rag_top_k
            )
            
            if relevant_docs:
                rag_context = "\n\nRelevant documents:\n"
                for doc_name, score, snippet in relevant_docs:
                    rag_context += f"\n[{doc_name}] (relevance: {score:.3f})\n{snippet}\n"
                
                system_content += rag_context
        
        messages.append({'role': 'system', 'content': system_content})
        
        # Add chat history
        for msg in st.session_state.chat_main[-10:]:  # Last 10 messages for context
            messages.append(msg)
        
        # Add current message
        messages.append({'role': 'user', 'content': user_input})
        
        # Call selected provider
        if provider == 'OpenAI':
            response = call_openai(messages, model, temperature)
        elif provider == 'Claude':
            response = call_anthropic(messages, model, temperature)
        elif provider == 'Gemini':
            response = call_gemini(messages, model, temperature)
        else:  # Local
            response = call_local(messages, model)
        
        # Update history
        st.session_state.chat_main.append({'role': 'user', 'content': user_input})
        st.session_state.chat_main.append({'role': 'assistant', 'content': response})
        
        st.rerun()

# =============================================================================
# FLOATING CHAT BUBBLE
# =============================================================================

# Floating chat bubble HTML
st.markdown("""
<div class="chat-bubble" onclick="toggleFloatChat()">
    <span class="chat-bubble-icon">💬</span>
</div>

<script>
function toggleFloatChat() {
    // This would toggle the floating chat
    // In Streamlit, we handle this via session state
    window.parent.postMessage({type: 'streamlit:toggleChat'}, '*');
}
</script>
""", unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"Provider: {provider}")

with col2:
    st.caption(f"Messages: {len(st.session_state.chat_main)}")

with col3:
    st.caption(f"Documents: {len(st.session_state.rag_docs)}")

st.markdown("""
<div style='text-align: center; color: #94A3B8; font-size: 0.875rem; margin-top: 1rem;'>
    💡 <b>Tip:</b> Set API keys as environment variables for full functionality<br/>
    Supports: OpenAI, Anthropic Claude, Google Gemini, Local Ollama
</div>
""", unsafe_allow_html=True)
