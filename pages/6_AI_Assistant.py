# pages/6_AI_Assistant.py — Multi‑provider AI Assistant (EN)
# ============================================================================
# Role
#   • Four parallel chat panels (A/B/C/D) with selectable providers: OpenAI, Claude, Gemini, Local (Ollama/HTTP)
#   • Project‑aware: can inject Nowcasting Lab context (target span, best models, top correlations) into system prompt
#   • Lightweight RAG: upload docs (txt/csv/md/json/pdf*) → TF‑IDF retrieval → augment prompt
#   • Safe fallbacks: missing keys/libs/providers handled gracefully
#   • Non‑streaming by design (Streamlit safe) with simple, fast UX
#
# *PDF support uses PyPDF2 if installed; otherwise skipped with notice.
# ============================================================================

from __future__ import annotations

import os
import io
import json
import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Optional deps (lazy):
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None  # type: ignore

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore

try:
    import requests  # for local HTTP / Ollama
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore

# ----------------------------------------------------------------------------
# State (robust import with fallback)
# ----------------------------------------------------------------------------
try:
    from utils.state import AppState  # type: ignore
except Exception:
    class _State:
        def __init__(self) -> None:
            self.y_monthly: pd.Series | None = None
            self.panel_monthly: pd.DataFrame | None = None
            self.bt_results: dict[str, pd.Series] = {}
            self.bt_metrics: pd.DataFrame | None = None
    class AppState:  # type: ignore
        @staticmethod
        def init() -> _State:
            if "_app" not in st.session_state:
                st.session_state["_app"] = _State()
            return st.session_state["_app"]  # type: ignore
        @staticmethod
        def get() -> _State:
            return AppState.init()

state = AppState.init()

# ----------------------------------------------------------------------------
# Context builder (project‑aware prompt enrichment)
# ----------------------------------------------------------------------------

def build_project_context(max_corr: int = 6) -> str:
    lines: list[str] = ["Project: ISTAT Unemployment Nowcasting Lab."]
    # target span
    if state.y_monthly is not None and not state.y_monthly.empty:
        ymin, ymax = state.y_monthly.index.min().date(), state.y_monthly.index.max().date()
        lines.append(f"Target coverage: {ymin} → {ymax} ({len(state.y_monthly)} months).")
        try:
            last = float(state.y_monthly.iloc[-1])
            lines.append(f"Latest target value: {last:.3f}.")
        except Exception:
            pass
    # best model from metrics
    if state.bt_metrics is not None and not state.bt_metrics.empty:
        row = state.bt_metrics.sort_values("MAE").iloc[0]
        lines.append(f"Best backtest model by MAE: {row['model']} (MAE={row['MAE']:.3f}, RMSE={row['RMSE']:.3f}).")
    # top correlations
    try:
        if state.panel_monthly is not None and not state.panel_monthly.empty and state.y_monthly is not None:
            y, X = state.y_monthly.align(state.panel_monthly.select_dtypes(include=[np.number]), join="inner")
            corr = X.corrwith(y).sort_values(ascending=False).dropna()
            top = corr.head(max_corr)
            if len(top) > 0:
                pairs = ", ".join([f"{k}:{v:.3f}" for k, v in top.items()])
                lines.append(f"Top correlations with target: {pairs}.")
    except Exception:
        pass
    return "\n".join(lines)

# ----------------------------------------------------------------------------
# RAG store (simple TF‑IDF)
# ----------------------------------------------------------------------------

@dataclass
class RagDoc:
    name: str
    text: str


def read_file_text(file) -> str:
    name = getattr(file, 'name', 'file')
    try:
        if name.lower().endswith((".txt", ".md")):
            return file.read().decode("utf-8", errors="ignore")
        if name.lower().endswith(".json"):
            obj = json.load(file)
            snippet = json.dumps(obj, indent=2)
            return snippet[:20000]
        if name.lower().endswith(".csv"):
            df = pd.read_csv(file)
            return df.head(200).to_csv(index=False)
        if name.lower().endswith(".pdf"):
            try:
                import PyPDF2  # type: ignore
                reader = PyPDF2.PdfReader(file)
                pages = []
                for i, p in enumerate(reader.pages[:40]):
                    pages.append(p.extract_text() or "")
                return "\n\n".join(pages)
            except Exception:
                return "(PDF parsing unavailable — install PyPDF2)"
        # fallback: binary -> empty
        return ""
    except Exception:
        return ""


def ensure_rag_index(key: str, docs: list[RagDoc]):
    if TfidfVectorizer is None:
        st.info("RAG disabled (scikit‑learn missing). Install scikit‑learn to enable.")
        return None
    texts = [d.text for d in docs]
    vec = TfidfVectorizer(max_features=25000, ngram_range=(1, 2))
    X = vec.fit_transform(texts) if texts else None
    st.session_state.setdefault("rag", {})
    st.session_state["rag"][key] = {"vectorizer": vec, "matrix": X, "docs": docs}
    return st.session_state["rag"][key]


def rag_search(key: str, query: str, k: int = 3) -> list[tuple[str, float]]:
    rag = st.session_state.get("rag", {}).get(key)
    if not rag or TfidfVectorizer is None or cosine_similarity is None:
        return []
    vec, X, docs = rag["vectorizer"], rag["matrix"], rag["docs"]
    if X is None or X.shape[0] == 0:
        return []
    qv = vec.transform([query])
    sim = cosine_similarity(qv, X).ravel()
    idx = np.argsort(sim)[::-1][:k]
    return [(docs[i].name, float(sim[i])) for i in idx]


def rag_snippets(key: str, query: str, k: int = 3) -> list[str]:
    rag = st.session_state.get("rag", {}).get(key)
    if not rag or TfidfVectorizer is None or cosine_similarity is None:
        return []
    vec, X, docs = rag["vectorizer"], rag["matrix"], rag["docs"]
    if X is None or X.shape[0] == 0:
        return []
    qv = vec.transform([query])
    sim = cosine_similarity(qv, X).ravel()
    idx = np.argsort(sim)[::-1][:k]
    out = []
    for i in idx:
        txt = docs[i].text
        out.append(txt[:1200])
    return out

# ----------------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------------

@dataclass
class ChatRequest:
    system: str
    user: str
    context: str | None
    model: str


def call_openai(req: ChatRequest) -> str:
    if openai is None:
        return "[OpenAI client not installed]"
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "[OPENAI_API_KEY missing in environment]"
    try:
        client = openai.OpenAI(api_key=key)  # type: ignore[attr-defined]
    except Exception:
        # older SDK style
        openai.api_key = key  # type: ignore[attr-defined]
        client = None
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    if req.context:
        messages.append({"role": "system", "content": f"Project context:\n{req.context}"})
    messages.append({"role": "user", "content": req.user})
    try:
        if client is not None:
            resp = client.chat.completions.create(model=req.model, messages=messages)  # type: ignore
            return resp.choices[0].message.content  # type: ignore
        else:
            # legacy
            resp = openai.ChatCompletion.create(model=req.model, messages=messages)  # type: ignore
            return resp["choices"][0]["message"]["content"]  # type: ignore
    except Exception as e:
        return f"[OpenAI error: {e}]"


def call_anthropic(req: ChatRequest) -> str:
    if anthropic is None:
        return "[anthropic package not installed]"
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return "[ANTHROPIC_API_KEY missing in environment]"
    try:
        client = anthropic.Anthropic(api_key=key)
        sys = req.system + ("\n" + req.context if req.context else "")
        msg = client.messages.create(
            model=req.model,
            max_tokens=1024,
            system=sys.strip(),
            messages=[{"role": "user", "content": req.user}],
        )
        parts = []
        for p in msg.content:
            parts.append(getattr(p, "text", ""))
        return "".join(parts)
    except Exception as e:
        return f"[Anthropic error: {e}]"


def call_gemini(req: ChatRequest) -> str:
    if genai is None:
        return "[google.generativeai not installed]"
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return "[GOOGLE_API_KEY missing in environment]"
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(req.model)
        sys = req.system + ("\n" + (req.context or ""))
        prompt = sys + "\n\nUser:\n" + req.user
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "")
    except Exception as e:
        return f"[Gemini error: {e}]"


def call_local(req: ChatRequest) -> str:
    """Call a local HTTP endpoint (e.g., Ollama /api/chat). Set OLLAMA_ENDPOINT env or default http://localhost:11434/api/chat"""
    if requests is None:
        return "[requests not installed]"
    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/chat")
    payload = {
        "model": req.model,
        "messages": [
            *( [{"role": "system", "content": req.system}] if req.system else [] ),
            *( [{"role": "system", "content": f"Project context:\n{req.context}"}] if req.context else [] ),
            {"role": "user", "content": req.user},
        ],
        "stream": False,
    }
    try:
        r = requests.post(endpoint, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Ollama returns {'message': {'content': '...'}} or a list of messages
        if isinstance(data, dict) and "message" in data:
            return data["message"].get("content", "")
        if isinstance(data, dict) and "messages" in data:
            return "\n".join([m.get("content", "") for m in data["messages"]])
        return str(data)[:4000]
    except Exception as e:
        return f"[Local endpoint error: {e}]"

# ----------------------------------------------------------------------------
# Chat UI utilities
# ----------------------------------------------------------------------------

PERSONAS = {
    "Research Analyst": "You are a rigorous labor‑market research analyst. Be concise, quantify uncertainty, cite formulas, and offer tests to validate claims.",
    "Data Scientist": "You are a pragmatic data scientist. Offer stepwise plans, edge cases, and code snippets in Python.",
    "Explainer": "You are a teacher. Prefer plain language and intuitive analogies. Avoid jargon.",
}

DEFAULT_SYSTEM = (
    "Follow these rules: 1) Be direct and precise. 2) Challenge assumptions and state risks. "
    "3) If uncertain, say 'uncertain' and propose a verification test. "
    "4) Prefer lists of 3–7 items. 5) For quantitative points, show steps."
)

PROVIDERS = ["OpenAI", "Claude", "Gemini", "Local"]
DEFAULT_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "Claude": "claude-3-5-sonnet-latest",
    "Gemini": "gemini-1.5-flash",
    "Local": "qwen2.5:7b",
}

# Keep per‑panel histories in session_state
for slot in ["A", "B", "C", "D"]:
    st.session_state.setdefault(f"chat_{slot}", [])  # list of dicts: {role, content}

# ----------------------------------------------------------------------------
# Layout: Controls bar + four chat panels
# ----------------------------------------------------------------------------

st.title("🤖 AI Assistant (Multi‑provider)")

with st.expander("Project context (auto‑injected)", expanded=True):
    ctx = build_project_context()
    st.code(ctx, language="markdown")

with st.expander("RAG — Upload documents (optional)"):
    rag_key = "rag_docs"
    up_files = st.file_uploader("Upload .txt/.md/.json/.csv/.pdf (up to ~40 pages)", type=["txt", "md", "json", "csv", "pdf"], accept_multiple_files=True)
    docs: list[RagDoc] = []
    if up_files:
        for f in up_files:
            txt = read_file_text(f)
            docs.append(RagDoc(name=getattr(f, 'name', 'doc'), text=txt))
        idx = ensure_rag_index(rag_key, docs)
        if docs:
            st.success(f"Indexed {len(docs)} document(s) for retrieval.")
    rag_topk = st.slider("# retrieved snippets", 0, 5, 3)

st.markdown("---")

# Shared controls for all panels
colc1, colc2, colc3, colc4 = st.columns([1.2, 1, 1, 1])
with colc1:
    persona = st.selectbox("Persona", list(PERSONAS.keys()), index=0)
with colc2:
    inject_ctx = st.checkbox("Inject project context", value=True)
with colc3:
    use_rag = st.checkbox("Use RAG snippets", value=False)
with colc4:
    temperature = st.slider("Creativity", 0.0, 1.0, 0.2, step=0.05)

system_preamble = DEFAULT_SYSTEM + "\n" + PERSONAS[persona]
ctx_text = ctx if inject_ctx else ""

# ----------------------------------------------------------------------------
# Render four chat panels
# ----------------------------------------------------------------------------

def chat_panel(slot: str):
    st.markdown(f"#### Panel {slot}")
    cols = st.columns([1, 1])
    with cols[0]:
        provider = st.selectbox(f"Provider {slot}", PROVIDERS, index=slot_index(slot))
    with cols[1]:
        model = st.text_input(f"Model {slot}", value=DEFAULT_MODELS.get(provider, ""))

    # history
    history_key = f"chat_{slot}"
    hist: list[dict] = st.session_state.get(history_key, [])

    # display history (compact bubbles)
    for msg in hist[-12:]:
        role = msg.get("role", "user")
        if role == "user":
            st.markdown(f"<div class='card' style='background:#EEF2FF;border-color:#E5E7EB'><b>User:</b> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='card'><b>{provider}:</b> {msg['content']}</div>", unsafe_allow_html=True)

    prompt = st.text_area(f"Your message ({slot})", height=120, key=f"prompt_{slot}")
    cbtn1, cbtn2, cbtn3 = st.columns([1, 1, 1])
    with cbtn1:
        send = st.button(f"Send {slot}", use_container_width=True)
    with cbtn2:
        clear = st.button(f"Clear {slot}", use_container_width=True)
    with cbtn3:
        paste_ctx = st.button(f"Paste context {slot}", use_container_width=True)

    if clear:
        st.session_state[history_key] = []
        st.experimental_rerun()

    if paste_ctx:
        st.session_state[f"prompt_{slot}"] = (prompt + "\n\n" + ctx_text).strip()
        st.experimental_rerun()

    if send and prompt.strip():
        # Build RAG text (optional)
        rag_text = ""
        if use_rag and rag_topk > 0:
            snippets = rag_snippets(rag_key, prompt, k=rag_topk)
            if snippets:
                rag_text = "\n\n".join([f"[Snippet {i+1}]\n{snip}" for i, snip in enumerate(snippets)])
        # Compose final system & user
        system = system_preamble
        context_block = (ctx_text + ("\n\nRAG snippets:\n" + rag_text if rag_text else "")) if ctx_text or rag_text else None
        req = ChatRequest(system=system, user=prompt, context=context_block, model=model)

        # Call provider
        if provider == "OpenAI":
            out = call_openai(req)
        elif provider == "Claude":
            out = call_anthropic(req)
        elif provider == "Gemini":
            out = call_gemini(req)
        else:
            out = call_local(req)

        # Update history
        hist.append({"role": "user", "content": prompt})
        hist.append({"role": "assistant", "content": out})
        st.session_state[history_key] = hist
        st.session_state[f"prompt_{slot}"] = ""
        st.experimental_rerun()


def slot_index(slot: str) -> int:
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    return mapping.get(slot, 0)

# Two rows × two panels
r1c1, r1c2 = st.columns(2)
with r1c1:
    chat_panel("A")
with r1c2:
    chat_panel("B")

r2c1, r2c2 = st.columns(2)
with r2c1:
    chat_panel("C")
with r2c2:
    chat_panel("D")

# ----------------------------------------------------------------------------
# Safety notes
# ----------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Set environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY. "
    "For Local, set OLLAMA_ENDPOINT (default http://localhost:11434/api/chat). "
    "PDF parsing requires PyPDF2; RAG indexing requires scikit‑learn."
)
