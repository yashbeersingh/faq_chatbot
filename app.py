import streamlit as st
import pandas as pd
import tempfile
import os
from core.faq_store import merge_faqs
from core.loader import load_faqs
from core.embedder import Embedder
from core.retriever import Retriever
from core.memory import ConversationMemory
from core.csv_validator import validate_faq_csv
from core.reindexer import rebuild_retriever

from analytics.dashboard import show_dashboard
from analytics.logger import log_event, init_analytics_file

from utils.config import FAQ_PATH, HIGH_THRESHOLD, MEDIUM_THRESHOLD

# =====================================================
# MUST be first
# =====================================================
st.set_page_config(page_title="Smart FAQ Chatbot", layout="centered")

# =====================================================
# Init analytics file (Phase 2)
# =====================================================
init_analytics_file()

# =====================================================
# Sidebar – Analytics
# =====================================================
with st.sidebar:
    if st.checkbox("📊 Analytics Dashboard"):
        show_dashboard()
        st.stop()

# =====================================================
# Sidebar – Admin Controls (Phase 3)
# =====================================================
with st.sidebar:
    st.header("🔐 Admin Controls")

    uploaded_file = st.file_uploader(
        "Upload FAQ CSV",
        type=["csv"],
        help="Upload updated FAQ file"
    )

    if uploaded_file:
        temp_path = os.path.join(tempfile.gettempdir(), "uploaded_faqs.csv")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df_new = pd.read_csv(temp_path)
        errors = validate_faq_csv(df_new)

        if errors:
            st.error("CSV validation failed:")
            for e in errors:
                st.write(f"- {e}")
        else:
            st.success("CSV validation successful")

            if st.button("🚀 Reindex FAQs"):
                with st.spinner("Merging & reindexing FAQs..."):
            # 1. Merge into master FAQ store
                    merge_faqs(df_new)

                    # 2. Clear cached retriever so it reloads from FAQ_PATH
                    st.cache_resource.clear()
        # 3. Reset chat
                    st.session_state.chat = []

                st.success("FAQs merged and reindexed successfully")
                st.rerun()
            

# =====================================================
# Setup (cached resources)
# =====================================================
@st.cache_resource
def setup():
    df = load_faqs(FAQ_PATH)
    embedder = Embedder()
    retriever = Retriever(df, embedder)
    memory = ConversationMemory()
    return df, retriever, memory

df, retriever, memory = setup()

# =====================================================
# Session State Initialization
# =====================================================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "reset_input" not in st.session_state:
    st.session_state.reset_input = False

# =====================================================
# Helper
# =====================================================
def submit_query(query: str):
    st.session_state.pending_query = query
    st.session_state.reset_input = True

# =====================================================
# UI Header
# =====================================================
st.title("💬 Smart FAQ Chatbot")

# =====================================================
# Render Chat History
# =====================================================
for i, msg in enumerate(st.session_state.chat):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        for j, opt in enumerate(msg.get("options", [])):
            key = f"opt_{i}_{j}_{hash(opt)}"
            if st.button(opt, key=key):
                submit_query(opt)
                st.rerun()

# =====================================================
# Reset input BEFORE widget creation
# =====================================================
if st.session_state.reset_input:
    st.session_state.current_query = ""
    st.session_state.reset_input = False

# =====================================================
# Input box
# =====================================================
st.text_input(
    "Type your question",
    key="current_query",
    placeholder="Start typing…"
)

# =====================================================
# Live Suggestions
# =====================================================
if st.session_state.current_query.strip():
    suggestions = retriever.semantic_search(
        st.session_state.current_query, top_k=5
    )

    st.caption("Suggestions")
    for i, row in suggestions.iterrows():
        key = f"live_{i}_{hash(row.question)}"
        if st.button(row.question, key=key):
            submit_query(row.question)
            st.rerun()

# =====================================================
# Ask button
# =====================================================
if st.button("Ask"):
    if st.session_state.current_query.strip():
        submit_query(st.session_state.current_query)
        st.rerun()

# =====================================================
# Process Query
# =====================================================
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

    st.session_state.chat.append({
        "role": "user",
        "content": query
    })

    results = retriever.search_with_context(query, memory)
    decision = retriever.decide(results)

    best = results.iloc[0]
    scores = results["score"].tolist()

    # -------------------------------
    # ANSWER
    # -------------------------------
    if decision == "answer":
        memory.update(query, best.subcategory)

        log_event(
            query=query,
            decision="answer",
            scores=scores,
            chosen_faq=best.question,
            subcategory=best.subcategory,
        )

        st.session_state.chat.append({
            "role": "assistant",
            "content": f"**Answer:**\n\n{best.answer}"
        })

        related = results.iloc[1:4]
        if not related.empty:
            st.session_state.chat.append({
                "role": "assistant",
                "content": "**🔗 Best suitable FAQs:**",
                "options": related.question.tolist()
            })

    # -------------------------------
    # SUGGEST
    # -------------------------------
    elif decision == "suggest":
        log_event(
            query=query,
            decision="suggest",
            scores=scores,
            chosen_faq=None,
            subcategory=None,
        )

        st.session_state.chat.append({
            "role": "assistant",
            "content": "**Did you mean one of these?**",
            "options": results.head(5).question.tolist()
        })

    # -------------------------------
    # FALLBACK
    # -------------------------------
    else:
        log_event(
            query=query,
            decision="fallback",
            scores=scores,
            chosen_faq=None,
            subcategory=None,
        )

        st.session_state.chat.append({
            "role": "assistant",
            "content": (
                "I couldn’t find an exact match.\n\n"
                "**Related topics you may explore:**"
            ),
            "options": results.head(5).question.tolist()
        })

    st.rerun()
