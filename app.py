"""
app.py — Streamlit entry point for the Personal Finance Advisor.
Run with: uv run streamlit run app.py
"""

import html

import chromadb
import ollama
import streamlit as st

from agents import AdvisorAgent, ExtractionAgent, QueryAgent
from rag import CHROMA_DIR, COLLECTION_NAME, retrieve_multi

def _escape_dollars(stream):
    for token in stream:
        yield token.replace("$", r"\$")

# ---------------------------------------------------------------------------
# Startup checks (run once at module load)
# ---------------------------------------------------------------------------

try:
    ollama.list()
except Exception:
    st.error("Ollama is not running. Start it with: ollama serve")
    st.stop()

try:
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _col = _client.get_or_create_collection(name=COLLECTION_NAME)
    if _col.count() == 0:
        st.error("No documents indexed. Run: uv run python ingest.py first.")
        st.stop()
except Exception as e:
    st.error(f"ChromaDB error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Personal Finance Advisor")
st.caption("Powered by Qwen Family")

with st.form("advice_form"):
    user_input = st.text_area("Describe your financial situation:", height=150)
    run_button = st.form_submit_button("Get Advice")

if run_button:
    if not user_input.strip():
        st.warning("Please describe your financial situation before clicking Get Advice.")
    else:
        try:
            # Step 1 — Extraction
            st.markdown("---")
            st.markdown("#### Step 1 — Extracting your financial profile")
            with st.spinner("Analysing your situation..."):
                profile = ExtractionAgent().run(user_input)
            st.success("Profile extracted.")
            st.json(profile)

            # Step 2 — Query reformulation
            st.markdown("---")
            st.markdown("#### Step 2 — Identifying relevant topics")
            with st.spinner("Generating retrieval queries..."):
                queries = QueryAgent().run(profile)
            st.success("Topics identified.")
            for i, q in enumerate(queries, 1):
                st.write(f"{i}. {q}")

            # Step 3 — RAG retrieval (silent)
            st.markdown("---")
            st.markdown("#### Step 3 — Retrieving relevant content")
            with st.spinner("Searching the knowledge base..."):
                chunks = retrieve_multi(queries, k_per_query=3)
            st.success(f"Retrieved {len(chunks)} relevant passages.")

            # Step 4 — Advisor (streaming)
            st.markdown("---")
            st.markdown("#### Step 4 — Generating your personalised advice")
            full_response = st.write_stream(_escape_dollars(AdvisorAgent().run(profile, chunks)))

            # Sources
            st.markdown("---")
            st.markdown("#### Sources consulted")
            by_source: dict[str, list[dict]] = {}
            for chunk in chunks:
                by_source.setdefault(chunk["source"], []).append(chunk)
            for source, source_chunks in sorted(by_source.items()):
                with st.expander(f"{source} ({len(source_chunks)} passage{'s' if len(source_chunks) > 1 else ''})"):
                    for i, chunk in enumerate(source_chunks, 1):
                        st.markdown(f"**Passage {i}**")
                        escaped_text = html.escape(chunk["text"]).replace('\n', '<br>')
                        st.markdown(f'<div style="opacity: 0.8; font-size: 0.85em; margin-bottom: 1rem; line-height: 1.5;">{escaped_text}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
