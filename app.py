"""
app.py — Streamlit entry point for the Personal Finance Advisor.
Run with: uv run streamlit run app.py
"""

import ollama
import streamlit as st

from agents import AdvisorAgent, ExtractionAgent, QueryAgent
from rag import get_collection_count, retrieve_multi


def _escape_dollars(stream):
    for token in stream:
        yield token.replace("$", r"\$")


# ---------------------------------------------------------------------------
# Cached resources (instantiated once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_extraction_agent():
    return ExtractionAgent()

@st.cache_resource
def get_query_agent():
    return QueryAgent()

@st.cache_resource
def get_advisor_agent():
    return AdvisorAgent()


# ---------------------------------------------------------------------------
# Startup checks (run once at module load)
# ---------------------------------------------------------------------------

try:
    ollama.list()
except Exception:
    st.error("Ollama is not running. Start it with: ollama serve")
    st.stop()

try:
    if get_collection_count() == 0:
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
            with st.status("Extracting your financial profile...", expanded=False) as s:
                profile = get_extraction_agent().run(user_input)
                s.update(label="Financial profile extracted", state="complete")

            profile_rows = [(k.replace("_", " ").title(), str(v)) for k, v in profile.items() if k != "summary"]
            st.dataframe(
                {"Field": [r[0] for r in profile_rows], "Value": [r[1] for r in profile_rows]},
                use_container_width=True,
                hide_index=True,
            )
            if "summary" in profile:
                st.caption(f"Summary: {profile['summary']}")

            # Step 2 — Query reformulation
            with st.status("Identifying relevant topics...", expanded=False) as s:
                queries = get_query_agent().run(profile)
                s.update(label="Topics identified", state="complete")

            for i, q in enumerate(queries, 1):
                st.write(f"{i}. {q}")

            # Step 3 — RAG retrieval
            with st.status("Searching the knowledge base...", expanded=False) as s:
                chunks = retrieve_multi(queries, k_per_query=3)
                s.update(label=f"Retrieved {len(chunks)} relevant passages", state="complete")

            # Step 4 — Advisor (streaming)
            st.markdown("#### Your personalised advice")
            st.write_stream(_escape_dollars(get_advisor_agent().run(profile, chunks)))

            # Sources
            st.markdown("#### Sources consulted")
            by_source: dict[str, list[dict]] = {}
            for chunk in chunks:
                by_source.setdefault(chunk["source"], []).append(chunk)
            for source, source_chunks in sorted(by_source.items()):
                with st.expander(f"{source} ({len(source_chunks)} passage{'s' if len(source_chunks) > 1 else ''})"):
                    for i, chunk in enumerate(source_chunks, 1):
                        st.markdown(f"**Passage {i}**")
                        st.caption(chunk["text"])

        except Exception as e:
            st.error(f"Something went wrong: {e}")
