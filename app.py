"""
app.py — Streamlit entry point for the Personal Finance Advisor.
Run with: uv run streamlit run app.py
"""

import ollama
import streamlit as st

from agents import AdvisorAgent, ExtractionAgent, QueryAgent
from rag import get_collection_count, merge_dedupe, retrieve_per_query


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

st.info(
    "**Before you start:** describe your current financial situation as clearly as possible. "
    "For the best results, try to mention each of the following points — they are the same fields "
    "the system extracts to build your profile:\n\n"
    "- **Monthly income** (e.g. \"I earn 2,000 $ per month\")\n"
    "- **Monthly expenses** (e.g. \"my expenses are around 1,200 $\")\n"
    "- **Savings** (e.g. \"I have 5,000 $ set aside\")\n"
    "- **Debts** (e.g. \"I have a 1,000 $ credit card debt\" or \"no debts\")\n"
    "- **Goals** (e.g. \"I want to buy a house\" or \"build an emergency fund\")\n"
    "- **Risk tolerance** (low, medium, or high)\n"
    "- **Time horizon** (e.g. \"in 3 years\")\n\n"
    "**Note:** if you want to consult the sources used by the advisor, tick the "
    "*Show sources* box **before** clicking **Get Advice** — toggling it afterwards "
    "will reset the results."
)

with st.form("advice_form"):
    user_input = st.text_area("Describe your financial situation:", height=150)
    run_button = st.form_submit_button("Get Advice")

show_sources = st.checkbox("Show sources", value=False)

if run_button:
    if not user_input.strip():
        st.warning("Please describe your financial situation before clicking Get Advice.")
    else:
        try:
            # Step 1 — Extraction
            with st.spinner("Extracting your financial profile..."):
                profile = get_extraction_agent().run(user_input)
            st.success("Financial profile extracted")

            profile_rows = [(k.replace("_", " ").title(), str(v)) for k, v in profile.items() if k != "summary"]
            st.dataframe(
                {"Field": [r[0] for r in profile_rows], "Value": [r[1] for r in profile_rows]},
                width="stretch",
                hide_index=True,
            )
            # Step 2 — Query reformulation
            with st.spinner("Identifying relevant topics..."):
                queries = get_query_agent().run(profile)
            st.success("Topics identified")

            for i, q in enumerate(queries, 1):
                q_display = q[:1].upper() + q[1:] if q else q
                st.text(f"{i}. {q_display}")

            # Step 3 — RAG retrieval
            with st.spinner("Searching the knowledge base..."):
                per_query_chunks = retrieve_per_query(queries, k_per_query=3)
                chunks = merge_dedupe(per_query_chunks)
            st.success(f"Retrieved {len(chunks)} relevant passages")

            # Step 4 — Advisor (streaming)
            st.markdown("#### Your personalised advice")
            st.write_stream(_escape_dollars(get_advisor_agent().run(profile, chunks)))

            # Sources (per-query, with similarity distances)
            if show_sources:
                st.markdown("#### Sources")
                for i, (q, q_chunks) in enumerate(zip(queries, per_query_chunks), 1):
                    q_display = q[:1].upper() + q[1:] if q else q
                    safe_label = q_display.replace("`", "").replace("$", "")
                    with st.expander(f"Query {i}: {safe_label}"):
                        for j, chunk in enumerate(q_chunks, 1):
                            st.markdown(
                                f"**Hit {j}** — `{chunk['source']}` "
                                f"(distance = `{chunk['distance']:.3f}`)"
                            )
                            st.text(chunk["text"])

        except Exception as e:
            st.error(f"Something went wrong: {e}")
