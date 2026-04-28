# Personal Finance Advisor

A local personal finance advisor that combines a three-agent pipeline with a RAG system. The user describes their financial situation in natural language; the system extracts a structured profile, reformulates it into targeted retrieval queries, retrieves relevant content from a finance corpus, and generates a grounded, personalised recommendation — all running entirely offline.

---

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running

---

## Setup

```bash
uv add ollama chromadb pdfplumber streamlit

ollama pull qwen3.5:2b
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b

uv run python ingest.py
```

The last command parses all documents in `data/`, chunks them, embeds each chunk, and
stores everything in a local ChromaDB database (`chroma_db/`). Run it once before
launching the app. Use `--force` to re-ingest.

---

## Usage

```bash
uv run streamlit run app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`).

### Example Input

To obtain the best personalised response from the Agents, you should mention the following key points in your description, as the system extracts these specific fields to build your profile:

* **Monthly Income**: Provide your regular income (e.g., "I earn 2000 (net or gross) a month").
* **Monthly Expenses**: Estimate your fixed and variable expenses (e.g., "My expenses are around 1200 a month").
* **Savings**: State your current savings (e.g., "I have 5000 in savings").
* **Debts**: Mention any outstanding loans or debts (e.g., "I have a 1000 credit card debt").
* **Goals**: Specify your main financial objective (e.g., "I want to buy a car") or (e.g., "I want to provide a good education to my children with a famous university").
* **Risk Tolerance**: Express how much risk you can handle ("low", "medium", or "high").
* **Time Horizon**: Give a timeframe for your goal (e.g., "in 3 years").

*Example:*
> "I earn 2000 net a month and my expenses are around 1200. I have 5000 in savings and a 1000 credit card debt. My goal is to buy a car in 3 years and my risk tolerance is low."

---

## Architecture

```
User input (natural language)
        ↓
[ExtractionAgent — qwen3.5:2b]
  structured JSON profile
        ↓
[QueryAgent — qwen3.5:2b]
  3 targeted retrieval queries
        ↓
[RAG retrieval — qwen3-embedding:0.6b + ChromaDB]
  top-3 chunks per query → merge → deduplicate
        ↓
[AdvisorAgent — qwen3.5:4b, streaming]
  personalised advice rendered token by token
```

### Agents

| Agent | Model | Task |
|---|---|---|
| `ExtractionAgent` | qwen3.5:2b | Parse free-text into a structured financial profile in JSON format|
| `QueryAgent` | qwen3.5:2b | Translate profile into 3 targeted retrieval queries |
| `AdvisorAgent` | qwen3.5:4b | Generate grounded advice from profile + retrieved context |

### Model sizing rationale

Each agent is assigned the minimum model capacity sufficient for its task, implementing
an ascending capability ladder:

- **ExtractionAgent** — Extraction is a constrained parsing task: follow strict JSON schema, copy
  values verbatim, use "unknown" for missing fields. Minimal reasoning required.
- **QueryAgent** — Query reformulation requires semantic inference: understand the profile
  holistically and translate it into distinct, information-seeking phrases that maximise
  retrieval coverage.
- **AdvisorAgent** — Advisory generation requires synthesis and reasoning: integrate the retrieved
  context, ground all claims in sources, and produce coherent, structured advice.

### Why query reformulation improves RAG retrieval

A raw user message ("I earn 1500 EUR a month and want to buy a house in 5 years") is
optimised for human communication, not vector similarity search. The `QueryAgent`
decomposes the profile into 3 targeted, concept-level queries that each address a
distinct financial dimension. This increases the semantic diversity of the retrieval
and ensures the advisor receives context covering debt management, savings strategies,
and investment horizons — rather than only chunks superficially similar to the original
free-text input.

---

## Project structure

```
finance-advisor/
├── data/          # source documents (.pdf and .txt)
├── chroma_db/     # ChromaDB persistent storage (auto-created by ingest.py)
├── ingest.py      # document ingestion pipeline (run once)
├── rag.py         # RAG retrieval logic
├── agents.py      # ExtractionAgent, QueryAgent, AdvisorAgent
├── app.py         # Streamlit entry point
├── pyproject.toml # project dependencies and configuration
└── uv.lock        # project dependencies lock file
```
