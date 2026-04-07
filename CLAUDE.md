# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This is a university demo project (Laboratorio di Data Science). Keep changes focused and avoid over-engineering — simplicity and clarity are preferred over robustness or scalability.

## Commands

```bash
# Install dependencies
uv add ollama chromadb pdfplumber streamlit

# Pull required Ollama models (Ollama must be running)
ollama pull qwen3.5:2b
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b

# Ingest documents into ChromaDB (run once before starting the app)
uv run python ingest.py

# Re-ingest from scratch (clears existing collection)
uv run python ingest.py --force

# Run the app
uv run streamlit run app.py
```

## Architecture

The system is a 4-step pipeline triggered on each user submission:

1. **ExtractionAgent** (`qwen3.5:2b`) — converts free-text financial description into a structured JSON profile with fixed keys: `monthly_income`, `monthly_expenses`, `savings`, `debts`, `goals`, `risk_tolerance`, `time_horizon`, `summary`. Has one retry on JSON parse failure.

2. **QueryAgent** (`qwen3.5:2b`) — converts the profile into exactly 3 retrieval queries covering: immediate situation (debt/cash-flow), savings/investment strategy, and long-term goal. Falls back to profile summary strings on any failure.

3. **RAG retrieval** (`rag.py`) — embeds each query with `qwen3-embedding:0.6b` via Ollama, queries ChromaDB, returns top-3 chunks per query, merges and deduplicates by chunk ID. ChromaDB is lazily initialized as a module-level singleton.

4. **AdvisorAgent** (`qwen3.5:4b`, streaming) — receives the profile + retrieved chunks and yields tokens via `ollama.chat(..., stream=True)`. The app uses `st.write_stream()` to render tokens incrementally.

All agents use `think=False` to suppress chain-of-thought reasoning in the Qwen3 models.

**Ingestion** (`ingest.py`): reads `.pdf` and `.txt` files from `data/`, extracts text (pdfplumber for PDFs, filtering lines < 40 chars as headers/footers), chunks with 800-char windows / 160-char overlap at sentence boundaries, embeds with `qwen3-embedding:0.6b`, stores in ChromaDB collection `finance_docs` at `chroma_db/`.

**App startup** (`app.py`): checks Ollama is reachable and ChromaDB collection is non-empty before rendering the UI; calls `st.stop()` on failure.

## Key details

- The `$` symbol is explicitly avoided in AdvisorAgent output (causes Streamlit LaTeX rendering issues); `app.py` also escapes any `$` tokens from the stream with `_escape_dollars()`.
- Chunk IDs are `{filename}_{chunk_index}` — used for deduplication in `retrieve_multi`.
- Models are instantiated fresh per request (no state held between submissions).
