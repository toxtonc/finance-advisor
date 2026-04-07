"""
rag.py — RAG retrieval logic using ChromaDB and Ollama embeddings.
"""

from pathlib import Path

import chromadb
import ollama

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "finance_docs"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def get_collection_count() -> int:
    return _get_collection().count()


def retrieve_single(query: str, k: int = 3) -> list[dict]:
    """
    Embed the query and return the top-k most relevant chunks.
    Returns list of {"text": str, "source": str, "id": str}.
    """
    try:
        response = ollama.embed(model="qwen3-embedding:0.6b", input=query)
        vector = response.embeddings[0]
    except Exception as e:
        raise RuntimeError(f"Embedding query failed: {e}") from e

    collection = _get_collection()
    results = collection.query(
        query_embeddings=[vector],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas"],
    )

    chunks = []
    for doc, meta, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["ids"][0],
    ):
        chunks.append({"text": doc, "source": meta["source"], "id": chunk_id})

    return chunks


def retrieve_multi(queries: list[str], k_per_query: int = 3) -> list[dict]:
    """
    Retrieve top-k chunks for each query, merge, and deduplicate by chunk id.
    Returns deduplicated list of {"text": str, "source": str, "id": str}.
    """
    seen_ids = set()
    merged = []

    for query in queries:
        for chunk in retrieve_single(query, k=k_per_query):
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                merged.append(chunk)

    return merged
