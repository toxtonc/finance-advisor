"""
ingest.py — Parse, chunk, embed, and store documents into ChromaDB.
Run once before launching the app: uv run python ingest.py
Use --force to re-ingest even if the collection already has documents.
"""

import argparse
import sys
from pathlib import Path

import chromadb
import ollama
import pdfplumber

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "finance_docs"
CHUNK_SIZE = 800       # ~200 tokens (1 token ≈ 4 chars)
CHUNK_OVERLAP = 160    # ~40 tokens


def extract_text_pdf(path: Path) -> str:
    """Extract text from a PDF, filtering out short header/footer lines."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [
                line for line in text.splitlines()
                if len(line.strip()) >= 40
            ]
            pages.append("\n".join(lines))
    return "\n\n".join(pages)


def extract_text_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks. Tries to split on sentence boundaries
    ('. ', '! ', '? '); falls back to character split.
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Try to find the last sentence boundary in the window
            window = text[start:end]
            best = -1
            for sep in (". ", "! ", "? ", "\n\n", "\n"):
                idx = window.rfind(sep)
                if idx > best:
                    best = idx
            if best > chunk_size // 2:
                end = start + best + 1  # include the separator

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = end - overlap

    return chunks


def embed(text: str) -> list[float]:
    try:
        response = ollama.embed(model="qwen3-embedding:0.6b", input=text)
        return response.embeddings[0]
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}") from e


def ingest(force: bool = False) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0 and not force:
        print(f"Collection '{COLLECTION_NAME}' already has {collection.count()} chunks. "
              "Use --force to re-ingest.")
        return

    if force and collection.count() > 0:
        print("--force flag set. Deleting existing collection and re-ingesting...")
        client.delete_collection(name=COLLECTION_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

    files = list(DATA_DIR.iterdir())
    if not files:
        print(f"No files found in {DATA_DIR}. Add .pdf or .txt files and try again.")
        sys.exit(1)

    total_chunks = 0

    for path in sorted(files):
        if path.suffix.lower() == ".pdf":
            text = extract_text_pdf(path)
        elif path.suffix.lower() == ".txt":
            text = extract_text_txt(path)
        else:
            print(f"Skipping unsupported file: {path.name}")
            continue

        if not text.strip():
            print(f"  {path.name}: no text extracted, skipping.")
            continue

        chunks = chunk_text(text)
        print(f"  {path.name}: {len(chunks)} chunks", end="", flush=True)

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{path.name}_{i}"
            vector = embed(chunk)
            ids.append(chunk_id)
            embeddings.append(vector)
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk_id": i})
            print(".", end="", flush=True)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f" done.")

    print(f"\nIngestion complete. Total chunks in collection: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest even if the collection already has documents.")
    args = parser.parse_args()
    ingest(force=args.force)
