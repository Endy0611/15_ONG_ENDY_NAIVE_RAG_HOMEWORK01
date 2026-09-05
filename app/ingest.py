"""
Stage 1 of the pipeline: turn raw files in data/ into searchable vectors.
Flow: load file -> split into chunks -> embed each chunk -> stroe in chroma

Run directly to (re) build the index from scratch: 
poetry run pyhton -m app.ingest
"""

import os
from typing import List

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

from app.config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
)
from app.embeddings import embed_text

# --- loading ---

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_documents(data_dir: str = DATA_DIR) -> List[Tuple[str, str]]:
    """Return a list of (filename, full_text) for every .txt/.md/.pdf file in data_dir."""
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext in ("txt", "md"):
            text = _read_txt(path)
        elif ext == "pdf":
            text = _read_pdf(path)
        else: 
            continue  # skip anything we don't know how to read
        if text.strip():
            documents.append((filename, text))
    return documents

# --- Chunking ---
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Fixed-size chunking with overlap.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks

# --- Indexing ---
_CHROMA_SETTINGS =Settings(anonymized_telemetry=False)

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=_CHROMA_SETTINGS)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def build_index(data_dir: str = DATA_DIR) -> int: 
    """
    Wipe and rebuild the collection from scratch from whatever is in data_dir.
    Return the number of chunks indexed.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=_CHROMA_SETTINGS)
    try: 
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass # collection didn't exist yet - that's fine
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    documents = load_documents(data_dir)
    if not documents:
        raise FileNotFoundError(f"No .txt/.md/.pdf files found in {data_dir}")
    
    ids, texts, metadatas = [], [], []
    for filename, full_text in documents:
        for i, chunk in enumerate(chunk_text(full_text)):
            ids.append(f"{filename}::{i}")
            texts.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})
    embeddings = embed_text(texts)
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings) # type: ignore
    return len(texts)

if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks from {DATA_DIR} into ChromaDB at {CHROMA_DB_DIR}")