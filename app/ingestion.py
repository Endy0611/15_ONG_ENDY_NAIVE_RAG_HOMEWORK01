"""
Stage 1 of the pipeline: turn raw files in data/ into searchable vectors.
Flow: load file -> split into chunks -> embed each chunk -> store in chroma

Run directly to (re) build the index from scratch: 
poetry run python -m app.vector_store
"""

import os
from typing import List, Tuple

from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR

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

# --- Bonus: a second chunking strategy, for comparison ---
def chunk_text_by_paragraph(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Paragraph-aware chunking: groups whole paragraphs together until adding
    another one would exceed chunk_size, instead of cutting at a fixed
    character count. Never splits a paragraph in the middle unless that one
    paragraph alone is already bigger than chunk_size, in which case it falls
    back to fixed-size chunking for just that paragraph.
    """
    text = text.strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(para) > chunk_size:
            chunks.extend(chunk_text(para, chunk_size, overlap))
        else:
            current = para
    if current:
        chunks.append(current)
    return chunks