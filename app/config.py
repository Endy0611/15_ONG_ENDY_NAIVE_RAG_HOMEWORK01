""""
ALL the knobs students can turn are collected here, in one place, 
Change a alude here and the whole app picks it up - no hunting through files.
"""

# --- Models (must already be pulled in Ollama: `ollama pull <model>`)
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen3:4b"

# --- Storage ---
DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "documents"

# --- Chunking ---
CHUNK_SIZE = 800 #characters per chunk (not tokens - good enough for a baseline)
CHUNK_OVERLAP = 120 #characters shared between consecutive chunks

# --- Retrieval ---
TOP_K = 4 # how many chunks to pull back per question

# --- Generation ---
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below. If the answer is not contained in the context, "
    "say \"I don't have enough information in the documents to answer that.\" "
    "Do not use outside knowledge. Cite the source file name(s) you use"
)