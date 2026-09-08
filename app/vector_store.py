"""
Stage 1 of the pipeline (continued): embed chunks and stroe them in chroma.
"""

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_DB_DIR, COLLECTION_NAME, DATA_DIR
from app.embeddings import embed_text
from app.ingestion import chunk_text, load_documents

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
    print(f"Indexed {count} chunks from '{DATA_DIR}/' into ChromaDB at '{CHROMA_DB_DIR}/'.")