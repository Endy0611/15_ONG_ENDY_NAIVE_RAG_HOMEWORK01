"""
Bonus: same job as vector_store.py (embed chunks + store/search them), but
backed by Qdrant instead of ChromaDB, so the two can be compared.

Requires:
  1. A local Qdrant instance:
       docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  2. The qdrant-client package:
       poetry add qdrant-client

Run directly to (re)build the Qdrant index from data/:
    poetry run python -m app.vector_store_qdrant
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import COLLECTION_NAME, DATA_DIR
from app.embeddings import embed_text
from app.ingestion import chunk_text, load_documents

QDRANT_URL = "http://localhost:6333"
# nomic-embed-text outputs 768-dimensional vectors; update this if you swap models.
EMBED_DIM = 768


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def build_index_qdrant(data_dir: str = DATA_DIR) -> int:
    """
    Wipe and rebuild the Qdrant collection from scratch from whatever is in
    data_dir. Return the number of chunks indexed. Mirrors build_index() in
    vector_store.py so the two are an apples-to-apples comparison.
    """
    client = get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    documents = load_documents(data_dir)
    if not documents:
        raise FileNotFoundError(f"No .txt/.md/.pdf files found in {data_dir}")

    texts, metadatas = [], []
    for filename, full_text in documents:
        for i, chunk in enumerate(chunk_text(full_text)):
            texts.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i, "text": chunk})

    embeddings = embed_text(texts)
    points = [
        PointStruct(id=i, vector=embedding, payload=meta)
        for i, (embedding, meta) in enumerate(zip(embeddings, metadatas))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


if __name__ == "__main__":
    count = build_index_qdrant()
    print(f"Indexed {count} chunks from '{DATA_DIR}/' into Qdrant at '{QDRANT_URL}'.")