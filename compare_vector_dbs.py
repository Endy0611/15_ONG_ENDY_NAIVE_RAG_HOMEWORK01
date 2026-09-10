"""
Bonus: compare ChromaDB vs Qdrant on the same documents and the same question.

Setup (one-time):
    poetry add qdrant-client
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Build both indexes first:
    poetry run python -m app.vector_store          # ChromaDB
    poetry run python -m app.vector_store_qdrant    # Qdrant

Then run:
    poetry run python compare_vector_dbs.py "What is RAG?"

Note on the numbers printed: ChromaDB reports "distance" (lower = more
similar). Qdrant here is configured with cosine similarity and reports
"score" (higher = more similar). They are not the same scale — this script
compares which chunks and how fast each retrieves, not a shared distance
number.
"""

import sys
import time

from app.config import COLLECTION_NAME, TOP_K
from app.embeddings import embed_query
from app.vector_store import get_collection
from app.vector_store_qdrant import get_qdrant_client


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is RAG?"
    query_embedding = embed_query(question)

    print(f"Question: {question}\n")

    # --- ChromaDB ---
    t0 = time.perf_counter()
    chroma_collection = get_collection()
    chroma_results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["metadatas", "distances"],
    )
    chroma_ms = (time.perf_counter() - t0) * 1000

    print(f"=== ChromaDB ({chroma_ms:.1f} ms) ===")
    for meta, dist in zip(chroma_results["metadatas"][0], chroma_results["distances"][0]):
        print(f"  source={meta.get('source')} distance={dist:.4f}")
    print()

    # --- Qdrant ---
    t0 = time.perf_counter()
    qdrant_client = get_qdrant_client()
    qdrant_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=TOP_K,
    )
    qdrant_ms = (time.perf_counter() - t0) * 1000

    print(f"=== Qdrant ({qdrant_ms:.1f} ms) ===")
    for point in qdrant_results.points:
        source = point.payload.get("source") if point.payload else "unknown"
        print(f"  source={source} score={point.score:.4f}")


if __name__ == "__main__":
    main()