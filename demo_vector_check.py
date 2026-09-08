"""
Standalone script (Step 4): embed one test question, search the vector db, print top 3 results.
Run: poetry run python demo_vector_check.py "your question here"
"""

import sys

from app.embeddings import embed_query
from app.vector_store import get_collection

question = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"

collection = get_collection()
query_embedding = embed_query(question)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

print(f"Question: {question}\n")
for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), start=1):
    print(f"[{i}] source={meta.get('source')} distance={dist:.4f}")
    print(doc[:200])
    print()