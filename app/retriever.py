"""
Stage 2 of the pipeline: given a question, find the chunks most likely to contain the answer.
"""

from typing import List, TypedDict

from app.config import TOP_K
from app.embeddings import embed_query
from app.vector_store import get_collection

class RetrievedChunk(TypedDict):
    text: str
    source: str
    chunk_index: int
    distance: float

def retrieve(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k,
        include = ["documents","metadatas", "distances"], # type: ignore
    )

    chunks: List[RetrievedChunk] = []
    documents = results["documents"][0] #type: ignore
    metadatas = results["metadatas"][0] #type: ignore
    distances = results["distances"][0] #type: ignore
    for text, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "distance": distance,
            } # type:ignore
        )
    return chunks
