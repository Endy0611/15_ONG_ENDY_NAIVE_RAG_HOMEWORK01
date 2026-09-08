"""
Connects the retriever and the generator into one flow: question in, grounded answer out.
"""

from app.config import DISTANCE_THRESHOLD
from app.generator import generate_answer
from app.retriever import retrieve

def answer_question(query: str, top_k: int | None = None) -> dict:
    chunks = retrieve(query, top_k) if top_k else retrieve(query)

    # if even the closest chunk is a weak match, don't ask the LLM to guess
    if not chunks or chunks[0]["distance"] > DISTANCE_THRESHOLD:
        return {"answer": "I could not find this in your documents.", "sources": [], "chunks": chunks}

    answer = generate_answer(query, chunks)
    sources = sorted({c["source"] for c in chunks})
    return {"answer": answer, "sources": sources, "chunks": chunks}