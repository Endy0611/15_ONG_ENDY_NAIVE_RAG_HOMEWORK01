"""
The whole app, end to end, exposed as a tiny API = no framework magic,
just three plain functions wired together.

Run it: 
    poetry run uvicorn app.main:app --reload

Then test it with curl (see README.md for full examples): 
    curl -X POST http://127.0.0.1:8000/ingest
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
        -d '{"question": "What is the vacation policy?"}'

Or run it as a plain terminal chat loop (no server needed):
    poetry run python -m app.main
"""

import logging

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.pipeline import answer_question
from app.vector_store import build_index

app = FastAPI(title="Baseline Chat-with-Documents API")

class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ingest")
def ingest():
    """(Re)build the vector index from everything in data/."""
    try:
        count = build_index()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"chunks_indexed": count}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """The full retrieve -> augment -> generate loop for one question."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    result = answer_question(req.question, req.top_k)
    return ChatResponse(answer=result["answer"], sources=result["sources"])

def run_chat_loop():
    """Terminal chat loop: ask a question, print the answer, ask again. Type exit to quit."""
    print("Baseline RAG chat. Type your question, or 'exit' to quit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        result = answer_question(question)

        # Bonus: show the retrieved chunks before the answer, so you can see
        # exactly what context the model was given to work with.
        if result["chunks"]:
            print("\nRetrieved chunks:")
            for i, c in enumerate(result["chunks"], start=1):
                preview = c["text"][:150].replace("\n", " ")
                print(f"  [{i}] {c['source']} (distance={c['distance']:.4f}): {preview}...")
            print()

        print(f"Answer: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print()

if __name__ == "__main__":
    run_chat_loop()