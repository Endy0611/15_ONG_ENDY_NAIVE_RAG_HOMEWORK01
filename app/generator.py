"""
Stage 3 of the pipeline: turn (question + retrieved chunks) into a grounded
answer by handing it all to the local LLM.
"""

from typing import List

import ollama

from app.config import GEN_MODEL, SYSTEM_PROMPT
from app.retriever import RetrievedChunk

def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:

    """
    This is the 'prompt-augmentation' step: we hand-assemble a prompt that contains only the retrieved context,
    numbered so the model (and students, reading the logs) can see exactly what it was given to work with.
    """
    if not chunks:
        context_block = "(no relevant context was found)"
    else: 
        context_block = "\n\n".join(
            f"[{i+1}] Source: {c['source']}\n{c['text']}"
            for i, c in enumerate(chunks)
        )
    return (
        f"Context: \n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above. If there is not enough context provided just say so."
    )

def generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    prompt = build_prompt(query, chunks)
    response = ollama.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # qwen3 is a reasoning model and "thinks" before answering by default.
        # On longer questions it can burn its whole token budget on that hidden
        # step and return empty content. Turning thinking off makes it answer
        # directly, which is all a baseline RAG app needs.
        think=False,
    )
    content = response.message.content #type:ignore
    if not content or not content.strip():
        return "The model returned an empty response. Try rephrasing the question."
    return content