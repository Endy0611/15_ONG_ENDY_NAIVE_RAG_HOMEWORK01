""""
Stage 3 of the pipeline: turn (question + retrivev chunks) into a grounded 
answer by handing it all to the local LLM.
"""

from typing import List

import ollama

from app.config import GEN_MODEL, GEN_MODEL, SYSTEM_PROMPT
from app.retrieval import RetrievedChunk

def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:

    """
    This is the 'prompt-augmentation' step: we hand-assemble a prompt that contains only the retrieved context,
    numbered so the model (and students, reading the logs) can see exactly what it was given to work with.
    """
    if not chunks:
        context_block = "(no relevant context was found)"
    else: 
        context_block = "\n\n".join(
            f"[{i+1}] Souce: {c['source']}\n{c['text']}"
            for i, c in enumerate(chunks)
        )
    return (
        f"Context: \n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the conteext above. If there is not enough context provided just say so."
    )

def generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    prompt = build_prompt(query, chunks)
    response = ollama.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.message.content #type:ignore
        