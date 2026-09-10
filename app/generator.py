"""
Stage 3 of the pipeline: turn (question + retrieved chunks) into a grounded
answer by handing it all to the local LLM.
"""

import re
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

def _strip_thinking(text: str) -> str:
    """
    qwen3 (and other reasoning models) can emit a <think>...</think> block
    before the real answer. The `think=False` API flag only works if the
    Ollama *server* binary is new enough to honor it — if it's ignored, the
    thinking block leaks straight into message.content as plain text. This
    strips it out either way, so the app never shows raw reasoning to the user.
    """
    if "</think>" in text:
        # keep only what comes after the closing tag (the real answer)
        text = text.split("</think>", 1)[1]
    text = re.sub(r"^\s*<think>", "", text)
    return text.strip()

def generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    prompt = build_prompt(query, chunks)
    response = ollama.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # qwen3 is a reasoning model and "thinks" before answering by default.
        # This flag asks the server to skip that step. Some Ollama server
        # versions ignore it, so _strip_thinking() below is a backup.
        think=False,
    )
    content = response.message.content or "" #type:ignore
    content = _strip_thinking(content)
    if not content:
        return "The model returned an empty response. Try rephrasing the question."
    return content