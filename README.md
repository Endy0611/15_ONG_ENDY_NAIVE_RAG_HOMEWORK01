# RAG Baseline App

A local "chat with documents" app. Reads files from `data/`, chunks them, embeds them, stores them in a vector database, and answers questions grounded in those documents using a local LLM.

## Stack

- **Embedding model**: `nomic-embed-text` (via Ollama)
- **Generation model**: `qwen3:4b` (via Ollama)
- **Vector database**: ChromaDB (local, persistent, no server setup needed)

## How to run

```bash
# 0. (optional) make Poetry create .venv/ inside this folder, one-time setting
poetry config virtualenvs.in-project true

# 1. Install deps
poetry install

# 2. Make sure Ollama is running and models are pulled
ollama serve
ollama pull nomic-embed-text
ollama pull qwen3:4b

# 3. Build the index from data/
poetry run python -m app.vector_store

# 4. (optional) sanity-check retrieval on its own
poetry run python demo_vector_check.py "What is RAG?"

# 5. Chat
poetry run python -m app.main

# 6. RestAPI
poetry run uvicorn app.main:app --reload
```

Type a question, get an answer. Type `exit` to quit.

## Chunking strategy

Fixed-size character chunking (800 chars, 120 char overlap). Chosen because it's simple and works on any document regardless of structure, and the overlap keeps an idea from being fully lost at a chunk boundary. Trade-off: it can still cut a sentence in half, which a semantic chunking approach would avoid at the cost of being more expensive to compute.

## Project structure

```
app/
├── main.py         # chat loop
├── pipeline.py      # retriever + generator -> question in, answer out
├── retriever.py      # embed question -> search ChromaDB -> top-k chunks
├── generator.py       # build prompt -> call Ollama -> answer
├── ingestion.py         # load + chunk documents
├── vector_store.py       # embed + store chunks in ChromaDB
├── embeddings.py           # Ollama embedding calls
└── config.py                # all tunable settings
demo_vector_check.py  # standalone retrieval check
data/                 # your source documents
```