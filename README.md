# RAG Baseline App

A minimal, no-framework-magic RAG (Retrieval-Augmented Generation) API. It exposes two endpoints — `/ingest` to build a vector index from documents in `data/`, and `/chat` to ask questions grounded in those documents — using a fully local stack (Ollama + ChromaDB).

The whole pipeline is three plain functions wired together:

```
data/*.pdf,.txt,.md ──ingest──▶ ChromaDB (local vector store)
                                       │
question ──embed──▶ retrieve top-k chunks ──▶ build prompt ──▶ Ollama LLM ──▶ answer
```

## Tech stack

| Piece | Tool |
|---|---|
| API framework | FastAPI + Uvicorn |
| Vector store | ChromaDB (local, persisted to disk) |
| Embedding model | `nomic-embed-text` (via [Ollama](https://ollama.com)) |
| Generation model | `qwen3:4b` (via Ollama) |
| PDF parsing | pypdf |
| Package manager | Poetry |
| Python | 3.14+ |

## Project structure

```
rag-baseling-app/
├── app/
│   ├── main.py         # FastAPI app: /ingest and /chat endpoints
│   ├── config.py        # All tunable settings (models, chunk size, top_k, prompt)
│   ├── embeddings.py     # Wraps Ollama embedding calls (shared by ingest + retrieval)
│   ├── ingest.py         # Load docs -> chunk -> embed -> store in ChromaDB
│   ├── retrieval.py       # Embed query -> similarity search -> top-k chunks
│   └── generate.py        # Build augmented prompt -> call Ollama chat -> answer
├── data/                # Drop your .txt / .md / .pdf source documents here
├── chroma_db/           # Auto-created on first ingest — persisted vector index
├── pyproject.toml
└── poetry.lock
```

## Prerequisites

1. **Python 3.14+**
2. **[Poetry](https://python-poetry.org/docs/#installation)** for dependency management
3. **[Ollama](https://ollama.com/download)** installed and running locally
4. Pull the two models this app uses:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen3:4b
   ```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Endy0611/rag-baseline-app.git
cd rag-baseline-app

# 2. Install dependencies
poetry install

# 3. Make sure Ollama is running (in a separate terminal)
ollama serve
```

Put whatever `.pdf`, `.txt`, or `.md` files you want to test against into the `data/` folder (a sample PDF is already included).

## Running the app

```bash
poetry run uvicorn app.main:app --reload
```

The API is now live at `http://127.0.0.1:8000` (interactive docs at `http://127.0.0.1:8000/docs`).

### Step 1 — Build the index

Run this once (and again any time you change files in `data/`):

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

Response:
```json
{"chunks_indexed": 42}
```

You can also run ingestion directly from the CLI without starting the server:

```bash
poetry run python -m app.ingest
```

### Step 2 — Ask a question

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the vacation policy?"}'
```

Response:
```json
{
  "answer": "...",
  "sources": ["policy.pdf"]
}
```

Optional: override how many chunks are retrieved per question with `top_k`:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the vacation policy?", "top_k": 6}'
```

## Configuration

Every tunable knob lives in `app/config.py` — nothing else needs to be touched to change behavior:

| Setting | Meaning | Default |
|---|---|---|
| `EMBED_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `GEN_MODEL` | Ollama chat/generation model | `qwen3:4b` |
| `DATA_DIR` | Folder scanned for source documents | `data` |
| `CHROMA_DB_DIR` | Where the vector index is persisted | `chroma_db` |
| `COLLECTION_NAME` | ChromaDB collection name | `documents` |
| `CHUNK_SIZE` | Characters per chunk | `800` |
| `CHUNK_OVERLAP` | Overlapping characters between chunks | `120` |
| `TOP_K` | Chunks retrieved per question | `4` |
| `SYSTEM_PROMPT` | Instruction given to the LLM (context-only answers) | see file |

## How it works (pipeline stages)

1. **Ingest** (`app/ingest.py`) — reads every `.txt`/`.md`/`.pdf` in `data/`, splits each into fixed-size overlapping chunks, embeds them via Ollama, and stores them (with source filename + chunk index metadata) in a local ChromaDB collection. Re-running `/ingest` wipes and rebuilds the collection from scratch.
2. **Retrieve** (`app/retrieval.py`) — embeds the incoming question with the same embedding model, then does a similarity search against ChromaDB to pull back the top-k most relevant chunks.
3. **Generate** (`app/generate.py`) — assembles a numbered context block from the retrieved chunks plus the question into a single prompt, and sends it to the local Ollama chat model. The system prompt instructs the model to answer only from the given context and to cite source filenames.

## Troubleshooting

- **`FileNotFoundError` on `/ingest`** — `data/` has no `.txt`/`.md`/`.pdf` files; add at least one.
- **Connection errors calling Ollama** — make sure `ollama serve` is running and that `nomic-embed-text` / `qwen3:4b` have been pulled (`ollama list` to check).
- **Empty or irrelevant answers** — try increasing `TOP_K` or lowering `CHUNK_SIZE` in `app/config.py`, then re-run `/ingest`.
- **Stale results after updating documents** — you must re-run `/ingest`; the vector store isn't updated automatically when files in `data/` change.

## Notes for testers

- This is a **baseline/teaching implementation** — chunking is fixed-size character splitting (not semantic), and there's no auth, rate limiting, or streaming.
- The vector index (`chroma_db/`) is local to your machine — delete that folder to fully reset, then re-run `/ingest`.
- Swap models by editing `EMBED_MODEL`/`GEN_MODEL` in `app/config.py` — just make sure you `ollama pull` whatever you switch to first.