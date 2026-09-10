# RAG Baseline App

A local "chat with documents" app. It reads files from `data/`, splits them into chunks, embeds those chunks, stores them in a vector database, and answers questions grounded in those documents using a local LLM — the five-stage Naive RAG pipeline (ingest → chunk → embed → retrieve → generate).

## Stack

- **Embedding model**: `nomic-embed-text` (via Ollama)
- **Generation model**: `qwen3:4b` (via Ollama)
- **Vector database**: ChromaDB (local, persistent, no server setup needed)

## Project structure

```
app/
├── main.py            # FastAPI app + terminal chat loop
├── pipeline.py         # retriever + generator -> question in, answer out
├── retriever.py         # embed question -> search ChromaDB -> top-k chunks
├── generator.py          # build prompt -> call Ollama -> answer
├── ingestion.py            # load + chunk documents
├── vector_store.py          # embed + store chunks in ChromaDB
├── embeddings.py              # Ollama embedding calls (shared by ingestion + retrieval)
└── config.py                    # all tunable settings, in one place
demo_vector_check.py    # standalone retrieval sanity check (Step 4)
data/                    # source documents (.txt / .md / .pdf)
chroma_db/                # persistent vector store (created by build_index)
```

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

# 5. Chat (terminal loop)
poetry run python -m app.main

# 6. Or run it as a REST API instead
poetry run uvicorn app.main:app --reload
```

Type a question, get an answer. Type `exit` to quit the terminal chat loop.

### REST API

```bash
# (re)build the index
curl -X POST http://127.0.0.1:8000/ingest

# ask a question
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## Chunking strategy

Fixed-size character chunking (`CHUNK_SIZE=800` chars, `CHUNK_OVERLAP=120` chars), implemented in `app/ingestion.py`. Chosen because it's simple, has no dependencies, and works on any document regardless of structure — no need to detect sentence or paragraph boundaries. The 120-char overlap keeps an idea from being fully lost when it straddles a chunk boundary.

Trade-off: it can still cut a sentence in half mid-thought, which a semantic or sentence-aware chunking approach would avoid, at the cost of being slower and more complex to implement.

## Retrieval & grounding

- Top `TOP_K=4` chunks are retrieved per question (`app/retriever.py`).
- If the closest chunk's distance is still above `DISTANCE_THRESHOLD=1.0` (`app/config.py`), the app assumes the question isn't covered by the documents and returns a fixed "I could not find this in your documents." message instead of asking the LLM to guess (`app/pipeline.py`).
- The system prompt (`app/config.py`) instructs the model to answer only from the provided context, with a fixed fallback line, no outside knowledge, and no self-cited filenames (sources are reported separately by the app).

Note: `DISTANCE_THRESHOLD` was picked by eyeballing example scores, not calibrated against this specific embedding model's distance distribution. If on-topic questions are ever wrongly rejected (or off-topic ones wrongly answered), that threshold — and/or explicitly setting Chroma's distance metric via `hnsw:space` at collection-creation time — is the first thing to tune.

## Known issues fixed in this pass

- `app/generator.py` imported `GEN_MODEL` twice from `app.config` — harmless but removed.
- The REST API docstring/example in `app/main.py` called `/chat` with `{"query": ...}`, but `ChatRequest` expects `{"question": ...}` — the example would have failed with a validation error. Fixed to match the actual schema.
- Assorted comment/docstring typos across `app/config.py`, `app/embeddings.py`, `app/generator.py`, `app/ingestion.py`, and `app/vector_store.py` (e.g. "Souce" → "Source", "conteext" → "context" in the text actually sent to the LLM as part of the prompt).
- `qwen3:4b` is a reasoning model and "thinks" before answering by default. On some questions (observed on "difference between vector and embedding") it burned its whole token budget on hidden reasoning and returned an **empty** answer, even though retrieval and sources worked fine. Fixed by passing `think=False` in `app/generator.py`'s `ollama.chat()` call, and by adding a fallback message if content ever comes back empty.

---

## Reflection

Building the pipeline stage by stage made the whole idea of RAG click in a way just reading about it didn't. Wiring `ingestion.py` → `embeddings.py` → `vector_store.py` for the offline half, then `retriever.py` → `generator.py` for the online half, and finally joining both in `pipeline.py`, made it obvious where a bad answer actually comes from: a chunking problem, a retrieval miss, or a generation problem, instead of one big mystery box. Keeping every tunable value in `config.py` also paid off — adjusting `CHUNK_SIZE`, `TOP_K`, or the distance threshold to see how answers changed was a one-line edit instead of a hunt through files. Using `demo_vector_check.py` to check retrieval on its own, before ever calling the LLM, was the single most useful step: it caught weak or irrelevant retrieval early, before it could be masked by the generator confidently answering anyway.

The harder part than expected was the "not covered" case, and a related surprise: `qwen3:4b`'s default "thinking" mode occasionally returned a completely empty answer on a harder question, even though the correct chunks were retrieved. That took some tracing to realize it wasn't a retrieval or prompt problem at all, but the model spending its whole token budget on hidden reasoning. Deciding when a question is genuinely outside the documents versus just phrased differently from the source text also isn't obvious from a single top-1 distance score, and a fixed threshold doesn't generalize well across question types.

For an improvement, I'd add re-ranking: retrieve a wider top-k with the embedding model, then use a cross-encoder (or the LLM itself) to re-score and reorder those candidates before generation. That should catch cases where the right chunk is retrieved but not ranked first, without touching the chunking or embedding stages at all.

---

## Test Log

Ran via `poetry run python -m app.main`. (Note: the terminal chat loop only prints `Sources:`, not per-chunk distance scores — distances can be seen instead via `demo_vector_check.py`.)

### Q1 (on-topic)
**Question:** what is rag?
**Sources:** intro_to_rag.txt, vector_database_and_embeddings.txt
**Answer:** RAG (Retrieval-Augmented Generation) is a technique that combines a language model with an external knowledge source so that answers are grounded in real documents instead of relying only on what the model memorized during training.

### Q2 (on-topic)
**Question:** what is chunking strategy?
**Sources:** chunking_strategy.txt, vector_database_and_embeddings.txt
**Answer:** Chunking strategy refers to the approach of breaking a document into smaller pieces (chunks) before embedding and storage, with common methods including fixed-size chunking (with overlap), recursive/structure-aware chunking, and semantic chunking.

### Q3 (on-topic)
**Question:** what is vector_db?
**Sources:** advanced_rag.txt, vector_database_and_embeddings.txt
**Answer:** A vector database is a specialized storage system built to hold large numbers of vectors and search through them efficiently, using indexing structures that let it quickly find vectors most similar to a given query vector.

### Q4 (on-topic)
**Question:** different between vector and embedding?
**Sources:** advanced_rag.txt, vector_database_and_embeddings.txt
**Answer:** An embedding is a vector (list of numbers) that represents the meaning of text.
**Note:** grounded correctly (no hallucination), but the source documents never explicitly contrast "vector" as a general term vs. "embedding" — so the answer only defines embedding rather than fully answering the comparison. A documentation gap, not an app bug.

### Q5 (off-topic / not in documents)
**Question:** what is water?
**Sources:** (none)
**Answer:** I could not find this in your documents.