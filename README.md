# RAG Baseline App

A local "chat with documents" app. It reads files from `data/`, splits them into chunks, embeds those chunks, stores them in a vector database, and answers questions grounded in those documents using a local LLM.

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
curl -X POST http://127.0.0.1:8000/ingest

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## Chunking strategy

Fixed-size character chunking (`CHUNK_SIZE=800` chars, `CHUNK_OVERLAP=120` chars), implemented in `app/ingestion.py`. Chosen because it's simple, has no dependencies, and works on any document regardless of structure — no need to detect sentence or paragraph boundaries. The 120-char overlap keeps an idea from being fully lost when it straddles a chunk boundary.

Trade-off: it can still cut a sentence in half mid-thought, which a paragraph-aware or semantic chunking approach would avoid, at the cost of being more complex to implement (see Bonus #1 below for a direct comparison).

## Embedding model & vector database

- **Embedding model**: `nomic-embed-text`, run locally via Ollama.
- **Vector database**: ChromaDB, local and persistent (`chroma_db/` folder), no server to run.
- **Generation model**: `qwen3:4b`, run locally via Ollama.

---

## Bonus challenges

### 1. Compare two chunking strategies

`compare_chunking.py` runs the existing fixed-size chunker and a new paragraph-aware chunker (groups whole paragraphs up to `CHUNK_SIZE`, only falling back to fixed-size splitting if a single paragraph is longer than that) on every file in `data/`:

```bash
poetry run python compare_chunking.py
```

| File | Fixed-size chunks | Paragraph-aware chunks |
|---|---|---|
| advanced_rag.txt | 4 | 4 |
| chunking_strategy.txt | 3 | 4 |
| intro_to_rag.txt | 3 | 4 |
| prompt_building_generate.txt | 3 | 4 |
| vector_database_and_embeddings.txt | 4 | 5 |

The paragraph-aware strategy produces one extra chunk on 4 of 5 files — it ends a chunk earlier rather than merge across a paragraph boundary, trading a slightly higher chunk count for never splitting a paragraph mid-sentence.

### 2. Compare two vector databases

`vector_store_qdrant.py` mirrors `vector_store.py` but stores vectors in Qdrant instead of ChromaDB. `compare_vector_dbs.py` builds/queries both and reports timing.

```bash
poetry add qdrant-client
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
poetry run python -m app.vector_store
poetry run python -m app.vector_store_qdrant
poetry run python compare_vector_dbs.py "What is RAG?"
```

Actual result (17 chunks indexed in both):

```
=== ChromaDB (315.0 ms) ===
  source=intro_to_rag.txt distance=0.6910
  source=intro_to_rag.txt distance=0.7372
  source=intro_to_rag.txt distance=0.7460
  source=vector_database_and_embeddings.txt distance=0.8465

=== Qdrant (27.9 ms) ===
  source=intro_to_rag.txt score=0.6545
  source=intro_to_rag.txt score=0.6314
  source=intro_to_rag.txt score=0.6270
  source=vector_database_and_embeddings.txt score=0.5767
```

Same 4 chunks, same order, from both — retrieval is correct either way. Differences: Qdrant was ~11x faster on this small dataset; ChromaDB reports distance (lower = closer) while Qdrant here reports cosine score (higher = closer), so the numbers aren't on the same scale; and ChromaDB persists automatically to a local folder, while Qdrant's data lives only inside its Docker container unless you mount a volume (`-v "$(pwd)/qdrant_storage:/qdrant/storage"`).

### 3. Print retrieved chunks before the answer

Done in `app/main.py`'s chat loop — each retrieved chunk's source, distance, and a text preview print before `Answer:`.

### 4. "Not found" fallback when no strong match

Done in `app/pipeline.py` via `DISTANCE_THRESHOLD` — confirmed working below (see "what is API Gateway?").

---

## Reflection

Building the pipeline stage by stage made the whole idea of RAG click in a way just reading about it didn't. Wiring `ingestion.py` → `embeddings.py` → `vector_store.py` for the offline half, then `retriever.py` → `generator.py` for the online half, and finally joining both in `pipeline.py`, made it obvious where a bad answer actually comes from: a chunking problem, a retrieval miss, or a generation problem, instead of one big mystery box. Keeping every tunable value in `config.py` also paid off — adjusting `CHUNK_SIZE`, `TOP_K`, or the distance threshold to see how answers changed was a one-line edit instead of a hunt through files. Using `demo_vector_check.py` to check retrieval on its own, before ever calling the LLM, was the single most useful step: it caught weak or irrelevant retrieval early, before it could be masked by the generator confidently answering anyway.

The harder part than expected was the "not covered" case, and a related surprise: `qwen3:4b`'s default "thinking" mode occasionally returned a completely empty answer on a harder question, even though the correct chunks were retrieved. That took some tracing to realize it wasn't a retrieval or prompt problem at all, but the model spending its whole token budget on hidden reasoning before ever writing a final answer. Deciding when a question is genuinely outside the documents versus just phrased differently from the source text also isn't obvious from a single top-1 distance score, and a fixed threshold doesn't generalize well across question types.

For an improvement, I'd add re-ranking: retrieve a wider top-k with the embedding model, then use a cross-encoder (or the LLM itself) to re-score and reorder those candidates before generation. That should catch cases where the right chunk is retrieved but not ranked first, without touching the chunking or embedding stages at all.

---

## Test Log

Ran via `poetry run python -m app.main`.

### Q1 (on-topic)
**Question:** What is rag?
**Retrieved:** intro_to_rag.txt (0.6910), intro_to_rag.txt (0.7372), intro_to_rag.txt (0.7460), vector_database_and_embeddings.txt (0.8465)
**Answer:** RAG (Retrieval-Augmented Generation) is a technique that combines a language model with an external knowledge source so that answers are grounded in real documents instead of relying only on what the model memorized during training. It has two main pipelines: an offline pipeline that processes documents into a vector database, and an online pipeline that retrieves relevant documents for answering questions.
**Sources:** intro_to_rag.txt, vector_database_and_embeddings.txt

### Q2 (on-topic)
**Question:** what is vector_db?
**Retrieved:** vector_database_and_embeddings.txt (0.4882), vector_database_and_embeddings.txt (0.6799), vector_database_and_embeddings.txt (0.7071), advanced_rag.txt (0.9020)
**Answer:** A specialized storage system built to hold large numbers of vectors and search through them efficiently.
**Sources:** advanced_rag.txt, vector_database_and_embeddings.txt

### Q3 (on-topic)
**Question:** Difference between basic rag and advanced rag?
**Retrieved:** intro_to_rag.txt (0.7349), advanced_rag.txt (0.7734), intro_to_rag.txt (0.7902), advanced_rag.txt (0.8051)
**Answer:** Basic RAG (naive) keeps every stage simple: fixed-size chunking, a single embedding model, a single similarity search, and a single call to the language model. Advanced RAG adds extra stages such as re-ranking retrieved chunks, rewriting the user's query before searching, or combining multiple retrieval strategies to address weaknesses like poor retrieval precision or missed multi-part answers. These advanced techniques add complexity and extra latency due to additional model calls or search passes.
**Sources:** advanced_rag.txt, intro_to_rag.txt

### Q4 (on-topic)
**Question:** different between vector and embedding?
**Answer:** An embedding is a vector (list of numbers) that represents the meaning of text.
**Note:** grounded correctly (no hallucination), but the source documents never explicitly contrast "vector" as a general term vs. "embedding" — so the answer only defines embedding rather than fully answering the comparison. A documentation gap, not an app bug.
**Sources:** advanced_rag.txt, vector_database_and_embeddings.txt

### Q5 (off-topic / not in documents)
**Question:** what is API Gateway?
**Retrieved:** intro_to_rag.txt (1.0970), intro_to_rag.txt (1.1147), vector_database_and_embeddings.txt (1.1282), vector_database_and_embeddings.txt (1.1413) — all above `DISTANCE_THRESHOLD=1.0`
**Answer:** I could not find this in your documents.
**Sources:** (none)

### Q6 (off-topic / not in documents)
**Question:** what is water?
**Answer:** I could not find this in your documents.
**Sources:** (none)