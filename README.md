# RAG Pipeline

A local Retrieval-Augmented Generation system for querying a PDF library: FastAPI backend (Qdrant + Ollama), React chat frontend, and a Ragas-based eval pipeline.

---

## How It Works

```
[ingest.py]
  backend/data/*.pdf
    -> loader.py       (load pages)
    -> chunker.py      (split into chunks)
    -> embedder.py     (encode chunks to normalized vectors)
    -> qdrant_store.py (build collection, persist to qdrant_storage/)

[api/app.py + routes/chat.py]  (used by the frontend)
  POST /chat
    -> qdrant_store.py (load persisted collection at startup)
    -> embedder.py     (encode query)
    -> retriever.py    (top-k similar chunks above similarity threshold)
    -> generator.py    (stream answer + citations over SSE)

[main.py]  (CLI alternative to the API, same pipeline, no server)

[eval/run_eval.py]
  testset.json
    -> retriever.py + generator.py (run the real pipeline per question)
    -> ragas_eval.py   (judge each answer: faithfulness, relevancy, precision, recall)
    -> aggregate mean/p50/p95 + retrieval hit-rate + latency -> results/*.json
```

---

## Usage

### Backend + ingestion

```bash
cd backend
uv sync                     # install deps (see pyproject.toml)
# Put PDFs in backend/data/, then:
python ingest.py            # build and save the Qdrant collection (run once, or after adding PDFs)

python main.py               # query from the terminal
# or
fastapi dev api/app.py       # run the API for the frontend (localhost:5173 is CORS-allowed)
```

### Frontend

```bash
cd frontend
npm install
npm run dev                  # Vite dev server, talks to the FastAPI backend
```

### Eval

```bash
cd eval
# Disabled by default (slow local-LLM judge). Set in backend/.env:
#   RAGAS_EVAL_ENABLED=true
python run_eval.py --smoke 3   # quick smoke test
python run_eval.py              # full run against testset.json -> results/baseline.json
```

---

## Key Dependencies

- `fastapi` - API server, SSE streaming chat endpoint
- `qdrant-client` - vector store
- `sentence-transformers` - `BAAI/bge-small-en-v1.5` embedding model
- `langchain-ollama` - LLM integration (`llama3.2` for generation)
- `langsmith` - tracing across chat, retrieval, and generation
- `ragas` - RAG evaluation metrics, judged locally via `llama3.1:8b` through Ollama
- `react`, `vite`, `tailwindcss` - chat frontend
- Ollama running locally with `llama3.2` and `llama3.1:8b` pulled
