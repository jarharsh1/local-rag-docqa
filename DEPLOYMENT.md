# Deployment guide

An **Ollama-first clinical document-QA (RAG)** API with a bundled browser UI.
Runs as local Python, a single Docker container, or the full `docker compose`
stack (API + Ollama). Falls back to a deterministic **offline backend** when no
model server is present, so it is always runnable.

## Architecture

```
  browser ──► FastAPI (ragdocqa.api.main)
  (web/ UI)    GET  /         static chat UI
               GET  /health   backend + index size
               POST /ingest   add a document (chunk + embed)
               POST /ask      retrieve + grounded answer + citations
                    │
                    ▼
             RagPipeline  →  embedder + vector store + LLM
                    │ (Ollama, or deterministic offline)
                    ▼
              Ollama model server
```

The bundled synthetic protocol (`data/clinical_protocol.md`) is pre-ingested at
startup so the UI works immediately; add more via `POST /ingest`.

## 1. Local (Python)

```bash
pip install -r requirements.txt
RAG_OFFLINE=1 uvicorn ragdocqa.api.main:app --reload       # instant, no Ollama
# or, with Ollama:
ollama pull llama3.2 && ollama pull nomic-embed-text
uvicorn ragdocqa.api.main:app --reload
```

Open <http://localhost:8000> (UI) or <http://localhost:8000/docs> (API).

## 2. Single Docker container

```bash
docker build -t local-rag-docqa .
docker run -p 8000:8000 -e RAG_OFFLINE=1 local-rag-docqa
# or point at a host Ollama:
docker run -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 local-rag-docqa
```

## 3. Full stack (docker compose)

```bash
cp .env.example .env
docker compose up --build       # ollama-init pulls the models on first run
# open http://localhost:8000
```

## 4. Cloud notes

- Stateless container; scale horizontally behind a load balancer (the in-memory
  index is per-instance — front it with a shared store for multi-instance
  persistence, or re-ingest at startup).
- Set `OLLAMA_HOST` to a managed/sidecar Ollama in production.
- `GET /health` is a ready-made liveness/readiness probe.

## Configuration

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `RAG_LLM_MODEL` / `RAG_EMBED_MODEL` | `llama3.2` / `nomic-embed-text` | models |
| `RAG_OFFLINE` | `0` | `1` forces the offline backend |
| `RAG_TOP_K` | `4` | chunks retrieved per query |
