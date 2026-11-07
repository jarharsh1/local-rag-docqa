# Maritime Compliance & SOP Assistant — Local RAG

An **Ollama-first**, fully local Retrieval-Augmented Generation service that
answers questions over fleet operating manuals, safety regulations and port
procedures — a **Bahri-style maritime use case** (The National Shipping Company
of Saudi Arabia operates ~89 vessels across oil, chemicals, dry bulk, gas and
logistics; crews and ops teams need fast, cited answers from a large body of
SOPs and regulations).

> The sample documents here are illustrative and fictional — not official
> documents of any real company.

## Why local

Compliance and operational documents are sensitive. This service runs entirely
on your own infrastructure: **Ollama** for the LLM and embeddings, an in-process
vector index, no data leaving the host. When Ollama is unreachable (e.g. CI), it
falls back to a deterministic offline backend so the pipeline and tests still run.

## How it works

```
documents ──chunk──► embed ──► in-memory vector store
                                      │
question ──embed──► cosine search ──► top-k chunks ──► LLM (grounded prompt) ──► answer + citations
```

- **Backends** (`core/backends.py`): `OllamaEmbedder` / `OllamaLLM` when Ollama
  is up; otherwise `HashingEmbedder` (deterministic hashed bag-of-words) +
  `ExtractiveLLM` (returns the most relevant context sentences).
- **Chunking** (`core/chunking.py`): overlapping windows on sentence/paragraph
  boundaries.
- **Vector store** (`core/vector_store.py`): pure-Python cosine similarity, with
  source metadata for citations.
- **Pipeline** (`core/pipeline.py`): `ingest()` and `ask()`.
- **API** (`api/main.py`): `/health`, `/ingest`, `/ask`.

## Quickstart

```bash
pip install -r requirements.txt

# Option A — with Ollama (recommended for real answers)
ollama pull llama3.2 && ollama pull nomic-embed-text
python demo.py

# Option B — no Ollama (deterministic offline backend)
RAG_OFFLINE=1 python demo.py

# Serve the API
uvicorn ragdocqa.api.main:app --reload   # see /docs
```

### Example

```bash
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"text": "Ballast water exchange must be at least 200 nautical miles from land.", "source": "sop.md"}'

curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How far from land must ballast water exchange happen?"}'
```

```json
{
  "answer": "Ballast water exchange must be at least 200 nautical miles from land.",
  "sources": [{ "rank": 1, "source": "sop.md", "score": 0.83, "excerpt": "..." }]
}
```

## Configuration (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `RAG_LLM_MODEL` | `llama3.2` | generation model |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `RAG_OFFLINE` | `0` | `1` forces the offline backend |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `600` / `100` | chunking |
| `RAG_TOP_K` | `4` | chunks retrieved per query |

## Tests

```bash
pytest -q          # runs offline; no Ollama needed
```

CI (GitHub Actions) lints, runs the tests and the demo with `RAG_OFFLINE=1`, and
builds the Docker image.
