# RAG Engine — Local Vector RAG

An **Ollama-first**, fully local Retrieval-Augmented Generation engine that
answers questions over **any text corpus** — with citations, running entirely on
your own infrastructure. Ingest documents, embed them into an in-process vector
store, and get grounded, cited answers.

> Ships with a tiny neutral sample knowledge base so it runs out of the box.
> Replace `data/sample_knowledge_base.md` with your own corpus — policies,
> manuals, contracts, protocols, FAQs — and nothing else changes.

## Use cases

The engine is the same; only the corpus changes. Examples:

| Domain | Corpus you ingest | Example question |
|--------|-------------------|------------------|
| **Healthcare** | clinical protocols, guidelines | "What's first-line for type 2 diabetes?" |
| **Legal** | contracts, charter parties, policies | "What's the demurrage rate on late discharge?" |
| **Customer support** | help center, FAQs, SLAs | "How long do I have to request a refund?" |
| **HR / IT** | employee handbook, runbooks | "What are the password requirements?" |
| **Finance / compliance** | regulations, filings | "What's the data-retention period?" |
| **Research** | papers, technical docs | "What method did the authors use for X?" |

## Stack

- **Backend:** FastAPI (an async Flask-equivalent — serves the HTTP API and auto
  docs at `/docs`).
- **Frontend:** a lightweight browser UI (`web/`) served at `/` — ingest a
  document and ask questions, with a citations panel and one-click example
  questions. **This is where the user interacts.**
- **Model:** Ollama (local LLM + embeddings), with a deterministic offline
  fallback so the pipeline and tests run with no model server (CI-safe).
- **Deploy:** single Docker image or `docker compose` (API + Ollama). See
  [DEPLOYMENT.md](DEPLOYMENT.md).

Open **<http://localhost:8000>** once running.

## How it works

```
documents ─chunk─► embed ─► in-memory vector store
question ─embed─► cosine search ─► top-k chunks ─► LLM (grounded prompt) ─► answer + citations
```

- **Backends** (`core/backends.py`): Ollama embeddings/LLM when up; otherwise a
  deterministic `HashingEmbedder` + `ExtractiveLLM`.
- **Chunking** (`core/chunking.py`): overlapping windows on sentence boundaries.
- **Vector store** (`core/vector_store.py`): pure-Python cosine, with source
  metadata for citations.
- **Pipeline** (`core/pipeline.py`): `ingest()` and `ask()`.
- **API** (`api/main.py`): `/`, `/health`, `/ingest`, `/ask`.

## Quickstart

```bash
pip install -r requirements.txt

# Offline (no Ollama) — instant:
RAG_OFFLINE=1 uvicorn ragdocqa.api.main:app --reload

# With Ollama for natural-language answers:
ollama pull llama3.2 && ollama pull nomic-embed-text
uvicorn ragdocqa.api.main:app --reload

python demo.py          # CLI demo (RAG_OFFLINE=1 to force offline)
```

### Example

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How long do I have to request a refund?"}'
```

```json
{
  "answer": "A customer may request a refund within 30 days of purchase...",
  "sources": [{ "rank": 1, "source": "sample_knowledge_base.md", "score": 0.52, "excerpt": "..." }]
}
```

## Configuration (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `RAG_LLM_MODEL` / `RAG_EMBED_MODEL` | `llama3.2` / `nomic-embed-text` | models |
| `RAG_OFFLINE` | `0` | `1` forces the offline backend |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `600` / `100` | chunking |
| `RAG_TOP_K` | `4` | chunks retrieved per query |

## Tests

```bash
pytest -q          # offline; no Ollama needed
```

CI (GitHub Actions) lints, runs the tests and the demo with `RAG_OFFLINE=1`, and
builds the Docker image.
