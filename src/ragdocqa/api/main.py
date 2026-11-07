"""FastAPI serving layer for the local clinical document-QA service.

Endpoints:
  GET  /         - the browser chat UI (static)
  GET  /health   - liveness + which backend (ollama/offline) is active
  POST /ingest   - add a document (raw text) to the index
  POST /ask      - ask a question; returns a grounded answer + citations
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragdocqa.core.backends import OllamaLLM
from ragdocqa.core.pipeline import RagPipeline

_WEB_DIR = Path(__file__).resolve().parents[3] / "web"
_SAMPLE = Path("data/clinical_protocol.md")

app = FastAPI(
    title="Clinical Document QA (Local RAG)",
    version="0.1.0",
    description="Ollama-first retrieval-augmented QA over clinical protocols, "
    "guidelines and discharge summaries — fully local, with citations.",
)

# One in-process pipeline/index for the app's lifetime.
_pipeline = RagPipeline()

# Pre-ingest the bundled synthetic protocol so the UI works immediately.
if _SAMPLE.exists():
    try:
        _pipeline.ingest(_SAMPLE.read_text(encoding="utf-8"), source=_SAMPLE.name)
    except Exception as exc:  # pragma: no cover - startup best-effort
        print(f"[startup] could not ingest {_SAMPLE}: {exc}", file=sys.stderr)


class IngestRequest(BaseModel):
    text: str = Field(
        ...,
        examples=["Metformin is the preferred initial pharmacologic agent for type 2 diabetes unless contraindicated."],
    )
    source: str = Field("inline", examples=["protocol.txt"])


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(
        ..., examples=["What is the first-line therapy for type 2 diabetes?"]
    )
    top_k: int | None = Field(None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    backend = "ollama" if isinstance(_pipeline.llm, OllamaLLM) else "offline"
    return {"status": "ok", "backend": backend, "indexed_chunks": len(_pipeline.store)}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    added = _pipeline.ingest(req.text, req.source)
    return IngestResponse(chunks_added=added, total_chunks=len(_pipeline.store))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = _pipeline.ask(req.question, req.top_k)
    return AskResponse(**result)


# --- Browser UI ------------------------------------------------------------ #
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(_WEB_DIR / "index.html"))
