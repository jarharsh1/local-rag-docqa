"""FastAPI serving layer for the local RAG document-QA service.

Endpoints:
  GET  /health   - liveness + which backend (ollama/offline) is active
  POST /ingest   - add a document (raw text) to the index
  POST /ask      - ask a question; returns a grounded answer + citations
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ragdocqa.core.backends import OllamaLLM
from ragdocqa.core.pipeline import RagPipeline

app = FastAPI(
    title="Maritime Compliance & SOP Assistant (Local RAG)",
    version="0.1.0",
    description="Ollama-first retrieval-augmented QA over fleet operating manuals, "
    "safety regulations and port procedures — a Bahri-style maritime use case.",
)

# One in-process pipeline/index for the app's lifetime.
_pipeline = RagPipeline()


class IngestRequest(BaseModel):
    text: str = Field(
        ...,
        examples=["Before any cargo transfer, the officer on watch must confirm the ship-shore safety checklist is complete and signed by both parties."],
    )
    source: str = Field("inline", examples=["ops-manual-cargo-transfer.txt"])


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(
        ..., examples=["What must be checked before a cargo transfer begins?"]
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
