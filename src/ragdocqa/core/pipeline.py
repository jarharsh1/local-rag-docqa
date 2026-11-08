"""The RAG pipeline: ingest documents, then answer questions with citations."""

from __future__ import annotations

from ragdocqa.config import settings
from ragdocqa.core.backends import get_backends
from ragdocqa.core.chunking import chunk_text
from ragdocqa.core.vector_store import VectorStore

_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the
context below. If the context does not contain the answer, say you don't know.
Cite nothing inline; a separate citation list is added by the caller.

CONTEXT:
{context}

QUESTION:
{question}
"""


class RagPipeline:
    def __init__(self, store: VectorStore | None = None):
        self.embedder, self.llm = get_backends()
        self.store = store if store is not None else VectorStore()

    def ingest(self, text: str, source: str) -> int:
        """Chunk + embed + store a document. Returns the number of chunks added."""
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return 0
        embeddings = self.embedder.embed(chunks)
        self.store.add(chunks, embeddings, source)
        return len(chunks)

    def reset(self) -> None:
        """Drop all ingested records. The shared in-process store is otherwise
        append-only, so this lets callers reclaim it between corpora/sessions."""
        self.store.clear()

    def ask(self, question: str, top_k: int | None = None) -> dict:
        """Retrieve relevant chunks and generate a grounded answer with sources."""
        if len(self.store) == 0:
            return {"answer": "No documents have been ingested yet.", "sources": []}

        k = top_k or settings.top_k
        q_emb = self.embedder.embed([question])[0]
        hits = self.store.search(q_emb, k)

        context = "\n\n".join(f"[{i + 1}] {rec.text}" for i, (rec, _) in enumerate(hits))
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.llm.generate(prompt)

        sources = [
            {
                "rank": i + 1,
                "source": rec.source,
                "score": round(score, 4),
                "excerpt": rec.text[:200],
            }
            for i, (rec, score) in enumerate(hits)
        ]
        return {"answer": answer, "sources": sources}
