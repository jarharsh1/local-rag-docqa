"""The RAG pipeline: ingest documents, then answer questions with citations."""

from __future__ import annotations

from ragdocqa.config import settings
from ragdocqa.core.backends import HashingEmbedder, _tokenize, get_backends
from ragdocqa.core.chunking import chunk_text
from ragdocqa.core.vector_store import VectorStore

# Short/function words carry no topical signal; ignore them when deciding
# whether an answer is lexically grounded in the retrieved context.
_STOP = {
    "what", "which", "when", "where", "who", "how", "why", "is", "are", "the",
    "a", "an", "of", "to", "and", "or", "for", "on", "in", "at", "did", "was",
    "were", "do", "does", "with", "that", "this", "it", "any", "i", "have",
}


def _content_words(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in _STOP and len(t) > 2}


def _has_lexical_overlap(question: str, texts: list[str]) -> bool:
    """True if any question content word appears in any retrieved chunk.

    The offline hashed embedder collides tokens into a fixed-dimension space, so
    cosine similarity never scores a clean zero even for unrelated text and the
    store always returns chunks. This guard distinguishes genuine hits from that
    noise floor; it is only applied for the offline embedder (a real embedder can
    match semantically without sharing surface words)."""
    q = _content_words(question)
    if not q:
        return False
    return any(q & _content_words(t) for t in texts)

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

        k = settings.top_k if top_k is None else top_k
        q_emb = self.embedder.embed([question])[0]
        hits = self.store.search(q_emb, k)

        # Offline only: if the question shares no content word with any retrieved
        # chunk, the hits are pure hashed-embedder collision noise — answer
        # honestly rather than citing unrelated chunks. A real embedder can match
        # paraphrases without shared words, so the guard is skipped for it.
        if isinstance(self.embedder, HashingEmbedder) and not _has_lexical_overlap(
            question, [rec.text for rec, _ in hits]
        ):
            return {
                "answer": "The documents do not appear to answer that question.",
                "sources": [],
            }

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
