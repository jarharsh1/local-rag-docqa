"""A tiny in-memory vector store with cosine similarity.

Kept dependency-free (pure Python) so the service runs anywhere. Each record
holds the chunk text, its embedding, and source metadata for citations.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field


@dataclass
class Record:
    id: int
    text: str
    embedding: list[float]
    source: str
    meta: dict = field(default_factory=dict)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorStore:
    def __init__(self):
        self._records: list[Record] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, texts: list[str], embeddings: list[list[float]], source: str) -> None:
        for text, emb in zip(texts, embeddings):
            self._records.append(
                Record(id=len(self._records), text=text, embedding=emb, source=source)
            )

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Record, float]]:
        scored = [(r, cosine(query_embedding, r.embedding)) for r in self._records]
        # nlargest is O(n log k) vs a full O(n log n) sort; equivalent result and
        # tie-ordering to sorted(..., reverse=True)[:top_k].
        return heapq.nlargest(top_k, scored, key=lambda t: t[1])

    def clear(self) -> None:
        self._records.clear()
