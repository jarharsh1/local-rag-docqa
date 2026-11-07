"""Pluggable embedding + LLM backends.

Ollama is the intended runtime. When it is unreachable (or RAG_OFFLINE=1), the
service falls back to a deterministic offline backend so the whole pipeline —
and the test suite — runs with no external model server.

Offline embedder: a fixed-dimension hashed bag-of-words vector (character- and
token-hashing), L2-normalised. It has no semantic understanding but is stable
and good enough to rank lexically-overlapping chunks, which keeps retrieval
meaningful in tests.

Offline LLM: an extractive responder that stitches the most relevant retrieved
sentences into a grounded answer, so /ask returns something sensible offline.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

from ragdocqa.config import settings

_EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


# --------------------------------------------------------------------------- #
# Offline (deterministic) backends
# --------------------------------------------------------------------------- #
class HashingEmbedder:
    """Deterministic hashed bag-of-words embedder (no model required)."""

    dim = _EMBED_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


class ExtractiveLLM:
    """Offline answerer: returns the context sentences most relevant to the
    question. The prompt carries the retrieved context under a CONTEXT header
    and the question under QUESTION; we parse those back out here."""

    def generate(self, prompt: str) -> str:
        context, question = _split_prompt(prompt)
        sentences = _sentences(context)
        if not sentences:
            return "I don't have enough context to answer that."
        q_tokens = set(_tokenize(question))
        scored = sorted(
            sentences,
            key=lambda s: len(q_tokens & set(_tokenize(s))),
            reverse=True,
        )
        top = [s for s in scored[:2] if q_tokens & set(_tokenize(s))]
        if not top:
            return "The documents do not appear to answer that question."
        return " ".join(top)


def _split_prompt(prompt: str) -> tuple[str, str]:
    context, question = "", prompt
    if "CONTEXT:" in prompt and "QUESTION:" in prompt:
        context = prompt.split("CONTEXT:", 1)[1].split("QUESTION:", 1)[0].strip()
        question = prompt.split("QUESTION:", 1)[1].strip()
    return context, question


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# Ollama backends
# --------------------------------------------------------------------------- #
class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        with httpx.Client(timeout=60) as client:
            for t in texts:
                r = client.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                )
                r.raise_for_status()
                out.append(r.json()["embedding"])
        return out


class OllamaLLM:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> str:
        with httpx.Client(timeout=120) as client:
            r = client.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            return r.json()["response"].strip()


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _ollama_up(host: str) -> bool:
    try:
        with httpx.Client(timeout=2) as client:
            return client.get(f"{host.rstrip('/')}/api/tags").status_code == 200
    except Exception:
        return False


def get_backends() -> tuple[Embedder, LLM]:
    """Return (embedder, llm): Ollama when reachable and not forced offline,
    otherwise the deterministic offline pair."""
    if not settings.offline and _ollama_up(settings.ollama_host):
        return (
            OllamaEmbedder(settings.ollama_host, settings.embed_model),
            OllamaLLM(settings.ollama_host, settings.llm_model),
        )
    return HashingEmbedder(), ExtractiveLLM()
