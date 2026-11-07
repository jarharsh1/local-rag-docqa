"""Runtime configuration, read from environment with sensible local defaults.

The service is Ollama-first but degrades to a deterministic offline backend
when Ollama is unreachable (e.g. in CI), so tests run without a model server.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "llama3.2")
    embed_model: str = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
    # Force the offline backend regardless of Ollama availability (used in CI).
    offline: bool = os.getenv("RAG_OFFLINE", "0") == "1"
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))


settings = Settings()
