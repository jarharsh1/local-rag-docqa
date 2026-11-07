"""End-to-end CLI demo: ingest the sample knowledge base and ask questions.

    python demo.py

Runs against Ollama if it is up, otherwise the deterministic offline backend
(so it works with no model server). Set RAG_OFFLINE=1 to force offline.
"""

from pathlib import Path

from ragdocqa.core.pipeline import RagPipeline

QUESTIONS = [
    "How long do I have to request a refund?",
    "What is the SLA for priority incidents?",
    "How long is personal data retained?",
]


def main() -> None:
    doc = Path("data/sample_knowledge_base.md").read_text(encoding="utf-8")
    pipe = RagPipeline()
    n = pipe.ingest(doc, source="sample_knowledge_base.md")
    backend = type(pipe.llm).__name__
    print(f"Ingested {n} chunks (backend: {backend})\n")

    for q in QUESTIONS:
        result = pipe.ask(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        top = result["sources"][0] if result["sources"] else None
        if top:
            print(f"   [top source: {top['source']} score={top['score']}]")
        print()


if __name__ == "__main__":
    main()
