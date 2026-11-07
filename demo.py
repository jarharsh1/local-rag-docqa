"""End-to-end CLI demo: ingest the sample SOP doc and answer a few questions.

    python demo.py

Runs against Ollama if it is up, otherwise the deterministic offline backend
(so it works with no model server). Set RAG_OFFLINE=1 to force offline.
"""

from pathlib import Path

from ragdocqa.core.pipeline import RagPipeline

QUESTIONS = [
    "What must be checked before a cargo transfer begins?",
    "How far from land must ballast water exchange be carried out?",
    "What is required for chemical cargoes sensitive to oxygen?",
]


def main() -> None:
    doc = Path("data/fleet_sop.md").read_text(encoding="utf-8")
    pipe = RagPipeline()
    n = pipe.ingest(doc, source="fleet_sop.md")
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
