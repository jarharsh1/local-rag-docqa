"""End-to-end: ingest the sample knowledge base via the API, then ask."""

from pathlib import Path

from fastapi.testclient import TestClient

from ragdocqa.api.main import app

client = TestClient(app)


def test_health_reports_offline_backend():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "offline"  # conftest forces RAG_OFFLINE=1


def test_ingest_then_ask_with_citation():
    doc = Path("data/sample_knowledge_base.md").read_text(encoding="utf-8")
    r = client.post("/ingest", json={"text": doc, "source": "sample_knowledge_base.md"})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1

    r = client.post(
        "/ask",
        json={"question": "How long do I have to request a refund?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sources"], "expected at least one citation"
    # Offline extractive answer should surface the 30-day refund window.
    assert "30 days" in body["answer"].lower()


def test_ask_is_well_formed():
    r = client.post("/ask", json={"question": "anything"})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_index_and_static_served():
    assert "RAG Engine" in client.get("/").text
    assert "/ask" in client.get("/static/app.js").text
