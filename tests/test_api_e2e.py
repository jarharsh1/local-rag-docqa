"""End-to-end: ingest the sample clinical protocol via the API, then ask."""

from pathlib import Path

from fastapi.testclient import TestClient

from ragdocqa.api.main import app

client = TestClient(app)


def test_health_reports_offline_backend():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "offline"  # conftest forces RAG_OFFLINE=1


def test_ingest_then_ask_with_citation():
    doc = Path("data/clinical_protocol.md").read_text(encoding="utf-8")
    r = client.post("/ingest", json={"text": doc, "source": "clinical_protocol.md"})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1

    r = client.post(
        "/ask",
        json={"question": "What is the first-line pharmacologic agent for type 2 diabetes?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sources"], "expected at least one citation"
    # Offline extractive answer should surface metformin.
    assert "metformin" in body["answer"].lower()


def test_ask_is_well_formed():
    r = client.post("/ask", json={"question": "anything"})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_index_and_static_served():
    assert "Clinical Document QA" in client.get("/").text
    assert "/ask" in client.get("/static/app.js").text
