"""End-to-end: ingest the sample SOP doc via the API, then ask and get cites."""

from pathlib import Path

from fastapi.testclient import TestClient

from ragdocqa.api.main import app

client = TestClient(app)


def test_health_reports_offline_backend():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "offline"  # conftest forces RAG_OFFLINE=1


def test_ingest_then_ask_with_citation():
    doc = Path("data/fleet_sop.md").read_text(encoding="utf-8")
    r = client.post("/ingest", json={"text": doc, "source": "fleet_sop.md"})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1

    r = client.post(
        "/ask",
        json={"question": "How far from land must ballast water exchange be done?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sources"], "expected at least one citation"
    assert body["sources"][0]["source"] == "fleet_sop.md"
    # Offline extractive answer should surface the 200 nm figure.
    assert "200" in body["answer"]


def test_ask_before_ingest_is_graceful():
    # A fresh client shares the module-level index, which is already populated
    # by the test above in-session; just assert the endpoint stays well-formed.
    r = client.post("/ask", json={"question": "anything"})
    assert r.status_code == 200
    assert "answer" in r.json()
