"""Unit tests for chunking, the vector store, and the offline backends."""

from ragdocqa.core.backends import ExtractiveLLM, HashingEmbedder
from ragdocqa.core.chunking import chunk_text
from ragdocqa.core.vector_store import VectorStore, cosine


def test_chunking_overlap_and_coverage():
    text = "a. " * 500  # ~1500 chars
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 220 for c in chunks)  # allow boundary slack


def test_chunking_empty():
    assert chunk_text("   ", 100, 10) == []


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder()
    a1 = emb.embed(["refund within 30 days of purchase"])[0]
    a2 = emb.embed(["refund within 30 days of purchase"])[0]
    assert a1 == a2
    # L2 norm ~ 1 for non-empty text
    assert abs(sum(x * x for x in a1) - 1.0) < 1e-6


def test_similar_text_scores_higher():
    emb = HashingEmbedder()
    q = emb.embed(["how long to request a refund"])[0]
    related = emb.embed(["a customer may request a refund within 30 days"])[0]
    unrelated = emb.embed(["priority incidents are acknowledged within one hour"])[0]
    assert cosine(q, related) > cosine(q, unrelated)


def test_vector_store_search_orders_by_similarity():
    emb = HashingEmbedder()
    store = VectorStore()
    texts = ["refund requests are accepted within 30 days", "invoices are payable within fourteen days"]
    store.add(texts, emb.embed(texts), source="t.md")
    q = emb.embed(["how long to request a refund"])[0]
    hits = store.search(q, top_k=2)
    assert hits[0][0].text == "refund requests are accepted within 30 days"


def test_extractive_llm_uses_context():
    llm = ExtractiveLLM()
    prompt = (
        "CONTEXT:\n[1] A customer may request a refund within 30 days of "
        "purchase. [2] Digital goods are non-refundable once downloaded.\n\n"
        "QUESTION:\nHow long do I have to request a refund?"
    )
    answer = llm.generate(prompt)
    assert "30 days" in answer.lower()
