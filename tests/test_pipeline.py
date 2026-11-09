"""Regression tests for the offline pipeline's ask() behavior."""

from ragdocqa.core.pipeline import RagPipeline


def _pipeline_with_corpus() -> RagPipeline:
    p = RagPipeline()
    p.reset()  # drop anything a shared backend/store might carry
    p.ingest(
        "A customer may request a refund within 30 days of purchase. "
        "Refunds are issued to the original payment method within five business days. "
        "Priority incidents affecting production are acknowledged within one hour. "
        "Personal data is retained for twenty-four months after the last account activity. "
        "Passwords must be at least twelve characters and stored as salted hashes.",
        source="kb.md",
    )
    return p


def test_top_k_zero_is_honoured_not_defaulted():
    # top_k=0 is falsy; `top_k or settings.top_k` wrongly fell back to the default.
    # With the fix, requesting 0 results returns an empty citation list.
    p = _pipeline_with_corpus()
    result = p.ask("How long do I have to request a refund?", top_k=0)
    assert result["sources"] == []


def test_offline_unanswerable_question_cites_nothing():
    # The hashed embedder's collision noise floor makes the store return chunks
    # even for a question with no shared vocabulary; the guard must suppress them.
    p = _pipeline_with_corpus()
    result = p.ask("xyzzy quux zzz unmatchedword")
    assert result["sources"] == []
    assert "do not appear to answer" in result["answer"].lower()


def test_offline_answerable_question_still_cites():
    p = _pipeline_with_corpus()
    result = p.ask("How long do I have to request a refund?")
    assert result["sources"], "an answerable question must still return citations"
    assert "30 days" in result["answer"].lower()
