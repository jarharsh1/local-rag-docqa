# Code Review — `local-rag-docqa`

A reviewer-facing deep-dive on the correctness and robustness findings fixed in
the second review round, with root causes, the actual before/after from each fix
commit, and the regression test that locks the behavior in.

Scope: the offline (CI/test) path — `HashingEmbedder` (hashed bag-of-words) +
`ExtractiveLLM` — which is what the test suite exercises. Fixes are written so
the Ollama-backed path is unchanged.

Commits reviewed: `78cb246`, `f3b0e44`, `b5b15d5`.

---

## 1. Startup pre-ingest silently did nothing off-root — `src/ragdocqa/api/main.py:24`

**Where.** Module load time, `_SAMPLE` definition.

**Root cause.** `_WEB_DIR` was resolved from `__file__`, but the sample corpus
was a **CWD-relative** path:

```python
_WEB_DIR = Path(__file__).resolve().parents[3] / "web"
_SAMPLE = Path("data/sample_knowledge_base.md")   # relative to the process CWD
```

The pre-ingest is guarded by `if _SAMPLE.exists():`, so when the server was
started from anywhere other than the repo root the guard was simply false and
the bundled knowledge base was never ingested — no error, just an empty index
and a UI that answers "No documents have been ingested yet." The inconsistency
with `_WEB_DIR` (which *was* anchored to `__file__`) is the tell.

**Fix (`78cb246`).** Anchor both to a single package-relative root:

```python
_ROOT = Path(__file__).resolve().parents[3]
_WEB_DIR = _ROOT / "web"
_SAMPLE = _ROOT / "data" / "sample_knowledge_base.md"
```

**Regression test** (`tests/test_api_e2e.py`): asserts the path is CWD-independent.

```python
def test_sample_path_is_absolute_and_cwd_independent():
    assert api_main._SAMPLE.is_absolute()
    assert api_main._SAMPLE.exists()
```

---

## 2. `top_k=0` falsy-zero fallback — `src/ragdocqa/core/pipeline.py` (`ask`)

**Where.** `RagPipeline.ask`, the `k = ...` line.

**Root cause.** `top_k or settings.top_k` collapses the caller's explicit
request for **zero** results into the default (4), because `0` is falsy:

```python
k = top_k or settings.top_k     # top_k=0 → falls through to settings.top_k
```

The API clamps `top_k` to `ge=1`, so this wasn't reachable over HTTP, but the
library API (`pipe.ask(q, top_k=0)`) and any future caller hit it.

**Fix (`f3b0e44`).** Treat `None` (the real "unset" sentinel) explicitly:

```python
k = settings.top_k if top_k is None else top_k
```

**Regression test** (`tests/test_pipeline.py`): fails on the old code (which
returned 4 sources), passes now.

```python
def test_top_k_zero_is_honoured_not_defaulted():
    p = _pipeline_with_corpus()
    result = p.ask("How long do I have to request a refund?", top_k=0)
    assert result["sources"] == []
```

---

## 3. Offline noise-floor chunks were cited for unanswerable questions — `src/ragdocqa/core/pipeline.py` (`ask`)

**Where.** `RagPipeline.ask`, between retrieval and prompt assembly.

**Root cause.** `HashingEmbedder` hashes tokens into a fixed 256-dim space.
Unrelated texts therefore collide and never produce a clean-zero cosine — the
store returns `top_k` chunks for *any* query, including pure gibberish. The
pipeline then cited those chunks even though the `ExtractiveLLM` could not
actually answer. The citations looked authoritative but were noise.

**Fix (`f3b0e44`).** A content-word overlap guard, applied **only** when the
offline embedder is in use:

```python
if isinstance(self.embedder, HashingEmbedder) and not _has_lexical_overlap(
    question, [rec.text for rec, _ in hits]
):
    return {
        "answer": "The documents do not appear to answer that question.",
        "sources": [],
    }
```

with the helper (stopword- and length-filtered so function words don't create
false overlap):

```python
def _content_words(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in _STOP and len(t) > 2}

def _has_lexical_overlap(question: str, texts: list[str]) -> bool:
    q = _content_words(question)
    if not q:
        return False
    return any(q & _content_words(t) for t in texts)
```

**Design trade-off (deliberate).** The guard is gated on
`isinstance(self.embedder, HashingEmbedder)`. A real (Ollama) embedder can match
a paraphrase that shares **no** surface words with the question — exactly the
case a lexical guard would wrongly suppress — so the guard is offline-only. The
cost is that the two paths now differ in one branch; the alternative (always
guarding) would degrade genuine semantic retrieval, which is worse.

**Regression tests** (`tests/test_pipeline.py`): one for suppression, one to
prove answerable questions are unaffected.

```python
def test_offline_unanswerable_question_cites_nothing():
    result = _pipeline_with_corpus().ask("xyzzy quux zzz unmatchedword")
    assert result["sources"] == []
    assert "do not appear to answer" in result["answer"].lower()

def test_offline_answerable_question_still_cites():
    result = _pipeline_with_corpus().ask("How long do I have to request a refund?")
    assert result["sources"]
    assert "30 days" in result["answer"].lower()
```

---

## 4. Full sort where a partial select suffices — `src/ragdocqa/core/vector_store.py:46`

**Where.** `VectorStore.search`.

**Root cause.** The store scored every record then `sort`ed the whole list
(`O(n log n)`) only to slice the top `k`.

**Fix (`b5b15d5`).** `heapq.nlargest` is `O(n log k)` and returns the identical
ordering (including tie-breaking) as `sorted(..., reverse=True)[:k]`:

```python
return heapq.nlargest(top_k, scored, key=lambda t: t[1])
```

No behavioral change — a pure efficiency fix — so it is covered by the existing
ranking tests rather than a new one.

---

## 5. 2-second Ollama probe on every pipeline construction — `src/ragdocqa/core/backends.py` (`get_backends`)

**Where.** `get_backends`, called from every `RagPipeline.__init__`.

**Root cause.** Each `RagPipeline()` ran `_ollama_up()`, a live HTTP probe with
a 2s timeout. Constructing several pipelines (or doing so per request) paid that
latency repeatedly, even though the selection is fixed for the process lifetime.

**Fix (`b5b15d5`).** Memoize the selection:

```python
@lru_cache(maxsize=1)
def get_backends() -> tuple[Embedder, LLM]:
    ...
```

**Design trade-off (deliberate).** The cache key is empty because selection
depends only on process-lifetime settings, and the returned backends are
stateless and safe to share. The documented caveat: a test that flips
`RAG_OFFLINE`/`OLLAMA_HOST` mid-process must call `get_backends.cache_clear()`,
or it will see the first selection. This is called out in the docstring.

---

## Verified correct, not changed

- **Chunking** (`chunking.py`) covers the whole source with correct overlap and
  cannot infinite-loop (`start` always advances via `max(end - overlap, start+1)`);
  empty input returns `[]`, and `overlap >= chunk_size` raises. A coverage test
  added in the prior round locates each chunk in source order and asserts no gap.
- **Cosine** guards zero-norm vectors and raises on dimension mismatch.
- **`ExtractiveLLM`** parses the `CONTEXT:`/`QUESTION:` prompt correctly and
  degrades to a clear fallback string when no sentence overlaps the question.
- **`/reset`** and `VectorStore.clear()` (prior round) remain the supported way
  to reclaim the append-only in-process store.
