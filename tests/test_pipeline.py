"""Pipeline control-flow tests with all external services mocked."""
import pytest
from langchain_core.documents import Document

import main_pipeline as mp

USER = {"user_id": "u1", "clearance_group": "cardiology"}


class FakeEmbeddings:
    def embed_query(self, q):
        return [0.1] * 8


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_by_vector_with_score(self, vec, k, filter):
        self.calls.append(filter)
        return self.results


class FakeChain:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        assert "<context>" not in payload["question"]
        return self.answer


@pytest.fixture
def wire(monkeypatch):
    stored = []

    def _wire(results, answer="Labetalol 20 mg IV [S1]", hit=None):
        store, chain = FakeStore(results), FakeChain(answer)
        monkeypatch.setattr(mp, "get_embeddings", lambda: FakeEmbeddings())
        monkeypatch.setattr(mp, "get_vector_store", lambda: store)
        monkeypatch.setattr(mp, "_generation_chain", lambda: chain)
        monkeypatch.setattr(mp.cache, "lookup", lambda emb, grp: hit)
        monkeypatch.setattr(mp.cache, "store", lambda *a, **k: stored.append(a))
        return store, chain, stored

    return _wire


def doc(group="cardiology", text="Labetalol 20 mg IV over 2 minutes."):
    return Document(page_content=text, metadata={
        "clearance_group": group, "chunk_id": f"{group}#g#p1#c0", "source_document": "g.pdf", "page_number": 1,
    })


def test_happy_path_returns_sources_and_contexts_and_caches(wire):
    store, chain, stored = wire([(doc(), 0.9)])
    r = mp.execute_clinical_query(USER, "adult hypertensive crisis dose?")
    assert r.grounded and not r.cached
    assert r.sources[0]["source_document"] == "g.pdf"
    assert r.contexts == ["Labetalol 20 mg IV over 2 minutes."]
    assert store.calls == [{"clearance_group": {"$eq": "cardiology"}}]
    assert len(stored) == 1


def test_no_relevant_docs_skips_llm_and_cache(wire):
    _, chain, stored = wire([(doc(), 0.05)])  # below RETRIEVAL_MIN_SCORE
    r = mp.execute_clinical_query(USER, "q?")
    assert not r.grounded and r.answer == mp.INSUFFICIENT_CONTEXT_ANSWER
    assert chain.calls == 0 and stored == []


def test_cross_tenant_docs_are_dropped_even_if_filter_fails(wire):
    _, chain, _ = wire([(doc(group="oncology"), 0.99)])
    r = mp.execute_clinical_query(USER, "q?")
    assert chain.calls == 0 and not r.grounded


def test_refusals_are_not_cached(wire):
    _, _, stored = wire([(doc(), 0.9)], answer=mp.INSUFFICIENT_CONTEXT_ANSWER)
    r = mp.execute_clinical_query(USER, "q?")
    assert not r.grounded and r.sources == [] and stored == []


def test_cache_hit_short_circuits(wire):
    from cache import CachedAnswer
    store, chain, _ = wire([], hit=CachedAnswer(answer="cached", sources=[{"id": "S1"}]))
    r = mp.execute_clinical_query(USER, "q?")
    assert r.cached and r.answer == "cached"
    assert chain.calls == 0 and store.calls == []


def test_use_cache_false_bypasses_cache(wire, monkeypatch):
    _, _, stored = wire([(doc(), 0.9)])
    monkeypatch.setattr(mp.cache, "lookup", lambda *a: pytest.fail("cache must be bypassed"))
    mp.execute_clinical_query(USER, "q?", use_cache=False)
    assert stored == []


@pytest.mark.parametrize("profile", [{}, {"clearance_group": ""}, {"clearance_group": "radiology"},
                                     {"clearance_group": "cardiology)|(@x"}])
def test_authorisation_denied(profile, wire):
    wire([])
    with pytest.raises((PermissionError, ValueError)):
        mp.execute_clinical_query(profile, "q?")


@pytest.mark.parametrize("q", ["", "   ", "x" * 5000])
def test_question_validation(q, wire):
    wire([])
    with pytest.raises(ValueError):
        mp.execute_clinical_query(USER, q)
