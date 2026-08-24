"""Unit tests for Phase 4 of user preferences: the RAG retriever.

Covers the deterministic pieces only — mocked vector engine and preference
lookup, no graph database, no LLM:
- Flag off (empty lookup): the vector search is issued with ``limit=top_k``
  and the returned order is untouched — byte-identical to today.
- Weights matching the chunk collection: ``limit == wide_search_top_k`` (the
  search path's one "fetch extra candidates" knob), a stable re-sort by
  ``score * personal_factor(w, influence, distance_space=True)``, a trim back
  to ``top_k``, and — the assertion that separates a real re-rank from a
  reorder — a chunk outside the raw top_k entering the final list (Demo 4b).
- An empty weights dict short-circuits: no wide fetch, no sort.
- Weights that only point at graph entities (no key in the chunk collection)
  do not widen the fetch: the extra work could never change membership.
- The sessionless completion passes the preference text as
  ``conversation_history``.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

import cognee.modules.retrieval.completion_retriever as retriever_module
from cognee.modules.retrieval.completion_retriever import (
    CompletionRetriever,
    _stable_sort_by_personal_distance,
)
from cognee.modules.user_preferences import personal_factor

INFLUENCE = 0.3
WIDE_SEARCH_TOP_K = 6


def _chunk(chunk_id: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        score=score,
        payload={"id": chunk_id, "text": f"text of {chunk_id}"},
    )


class FakeVectorEngine:
    """Records search kwargs and serves the first ``limit`` seeded chunks."""

    def __init__(self, chunks: List[Any]):
        self.chunks = chunks
        self.search_calls: List[Dict[str, Any]] = []
        self.retrieve_calls: List[Dict[str, Any]] = []

    async def search(self, collection_name, query, **kwargs):
        self.search_calls.append({"collection_name": collection_name, **kwargs})
        return self.chunks[: kwargs["limit"]]

    async def retrieve(self, collection_name, data_point_ids):
        self.retrieve_calls.append(
            {"collection_name": collection_name, "ids": list(data_point_ids)}
        )
        ids = {str(data_point_id) for data_point_id in data_point_ids}
        return [chunk for chunk in self.chunks if str(chunk.payload.get("id", chunk.id)) in ids]


def _patch_lookup(monkeypatch, result: Tuple[str, Dict[str, float]]):
    text, weights = result

    async def fake_load_preference_text():
        return text

    async def fake_load_preference_weights():
        return weights

    monkeypatch.setattr(retriever_module, "load_preference_text", fake_load_preference_text)
    monkeypatch.setattr(retriever_module, "load_preference_weights", fake_load_preference_weights)


def _patch_engine(monkeypatch, chunks: List[Any]) -> FakeVectorEngine:
    engine = FakeVectorEngine(chunks)

    async def fake_get_vector_engine_async():
        return engine

    monkeypatch.setattr(retriever_module, "get_vector_engine_async", fake_get_vector_engine_async)
    return engine


def _patch_influence(monkeypatch, influence: float = INFLUENCE):
    config = SimpleNamespace(personalization_influence=influence)
    monkeypatch.setattr(retriever_module, "get_base_config", lambda: config)


def _patch_hybrid(monkeypatch, chunks):
    calls = []

    class FakeHybridRetriever:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        async def get_retrieved_objects(self, query):
            return chunks

    monkeypatch.setattr(retriever_module, "HybridChunksRetriever", FakeHybridRetriever)
    return calls


@pytest.mark.asyncio
class TestGetRetrievedObjects:
    async def test_flag_off_keeps_production_hybrid_pool_and_order(self, monkeypatch):
        chunks = [_chunk("a", 0.1), _chunk("b", 0.2), _chunk("c", 0.3)]
        calls = _patch_hybrid(monkeypatch, chunks)
        _patch_lookup(monkeypatch, ("", {}))
        _patch_influence(monkeypatch)

        result = await CompletionRetriever(top_k=3).get_retrieved_objects("query")

        assert result == chunks
        assert calls[0]["top_k"] == 3
        assert calls[0]["candidate_pool_size"] == 30
        assert calls[0]["preference_weights"] == {}
        assert calls[0]["personalization_influence"] == 0.0

    async def test_empty_weights_dict_short_circuits_even_with_text(self, monkeypatch):
        chunks = [_chunk("a", 0.1), _chunk("b", 0.2)]
        calls = _patch_hybrid(monkeypatch, chunks)
        _patch_lookup(monkeypatch, ("## What this user prefers\nconcise answers", {}))
        _patch_influence(monkeypatch)

        result = await CompletionRetriever(top_k=2).get_retrieved_objects("query")

        assert result == chunks
        assert calls[0]["candidate_pool_size"] == 30
        assert calls[0]["preference_weights"] == {}

    async def test_weights_widen_hybrid_pool_and_forward_personalization(self, monkeypatch):
        chunks = [_chunk("a", 0.10), _chunk("e", 0.25)]
        engine = _patch_engine(monkeypatch, chunks)
        calls = _patch_hybrid(monkeypatch, chunks)
        _patch_lookup(monkeypatch, ("", {"e": 0.95}))
        _patch_influence(monkeypatch)

        retriever = CompletionRetriever(
            top_k=2,
            wide_search_top_k=WIDE_SEARCH_TOP_K,
            candidate_pool_size=2,
        )
        result = await retriever.get_retrieved_objects("query")

        assert result == chunks
        assert engine.retrieve_calls[0]["collection_name"] == "DocumentChunk_text"
        assert calls[0]["candidate_pool_size"] == WIDE_SEARCH_TOP_K
        assert calls[0]["preference_weights"] == {"e": 0.95}
        assert calls[0]["personalization_influence"] == INFLUENCE

    async def test_entity_only_weights_do_not_widen_the_pool(self, monkeypatch):
        chunks = [_chunk("a", 0.1), _chunk("b", 0.2)]
        engine = _patch_engine(monkeypatch, chunks)
        calls = _patch_hybrid(monkeypatch, chunks)
        _patch_lookup(monkeypatch, ("", {"entity-1": 0.95, "entity-2": 0.1}))
        _patch_influence(monkeypatch)

        retriever = CompletionRetriever(
            top_k=2,
            wide_search_top_k=WIDE_SEARCH_TOP_K,
            candidate_pool_size=2,
        )
        result = await retriever.get_retrieved_objects("query")

        assert result == chunks
        assert len(engine.retrieve_calls) == 1
        assert calls[0]["candidate_pool_size"] == 2
        assert calls[0]["preference_weights"] == {}
        assert calls[0]["personalization_influence"] == 0.0

    async def test_collection_check_failure_fails_open_to_wide_pool(self, monkeypatch):
        chunks = [_chunk("a", 0.10), _chunk("b", 0.11)]
        engine = _patch_engine(monkeypatch, chunks)
        calls = _patch_hybrid(monkeypatch, chunks)

        async def broken_retrieve(collection_name, data_point_ids):
            raise RuntimeError("boom")

        engine.retrieve = broken_retrieve
        _patch_lookup(monkeypatch, ("", {"a": 0.0}))
        _patch_influence(monkeypatch)

        retriever = CompletionRetriever(
            top_k=2,
            wide_search_top_k=WIDE_SEARCH_TOP_K,
            candidate_pool_size=2,
        )
        await retriever.get_retrieved_objects("query")

        assert calls[0]["candidate_pool_size"] == WIDE_SEARCH_TOP_K
        assert calls[0]["preference_weights"] == {"a": 0.0}

    async def test_wide_search_top_k_never_shrinks_below_top_k(self, monkeypatch):
        chunks = [_chunk(f"c{index}", 0.1 * (index + 1)) for index in range(4)]
        _patch_engine(monkeypatch, chunks)
        calls = _patch_hybrid(monkeypatch, chunks)
        _patch_lookup(monkeypatch, ("", {"c0": 0.9}))
        _patch_influence(monkeypatch)

        retriever = CompletionRetriever(
            top_k=4,
            wide_search_top_k=2,
            candidate_pool_size=1,
        )
        await retriever.get_retrieved_objects("query")

        assert calls[0]["candidate_pool_size"] == 4


class TestStableSort:
    def test_unmatched_chunks_keep_engine_order(self):
        chunks = [_chunk("a", 0.2), _chunk("b", 0.2), _chunk("c", 0.2)]
        result = _stable_sort_by_personal_distance(chunks, {"zzz": 0.9}, INFLUENCE)
        assert result == chunks

    def test_neutral_weight_is_an_exact_no_op(self):
        chunks = [_chunk("a", 0.2), _chunk("b", 0.2)]
        result = _stable_sort_by_personal_distance(chunks, {"a": 0.5, "b": 0.5}, INFLUENCE)
        assert result == chunks


@pytest.mark.asyncio
class TestSessionlessGuidance:
    async def test_preference_text_passed_as_conversation_history(self, monkeypatch):
        _patch_lookup(monkeypatch, ("PREFS", {}))
        captured = {}

        async def fake_generate_completion(**kwargs):
            captured.update(kwargs)
            return "answer"

        monkeypatch.setattr(retriever_module, "generate_completion", fake_generate_completion)

        retriever = CompletionRetriever(top_k=2)
        result = await retriever._generate_completion_without_session("q", "ctx")

        assert result == ["answer"]
        assert captured["conversation_history"] == "PREFS"
        assert captured["context"] == "ctx"

    async def test_empty_preference_text_passes_falsy_history(self, monkeypatch):
        _patch_lookup(monkeypatch, ("", {}))
        captured = {}

        async def fake_generate_completion(**kwargs):
            captured.update(kwargs)
            return "answer"

        monkeypatch.setattr(retriever_module, "generate_completion", fake_generate_completion)

        retriever = CompletionRetriever(top_k=2)
        await retriever._generate_completion_without_session("q", "ctx")

        # generate_completion treats a falsy history as "no layer", so the
        # system prompt stays byte-identical to the un-personalized path.
        assert captured["conversation_history"] == ""
