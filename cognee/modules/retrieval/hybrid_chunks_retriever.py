"""Hybrid lexical/vector retrieval with canonical-page-aware selection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.utils.chunk_metadata import (
    canonical_page_key,
    normalize_search_text,
    parse_json_front_matter,
    split_json_front_matter,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridChunksRetriever")

DEFAULT_MAX_CONTEXT_CHARS = 15_000
DEFAULT_CANDIDATE_POOL = 30
RRF_CONSTANT = 60


@dataclass
class _Candidate:
    identity: str
    payload: dict[str, Any]
    vector_rank: int | None = None
    lexical_rank: int | None = None


def fuse_chunk_results(
    query: str,
    vector_results: Any,
    lexical_results: Any,
    *,
    top_k: int,
    page_limit: int | None = None,
    max_chunks_per_page: int = 1,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> list[ScoredResult]:
    """Fuse ranked channels, then select diverse pages and bounded context."""
    if top_k < 1 or max_chunks_per_page < 1 or max_context_chars < 1:
        raise ValueError("hybrid retrieval limits must be positive")
    page_limit = min(top_k, page_limit if page_limit is not None else top_k)
    candidates: dict[str, _Candidate] = {}

    for rank, result in enumerate(vector_results or (), start=1):
        payload = getattr(result, "payload", None)
        if not isinstance(payload, dict):
            continue
        identity = _candidate_identity(result, payload)
        candidates[identity] = _Candidate(identity, payload, vector_rank=rank)

    for rank, value in enumerate(lexical_results or (), start=1):
        if not isinstance(value, tuple) or len(value) != 2:
            continue
        payload, lexical_score = value
        if not isinstance(payload, dict) or not isinstance(lexical_score, (int, float)):
            continue
        if lexical_score <= 0:
            continue
        identity = _candidate_identity(payload, payload)
        candidate = candidates.get(identity)
        if candidate is None:
            candidate = _Candidate(identity, payload)
            candidates[identity] = candidate
        candidate.lexical_rank = rank

    ranked = sorted(
        (
            (_hybrid_relevance(query, candidate), candidate)
            for candidate in candidates.values()
            if not _historical_body(candidate.payload)
        ),
        key=lambda value: (-value[0], value[1].identity),
    )

    selected: list[tuple[float, _Candidate]] = []
    selected_pages: set[str] = set()
    page_counts: dict[str, int] = {}
    for relevance, candidate in ranked:
        page = canonical_page_key(candidate.payload, candidate.identity)
        if page in selected_pages:
            continue
        selected.append((relevance, candidate))
        selected_pages.add(page)
        page_counts[page] = 1
        if len(selected_pages) >= page_limit or len(selected) >= top_k:
            break

    if len(selected) < top_k and max_chunks_per_page > 1:
        selected_ids = {candidate.identity for _, candidate in selected}
        for relevance, candidate in ranked:
            if candidate.identity in selected_ids:
                continue
            page = canonical_page_key(candidate.payload, candidate.identity)
            if page not in selected_pages or page_counts[page] >= max_chunks_per_page:
                continue
            selected.append((relevance, candidate))
            selected_ids.add(candidate.identity)
            page_counts[page] += 1
            if len(selected) >= top_k:
                break

    return _bound_results(selected, max_context_chars=max_context_chars)


class HybridChunksRetriever(ChunksRetriever):
    """Retrieve page-diverse chunks from semantic and lexical channels."""

    def __init__(
        self,
        top_k: Optional[int] = 5,
        node_name: Optional[list[str]] = None,
        node_name_filter_operator: str = "OR",
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL,
        page_limit: int | None = None,
        max_chunks_per_page: int = 1,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ):
        super().__init__(
            top_k=top_k, node_name=node_name, node_name_filter_operator=node_name_filter_operator
        )
        self.candidate_pool_size = max(candidate_pool_size, self.top_k or 1)
        self.page_limit = page_limit
        self.max_chunks_per_page = max_chunks_per_page
        self.max_context_chars = max_context_chars

    async def get_retrieved_objects(self, query: str) -> Any:
        unified = await get_unified_engine()
        vector_search = unified.vector.search(
            "DocumentChunk_text",
            query,
            limit=self.candidate_pool_size,
            include_payload=True,
            node_name=self.node_name,
            node_name_filter_operator=self.node_name_filter_operator,
        )
        lexical_retriever = BM25ChunksRetriever(
            top_k=self.candidate_pool_size,
            with_scores=True,
            node_name=self.node_name,
            node_name_filter_operator=self.node_name_filter_operator,
        )
        vector_result, lexical_result = await asyncio.gather(
            vector_search,
            lexical_retriever.get_retrieved_objects(query),
            return_exceptions=True,
        )
        if isinstance(vector_result, CollectionNotFoundError):
            vector_result = []
        elif isinstance(vector_result, Exception):
            logger.warning("Vector candidate retrieval failed: %s", vector_result)
            vector_result = []
        if isinstance(lexical_result, Exception):
            logger.warning("Lexical candidate retrieval failed: %s", lexical_result)
            lexical_result = []
        if not vector_result and not lexical_result:
            raise NoDataError("No document chunks were available for hybrid retrieval.")
        return fuse_chunk_results(
            query,
            vector_result,
            lexical_result,
            top_k=self.top_k or 1,
            page_limit=self.page_limit,
            max_chunks_per_page=self.max_chunks_per_page,
            max_context_chars=self.max_context_chars,
        )


def _candidate_identity(result: Any, payload: dict[str, Any]) -> str:
    value = getattr(result, "id", None)
    if value is None:
        value = payload.get("id")
    if value is None:
        metadata = parse_json_front_matter(payload.get("text"))
        value = metadata.get("document_id")
    return str(value) if value is not None else canonical_page_key(payload, repr(payload))


def _hybrid_relevance(query: str, candidate: _Candidate) -> float:
    relevance = 0.0
    if candidate.vector_rank is not None:
        relevance += 1 / (RRF_CONSTANT + candidate.vector_rank)
    if candidate.lexical_rank is not None:
        relevance += 1 / (RRF_CONSTANT + candidate.lexical_rank)

    metadata, answer_body = split_json_front_matter(candidate.payload.get("text"))
    query_normalized = normalize_search_text(query)
    query_tokens = set(query_normalized.split())
    aliases = metadata.get("aliases")
    names = [metadata.get("title"), *(aliases if isinstance(aliases, list) else [])]
    normalized_names = [normalize_search_text(name) for name in names]
    if query_normalized and query_normalized in normalized_names:
        relevance += 0.12
    elif any(set(name.split()) <= query_tokens for name in normalized_names if name):
        relevance += 0.06

    section_path = metadata.get("section_path")
    if isinstance(section_path, list):
        section_tokens = set(normalize_search_text(" ".join(map(str, section_path))).split())
        relevance += min(3, len(query_tokens & section_tokens)) * 0.015
    answer_tokens = set(normalize_search_text(answer_body).split())
    # A tiny long-term overlap tie-breaker helps choose an answer-bearing chunk
    # within an already-ranked page without overriding title/section/status policy.
    specific_query_tokens = {token for token in query_tokens if len(token) >= 7}
    relevance += min(3, len(specific_query_tokens & answer_tokens)) * 0.003
    if _status_stub(candidate.payload):
        relevance -= 0.04
    return relevance


def _status_stub(payload: dict[str, Any]) -> bool:
    metadata = parse_json_front_matter(payload.get("text"))
    return metadata.get("chunk_kind") == "status" or metadata.get("content_excluded") is True


def _historical_body(payload: dict[str, Any]) -> bool:
    metadata = parse_json_front_matter(payload.get("text"))
    return metadata.get("active") is False and not _status_stub(payload)


def _bound_results(
    selected: list[tuple[float, _Candidate]],
    *,
    max_context_chars: int,
) -> list[ScoredResult]:
    bounded: list[ScoredResult] = []
    used = 0
    for relevance, candidate in selected:
        payload = dict(candidate.payload)
        text = payload.get("text")
        if not isinstance(text, str):
            continue
        separator = 1 if bounded else 0
        remaining = max_context_chars - used - separator
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
            payload["text"] = text
        used += separator + len(text)
        bounded.append(
            ScoredResult(
                id=_result_uuid(candidate.identity),
                score=-relevance,
                payload=payload,
            )
        )
    return bounded


def _result_uuid(identity: str) -> UUID:
    try:
        return UUID(identity)
    except (TypeError, ValueError, AttributeError):
        return uuid5(NAMESPACE_URL, identity)
