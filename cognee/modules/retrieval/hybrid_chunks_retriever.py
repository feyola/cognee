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
from cognee.modules.user_preferences import personal_factor
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
    result_id: Any = None
    vector_rank: int | None = None
    lexical_rank: int | None = None


def _personalized_hybrid_relevance(
    query: str,
    candidate: _Candidate,
    preference_weights: dict[str, float],
    influence: float,
) -> float:
    """Apply Cognee preference weights in the hybrid retriever's score space."""
    relevance = _hybrid_relevance(query, candidate)
    chunk_id = candidate.payload.get("id") or candidate.result_id
    if chunk_id is None:
        chunk_id = candidate.identity
    weight = preference_weights.get(str(chunk_id))
    if weight is None:
        return relevance
    return relevance * personal_factor(weight, influence, distance_space=False)


def fuse_chunk_results(
    query: str,
    vector_results: Any,
    lexical_results: Any,
    *,
    top_k: int,
    page_limit: int | None = None,
    max_chunks_per_page: int = 1,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    preference_weights: Optional[dict[str, float]] = None,
    personalization_influence: float = 0.0,
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
        candidates[identity] = _Candidate(
            identity,
            payload,
            result_id=getattr(result, "id", None) or payload.get("id"),
            vector_rank=rank,
        )

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
            candidate = _Candidate(identity, payload, result_id=payload.get("id"))
            candidates[identity] = candidate
        candidate.lexical_rank = rank

    ranked = sorted(
        (
            (
                _personalized_hybrid_relevance(
                    query,
                    candidate,
                    preference_weights or {},
                    personalization_influence,
                ),
                candidate,
            )
            for candidate in candidates.values()
            if not _historical_body(candidate.payload)
        ),
        key=lambda value: (-value[0], value[1].identity),
    )

    selected: list[tuple[float, _Candidate]] = []
    selected_pages: set[str] = set()
    page_counts: dict[str, int] = {}
    # Explicit alias relations describe which source-body terms answer a
    # recognized query phrase. Reserve their distinct pages before generic
    # fused pages so multi-part questions retain every configured support page.
    # The hint is body-gated, so metadata alone cannot reserve a page.
    for relevance, candidate in ranked:
        if _alias_answer_hint_score(query, candidate.payload) <= 0:
            continue
        page = canonical_page_key(candidate.payload, candidate.identity)
        if page in selected_pages:
            continue
        selected.append((relevance, candidate))
        selected_pages.add(page)
        page_counts[page] = 1
        if len(selected_pages) >= page_limit or len(selected) >= top_k:
            break

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
        query_tokens = {token for token in normalize_search_text(query).split() if len(token) >= 4}
        page_query_coverage: dict[str, set[str]] = {}
        for _, candidate in selected:
            page = canonical_page_key(candidate.payload, candidate.identity)
            page_query_coverage.setdefault(page, set()).update(
                _answer_body_tokens(candidate.payload) & query_tokens
            )
        while len(selected) < top_k:
            eligible = [
                (relevance, candidate)
                for relevance, candidate in ranked
                if candidate.identity not in selected_ids
                and (page := canonical_page_key(candidate.payload, candidate.identity))
                in selected_pages
                and page_counts[page] < max_chunks_per_page
            ]
            if not eligible:
                break

            def secondary_score(value: tuple[float, _Candidate]) -> tuple[float, float, str]:
                relevance, candidate = value
                page = canonical_page_key(candidate.payload, candidate.identity)
                novel = _answer_body_tokens(
                    candidate.payload
                ) & query_tokens - page_query_coverage.get(page, set())
                answer_hint = _alias_answer_hint_score(query, candidate.payload)
                section_match = _exact_section_match_score(query, candidate.payload)
                return (
                    relevance + min(3, len(novel)) * 0.012 + answer_hint + section_match,
                    relevance,
                    candidate.identity,
                )

            relevance, candidate = max(eligible, key=secondary_score)
            page = canonical_page_key(candidate.payload, candidate.identity)
            selected.append((relevance, candidate))
            selected_ids.add(candidate.identity)
            page_counts[page] += 1
            page_query_coverage.setdefault(page, set()).update(
                _answer_body_tokens(candidate.payload) & query_tokens
            )

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
        preference_weights: Optional[dict[str, float]] = None,
        personalization_influence: float = 0.0,
    ):
        super().__init__(
            top_k=top_k, node_name=node_name, node_name_filter_operator=node_name_filter_operator
        )
        self.candidate_pool_size = max(candidate_pool_size, self.top_k or 1)
        self.page_limit = page_limit
        self.max_chunks_per_page = max_chunks_per_page
        self.max_context_chars = max_context_chars
        self.preference_weights = preference_weights or {}
        self.personalization_influence = personalization_influence

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
            preference_weights=self.preference_weights,
            personalization_influence=self.personalization_influence,
        )


def _candidate_identity(result: Any, payload: dict[str, Any]) -> str:
    # The vector result id and lexical payload id are backend-specific and may
    # differ for the same chunk. ES front matter provides the stable identity
    # required to fuse both channels without consuming context slots twice.
    metadata = parse_json_front_matter(payload.get("text"))
    value = metadata.get("document_id")
    if value is None:
        value = getattr(result, "id", None)
    if value is None:
        value = payload.get("id")
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
    alias_relations = metadata.get("alias_relations")
    related_aliases = (
        [normalize_search_text(name) for name in alias_relations if isinstance(name, str)]
        if isinstance(alias_relations, dict)
        else []
    )
    if query_normalized and query_normalized in normalized_names:
        relevance += 0.12
    elif any(
        _contains_normalized_phrase(query_normalized, name) for name in normalized_names
    ) or any(set(name.split()) <= query_tokens for name in related_aliases if name):
        relevance += 0.06

    section_path = metadata.get("section_path")
    if isinstance(section_path, list):
        section_tokens = set(normalize_search_text(" ".join(map(str, section_path))).split())
        relevance += min(3, len(query_tokens & section_tokens)) * 0.015
        normalized_sections = {normalize_search_text(str(value)) for value in section_path}
        if "summary" in normalized_sections or "overview" in normalized_sections:
            relevance += 0.05 if "summary" in normalized_sections else 0.02
        if "notes" in normalized_sections or "notes and references" in normalized_sections:
            relevance -= 0.02
    chunk_index = metadata.get("chunk_index")
    if isinstance(chunk_index, int) and not isinstance(chunk_index, bool) and chunk_index >= 0:
        relevance += 0.012 / (chunk_index + 1)
    if metadata.get("chunk_kind") == "table":
        relevance -= 0.006
    answer_tokens = set(normalize_search_text(answer_body).split())
    # A tiny long-term overlap tie-breaker helps choose an answer-bearing chunk
    # within an already-ranked page without overriding title/section/status policy.
    specific_query_tokens = {token for token in query_tokens if len(token) >= 7}
    relevance += min(3, len(specific_query_tokens & answer_tokens)) * 0.003
    # Exact alphanumeric identifiers are frequently the decisive evidence in
    # compact tables (E587, C729, EVE item/type codes).  Give the chunk that
    # actually contains the queried identifier enough weight to beat a generic
    # same-page overview selected only through shared aliases.  Metadata is not
    # considered here: aliases can discover the page but cannot prove a fact.
    identifier_tokens = {
        token
        for token in query_tokens
        if any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
    }
    relevance += min(2, len(identifier_tokens & answer_tokens)) * 0.04
    relevance += _alias_answer_hint_score(query, candidate.payload)
    if _status_stub(candidate.payload):
        relevance -= 0.04
    return relevance


def _contains_normalized_phrase(query: str, phrase: str) -> bool:
    """Match a title or alias as a contiguous phrase, not scattered query words."""
    if not query or not phrase:
        return False
    return f" {phrase} " in f" {query} "


def _status_stub(payload: dict[str, Any]) -> bool:
    metadata = parse_json_front_matter(payload.get("text"))
    return metadata.get("chunk_kind") == "status" or metadata.get("content_excluded") is True


def _answer_body_tokens(payload: dict[str, Any]) -> set[str]:
    """Return normalized answer-bearing tokens without front-matter matches."""
    _metadata, body = split_json_front_matter(payload.get("text"))
    return set(normalize_search_text(body).split())


def _alias_answer_hint_score(query: str, payload: dict[str, Any]) -> float:
    """Prefer answer text explicitly related to a matched page alias."""
    metadata, body = split_json_front_matter(payload.get("text"))
    relations = metadata.get("alias_relations")
    if not isinstance(relations, dict):
        return 0.0
    query_tokens = _token_roots(query)
    body_tokens = _token_roots(body)
    best_score = 0.0
    for trigger, answer_terms in relations.items():
        trigger_tokens = _token_roots(trigger) if isinstance(trigger, str) else set()
        if not trigger_tokens or not trigger_tokens <= query_tokens:
            continue
        if not isinstance(answer_terms, list):
            continue
        coverages: list[float] = []
        for answer_term in answer_terms:
            answer_tokens = _token_roots(answer_term) if isinstance(answer_term, str) else set()
            if answer_tokens:
                coverages.append(len(answer_tokens & body_tokens) / len(answer_tokens))
        complete = sum(coverage == 1.0 for coverage in coverages)
        partial = sum(coverage for coverage in coverages if coverage < 1.0)
        # Multiple independently configured answer phrases should promote the
        # chunk that actually contains the complete relation, not whichever
        # page overview happened to rank first for the shared alias.
        best_score = max(best_score, min(0.12, complete * 0.03 + partial * 0.008))
    return best_score


def _exact_section_match_score(query: str, payload: dict[str, Any]) -> float:
    """Favor a specifically named section only after its canonical page wins."""
    metadata = parse_json_front_matter(payload.get("text"))
    section_path = metadata.get("section_path")
    if not isinstance(section_path, list):
        return 0.0
    query_normalized = normalize_search_text(query)
    matching_size = max(
        (
            len(section.split())
            for value in section_path
            if (section := normalize_search_text(str(value)))
            and _contains_normalized_phrase(query_normalized, section)
        ),
        default=0,
    )
    return min(0.12, matching_size * 0.04)


def _token_roots(value: str) -> set[str]:
    tokens = normalize_search_text(value).split()
    return {token[:-1] if len(token) > 4 and token.endswith("s") else token for token in tokens}


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
            text = _truncate_preserving_front_matter(text, remaining)
            if text is None:
                break
            payload["text"] = text
        used += separator + len(text)
        bounded.append(
            ScoredResult(
                # ``identity`` may be an ES document id used only to fuse
                # backend copies. Citations must expose the real indexed
                # DocumentChunk UUID selected from the backend.
                id=_result_uuid(candidate.result_id or candidate.identity),
                score=-relevance,
                payload=payload,
            )
        )
    return bounded


def _truncate_preserving_front_matter(text: str, limit: int) -> str | None:
    """Bound text without emitting a partial metadata envelope."""
    normalized = text.lstrip("\ufeff\r\n").replace("\r\n", "\n")
    if len(normalized) <= limit:
        return normalized
    if not normalized.startswith("---\n"):
        return normalized[:limit]
    closing = normalized.find("\n---\n", 4)
    if closing < 0:
        return None
    body_start = closing + len("\n---\n")
    if body_start >= limit:
        return None
    return normalized[:body_start] + normalized[body_start:limit]


def _result_uuid(identity: Any) -> UUID:
    if isinstance(identity, UUID):
        return identity
    try:
        return UUID(str(identity))
    except (TypeError, ValueError, AttributeError):
        return uuid5(NAMESPACE_URL, str(identity))
