from typing import Any, Dict, List, Optional, Type

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.merge_results import conversational_reserve, merge_ranked
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.used_graph_elements import extract_from_scored_results
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.modules.retrieval.utils.references import append_chunk_evidence
from cognee.modules.retrieval.hybrid_chunks_retriever import (
    DEFAULT_CANDIDATE_POOL,
    DEFAULT_MAX_CONTEXT_CHARS,
    HybridChunksRetriever,
)
from cognee.modules.retrieval.utils.chunk_metadata import canonical_page_key
from cognee.base_config import get_base_config
from cognee.modules.user_preferences import (
    load_preference_text,
    load_preference_weights,
    personal_factor,
)
from cognee.modules.retrieval.utils.evidence import chunk_context_evidence

logger = get_logger("CompletionRetriever")

CURRENTNESS_GUIDANCE = (
    "Retrieval policy: use active answer-bearing content. Status chunks are warnings, not "
    "answers. If facts conflict, prefer the most specific section whose heading matches the "
    "question; do not silently combine contradictory values. Preserve the most precise numeric "
    "value supported by the selected evidence and do not round it. When one source gives a "
    "rounded value and a more specific matching section gives additional decimal places or a "
    "formula, use the section's more precise value and treat the shorter form only as an "
    "approximation.\n\n"
)


def _stable_sort_by_personal_distance(
    found_chunks: List[Any], weights: Dict[str, float], influence: float
) -> List[Any]:
    """Stable re-sort of ScoredResult chunks by personalized distance.

    ``score`` is a distance here (lower is better), so a preferred chunk's
    distance shrinks by ``personal_factor(..., distance_space=True)``. Chunk
    id comes from ``payload["id"]`` falling back to ``.id`` — the same rule
    ``extract_from_scored_results`` uses, so the ids match the ones the
    preference update wrote ``prefers`` edges for. Chunks with no matching
    weight keep their raw distance, and the sort is stable, so ties keep the
    vector engine's order.
    """

    def personalized_distance(chunk: Any) -> float:
        chunk_id = None
        payload = getattr(chunk, "payload", None)
        if isinstance(payload, dict):
            chunk_id = payload.get("id")
        if chunk_id is None:
            chunk_id = getattr(chunk, "id", None)
        weight = weights.get(str(chunk_id)) if chunk_id is not None else None
        if weight is None:
            return chunk.score
        return chunk.score * personal_factor(weight, influence, distance_space=True)

    return sorted(found_chunks, key=personalized_distance)


async def _weights_matching_collection(
    vector_engine: Any, weights: Dict[str, float], collection_name: str = "DocumentChunk_text"
) -> Dict[str, float]:
    """Keep only the prefers weights whose key is a row in this collection.

    Weight keys span every rated node — graph entities included — but only
    chunk ids can ever match a ``DocumentChunk_text`` result, so weights that
    point elsewhere must not trigger the wide fetch: the extra work would be
    guaranteed to change nothing. One id-lookup ``retrieve`` answers the
    question; on any error it fails open by returning the weights unfiltered,
    so a broken lookup costs a wider search, never a lost personalization.
    """
    try:
        rows = await vector_engine.retrieve(collection_name, list(weights))
    except Exception as error:
        logger.debug("Preference weight collection check failed open: %s", error)
        return weights

    present = set()
    for row in rows:
        payload = getattr(row, "payload", None)
        if isinstance(payload, dict) and payload.get("id") is not None:
            present.add(str(payload["id"]))
        row_id = getattr(row, "id", None)
        if row_id is not None:
            present.add(str(row_id))
    return {key: weight for key, weight in weights.items() if key in present}


class CompletionRetriever(BaseRetriever):
    """Retriever for LLM completions over page-diverse hybrid chunk retrieval."""

    def __init__(
        self,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 1,
        session_id: Optional[str] = None,
        response_model: Type = str,
        include_references: bool = False,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
        wide_search_top_k: Optional[int] = 100,
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_chunks_per_page: int = 2,
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 1
        self.wide_search_top_k = wide_search_top_k
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.response_model = response_model
        self.include_references = include_references
        self.node_name = node_name
        self.node_name_filter_operator = node_name_filter_operator
        self.candidate_pool_size = candidate_pool_size
        self.max_context_chars = max_context_chars
        self.max_chunks_per_page = max_chunks_per_page

    async def get_retrieved_objects(self, query: str) -> Any:
        weights = await load_preference_weights()
        if weights:
            vector_engine = await get_vector_engine_async()
            weights = await _weights_matching_collection(vector_engine, weights)

        candidate_pool_size = max(self.candidate_pool_size, self.top_k)
        if weights:
            candidate_pool_size = max(candidate_pool_size, self.wide_search_top_k or 0)

        candidate_context_limit = max(
            1,
            self.max_context_chars - len(CURRENTNESS_GUIDANCE),
        )
        retriever = HybridChunksRetriever(
            top_k=self.top_k,
            node_name=self.node_name,
            node_name_filter_operator=self.node_name_filter_operator,
            candidate_pool_size=candidate_pool_size,
            page_limit=min(3, self.top_k),
            max_chunks_per_page=self.max_chunks_per_page,
            max_context_chars=candidate_context_limit,
            preference_weights=weights,
            personalization_influence=(
                get_base_config().personalization_influence if weights else 0.0
            ),
        )
        results = await retriever.get_retrieved_objects(query)
        return _group_answer_chunks_by_page(results)

    def merge_retrieved_objects(self, primary: Any, secondary: Any) -> Any:
        return merge_ranked(
            primary,
            secondary,
            limit=self.top_k,
            secondary_reserve=conversational_reserve(self.top_k),
        )

    def extract_context_object_ids(self, retrieved_objects: Any) -> Optional[Dict[str, List[str]]]:
        """Extract node_ids from ScoredResult-like list for session QA."""
        if isinstance(retrieved_objects, list) and retrieved_objects:
            return extract_from_scored_results(retrieved_objects)
        return None

    def get_context_evidence(self, retrieved_objects: Any, dataset_id: Any = None):
        """Return the exact chunks concatenated into this RAG completion's context."""
        return chunk_context_evidence(retrieved_objects, dataset_id=dataset_id)

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """Combine retrieved chunk text with the production currentness policy."""
        if retrieved_objects:
            chunks_payload = [found_chunk.payload["text"] for found_chunk in retrieved_objects]
            return CURRENTNESS_GUIDANCE + "\n".join(chunks_payload)
        return ""

    def _completion_kwargs(self, context: str) -> dict:
        """Common kwargs for completion calls (no session)."""
        return {
            "context": context,
            "user_prompt_path": self.user_prompt_path,
            "system_prompt_path": self.system_prompt_path,
            "system_prompt": self.system_prompt,
            "response_model": self.response_model,
        }

    async def _generate_completion_without_session(self, query: str, context: str) -> List[Any]:
        """Generate completion without session; returns list of one completion."""
        kwargs = self._completion_kwargs(context)
        preference_text = await load_preference_text()
        completion = await generate_completion(
            query=query, conversation_history=preference_text, **kwargs
        )
        return [completion]

    async def append_references(self, completions: List[Any], retrieved_objects: Any) -> List[Any]:
        return append_chunk_evidence(
            completions,
            retrieved_objects,
            enabled=self.include_references and self.response_model is str,
        )

    async def get_completion_from_context(
        self,
        query: str,
        retrieved_objects: Any,
        context: Optional[Any] = None,
        effective_query: Optional[str] = None,
        turn_preparation=None,
    ) -> List[Any]:
        """Generate an LLM completion from retrieved context."""
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        use_session = user_id and cache_config.caching

        if use_session:
            sm = get_session_manager()
            used_graph_element_ids = self.extract_context_object_ids(retrieved_objects)
            completion = await sm.generate_completion_with_session(
                session_id=self.session_id,
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                response_model=self.response_model,
                summarize_context=False,
                used_graph_element_ids=used_graph_element_ids,
                max_context_chars=self.max_context_chars,
                effective_query=effective_query,
                turn_preparation=turn_preparation,
            )
            completions = [completion]
        else:
            completions = await self._generate_completion_without_session(query, context)

        return await self.append_references(completions, retrieved_objects)


def _group_answer_chunks_by_page(retrieved_objects: Any) -> Any:
    """Keep each selected page's supporting chunks adjacent for answer synthesis."""
    if not isinstance(retrieved_objects, list) or len(retrieved_objects) < 2:
        return retrieved_objects
    grouped: dict[str, list[Any]] = {}
    page_order: list[str] = []
    for index, result in enumerate(retrieved_objects):
        payload = getattr(result, "payload", None)
        if not isinstance(payload, dict):
            page = f"result:{index}"
        else:
            page = canonical_page_key(payload, f"result:{index}")
        if page not in grouped:
            grouped[page] = []
            page_order.append(page)
        grouped[page].append(result)
    return [result for page in page_order for result in grouped[page]]
