from typing import Any, Dict, List, Optional, Type

from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.utils.completion import generate_completion
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

logger = get_logger("CompletionRetriever")

CURRENTNESS_GUIDANCE = (
    "Retrieval policy: use active answer-bearing content. Status chunks are warnings, not "
    "answers. If facts conflict, prefer the most specific section whose heading matches the "
    "question; do not silently combine contradictory values. Preserve the most precise numeric "
    "value supported by the selected evidence and do not round it; when the same value appears "
    "at different precision, state the more precise form.\n\n"
)


class CompletionRetriever(BaseRetriever):
    """
    Retriever for handling LLM-based completion searches.
    """

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
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_chunks_per_page: int = 2,
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 1
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
        candidate_context_limit = max(
            1,
            self.max_context_chars - len(CURRENTNESS_GUIDANCE),
        )
        retriever = HybridChunksRetriever(
            top_k=self.top_k,
            node_name=self.node_name,
            node_name_filter_operator=self.node_name_filter_operator,
            candidate_pool_size=self.candidate_pool_size,
            page_limit=min(3, self.top_k),
            max_chunks_per_page=self.max_chunks_per_page,
            max_context_chars=candidate_context_limit,
        )
        results = await retriever.get_retrieved_objects(query)
        return _group_answer_chunks_by_page(results)

    def _extract_context_object_ids(self, retrieved_objects: Any) -> Optional[Dict[str, List[str]]]:
        """Extract node_ids from ScoredResult-like list for session QA."""
        if isinstance(retrieved_objects, list) and retrieved_objects:
            return extract_from_scored_results(retrieved_objects)
        return None

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """
        Retrieves relevant document chunks as context.

        Fetches document chunks based on a query from a vector engine and combines their text.
        Returns empty string if no chunks are found. Raises NoDataError if the collection is not
        found.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant document chunks.

        Returns:
        --------

            - str: A string containing the combined text of the retrieved document chunks, or an
              empty string if none are found.
        """
        if retrieved_objects:
            # Combine all chunks text returned from vector search (number of chunks is determined by top_k)
            chunks_payload = [found_chunk.payload["text"] for found_chunk in retrieved_objects]
            combined_context = CURRENTNESS_GUIDANCE + "\n".join(chunks_payload)
            return combined_context
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
        completion = await generate_completion(query=query, **kwargs)
        return [completion]

    async def get_completion_from_context(
        self,
        query: str,
        retrieved_objects: Any,
        context: Optional[Any] = None,
        effective_query: Optional[str] = None,
        turn_preparation=None,
    ) -> List[Any]:
        """
        Generates an LLM completion using the context.

        Retrieves context if not provided and generates a completion based on the query and
        context using an external completion generator.

        Parameters:
        -----------

            - query (str): The query string to be used for generating a completion.
            - context (Optional[Any]): Optional pre-fetched context to use for generating the
              completion; if None, it retrieves the context for the query. (default None)
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)
            - response_model (Type): The Pydantic model type for structured output. (default str)

        Returns:
        --------

            - Any: The generated completion based on the provided query and context.
        """
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        use_session = user_id and cache_config.caching

        if use_session:
            sm = get_session_manager()
            used_graph_element_ids = self._extract_context_object_ids(retrieved_objects)
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
                max_context_chars=getattr(self, "max_context_chars", None),
                effective_query=effective_query,
                turn_preparation=turn_preparation,
            )
            completions = [completion]
        else:
            completions = await self._generate_completion_without_session(query, context)

        # Both the session/cache branch and the non-session branch rejoin here so
        # logged-in/cached calls also receive references. Evidence is grounded in
        # each completion's own text, so a cache-hit answer never cites chunks
        # that share nothing with it.
        return append_chunk_evidence(
            completions,
            retrieved_objects,
            enabled=self.include_references and self.response_model is str,
        )


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
