import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.completion_retriever import (
    CURRENTNESS_GUIDANCE,
    CompletionRetriever,
)
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError


def test_currentness_guidance_preserves_precise_numeric_evidence():
    assert "most precise numeric value" in CURRENTNESS_GUIDANCE
    assert "do not round" in CURRENTNESS_GUIDANCE
    assert "more specific matching section" in CURRENTNESS_GUIDANCE


@pytest.fixture
def mock_vector_engine():
    """Create a mock vector engine."""
    engine = AsyncMock()
    engine.search = AsyncMock()
    return engine


@pytest.fixture
def mock_hybrid_retriever():
    instance = MagicMock()
    instance.get_retrieved_objects = AsyncMock()
    with patch(
        "cognee.modules.retrieval.completion_retriever.HybridChunksRetriever",
        return_value=instance,
    ) as factory:
        yield instance, factory


@pytest.mark.asyncio
async def test_get_context_success(mock_hybrid_retriever):
    """Test successful retrieval of context."""
    mock_result1 = MagicMock()
    mock_result1.payload = {"text": "Steve Rodger"}
    mock_result2 = MagicMock()
    mock_result2.payload = {"text": "Mike Broski"}

    hybrid, _ = mock_hybrid_retriever
    hybrid.get_retrieved_objects.return_value = [mock_result1, mock_result2]

    retriever = CompletionRetriever(top_k=2)

    objects = await retriever.get_retrieved_objects("test query")
    context = await retriever.get_context_from_objects("test query", objects)

    assert context == CURRENTNESS_GUIDANCE + "Steve Rodger\nMike Broski"
    hybrid.get_retrieved_objects.assert_awaited_once_with("test query")


@pytest.mark.asyncio
async def test_answer_context_groups_supporting_chunks_by_selected_page(mock_hybrid_retriever):
    def result(title: str, url: str, body: str):
        item = MagicMock()
        item.payload = {"text": (f'---\ntitle: "{title}"\ncanonical_url: "{url}"\n---\n\n{body}')}
        return item

    trading_url = "https://docs.example.org/Trading"
    tax_url = "https://docs.example.org/Tax"
    trading_overview = result("Trading", trading_url, "Station trading overview")
    tax = result("Tax", tax_url, "Rounded tax summary")
    trading_rate = result("Trading", trading_url, "Precise current rate")
    hybrid, _ = mock_hybrid_retriever
    hybrid.get_retrieved_objects.return_value = [trading_overview, tax, trading_rate]

    objects = await CompletionRetriever(top_k=3).get_retrieved_objects("station trading tax")

    assert objects == [trading_overview, trading_rate, tax]


@pytest.mark.asyncio
async def test_get_context_collection_not_found_error(mock_hybrid_retriever):
    """Test that missing hybrid data is surfaced as NoDataError."""
    hybrid, _ = mock_hybrid_retriever
    hybrid.get_retrieved_objects.side_effect = NoDataError("No data found")

    retriever = CompletionRetriever()

    with pytest.raises(NoDataError, match="No data found"):
        await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_get_context_empty_results(mock_vector_engine):
    """Test that empty string is returned when no chunks are found."""
    mock_vector_engine.search.return_value = []

    retriever = CompletionRetriever()

    context = await retriever.get_context_from_objects("test query", [])

    assert context == ""


@pytest.mark.asyncio
async def test_get_context_top_k_limit(mock_hybrid_retriever):
    """Test that top_k parameter limits the number of results."""
    mock_results = [MagicMock() for _ in range(2)]
    for i, result in enumerate(mock_results):
        result.payload = {"text": f"Chunk {i}"}

    hybrid, factory = mock_hybrid_retriever
    hybrid.get_retrieved_objects.return_value = mock_results

    retriever = CompletionRetriever(top_k=2)

    objects = await retriever.get_retrieved_objects("test query")
    context = await retriever.get_context_from_objects("test query", objects)

    assert context == CURRENTNESS_GUIDANCE + "Chunk 0\nChunk 1"
    factory.assert_called_once_with(
        top_k=2,
        node_name=None,
        node_name_filter_operator="OR",
        candidate_pool_size=30,
        page_limit=2,
        max_chunks_per_page=2,
        max_context_chars=15_000 - len(CURRENTNESS_GUIDANCE),
        preference_weights={},
        personalization_influence=0.0,
    )


@pytest.mark.asyncio
async def test_get_context_single_chunk(mock_hybrid_retriever):
    """Test get_context with single chunk result."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Single chunk text"}
    hybrid, _ = mock_hybrid_retriever
    hybrid.get_retrieved_objects.return_value = [mock_result]

    retriever = CompletionRetriever()

    objects = await retriever.get_retrieved_objects("test query")
    context = await retriever.get_context_from_objects("test query", objects)

    assert context == CURRENTNESS_GUIDANCE + "Single chunk text"


@pytest.mark.asyncio
async def test_get_completion_without_session(mock_vector_engine):
    """Test get_completion without session caching."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Chunk text"}
    mock_vector_engine.search.return_value = [mock_result]

    retriever = CompletionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.completion_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context("test query", None, "")

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_provided_context(mock_vector_engine):
    """Test get_completion with provided context."""
    retriever = CompletionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.completion_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context(
            "test query", None, context="Provided context"
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_session(mock_vector_engine):
    """Test get_completion with session caching enabled (SessionManager path)."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Chunk text", "id": "chunk1"}
    mock_vector_engine.search.return_value = [mock_result]

    retriever = CompletionRetriever(session_id="test_session")

    mock_user = MagicMock()
    mock_user.id = "test-user-id"

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.get_session_manager",
        ) as mock_get_sm,
        patch("cognee.modules.retrieval.completion_retriever.CacheConfig") as mock_cache_config,
        patch("cognee.modules.retrieval.completion_retriever.session_user") as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = mock_user
        mock_sm = MagicMock()
        mock_sm.generate_completion_with_session = AsyncMock(return_value="Generated answer")
        mock_get_sm.return_value = mock_sm

        completion = await retriever.get_completion_from_context(
            "test query", [mock_result], "test"
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"
    mock_sm.generate_completion_with_session.assert_awaited_once()
    call_kw = mock_sm.generate_completion_with_session.call_args.kwargs
    assert call_kw.get("used_graph_element_ids") == {"node_ids": ["chunk1"]}


@pytest.mark.asyncio
async def test_get_completion_with_session_no_user_id(mock_vector_engine):
    """Test get_completion with session config but no user ID."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Chunk text"}
    mock_vector_engine.search.return_value = [mock_result]

    retriever = CompletionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.completion_retriever.CacheConfig") as mock_cache_config,
        patch("cognee.modules.retrieval.completion_retriever.session_user") as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = None  # No user

        completion = await retriever.get_completion_from_context("test query", None, "")

    assert isinstance(completion, list)
    assert len(completion) == 1


@pytest.mark.asyncio
async def test_get_completion_with_response_model(mock_vector_engine):
    """Test get_completion with custom response model."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        answer: str

    mock_result = MagicMock()
    mock_result.payload = {"text": "Chunk text"}
    mock_vector_engine.search.return_value = [mock_result]

    retriever = CompletionRetriever(response_model=TestModel)

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value=TestModel(answer="Test answer"),
        ),
        patch("cognee.modules.retrieval.completion_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context("test query", None, None)

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert isinstance(completion[0], TestModel)


@pytest.mark.asyncio
async def test_init_defaults():
    """Test CompletionRetriever initialization with defaults."""
    retriever = CompletionRetriever()

    assert retriever.user_prompt_path == "context_for_question.txt"
    assert retriever.system_prompt_path == "answer_simple_question.txt"
    assert retriever.top_k == 1
    assert retriever.system_prompt is None
    assert retriever.node_name is None
    assert retriever.node_name_filter_operator == "OR"


@pytest.mark.asyncio
async def test_init_custom_params():
    """Test CompletionRetriever initialization with custom parameters."""
    retriever = CompletionRetriever(
        user_prompt_path="custom_user.txt",
        system_prompt_path="custom_system.txt",
        system_prompt="Custom prompt",
        top_k=10,
        node_name=["KEN", "src_type:figure"],
        node_name_filter_operator="AND",
    )

    assert retriever.user_prompt_path == "custom_user.txt"
    assert retriever.system_prompt_path == "custom_system.txt"
    assert retriever.system_prompt == "Custom prompt"
    assert retriever.top_k == 10
    assert retriever.node_name == ["KEN", "src_type:figure"]
    assert retriever.node_name_filter_operator == "AND"


@pytest.mark.asyncio
async def test_get_context_forwards_nodeset_filter_to_vector_search(mock_hybrid_retriever):
    """node_set filtering must be passed through to the vector engine (RAG_COMPLETION)."""
    hybrid, factory = mock_hybrid_retriever
    hybrid.get_retrieved_objects.return_value = []

    retriever = CompletionRetriever(
        top_k=30,
        node_name=["KEN", "src_type:figure"],
        node_name_filter_operator="AND",
    )

    await retriever.get_retrieved_objects("land cover")

    factory.assert_called_once_with(
        top_k=30,
        node_name=["KEN", "src_type:figure"],
        node_name_filter_operator="AND",
        candidate_pool_size=30,
        page_limit=3,
        max_chunks_per_page=2,
        max_context_chars=15_000 - len(CURRENTNESS_GUIDANCE),
        preference_weights={},
        personalization_influence=0.0,
    )


@pytest.mark.asyncio
async def test_get_context_missing_text_key(mock_vector_engine):
    """Test get_context handles missing text key in payload."""
    mock_result = MagicMock()
    mock_result.payload = {"other_key": "value"}

    mock_vector_engine.search.return_value = [mock_result]

    retriever = CompletionRetriever()

    with pytest.raises(KeyError):
        await retriever.get_context_from_objects("test query", [mock_result])
