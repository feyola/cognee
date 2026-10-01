"""Generic deployment regressions; no live provider or database needed."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
    OpenAICompatibleEmbeddingEngine,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api import (
    adapter as adapter_module,
)


def test_openai_compatible_honors_explicit_tokenizer(monkeypatch):
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setenv("HUGGINGFACE_TOKENIZER", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setattr(
        "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine.resolve_embedding_tokenizer",
        resolve,
    )
    engine = OpenAICompatibleEmbeddingEngine.__new__(OpenAICompatibleEmbeddingEngine)
    engine.model = "Qwen3-Embedding-4B-Q4_K_M.gguf"
    engine.get_tokenizer()
    assert calls[0].get("huggingface_tokenizer") == "Qwen/Qwen3-Embedding-4B"


def test_plain_completion_materializes_configured_stream(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="qualified"))]
    )

    async def chunks():
        yield object()

    async def completion(**kwargs):
        assert kwargs["stream"] is True
        return chunks()

    @asynccontextmanager
    async def rate_limit():
        yield

    monkeypatch.setattr(adapter_module.litellm, "acompletion", completion)
    monkeypatch.setattr(adapter_module.litellm, "stream_chunk_builder", lambda **_: response)
    monkeypatch.setattr(adapter_module, "get_active_token_sink", lambda: None)
    monkeypatch.setattr(adapter_module, "llm_rate_limiter_context_manager", rate_limit)
    adapter = adapter_module.GenericAPIAdapter.__new__(adapter_module.GenericAPIAdapter)
    adapter.model = "openai/compatible-model"
    adapter.name = "custom"
    adapter.api_key = "test-key"
    adapter.endpoint = "http://localhost:9999/v1"
    adapter.api_version = None
    assert (
        asyncio.run(adapter.acreate_str_output("test", "Reply briefly", stream=True)) == "qualified"
    )
