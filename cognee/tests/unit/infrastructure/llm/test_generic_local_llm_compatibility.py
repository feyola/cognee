import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api import (
    adapter as adapter_module,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
    _copy_reasoning_content_to_empty_content,
    _enforce_strict_json_schema,
    _llm_concurrency_context,
    _materialize_streaming_response,
)


def test_reasoning_content_fills_empty_content():
    message = SimpleNamespace(content=None, reasoning_content='{"answer": "ok"}')
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    assert _copy_reasoning_content_to_empty_content(response) is response
    assert message.content == '{"answer": "ok"}'


def test_existing_content_is_preserved():
    message = SimpleNamespace(content="final", reasoning_content="thinking")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    _copy_reasoning_content_to_empty_content(response)
    assert message.content == "final"


@pytest.mark.asyncio
async def test_non_streaming_response_is_preserved():
    response = SimpleNamespace(choices=[])

    assert await _materialize_streaming_response(response) is response


@pytest.mark.asyncio
async def test_streaming_response_is_materialized(monkeypatch):
    chunks = [SimpleNamespace(value="first"), SimpleNamespace(value="second")]
    messages = [{"role": "user", "content": "hello"}]
    materialized = SimpleNamespace(choices=[])
    builder_calls = []

    async def stream():
        for chunk in chunks:
            yield chunk

    def build_stream(*, chunks, messages):
        builder_calls.append((chunks, messages))
        return materialized

    monkeypatch.setattr("litellm.stream_chunk_builder", build_stream)

    assert await _materialize_streaming_response(stream(), messages=messages) is materialized
    assert builder_calls == [(chunks, messages)]


@pytest.mark.asyncio
async def test_empty_stream_is_rejected():
    async def empty_stream():
        if False:
            yield None

    with pytest.raises(RuntimeError, match="without response chunks"):
        await _materialize_streaming_response(empty_stream())


def test_strict_json_schema_closes_all_objects_without_mutating_input():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "child": {"$ref": "#/$defs/Child"},
                },
                "required": ["name"],
                "$defs": {
                    "Child": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "note": {"type": ["string", "null"]},
                        },
                    }
                },
            }
        },
    }

    strict = _enforce_strict_json_schema(response_format)
    schema = strict["json_schema"]["schema"]
    child_schema = schema["$defs"]["Child"]

    assert strict is not response_format
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["name", "child"]
    assert child_schema["additionalProperties"] is False
    assert child_schema["required"] == ["value", "note"]
    assert "additionalProperties" not in response_format["json_schema"]["schema"]


def test_strict_json_schema_normalizes_nested_discriminated_unions():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "SessionTurnAnalysis",
            "strict": True,
            "schema": SessionTurnAnalysis.model_json_schema(),
        },
    }

    strict = _enforce_strict_json_schema(response_format)
    items = strict["json_schema"]["schema"]["properties"]["candidate_context_updates"]["items"]

    assert len(items["anyOf"]) == 4

    def find_keywords(value, keywords):
        if isinstance(value, list):
            return [found for item in value for found in find_keywords(item, keywords)]
        if not isinstance(value, dict):
            return []
        found = [key for key in keywords if key in value]
        return found + [
            nested_found
            for nested_value in value.values()
            for nested_found in find_keywords(nested_value, keywords)
        ]

    assert find_keywords(strict["json_schema"]["schema"], {"oneOf", "discriminator"}) == []
    source_items = response_format["json_schema"]["schema"]["properties"][
        "candidate_context_updates"
    ]["items"]
    assert "oneOf" in source_items
    assert "discriminator" in source_items


def test_strict_json_schema_preserves_non_discriminated_one_of():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                ]
            }
        },
    }

    strict = _enforce_strict_json_schema(response_format)

    assert strict["json_schema"]["schema"]["oneOf"] == [
        {"type": "string"},
        {"type": "number"},
    ]
    assert "anyOf" not in strict["json_schema"]["schema"]


@pytest.mark.asyncio
async def test_local_llm_concurrency_limit(monkeypatch):
    monkeypatch.setenv("COGNEE_LLM_MAX_CONCURRENCY", "2")
    active = 0
    maximum = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal active, maximum
        async with _llm_concurrency_context():
            async with lock:
                active += 1
                maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1

    await asyncio.gather(*(worker() for _ in range(8)))
    assert maximum == 2


@pytest.mark.asyncio
async def test_concurrency_cap_composes_with_rate_limit_dispatch(monkeypatch):
    monkeypatch.setenv("COGNEE_LLM_MAX_CONCURRENCY", "1")
    rate_limit_entries = 0
    rate_limit_active = 0
    active_completions = 0
    maximum_completions = 0

    @asynccontextmanager
    async def governed_dispatch():
        nonlocal rate_limit_entries, rate_limit_active
        rate_limit_entries += 1
        rate_limit_active += 1
        try:
            yield
        finally:
            rate_limit_active -= 1

    async def completion(**_kwargs):
        nonlocal active_completions, maximum_completions
        assert rate_limit_active > 0
        active_completions += 1
        maximum_completions = max(maximum_completions, active_completions)
        await asyncio.sleep(0.01)
        active_completions -= 1
        message = SimpleNamespace(content="ok", reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(adapter_module, "llm_rate_limiter_context_manager", governed_dispatch)
    monkeypatch.setattr(adapter_module.litellm, "acompletion", completion)

    adapter = object.__new__(GenericAPIAdapter)
    adapter.model = "openai/test"
    adapter.api_key = "test"
    adapter.endpoint = "http://localhost"
    adapter.api_version = None

    await asyncio.gather(*(adapter.acreate_str_output("input", "system") for _ in range(4)))

    assert rate_limit_entries == 4
    assert maximum_completions == 1
