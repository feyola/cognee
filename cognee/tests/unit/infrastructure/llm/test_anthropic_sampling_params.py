"""Anthropic SDK 1.x drops sampling params from the typed create signature."""

import pytest
from pydantic import BaseModel


class _SimpleModel(BaseModel):
    value: str = "ok"


def _build_adapter(monkeypatch):
    anthropic = pytest.importorskip("anthropic", reason="anthropic package not installed")

    class FakeAsyncAnthropic:
        class messages:
            @staticmethod
            def create(*args, **kwargs):
                pass

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAsyncAnthropic)
    monkeypatch.setattr(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter.instructor.patch",
        lambda create, mode: object(),
    )

    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter import (
        AnthropicAdapter,
    )

    return AnthropicAdapter


async def _capture(adapter):
    """Swap the patched client for a recorder and run one completion."""
    seen = {}

    async def _record(*args, **kwargs):
        seen.update(kwargs)
        return _SimpleModel()

    adapter.aclient = _record
    await adapter.acreate_structured_output("input", "system", _SimpleModel)
    return seen


@pytest.mark.asyncio
async def test_temperature_travels_in_extra_body(monkeypatch):
    """LLM_TEMPERATURE reaches the API without tripping the typed signature."""
    AnthropicAdapter = _build_adapter(monkeypatch)
    adapter = AnthropicAdapter(
        api_key="test-key",
        model="claude-haiku-4-5",
        max_completion_tokens=1024,
        llm_args={"temperature": 0.0, "top_p": 0.9},
    )

    seen = await _capture(adapter)

    assert "temperature" not in seen, "sampling params raise TypeError as keywords"
    assert "top_p" not in seen
    assert seen["extra_body"] == {"temperature": 0.0, "top_p": 0.9}
    # Everything else still goes through as a typed keyword.
    assert seen["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_explicit_extra_body_wins(monkeypatch):
    """A caller reaching past the shim keeps its own value."""
    AnthropicAdapter = _build_adapter(monkeypatch)
    adapter = AnthropicAdapter(
        api_key="test-key",
        model="claude-haiku-4-5",
        max_completion_tokens=1024,
        llm_args={"temperature": 0.0, "extra_body": {"temperature": 1.0, "keep": True}},
    )

    seen = await _capture(adapter)

    assert seen["extra_body"] == {"temperature": 1.0, "keep": True}


@pytest.mark.asyncio
async def test_untouched_when_no_sampling_params(monkeypatch):
    """No sampling params configured means no extra_body is invented."""
    AnthropicAdapter = _build_adapter(monkeypatch)
    adapter = AnthropicAdapter(
        api_key="test-key",
        model="claude-haiku-4-5",
        max_completion_tokens=1024,
    )

    seen = await _capture(adapter)

    assert "extra_body" not in seen
