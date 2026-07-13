import asyncio
from types import SimpleNamespace

import pytest

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    _copy_reasoning_content_to_empty_content,
    _llm_concurrency_context,
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
