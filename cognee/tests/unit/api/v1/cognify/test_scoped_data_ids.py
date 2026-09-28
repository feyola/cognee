"""A targeted cognify must never fall back to all dataset items."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")


class _Session:
    def __init__(self, rows):
        self.rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def execute(self, _query):
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self.rows))


@pytest.mark.asyncio
async def test_scoped_cognify_passes_only_authorized_data_to_pipeline():
    dataset_id, data_id = uuid4(), uuid4()
    row = SimpleNamespace(id=data_id)
    executor = AsyncMock(return_value={})
    engine = MagicMock()
    engine.get_async_session.return_value = _Session([row])

    with (
        patch("cognee.api.v1.serve.state.get_remote_client", return_value=None),
        patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock),
        patch.object(cognify_module, "resolve_extractor", return_value="llm"),
        patch.object(cognify_module, "ensure_extractor_runtime", new_callable=AsyncMock),
        patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock, return_value=[]),
        patch.object(cognify_module, "get_dlt_tasks", new_callable=AsyncMock, return_value=[]),
        patch.object(cognify_module, "get_code_file_tasks", return_value=[]),
        patch.object(cognify_module, "get_code_repo_tasks", return_value=[]),
        patch.object(cognify_module, "get_pipeline_executor", return_value=executor),
        patch(
            "cognee.modules.pipelines.layers.resolve_authorized_user_datasets.resolve_authorized_user_datasets",
            new_callable=AsyncMock,
            return_value=(object(), [SimpleNamespace(id=dataset_id)]),
        ),
        patch("cognee.infrastructure.databases.relational.get_relational_engine", return_value=engine),
    ):
        await cognify_module.cognify(datasets=[dataset_id], data_ids=[data_id])

    assert executor.await_args.kwargs["data"] == [row]


@pytest.mark.asyncio
async def test_scoped_cognify_rejects_missing_id_without_running_pipeline():
    dataset_id, data_id = uuid4(), uuid4()
    executor = AsyncMock(return_value={})
    engine = MagicMock()
    engine.get_async_session.return_value = _Session([])

    with (
        patch("cognee.api.v1.serve.state.get_remote_client", return_value=None),
        patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock),
        patch.object(cognify_module, "resolve_extractor", return_value="llm"),
        patch.object(cognify_module, "ensure_extractor_runtime", new_callable=AsyncMock),
        patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock, return_value=[]),
        patch.object(cognify_module, "get_dlt_tasks", new_callable=AsyncMock, return_value=[]),
        patch.object(cognify_module, "get_code_file_tasks", return_value=[]),
        patch.object(cognify_module, "get_code_repo_tasks", return_value=[]),
        patch.object(cognify_module, "get_pipeline_executor", return_value=executor),
        patch(
            "cognee.modules.pipelines.layers.resolve_authorized_user_datasets.resolve_authorized_user_datasets",
            new_callable=AsyncMock,
            return_value=(object(), [SimpleNamespace(id=dataset_id)]),
        ),
        patch("cognee.infrastructure.databases.relational.get_relational_engine", return_value=engine),
        pytest.raises(ValueError, match="not in the authorized dataset"),
    ):
        await cognify_module.cognify(datasets=[dataset_id], data_ids=[data_id])

    executor.assert_not_awaited()
