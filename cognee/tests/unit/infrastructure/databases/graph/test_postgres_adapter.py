from contextlib import asynccontextmanager
import importlib.util
import sys
import types

import pytest

_created_asyncpg_stub = False
if importlib.util.find_spec("asyncpg") is None:
    asyncpg_stub = types.ModuleType("asyncpg")

    class DeadlockDetectedError(Exception):
        pass

    asyncpg_stub.DeadlockDetectedError = DeadlockDetectedError
    sys.modules["asyncpg"] = asyncpg_stub
    _created_asyncpg_stub = True

from cognee.infrastructure.databases.graph.postgres.adapter import PostgresAdapter  # noqa: E402
from cognee.tasks.temporal_graph.models import Timestamp  # noqa: E402

if _created_asyncpg_stub:
    del sys.modules["asyncpg"]


class _EmptyResult:
    def fetchall(self):
        return []


class _CapturingSession:
    def __init__(self, rows=None):
        self.statement = None
        self.params = None
        self.rows = rows or []

    async def execute(self, statement, params=None):
        self.statement = statement
        self.params = params
        if not self.rows:
            return _EmptyResult()

        class _RowsResult:
            def fetchall(inner_self):
                return self.rows

        return _RowsResult()


@pytest.mark.asyncio
async def test_get_neighborhood_casts_seed_parameter_to_text_array():
    adapter = PostgresAdapter.__new__(PostgresAdapter)
    session = _CapturingSession()

    @asynccontextmanager
    async def session_context():
        yield session

    adapter._session = session_context

    nodes, edges = await adapter.get_neighborhood(["a"], depth=1)

    assert nodes == []
    assert edges == []
    assert "unnest(CAST(:seeds AS text[]))" in str(session.statement)
    assert session.params == {"seeds": ["a"], "depth": 1}


@pytest.mark.asyncio
async def test_collect_time_ids_uses_safe_inclusive_jsonb_bounds():
    adapter = PostgresAdapter.__new__(PostgresAdapter)
    session = _CapturingSession(rows=[("timestamp-a",), ("timestamp-b",)])

    @asynccontextmanager
    async def session_context():
        yield session

    adapter._session = session_context

    ids = await adapter.collect_time_ids(
        time_from=Timestamp(year=2020),
        time_to=Timestamp(year=2021),
    )

    assert ids == ["timestamp-a", "timestamp-b"]
    statement = str(session.statement)
    assert "properties ->> 'time_at'" in statement
    assert "~ '^-?[0-9]+$'" in statement
    assert ">= :time_from" in statement
    assert "<= :time_to" in statement
    assert session.params == {
        "time_from": 1577836800000,
        "time_to": 1609459200000,
    }


@pytest.mark.asyncio
async def test_collect_time_ids_without_bounds_avoids_database_query():
    adapter = PostgresAdapter.__new__(PostgresAdapter)

    assert await adapter.collect_time_ids() == []


@pytest.mark.asyncio
async def test_collect_events_binds_ids_and_shapes_temporal_results():
    adapter = PostgresAdapter.__new__(PostgresAdapter)
    session = _CapturingSession(
        rows=[
            (
                "event-a",
                "Battle of Jita",
                {"description": "A major battle.", "location": "Jita"},
            ),
            ("event-b", "Second event", '{"description":"Another event."}'),
        ]
    )

    @asynccontextmanager
    async def session_context():
        yield session

    adapter._session = session_context

    result = await adapter.collect_events("'timestamp-a', 'timestamp-b'")

    assert session.params == {"ids": ["timestamp-a", "timestamp-b"]}
    assert "WITH RECURSIVE reachable" in str(session.statement)
    assert "CAST(:ids AS text[])" in str(session.statement)
    assert result == [
        {
            "events": [
                {
                    "id": "event-a",
                    "name": "Battle of Jita",
                    "description": "A major battle.",
                    "location": "Jita",
                },
                {
                    "id": "event-b",
                    "name": "Second event",
                    "description": "Another event.",
                },
            ]
        }
    ]


@pytest.mark.asyncio
async def test_collect_events_empty_ids_returns_empty_contract():
    adapter = PostgresAdapter.__new__(PostgresAdapter)

    assert await adapter.collect_events([]) == [{"events": []}]
