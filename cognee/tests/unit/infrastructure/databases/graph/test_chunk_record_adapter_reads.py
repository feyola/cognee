"""Native exact snapshot reads retain text/refs and perform no mutation queries."""

import json
import re
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
from cognee.infrastructure.databases.graph.postgres_demo.adapter import PostgresDemoAdapter
from cognee.infrastructure.databases.provenance import make_source_ref_key, make_source_run_ref


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres", "ladybug", "neo4j"])
async def test_exact_native_read_preserves_bytes_and_refs_without_writes(backend):
    chunk, dataset, data, run = [str(uuid4()) for _ in range(4)]
    source = make_source_ref_key(dataset, data)
    run_ref = make_source_run_ref(run, source)
    props = {
        "text": " \nα 猫\r\n ",
        "document_id": data,
        "chunk_index": 0,
        "metadata": {"index_fields": ["text"]},
    }
    if backend == "postgres":
        adapter = PostgresDemoAdapter.__new__(PostgresDemoAdapter)
        row = {
            "id": chunk,
            "name": "notes",
            "type": "DocumentChunk",
            "properties": props,
            "source_ref_keys": [source],
            "source_dataset_ids": [dataset],
            "source_run_ids": [run],
            "source_run_refs": [run_ref],
        }
        session = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: [row]))
            ),
            commit=AsyncMock(),
        )

        @asynccontextmanager
        async def sessionmaker():
            yield session

        adapter.sessionmaker = sessionmaker
    elif backend == "ladybug":
        adapter = LadybugAdapter.__new__(LadybugAdapter)
        adapter.query = AsyncMock(
            return_value=[
                [
                    chunk,
                    "notes",
                    "DocumentChunk",
                    json.dumps(props),
                    f"|{source}|",
                    f"|{dataset}|",
                    f"|{run}|",
                    f"|{run_ref}|",
                ]
            ]
        )
        adapter.checkpoint = AsyncMock()
    else:
        adapter = Neo4jAdapter.__new__(Neo4jAdapter)
        adapter.query = AsyncMock(
            return_value=[
                {
                    "id": chunk,
                    "properties": {**props, "id": chunk, "name": "notes", "type": "DocumentChunk"},
                    "srk": [source],
                    "sdi": [dataset],
                    "sri": [run],
                    "srr": [run_ref],
                }
            ]
        )
    result = await adapter.get_node_delete_data([chunk])
    assert set(result) == {chunk}
    snapshot = result[chunk]
    assert snapshot.node_id == chunk and snapshot.node_type == "DocumentChunk"
    assert snapshot.node_properties["text"] == props["text"]
    assert snapshot.source_ref_keys == [source]
    assert snapshot.source_dataset_ids == [dataset]
    assert snapshot.source_run_ids == [run]
    assert snapshot.source_run_refs == [run_ref]
    if backend == "postgres":
        statement, params = session.execute.await_args.args
        assert params == {"ids": [chunk]}
        session.execute.assert_awaited_once()
        session.commit.assert_not_awaited()
    else:
        statement, params = adapter.query.await_args.args
        assert params == {"ids": [chunk]}
        adapter.query.assert_awaited_once()
        if backend == "ladybug":
            adapter.checkpoint.assert_not_awaited()
    assert not re.search(
        r"\b(DELETE|SET|CREATE|MERGE|INSERT|UPDATE)\b", str(statement), re.IGNORECASE
    )
