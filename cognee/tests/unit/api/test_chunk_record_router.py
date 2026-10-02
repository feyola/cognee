"""Exact chunk evidence: authorization, honest gaps, integrity and byte bounds."""

import hashlib
import importlib
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cognee.exceptions import CogneeApiError
from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import (
    NodeDeleteData,
    make_chunk_source_ref_key,
    make_source_ref_key,
    make_source_run_ref,
)
from cognee.modules.users.exceptions import PermissionDeniedError

router = importlib.import_module("cognee.api.v1.search.routers.get_search_router")
reader = importlib.import_module("cognee.modules.retrieval.chunk_records")
dto = importlib.import_module("cognee.api.v1.search.chunk_record_dto")


@pytest.fixture
def lookup(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    chunk, document, run = uuid4(), uuid4(), uuid4()
    source = make_source_ref_key(dataset.id, document)
    props = {
        "text": ' \nUnicode α 猫 🙂\r\n"quoted"\n ',
        "document_id": str(document),
        "document_name": "notes.md",
        "chunk_index": 0,
        "content_hash": "stored-hash",
        "chunk_size": 32,
        "chunker_id": "fixture",
        "max_chunk_tokens": 4096,
        "cut_type": "limit",
        "belongs_to_set": ["notes"],
        "external_metadata": '{"source":"https://example.test/notes"}',
        "source_task": "extract",
        "source_pipeline": "cognify",
        "source_node_set": ["notes"],
        "source_user": str(dataset.owner_id),
    }
    snapshot = NodeDeleteData(
        str(chunk),
        "DocumentChunk",
        ["text"],
        props,
        [source],
        [str(dataset.id)],
        [str(run)],
        [make_source_run_ref(run, source)],
    )
    permission = AsyncMock(return_value=[dataset])
    read = AsyncMock(return_value={str(chunk): snapshot})
    graph = AsyncMock(return_value=SimpleNamespace(get_node_delete_data=read))
    events = []

    @asynccontextmanager
    async def scope(dataset_id, owner_id):
        events.append(("enter", dataset_id, owner_id))
        try:
            yield
        finally:
            events.append(("exit", dataset_id, owner_id))

    monkeypatch.setattr(reader, "get_specific_user_permission_datasets", permission)
    monkeypatch.setattr(reader, "set_database_global_context_variables", scope)
    monkeypatch.setattr(reader, "get_graph_engine", graph)
    app = FastAPI()
    app.include_router(router.get_search_router(), prefix="/api/v1/search")
    app.dependency_overrides[router.get_authenticated_user] = lambda: user

    @app.exception_handler(CogneeApiError)
    async def handler(_request, error):
        return JSONResponse(
            status_code=error.status_code, content={"detail": error.message, "name": error.name}
        )

    return SimpleNamespace(
        client=TestClient(app, raise_server_exceptions=False),
        app=app,
        user=user,
        dataset=dataset,
        chunk=chunk,
        document=document,
        run=run,
        snapshot=snapshot,
        permission=permission,
        read=read,
        graph=graph,
        events=events,
    )


def post(lookup, ids=None, **extra):
    return lookup.client.post(
        "/api/v1/search/chunk-records",
        json={
            "datasetId": str(lookup.dataset.id),
            "chunkIds": ids or [str(lookup.chunk)],
            **extra,
        },
    )


def test_exact_bytes_fields_provenance_and_owner_scope(lookup):
    response = post(lookup)
    assert response.status_code == 200, response.text
    wire = response.json()
    assert wire["contractVersion"] == "chunk-records/v1"
    assert wire["datasetId"] == str(lookup.dataset.id) and wire["missingChunkIds"] == []
    record = wire["records"][0]
    assert record["chunkId"] == str(lookup.chunk) and record["type"] == "DocumentChunk"
    assert record["text"] == lookup.snapshot.node_properties["text"]
    assert record["utf8Sha256"] == hashlib.sha256(record["text"].encode()).hexdigest()
    assert record["contentHash"] == "stored-hash"
    assert record["documentId"] == str(lookup.document) and record["chunkIndex"] == 0
    assert record["externalMetadata"] == lookup.snapshot.node_properties["external_metadata"]
    assert record["sourceNodeSet"] == ["notes"]
    assert record["sourceUser"] == str(lookup.dataset.owner_id)
    assert record["sourceRefKeys"] == lookup.snapshot.source_ref_keys
    assert record["sourceRunRefs"] == lookup.snapshot.source_run_refs
    assert record["sourceDatasetIds"] == [str(lookup.dataset.id)]
    assert record["sourceRunIds"] == [str(lookup.run)]
    lookup.permission.assert_awaited_once_with(lookup.user.id, "read", [lookup.dataset.id])
    lookup.read.assert_awaited_once_with([str(lookup.chunk)])
    assert lookup.events == [
        ("enter", lookup.dataset.id, lookup.dataset.owner_id),
        ("exit", lookup.dataset.id, lookup.dataset.owner_id),
    ]


def test_missing_foreign_and_unknown_ids_are_explicit_subset_gaps(lookup):
    absent = str(uuid4())
    response = post(lookup, [absent, str(lookup.chunk)])
    assert response.status_code == 200
    assert response.json()["missingChunkIds"] == [absent]
    assert [r["chunkId"] for r in response.json()["records"]] == [str(lookup.chunk)]


def test_request_order_not_backend_order(lookup):
    other = str(uuid4())
    snapshot = replace(lookup.snapshot, node_id=other)
    lookup.read.return_value = {other: snapshot, str(lookup.chunk): lookup.snapshot}
    response = post(lookup, [str(lookup.chunk), other])
    assert response.status_code == 200
    assert [r["chunkId"] for r in response.json()["records"]] == [str(lookup.chunk), other]


def test_chunk_scoped_v2_refs_are_preserved(lookup):
    source = make_chunk_source_ref_key(lookup.dataset.id, lookup.document, lookup.chunk)
    run_ref = make_source_run_ref(lookup.run, source)
    lookup.read.return_value = {
        str(lookup.chunk): replace(
            lookup.snapshot, source_ref_keys=[source], source_run_refs=[run_ref]
        )
    }
    response = post(lookup)
    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["sourceRefKeys"] == [source] and record["sourceRunRefs"] == [run_ref]


def test_legacy_gaps_are_null_or_empty_not_invented(lookup):
    snapshot = replace(
        lookup.snapshot,
        node_properties={"text": " \n "},
        source_ref_keys=[],
        source_dataset_ids=[],
        source_run_ids=[],
        source_run_refs=[],
    )
    lookup.read.return_value = {str(lookup.chunk): snapshot}
    response = post(lookup)
    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["text"] == " \n "
    assert all(
        record[field] is None
        for field in (
            "contentHash",
            "documentId",
            "documentName",
            "chunkIndex",
            "chunkSize",
            "chunkerId",
            "maxChunkTokens",
            "externalMetadata",
        )
    )
    assert all(
        record[field] == []
        for field in ("sourceRefKeys", "sourceDatasetIds", "sourceRunIds", "sourceRunRefs")
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("external_metadata", {"nested": ["raw", 1]}),
        ("source_node_set", "legacy-name"),
        ("source_user", None),
    ],
)
def test_native_json_field_representations_are_preserved(lookup, field, value):
    props = {**lookup.snapshot.node_properties, field: value}
    lookup.read.return_value = {str(lookup.chunk): replace(lookup.snapshot, node_properties=props)}
    record = post(lookup).json()["records"][0]
    assert (
        record[
            {
                "external_metadata": "externalMetadata",
                "source_node_set": "sourceNodeSet",
                "source_user": "sourceUser",
            }[field]
        ]
        == value
    )


def test_permission_denied_before_scope_or_backend(lookup):
    lookup.permission.side_effect = PermissionDeniedError(log=False)
    assert post(lookup).status_code == 403
    lookup.graph.assert_not_awaited()
    lookup.read.assert_not_awaited()
    assert lookup.events == []


def test_unauthenticated_never_checks_permission(lookup):
    def denied():
        raise HTTPException(status_code=401)

    lookup.app.dependency_overrides[router.get_authenticated_user] = denied
    assert post(lookup).status_code == 401
    lookup.permission.assert_not_awaited()
    lookup.graph.assert_not_awaited()


@pytest.mark.parametrize(
    "ids",
    [
        [],
        ["bad"],
        ["f" * 32],
        [str(uuid4()).upper()],
        [str(uuid4())] * 2,
        [str(uuid4()) for _ in range(101)],
    ],
)
def test_invalid_canonical_unique_or_bounded_ids_never_read(lookup, ids):
    response = lookup.client.post(
        "/api/v1/search/chunk-records", json={"datasetId": str(lookup.dataset.id), "chunkIds": ids}
    )
    assert response.status_code == 422
    lookup.permission.assert_not_awaited()
    lookup.read.assert_not_awaited()


def test_extra_fields_rejected_no_query_or_widening_option(lookup):
    assert post(lookup, query="notes", full=True).status_code == 422
    lookup.read.assert_not_awaited()


def test_one_hundred_unique_ids_are_supported(lookup):
    ids = [str(uuid4()) for _ in range(100)]
    lookup.read.return_value = {}
    response = post(lookup, ids)
    assert response.status_code == 200 and response.json()["records"] == []
    assert response.json()["missingChunkIds"] == ids
    lookup.read.assert_awaited_once_with(ids)


@pytest.mark.parametrize(
    "malformation",
    [
        "unexpected",
        "nonchunk",
        "wrong_identity",
        "not_snapshot",
        "not_dict",
        "missing_text",
        "invalid_index",
        "invalid_refs",
        "invalid_ref_array",
        "invalid_run_ref",
        "wrong_property_id",
        "wrong_property_type",
    ],
)
def test_malformed_backend_result_returns_no_partial_records(lookup, malformation):
    record = lookup.snapshot
    if malformation == "unexpected":
        lookup.read.return_value = {str(uuid4()): record}
    elif malformation == "not_dict":
        lookup.read.return_value = [record]
    elif malformation == "not_snapshot":
        lookup.read.return_value = {str(lookup.chunk): {"text": "not trusted"}}
    else:
        if malformation == "nonchunk":
            record = replace(record, node_type="Entity")
        if malformation == "wrong_identity":
            record = replace(record, node_id=str(uuid4()))
        if malformation == "missing_text":
            record = replace(record, node_properties={})
        if malformation == "invalid_index":
            record = replace(
                record, node_properties={**record.node_properties, "chunk_index": True}
            )
        if malformation == "invalid_refs":
            record = replace(record, source_ref_keys=["private-malformed-ref"])
        if malformation == "invalid_ref_array":
            record = replace(record, source_ref_keys="private-malformed-ref")
        if malformation == "invalid_run_ref":
            record = replace(record, source_run_refs=["private-malformed-ref"])
        if malformation == "wrong_property_id":
            record = replace(record, node_properties={**record.node_properties, "id": str(uuid4())})
        if malformation == "wrong_property_type":
            record = replace(record, node_properties={**record.node_properties, "type": "Entity"})
        lookup.read.return_value = {str(lookup.chunk): record}
    response = post(lookup)
    assert response.status_code == 502
    assert response.json()["name"] == "ChunkRecordIntegrityError"
    assert "records" not in response.json() and "private-malformed-ref" not in response.text


@pytest.mark.parametrize(
    "field", ["source_ref_keys", "source_dataset_ids", "source_run_refs", "orphan_run"]
)
def test_mixed_or_unscoped_provenance_never_disclosed(lookup, field):
    foreign = uuid4()
    source = make_source_ref_key(foreign, uuid4())
    if field == "source_ref_keys":
        record = replace(lookup.snapshot, source_ref_keys=[source])
    if field == "source_dataset_ids":
        record = replace(lookup.snapshot, source_dataset_ids=[str(foreign)])
    if field == "source_run_refs":
        record = replace(lookup.snapshot, source_run_refs=[make_source_run_ref(lookup.run, source)])
    if field == "orphan_run":
        record = replace(lookup.snapshot, source_run_ids=[str(foreign)], source_run_refs=[])
    lookup.read.return_value = {str(lookup.chunk): record}
    response = post(lookup)
    assert (
        response.status_code == 403
        and str(foreign) not in response.text
        and source not in response.text
    )
    assert "records" not in response.json()


def test_backend_failure_fixed_error_and_scope_teardown(lookup):
    lookup.read.side_effect = RuntimeError("private-backend-sentinel")
    response = post(lookup)
    assert response.status_code == 502 and "private-backend-sentinel" not in response.text
    assert lookup.events[-1][0] == "exit"


def test_unsupported_adapter_is_not_empty_fallback(lookup):
    lookup.read.side_effect = UnsupportedProvenanceCapability()
    assert post(lookup).status_code == 501


def test_absent_native_capability_is_explicitly_unsupported(lookup):
    lookup.graph.return_value = SimpleNamespace()
    assert post(lookup).status_code == 501


def test_mixed_batch_never_discloses_even_the_valid_record(lookup):
    foreign_chunk, foreign_dataset = str(uuid4()), uuid4()
    mixed = replace(
        lookup.snapshot,
        node_id=foreign_chunk,
        source_dataset_ids=[str(foreign_dataset)],
    )
    lookup.read.return_value = {str(lookup.chunk): lookup.snapshot, foreign_chunk: mixed}
    response = post(lookup, [str(lookup.chunk), foreign_chunk])
    assert response.status_code == 403
    assert "records" not in response.json()
    assert lookup.snapshot.node_properties["text"] not in response.text
    assert str(foreign_dataset) not in response.text


def test_whole_utf8_json_bound_not_character_count_or_truncation(lookup, monkeypatch):
    baseline = post(lookup)
    monkeypatch.setattr(dto, "MAX_CHUNK_RECORD_RESPONSE_BYTES", len(baseline.content))
    assert post(lookup).status_code == 200
    monkeypatch.setattr(dto, "MAX_CHUNK_RECORD_RESPONSE_BYTES", len(baseline.content) - 1)
    response = post(lookup)
    assert response.status_code == 413 and response.json()["name"] == "ChunkRecordResponseTooLarge"
    assert "records" not in response.json()
    assert len(baseline.content) > len(baseline.content.decode("utf-8"))


def test_openapi_is_versioned_camelcase_and_forbids_options(lookup):
    schemas = lookup.app.openapi()["components"]["schemas"]
    request = schemas["ChunkRecordsPayloadDTO"]
    assert request["additionalProperties"] is False
    assert request["properties"]["chunkIds"]["maxItems"] == 100
    response = schemas["ChunkRecordsResponseDTO"]
    assert response["properties"]["contractVersion"]["const"] == "chunk-records/v1"
    fields = schemas["ChunkRecordDTO"]["properties"]
    assert "utf8Sha256" in fields and "sourceRunRefs" in fields and "externalMetadata" in fields


def test_actual_one_mib_limit_is_enforced(lookup):
    props = {**lookup.snapshot.node_properties, "text": "x" * (1024 * 1024)}
    lookup.read.return_value = {str(lookup.chunk): replace(lookup.snapshot, node_properties=props)}
    response = post(lookup)
    assert response.status_code == 413
    assert response.json()["name"] == "ChunkRecordResponseTooLarge"
    assert "records" not in response.json()
