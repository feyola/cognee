"""Versioned wire contract for exact persisted chunk evidence, not ranking."""

import hashlib
from typing import Literal
from uuid import UUID

from fastapi.responses import JSONResponse
from pydantic import ConfigDict, Field, JsonValue, StrictInt, StrictStr, field_validator

from cognee.api.DTO import InDTO, OutDTO
from cognee.infrastructure.databases.provenance import NodeDeleteData
from cognee.modules.retrieval.chunk_records import (
    MAX_CHUNK_RECORD_IDS,
    MAX_CHUNK_RECORD_RESPONSE_BYTES,
    ChunkRecordIntegrityError,
    ChunkRecordResponseTooLarge,
)


class ChunkRecordsPayloadDTO(InDTO):
    model_config = ConfigDict(extra="forbid")
    dataset_id: UUID
    chunk_ids: list[UUID] = Field(min_length=1, max_length=MAX_CHUNK_RECORD_IDS)

    @field_validator("chunk_ids", mode="before")
    @classmethod
    def canonical_unique_ids(cls, value):
        if isinstance(value, list):
            for identity in value:
                if not isinstance(identity, (str, UUID)) or str(UUID(str(identity))) != str(
                    identity
                ):
                    raise ValueError("Chunk IDs must be canonical lowercase UUIDs")
            if len({str(identity) for identity in value}) != len(value):
                raise ValueError("Chunk IDs must be unique")
        return value


class ChunkRecordDTO(OutDTO):
    chunk_id: UUID
    type: Literal["DocumentChunk"]
    text: StrictStr
    utf8_sha256: StrictStr
    content_hash: StrictStr | None
    document_id: UUID | None
    document_name: StrictStr | None
    chunk_index: StrictInt | None = Field(ge=0)
    belongs_to_set: JsonValue
    external_metadata: JsonValue
    chunk_size: StrictInt | None = Field(ge=0)
    cut_type: StrictStr | None
    chunker_id: StrictStr | None
    max_chunk_tokens: StrictInt | None = Field(ge=1)
    source_task: StrictStr | None
    source_pipeline: StrictStr | None
    source_node_set: JsonValue
    source_user: JsonValue
    source_ref_keys: list[StrictStr]
    source_dataset_ids: list[UUID]
    source_run_ids: list[UUID]
    source_run_refs: list[StrictStr]


class ChunkRecordsResponseDTO(OutDTO):
    contract_version: Literal["chunk-records/v1"] = "chunk-records/v1"
    dataset_id: UUID
    records: list[ChunkRecordDTO]
    missing_chunk_ids: list[UUID]


_PROPERTY_FIELDS = (
    "content_hash",
    "document_id",
    "document_name",
    "chunk_index",
    "belongs_to_set",
    "external_metadata",
    "chunk_size",
    "cut_type",
    "chunker_id",
    "max_chunk_tokens",
    "source_task",
    "source_pipeline",
    "source_node_set",
    "source_user",
)


def render_chunk_records(
    dataset_id: UUID, snapshots: list[NodeDeleteData], missing_ids: list[UUID]
) -> JSONResponse:
    try:
        records = []
        for snapshot in snapshots:
            properties = snapshot.node_properties
            text = properties.get("text")
            if not isinstance(text, str):
                raise ChunkRecordIntegrityError()
            records.append(
                ChunkRecordDTO(
                    chunk_id=snapshot.node_id,
                    type=snapshot.node_type,
                    text=text,
                    utf8_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    **{field: properties.get(field) for field in _PROPERTY_FIELDS},
                    source_ref_keys=snapshot.source_ref_keys,
                    source_dataset_ids=snapshot.source_dataset_ids,
                    source_run_ids=snapshot.source_run_ids,
                    source_run_refs=snapshot.source_run_refs,
                )
            )
        dto = ChunkRecordsResponseDTO(
            dataset_id=dataset_id, records=records, missing_chunk_ids=missing_ids
        )
        response = JSONResponse(dto.model_dump(mode="json", by_alias=True))
    except ChunkRecordIntegrityError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ChunkRecordIntegrityError() from None
    if len(response.body) > MAX_CHUNK_RECORD_RESPONSE_BYTES:
        raise ChunkRecordResponseTooLarge()
    return response
