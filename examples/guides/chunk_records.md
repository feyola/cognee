# Exact persisted chunk records

`POST /api/v1/search/chunk-records` is a read-only, single-dataset evidence lookup.
Authenticate using an API key or the ordinary user session. It does not search,
complete, ingest, reindex, expand neighbors or change session memory.

Request (camelCase or snake_case field names):

```json
{"datasetId":"00000000-0000-0000-0000-000000000001","chunkIds":["00000000-0000-0000-0000-000000000002"]}
```

Send 1–100 **unique canonical lowercase UUIDs**. Extra fields are rejected. The
server checks read permission before selecting the dataset owner's graph backend.

The response has `contractVersion: "chunk-records/v1"`, `datasetId`, `records`,
and `missingChunkIds`. Records follow request order. Unknown IDs, including IDs
from another dataset, are omitted and included in `missingChunkIds`. A caller
requiring complete evidence must require an empty missing list and exactly the
full requested record ID set before using any record.

Each record contains:

- `chunkId`, `type: "DocumentChunk"`, exact `text`, and `utf8Sha256` computed from
  the unmodified text's UTF-8 bytes. This computed digest is not the stored
  `contentHash`, whose actual format/meaning depends on the ingestion profile.
- Actual stored `contentHash`, `documentId`, `documentName`, `chunkIndex`,
  `belongsToSet`, `externalMetadata`, `chunkSize`, `cutType`, `chunkerId`,
  `maxChunkTokens`, `sourceTask`, `sourcePipeline`, `sourceNodeSet`, `sourceUser`.
  Missing fields are explicit `null`, not invented defaults. `documentId` is
  the stored property; any equivalence with an accepted data UUID must be verified
  against provenance and the caller's accepted-generation catalog, not manufactured.
- Native normalized `sourceRefKeys`, `sourceDatasetIds`, `sourceRunIds`,
  `sourceRunRefs`. Empty lists mean the native read primitive returned no refs,
  including possible legacy gaps. They do **not** prove complete ownership or
  migration. No missing source, profile or hash field is inferred from text.

`belongsToSet`, `externalMetadata`, `sourceNodeSet`, and `sourceUser` preserve
JSON values exactly (string, object, array, scalar, or null), without parsing
serialized strings or changing their representation. The current document model
usually stores `externalMetadata` as a serialized JSON string, `sourceNodeSet`
as a list of node-set names, and `sourceUser` as a user UUID string. Legacy/custom
representations remain visible; validate the representation expected by your
accepted profile. For example these distinct wire values are not normalized:

```json
{"externalMetadata":"{\"source\":\"https://example.test/notes\"}","sourceNodeSet":["notes"],"sourceUser":"00000000-0000-0000-0000-000000000003"}
```

Native source serialization:

- Document ownership: `source_ref:v1:<dataset UUID>:<data UUID>`.
- Chunk ownership: `source_ref:v2:<dataset UUID>:<data UUID>:<owning chunk UUID>`.
- Run attachment: `source_run_ref:v1:<run UUID>:<source-ref key>` (the key is
  the complete v1 or v2 string above).

`sourceRefKeys` express dataset/data ownership; `sourceDatasetIds` are the native
dataset-filter index; `sourceRunIds` are the native run-filter index;
`sourceRunRefs` attach an exact run to an exact source ref. They are returned as
stored, not synthesized from each other. UUID-valued arrays serialize as canonical
UUID strings. Every ref must be representable within the requested dataset scope.
Foreign/mixed scopes, malformed refs, or run IDs without a dataset-bound run ref
fail without disclosing records or the foreign values. Empty legacy ref lists can
be inspected, but a strict current-generation evidence policy must treat them as gaps.

The entire camelCase UTF-8 JSON response is capped at **1 MiB**. Oversized responses
fail with typed `ChunkRecordResponseTooLarge` / HTTP 413; nothing is truncated.
Malformed, unexpected or non-chunk backend records fail all-or-nothing with
`ChunkRecordIntegrityError` / HTTP 502. Permission/scope failures are HTTP 403.
The full application renders request validation as HTTP 400 (isolated FastAPI
routers ordinarily use 422). Unsupported graph provenance reads fail explicitly
with HTTP 501; no visualization, similarity or synthetic-provenance fallback exists.

Backend selection is bounded to the exact requested IDs and count. The 1 MiB cap
is a **response** byte bound, not a claim that every backend preflights row bytes
before reading them. PostgreSQL, Ladybug and Neo4j implement the native read-only
snapshot method `get_node_delete_data`; its name reflects other uses in deletion
planning, but this endpoint calls only its read operation. Other adapters must
support that contract or report unsupported. Inspected/mocked adapter coverage
is distinct from live deployment qualification.

Record existence and exact bytes do not alone establish freshness or answer support.
Bind records to an accepted generation/run, data UUID, source and chunk/profile
catalog. Ranking results may provide identity/rank signals; do not treat a normalized
vector payload as authoritative text. This endpoint imposes no new requirements on
existing search/member workflows and never triggers reingestion of legacy corpora.
