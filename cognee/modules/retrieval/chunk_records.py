"""Exact, read-only chunk snapshots with non-leaking dataset provenance scope.

The native method is named ``get_node_delete_data`` because its snapshots also
serve deletion planning. It only reads records; no delete or cleanup is invoked.
Adapters without this capability must fail explicitly, not invent provenance.
"""

from uuid import UUID

from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)
from cognee.exceptions import CogneeApiError
from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance import NodeDeleteData
from cognee.infrastructure.databases.provenance.source_refs import (
    get_dataset_id_from_source_ref_key,
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
)
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
    get_specific_user_permission_datasets,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)
MAX_CHUNK_RECORD_IDS = 100
MAX_CHUNK_RECORD_RESPONSE_BYTES = 1024 * 1024


class ChunkRecordIntegrityError(CogneeApiError):
    def __init__(self):
        super().__init__(
            "Authoritative chunk records could not be read safely.",
            "ChunkRecordIntegrityError",
            502,
            remediation="Check stored record integrity; no partial records were returned.",
        )


class ChunkRecordResponseTooLarge(CogneeApiError):
    def __init__(self):
        super().__init__(
            "Chunk record response exceeds the 1 MiB UTF-8 JSON limit.",
            "ChunkRecordResponseTooLarge",
            413,
            remediation="Request fewer chunk IDs; records are never truncated.",
        )


def _check_source_scope(snapshot: NodeDeleteData, dataset_id: UUID) -> None:
    """Validate every persisted normalized ref before any record is disclosed."""
    arrays = (
        snapshot.source_ref_keys,
        snapshot.source_dataset_ids,
        snapshot.source_run_ids,
        snapshot.source_run_refs,
    )
    if any(not isinstance(values, list) for values in arrays):
        raise ChunkRecordIntegrityError()
    if any(not isinstance(value, str) for values in arrays for value in values):
        raise ChunkRecordIntegrityError()
    datasets = {UUID(value) for value in snapshot.source_dataset_ids}
    datasets.update(get_dataset_id_from_source_ref_key(ref) for ref in snapshot.source_ref_keys)
    scoped_runs = set()
    for run_ref in snapshot.source_run_refs:
        datasets.add(
            get_dataset_id_from_source_ref_key(get_source_ref_key_from_source_run_ref(run_ref))
        )
        scoped_runs.add(get_pipeline_run_id_from_source_run_ref(run_ref))
    runs = {UUID(value) for value in snapshot.source_run_ids}
    if datasets - {dataset_id} or runs - scoped_runs:
        # Orphan run IDs cannot be proven to belong to this dataset. Do not
        # expose even their values, or silently filter away mixed ownership.
        raise PermissionDeniedError(
            "Requested chunk provenance cannot be represented within the authorized dataset scope."
        )


async def read_chunk_record_snapshots(
    user: User, dataset_id: UUID, chunk_ids: list[UUID]
) -> tuple[list[NodeDeleteData], list[UUID]]:
    if not 1 <= len(chunk_ids) <= MAX_CHUNK_RECORD_IDS or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Expected 1-100 unique chunk IDs")
    datasets = await get_specific_user_permission_datasets(user.id, "read", [dataset_id])
    dataset = datasets[0]
    requested = [str(chunk_id) for chunk_id in chunk_ids]
    try:
        if not backend_access_control_enabled():
            # Global storage cannot establish dataset membership for legacy
            # snapshots with empty refs. Never disclose those by exact ID alone.
            raise UnsupportedProvenanceCapability()
        async with set_database_global_context_variables(dataset.id, dataset.owner_id):
            graph = await get_graph_engine()
            reader = getattr(graph, "get_node_delete_data", None)
            if not callable(reader):
                raise UnsupportedProvenanceCapability()
            found = await reader(requested)
        if not isinstance(found, dict) or len(found) > len(requested):
            raise ChunkRecordIntegrityError()
        seen = set()
        for key, snapshot in found.items():
            if (
                key not in requested
                or not isinstance(snapshot, NodeDeleteData)
                or snapshot.node_id != key
                or snapshot.node_id in seen
                or snapshot.node_type != "DocumentChunk"
                or not isinstance(snapshot.node_properties, dict)
                or snapshot.node_properties.get("id", key) != key
                or snapshot.node_properties.get("type", "DocumentChunk") != "DocumentChunk"
            ):
                raise ChunkRecordIntegrityError()
            seen.add(snapshot.node_id)
            _check_source_scope(snapshot, dataset_id)
        return (
            [found[chunk_id] for chunk_id in requested if chunk_id in found],
            [chunk_id for chunk_id in chunk_ids if str(chunk_id) not in found],
        )
    except CogneeApiError:
        raise
    except Exception:
        logger.exception("Exact chunk snapshot read failed")
        raise ChunkRecordIntegrityError() from None
