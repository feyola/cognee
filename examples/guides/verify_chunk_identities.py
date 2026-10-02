"""Verify chunk existence through the authenticated, dataset-scoped REST API.

Set COGNEE_BASE_URL and COGNEE_API_KEY, then pass a dataset UUID and 1-100 chunk
UUIDs: python examples/guides/verify_chunk_identities.py DATASET_ID CHUNK_ID ...

This read-only check never ingests, completes or modifies session memory. It
requires *every* supplied ID to exist in the chosen dataset's chunk index; an
omitted ID fails the check. Existence alone does not establish answer support,
source freshness or exact content: validate those separately before publishing.
"""

import argparse
import json
import os
from urllib.request import Request, urlopen
from uuid import UUID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id", type=UUID)
    parser.add_argument("chunk_ids", type=UUID, nargs="+")
    args = parser.parse_args()
    if len(args.chunk_ids) > 100:
        parser.error("At most 100 chunk IDs are supported per request")
    base_url = os.environ["COGNEE_BASE_URL"].rstrip("/")
    token = os.environ["COGNEE_API_KEY"]
    requested = list(dict.fromkeys(str(chunk_id) for chunk_id in args.chunk_ids))
    request = Request(
        f"{base_url}/api/v1/search/chunk-identities",
        data=json.dumps({"datasetId": str(args.dataset_id), "chunkIds": requested}).encode(),
        headers={"X-Api-Key": token, "Content-Type": "application/json"},
        method="POST",
    )
    # HTTP/auth/backend errors intentionally fail; do not treat them as no matches.
    with urlopen(request, timeout=30) as response:
        present = json.load(response)
    if not isinstance(present, list) or any(not isinstance(value, str) for value in present):
        raise ValueError("Invalid identity verification response")
    if not set(present).issubset(requested):
        raise ValueError("Server returned an identity outside the requested set")
    missing = [chunk_id for chunk_id in requested if chunk_id not in present]
    if missing:
        raise SystemExit(f"Not verified in this dataset: {', '.join(missing)}")
    print(f"Verified {len(requested)} chunk identities in the requested dataset")


if __name__ == "__main__":
    main()
