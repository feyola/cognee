"""Read exact chunk records; set COGNEE_BASE_URL/COGNEE_API_KEY, then supply
DATASET_UUID CHUNK_UUID ... . No ingestion, completion or implicit repair occurs.
Validate accepted generation/provenance/profile separately before publication.
"""

import argparse
import hashlib
import json
import os
from urllib.request import Request, urlopen
from uuid import UUID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id", type=UUID)
    parser.add_argument("chunk_ids", type=UUID, nargs="+")
    args = parser.parse_args()
    ids = [str(identity) for identity in args.chunk_ids]
    if not 1 <= len(ids) <= 100 or len(set(ids)) != len(ids):
        parser.error("Supply 1-100 unique chunk IDs")
    request = Request(
        os.environ["COGNEE_BASE_URL"].rstrip("/") + "/api/v1/search/chunk-records",
        data=json.dumps({"datasetId": str(args.dataset_id), "chunkIds": ids}).encode(),
        headers={"X-Api-Key": os.environ["COGNEE_API_KEY"], "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Response byte bound exceeded")
    value = json.loads(raw)
    if value["contractVersion"] != "chunk-records/v1" or value["datasetId"] != str(args.dataset_id):
        raise ValueError("Wrong record contract or dataset")
    records = value["records"]
    if value["missingChunkIds"] or [r["chunkId"] for r in records] != ids:
        raise ValueError("Incomplete or unexpected exact record set")
    for record in records:
        if (
            record["type"] != "DocumentChunk"
            or hashlib.sha256(record["text"].encode()).hexdigest() != record["utf8Sha256"]
        ):
            raise ValueError("Invalid authoritative chunk body")
        print(f"{record['chunkId']}: {len(record['text'].encode())} exact UTF-8 bytes")
    print(
        "Exact bodies verified; current-generation source/run/profile acceptance is a separate check."
    )


if __name__ == "__main__":
    main()
