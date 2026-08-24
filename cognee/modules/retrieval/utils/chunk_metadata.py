"""Helpers for document metadata embedded in chunk text."""

from __future__ import annotations

import json
import re
from typing import Any


def split_json_front_matter(text: Any) -> tuple[dict[str, Any], str]:
    """Return JSON-compatible Markdown front matter and its answer-bearing body."""
    if not isinstance(text, str):
        return {}, ""
    normalized = text.lstrip("\ufeff\r\n").replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized
    closing = normalized.find("\n---\n", 4)
    if closing < 0:
        return {}, normalized
    metadata: dict[str, Any] = {}
    for line in normalized[4:closing].splitlines():
        key, separator, encoded = line.partition(": ")
        if not separator or not key:
            continue
        try:
            metadata[key] = json.loads(encoded)
        except json.JSONDecodeError:
            continue
    return metadata, normalized[closing + len("\n---\n") :]


def parse_json_front_matter(text: Any) -> dict[str, Any]:
    """Parse Cognee input that uses JSON-compatible Markdown front matter."""
    return split_json_front_matter(text)[0]


def normalize_search_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def canonical_page_key(payload: dict[str, Any], fallback: str) -> str:
    metadata = parse_json_front_matter(payload.get("text"))
    for key in ("canonical_url", "document_id", "title"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("document_id", "document_name", "id"):
        value = payload.get(key)
        if value is not None and str(value):
            return str(value)
    return fallback
