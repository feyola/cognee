"""Helpers for document metadata embedded in chunk text."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_front_matter(text: Any) -> dict[str, Any]:
    """Parse Cognee input that uses JSON-compatible Markdown front matter."""
    if not isinstance(text, str) or not text.startswith("---\n"):
        return {}
    closing = text.find("\n---\n", 4)
    if closing < 0:
        return {}
    metadata: dict[str, Any] = {}
    for line in text[4:closing].splitlines():
        key, separator, encoded = line.partition(": ")
        if not separator or not key:
            continue
        try:
            metadata[key] = json.loads(encoded)
        except json.JSONDecodeError:
            continue
    return metadata


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
