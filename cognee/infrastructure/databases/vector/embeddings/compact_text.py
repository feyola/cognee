"""Helpers for keeping ad-hoc embedding inputs within configured model limits."""

import json
from typing import Any

DEFAULT_EMBEDDING_QUERY_MAX_CHARS = 8_000


def compact_embedding_text(
    text: str,
    *,
    max_chars: int = DEFAULT_EMBEDDING_QUERY_MAX_CHARS,
) -> str:
    """Bound text while retaining names and claims from structured responses."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return text

    compact: list[str] = []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None

    def collect(value: Any, *, include_string: bool = False) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                collect(nested, include_string=key in {"name", "description", "notes"})
        elif isinstance(value, list):
            for nested in value:
                collect(nested, include_string=include_string)
        elif include_string and isinstance(value, str) and value.strip():
            compact.append(value.strip())

    collect(parsed)
    if compact:
        text = "\n".join(dict.fromkeys(compact))
        if len(text) <= max_chars:
            return text

    first = max_chars // 2
    last = max_chars - first
    return text[:first] + "\n" + text[-last:]
