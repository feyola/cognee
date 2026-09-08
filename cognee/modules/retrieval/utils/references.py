"""Deterministic, LLM-free helpers for building reference (Evidence) blocks.

Evidence is grounded in the generated answer, not in whatever happened to be
retrieved before the LLM was called:

- ``format_chunk_references`` builds an Evidence block from retrieved vector
  payloads, keeping only chunks that share significant terms with the answer
  (the ``RAG_COMPLETION`` / chunk path, where candidates are the chunks the
  LLM actually read).
- ``build_answer_grounded_chunk_references`` is retained as a compatibility
  helper for callers that explicitly request a post-hoc similarity lookup. The
  graph completion path no longer uses it; graph provenance comes directly from
  the graph objects included in the LLM context.
- ``append_chunk_evidence`` / ``append_answer_grounded_evidence`` apply the
  above to a list of completions, one Evidence block per string completion.

All helpers are pure with respect to the LLM (no model calls) so they can be
unit tested in isolation. All return ``""`` (or the completions unchanged)
when there is nothing usable, and never raise on backend failures.
"""

import re
from collections import Counter
from typing import Any, List, Optional, Set, Tuple

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.databases.vector.embeddings.compact_text import (
    compact_embedding_text,
)
from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.utils.chunk_metadata import split_json_front_matter

logger = get_logger("references")

# Header emitted on its own line above the bullets. Kept here so both helpers
# and the wiring code agree on the exact literal.
EVIDENCE_HEADER = "Evidence:"

# Maximum length of a rendered text snippet (characters) before truncation.
_SNIPPET_MAX_CHARS = 1_200
_SNIPPET_HEAD_CHARS = 480

# Hard upper bound on bullets regardless of the requested limit (3-5 range).
_MAX_BULLETS = 5
_MIN_LIMIT = 3

# Vector collection holding document chunks (same one ChunksRetriever queries).
_CHUNK_COLLECTION = "DocumentChunk_text"

# How many vector candidates to fetch before answer-overlap filtering.
_CANDIDATE_POOL = 10


def _answer_body(text: str) -> str:
    """Remove machine metadata so overlap and snippets represent supporting prose."""
    _metadata, body = split_json_front_matter(text)
    lines = body.splitlines()
    while lines and (
        not lines[0].strip()
        or _is_provenance_banner(lines[0])
    ):
        lines.pop(0)
    return "\n".join(lines).strip()


def _is_provenance_banner(line: str) -> bool:
    """Recognize a quoted producer provenance banner without naming the producer."""
    label, separator, _details = line.lstrip().partition(":")
    return bool(separator) and label.startswith(">") and label[1:].strip().casefold().endswith(
        "provenance"
    )


# Common English words excluded from answer/chunk term overlap scoring.
_STOPWORDS = frozenset(
    """
    a about above after again all also an and any are as at be because been
    before being below between both but by can did do does doing down during
    each few for from further had has have having he her here hers him his how
    i if in into is it its just me more most my no nor not of off on once only
    or other our ours out over own same she should so some such than that the
    their theirs them then there these they this those through to too under
    until up very was we were what when where which while who whom why will
    with you your yours
    nodes edges source target relationship description notes id type name
    """.split()
)


def _significant_term_weights(text: str) -> dict[str, float]:
    """Weight claim terms so entity evidence beats response-format boilerplate."""
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if (len(token) >= 3 or any(character.isdigit() for character in token))
        and token not in _STOPWORDS
    ]
    counts = Counter(tokens)
    return {
        token: 1.0
        + min(count, 5)
        + min(len(token), 12) / 12
        # Alphanumeric codes and numeric values (E587, 7.5, etc.) are usually
        # the decisive part of a claim. Make a window containing them beat a
        # nearby paragraph with several generic overlapping nouns.
        + (8.0 if any(character.isdigit() for character in token) else 0.0)
        for token, count in counts.items()
    }


def _answer_phrases(text: str) -> set[str]:
    """Keep short answer phrases that can anchor a citation excerpt."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    phrases: set[str] = set()
    for size in (2, 3, 4):
        for index in range(len(tokens) - size + 1):
            group = tokens[index : index + size]
            significant = [
                token
                for token in group
                if (len(token) >= 3 or any(character.isdigit() for character in token))
                and token not in _STOPWORDS
            ]
            if len(significant) >= 2:
                phrases.add(" ".join(group))
    return phrases


def _clamp_limit(limit: int) -> int:
    """Clamp the requested bullet limit into the contracted 3-5 range."""
    if limit < _MIN_LIMIT:
        return _MIN_LIMIT
    if limit > _MAX_BULLETS:
        return _MAX_BULLETS
    return limit


def _clean_str(value: Any) -> Optional[str]:
    """Return a stripped string, or None if the value is unusable.

    Missing, null, non-string, or empty/whitespace-only values are treated as
    unusable (the common state for data indexed before reference fields
    existed).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        # Numbers etc. are not valid document names / text; reject defensively.
        return None
    stripped = value.strip()
    return stripped or None


def _snippet(
    text: str,
    focus_terms: Optional[Set[str]] = None,
    focus_weights: Optional[dict[str, float]] = None,
    focus_phrases: Optional[Set[str]] = None,
) -> str:
    """Return a compact supporting excerpt, focused on answer terms when supplied."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_MAX_CHARS:
        return collapsed
    start = 0
    if focus_terms:
        candidates = {0}
        for match in re.finditer(r"[a-z0-9]+", collapsed.lower()):
            if match.group() in focus_terms:
                candidates.add(max(0, match.start() - 80))
                candidates.add(max(0, match.start() - _SNIPPET_MAX_CHARS // 3))
        for phrase in focus_phrases or ():
            for match in re.finditer(rf"\b{re.escape(phrase)}\b", collapsed.lower()):
                candidates.add(max(0, match.start() - 120))

        def score(
            offset: int, window_chars: int = _SNIPPET_MAX_CHARS
        ) -> tuple[float, int, int, int, int]:
            excerpt = collapsed[offset : offset + window_chars]
            terms = set(re.findall(r"[a-z0-9]+", excerpt.lower()))
            covered = focus_terms & terms
            weighted = sum((focus_weights or {}).get(term, 1.0) for term in covered)
            phrase_score = sum(
                len(phrase.split()) * 50
                for phrase in focus_phrases or ()
                if phrase in excerpt.lower()
            )
            phrase_margin = max(
                (
                    min(match.start(), window_chars - match.end())
                    for phrase in focus_phrases or ()
                    for match in re.finditer(rf"\b{re.escape(phrase)}\b", excerpt.lower())
                ),
                default=0,
            )
            distinctive_margin = max(
                (
                    min(match.start(), window_chars - match.end())
                    for match in re.finditer(r"[a-z0-9]+", excerpt.lower())
                    if match.group() in covered
                    and any(character.isdigit() for character in match.group())
                ),
                default=0,
            )
            return (
                weighted + phrase_score,
                phrase_margin,
                distinctive_margin,
                len(covered),
                -offset,
            )

        start = max(candidates, key=score)
    if start > _SNIPPET_HEAD_CHARS:
        # Preserve the chunk's identifying lead (for example a ship/faction
        # infobox row) alongside the answer-focused region. This makes
        # multi-hop citations inspectable instead of silently dropping the
        # first link in the chain when the strongest answer term is later.
        head = collapsed[:_SNIPPET_HEAD_CHARS].rstrip()
        tail_limit = _SNIPPET_MAX_CHARS - len(head) - 3
        if focus_terms:
            start = max(candidates, key=lambda offset: score(offset, tail_limit))
        tail = collapsed[start : start + tail_limit].strip()
        excerpt = f"{head} … {tail}"
        if start + tail_limit < len(collapsed):
            excerpt = excerpt[:-1].rstrip() + "…"
        return excerpt
    excerpt = collapsed[start : start + _SNIPPET_MAX_CHARS]
    if start:
        excerpt = "…" + excerpt[1:]
    if start + _SNIPPET_MAX_CHARS < len(collapsed):
        excerpt = excerpt[:-1].rstrip() + "…"
    return excerpt


def _chunk_number(payload: dict) -> Optional[int]:
    """Resolve the 1-based display number from payload.

    Prefers an explicit ``chunk_number`` if present; otherwise derives it from
    the 0-based ``chunk_index`` as ``chunk_index + 1``. Returns None when no
    usable index information is present.
    """
    chunk_number = payload.get("chunk_number")
    if isinstance(chunk_number, bool):  # guard: bool is an int subclass
        chunk_number = None
    if isinstance(chunk_number, int) and chunk_number > 0:
        return chunk_number

    chunk_index = payload.get("chunk_index")
    if isinstance(chunk_index, bool):
        chunk_index = None
    if isinstance(chunk_index, int) and chunk_index >= 0:
        return chunk_index + 1

    return None


def _get_payload(obj: Any) -> Optional[dict]:
    """Extract a payload dict from a retrieved object.

    Retrieved objects are ``ScoredResult`` instances exposing ``.payload`` as a
    dict, but we also tolerate a raw dict or any object carrying a ``payload``
    attribute so the helper stays unit-testable without constructing a full
    ``ScoredResult``.
    """
    if isinstance(obj, dict):
        # Either the object IS the payload, or it wraps one under "payload".
        inner = obj.get("payload")
        if isinstance(inner, dict):
            return inner
        return obj

    payload = getattr(obj, "payload", None)
    if isinstance(payload, dict):
        return payload
    return None


def _provenance_suffix(
    source_id: Optional[str],
    data_id: Optional[str],
    chunk_id: Optional[str],
    dataset_id: Optional[str],
    node_sets: tuple[str, ...],
) -> str:
    """Render source, ingested-data, and internal-chunk identities when available.

    Lets a reader map the citation back to the ingested data item and the exact
    cited chunk, instead of only a (possibly auto-generated) document name and a
    positional chunk number.
    """
    parts = []
    if source_id:
        parts.append(f"source_id: {source_id}")
    if data_id:
        parts.append(f"data_id: {data_id}")
    if chunk_id:
        parts.append(f"chunk_id: {chunk_id}")
    if dataset_id:
        parts.append(f"dataset_id: {dataset_id}")
    if node_sets:
        parts.append(f"node_sets: {'|'.join(node_sets)}")
    return f" ({', '.join(parts)})" if parts else ""


def _dataset_provenance(payload: dict) -> tuple[Optional[str], tuple[str, ...]]:
    """Resolve the request dataset and exact node-set memberships for a chunk."""
    dataset_id = _clean_str(payload.get("dataset_id"))
    if dataset_id is None:
        active_dataset_id = current_dataset_id.get()
        if active_dataset_id is not None:
            dataset_id = str(active_dataset_id)

    raw_node_sets = payload.get("belongs_to_set")
    if isinstance(raw_node_sets, str):
        node_sets = (raw_node_sets.strip(),) if raw_node_sets.strip() else ()
    elif isinstance(raw_node_sets, (list, tuple, set)):
        node_sets = tuple(sorted({str(value).strip() for value in raw_node_sets if str(value).strip()}))
    else:
        node_sets = ()
    return dataset_id, node_sets


def _chunk_id(obj: Any, payload: dict) -> Optional[str]:
    """Resolve a stable chunk id for dedup, preferring the object id."""
    obj_id = getattr(obj, "id", None)
    if obj_id is not None:
        return str(obj_id)
    payload_id = payload.get("id")
    if payload_id is not None:
        return str(payload_id)
    # No stable id: fall back to (document_name, chunk_number) signature so we
    # still avoid duplicate bullets, computed by the caller from the payload.
    return None


def _significant_terms(text: str) -> Set[str]:
    """Lowercased alphanumeric terms of an answer, minus stopwords and stubs."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) >= 3 and token not in _STOPWORDS}


def format_chunk_references(
    retrieved_objects: Any, answer: Optional[str] = None, limit: int = 5
) -> str:
    """Build an Evidence block from retrieved vector payloads, grounded in the answer.

    Reads ``payload["document_name"]``, ``payload["chunk_number"]`` (falling back
    to ``payload["chunk_index"] + 1``), and ``payload["text"]`` from each
    retrieved object. Entries missing usable document name or chunk-number
    metadata are skipped. Results are deduplicated by chunk id and capped at
    3-5 bullets.

    When ``answer`` is provided, candidates that share no significant terms
    with the answer are dropped and the remainder is ranked by term overlap,
    so bullets reflect answer provenance rather than retrieval order. An
    answer with no significant terms cannot be grounded and yields ``""``.

    Parameters
    ----------
    retrieved_objects:
        An iterable of retrieved vector results (``ScoredResult``-like objects
        exposing a ``.payload`` dict), or raw payload dicts.
    answer:
        The generated answer text used to filter and rank candidates. When
        None, candidates keep their retrieval order unfiltered.
    limit:
        Desired maximum number of bullets, clamped into the 3-5 range.

    Returns
    -------
    str
        A multi-line Evidence block prefixed by an ``Evidence:`` header, or an
        empty string when nothing usable was found.
    """
    if not retrieved_objects:
        return ""

    try:
        iterator = list(retrieved_objects)
    except TypeError:
        return ""

    answer_terms: Optional[Set[str]] = None
    answer_term_weights: Optional[dict[str, float]] = None
    answer_phrases: Optional[Set[str]] = None
    if answer is not None:
        answer_term_weights = _significant_term_weights(answer)
        answer_terms = set(answer_term_weights)
        answer_phrases = _answer_phrases(answer)
        if not answer_terms:
            # Nothing to ground the citation in (e.g. "Yes."): omit Evidence
            # rather than presenting unverifiable retrieval order as provenance.
            return ""

    # (overlap, name, URL, source index, fallback number, body, provenance ids).
    candidates: List[
        Tuple[
            float,
            str,
            Optional[str],
            Optional[int],
            int,
            str,
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
            tuple[str, ...],
        ]
    ] = []
    seen: set = set()

    for obj in iterator:
        payload = _get_payload(obj)
        if payload is None:
            continue

        text = _clean_str(payload.get("text"))
        if text is None:
            continue
        metadata, _body = split_json_front_matter(text)
        body = _answer_body(text)
        document_name = _clean_str(metadata.get("title")) or _clean_str(
            payload.get("document_name")
        )
        canonical_url = _clean_str(metadata.get("canonical_url"))
        source_index = metadata.get("chunk_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            source_index = None
        number = _chunk_number(payload) or _chunk_number(metadata)

        # Document name and a chunk number are both required to ground the
        # citation; text is required for a meaningful snippet.
        if document_name is None or number is None or not body:
            continue

        chunk_id = _chunk_id(obj, payload)
        # document_id == the ingested Data item's id (cognify sets
        # Document.id = data.id), i.e. the dataId a caller needs to map a
        # citation back to the document they ingested.
        source_id = _clean_str(metadata.get("document_id"))
        data_id = _clean_str(payload.get("document_id"))
        dataset_id, node_sets = _dataset_provenance(payload)

        dedup_key = chunk_id or f"{document_name}#{number}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        score = 0
        if answer_terms is not None:
            # Search aliases may select a page, but only answer-bearing body
            # text can support a factual citation.  Treating a flat alias bag
            # as evidence can manufacture relations that the source never
            # states (for example co-located wormhole codes/classes).
            chunk_terms = set(re.findall(r"[a-z0-9]+", body.lower()))
            overlap_terms = answer_terms & chunk_terms
            score = sum(
                (answer_term_weights or {}).get(term, 1.0) for term in overlap_terms
            )
            if score == 0:
                # No term from the answer appears in this chunk: it is almost
                # certainly not a source of the answer.
                continue
            section_path = metadata.get("section_path")
            summary_section = False
            if isinstance(section_path, list):
                sections = {str(value).strip().casefold() for value in section_path}
                if "summary" in sections:
                    score += 3
                    summary_section = True
                elif "overview" in sections:
                    score += 1
                if "notes" in sections or "notes and references" in sections:
                    score -= 3
            if metadata.get("chunk_kind") == "table":
                if not any(
                    any(character.isdigit() for character in term)
                    for term in overlap_terms
                ):
                    score -= 2
            if metadata.get("chunk_kind") == "status" and overlap_terms & {
                "outdated",
                "historical",
                "excluded",
                "warning",
            }:
                score += 4
            title_terms = _significant_terms(document_name)
            title_in_answer = bool(title_terms) and title_terms <= answer_terms
            has_distinctive_overlap = any(
                len(term) >= 7 or any(character.isdigit() for character in term)
                for term in overlap_terms
            )
            if score <= 0 or (
                len(overlap_terms) < 2
                and not has_distinctive_overlap
                and not title_in_answer
                and not summary_section
            ):
                continue

        candidates.append(
            (
                score,
                document_name,
                canonical_url,
                source_index,
                number,
                body,
                source_id,
                data_id,
                chunk_id,
                dataset_id,
                node_sets,
            )
        )

    if not candidates:
        return ""

    if answer_terms is not None:
        # Stable sort: highest answer overlap first, retrieval order as tiebreak.
        candidates.sort(key=lambda candidate: -candidate[0])

    max_bullets = _clamp_limit(limit)
    bullets = [
        (
            f"- {document_name}: {canonical_url} "
            + (
                f"(source chunk {source_index:04d})"
                if source_index is not None
                else f"(chunk {number})"
            )
            if canonical_url
            else f"- chunk {number} of document {document_name}"
        )
        + _provenance_suffix(source_id, data_id, chunk_id, dataset_id, node_sets)
        + f': "{_snippet(body, answer_terms, answer_term_weights, answer_phrases)}"'
        for (
            _,
            document_name,
            canonical_url,
            source_index,
            number,
            body,
            source_id,
            data_id,
            chunk_id,
            dataset_id,
            node_sets,
        ) in candidates[:max_bullets]
    ]

    return EVIDENCE_HEADER + "\n" + "\n".join(bullets)


async def build_answer_grounded_chunk_references(
    answer: str, vector_engine: Any, limit: int = 5
) -> str:
    """Build an Evidence block by running the answer as a vector query over chunks.

    This grounds Evidence in the answer text itself, independent of how the
    original retrieval was done (graph traversal, triplets, ...): the answer is
    embedded once and matched against the existing chunk index, then candidates
    are additionally filtered by term overlap with the answer.

    Never raises: a missing collection or backend failure degrades to ``""``.

    Parameters
    ----------
    answer:
        The generated answer text to ground.
    vector_engine:
        A vector engine exposing ``search(collection, query, limit, include_payload)``.
    limit:
        Desired maximum number of bullets, clamped into the 3-5 range.

    Returns
    -------
    str
        A multi-line Evidence block prefixed by an ``Evidence:`` header, or an
        empty string when nothing usable was found or the search failed.
    """
    cleaned_answer = _clean_str(answer)
    if cleaned_answer is None or vector_engine is None:
        return ""

    try:
        found_chunks = await vector_engine.search(
            _CHUNK_COLLECTION,
            compact_embedding_text(cleaned_answer),
            limit=_CANDIDATE_POOL,
            include_payload=True,
        )
    except Exception as error:
        logger.debug(f"Answer-grounded chunk search failed: {error}")
        return ""

    return format_chunk_references(found_chunks, answer=cleaned_answer, limit=limit)


def append_chunk_evidence(
    completions: List[Any], retrieved_objects: Any, enabled: bool
) -> List[Any]:
    """Append an answer-grounded chunk Evidence block to string completions.

    Each string completion gets its own Evidence block, filtered and ranked by
    that completion's text against the retrieved candidates. Non-string
    completions (structured response models) are never touched, and an empty
    Evidence block leaves the completion unchanged.
    """
    if not enabled:
        return completions

    appended: List[Any] = []
    for completion in completions:
        if not isinstance(completion, str):
            appended.append(completion)
            continue
        evidence = format_chunk_references(retrieved_objects, answer=completion)
        appended.append(f"{completion}\n\n{evidence}" if evidence else completion)
    return appended


async def append_answer_grounded_evidence(completions: List[Any], enabled: bool) -> List[Any]:
    """Append an answer-grounded Evidence block to string completions.

    Each string completion is run as a vector query against the chunk index
    (see :func:`build_answer_grounded_chunk_references`). Non-string completions
    are never touched; any backend failure degrades to no Evidence.
    """
    if not enabled:
        return completions

    try:
        from cognee.infrastructure.databases.vector import get_vector_engine_async

        vector_engine = await get_vector_engine_async()
    except Exception as error:
        logger.debug(f"Unable to obtain vector engine for references: {error}")
        return completions

    appended: List[Any] = []
    for completion in completions:
        if not isinstance(completion, str):
            appended.append(completion)
            continue
        evidence = await build_answer_grounded_chunk_references(completion, vector_engine)
        appended.append(f"{completion}\n\n{evidence}" if evidence else completion)
    return appended
