import json
from uuid import uuid4

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.retrieval.hybrid_chunks_retriever import fuse_chunk_results
from cognee.modules.retrieval.utils.chunk_metadata import parse_json_front_matter
from cognee.modules.retrieval.utils.references import format_chunk_references


def _text(
    title: str,
    *,
    url: str | None = None,
    aliases: list[str] | None = None,
    section: list[str] | None = None,
    active: bool = True,
    kind: str = "prose",
    body: str = "Current answer-bearing content.",
) -> str:
    metadata = {
        "title": title,
        "canonical_url": url or f"https://wiki.eveuniversity.org/{title.replace(' ', '_')}",
        "aliases": aliases or [],
        "section_path": section or ["Overview"],
        "active": active,
        "chunk_kind": kind,
        "content_excluded": kind == "status",
    }
    encoded = "\n".join(f"{key}: {json.dumps(value)}" for key, value in metadata.items())
    return f"---\n{encoded}\n---\n\n# {title}\n\n{body}"


def _result(text: str, *, id_=None, chunk_index: int = 0) -> ScoredResult:
    id_ = id_ or uuid4()
    return ScoredResult(
        id=id_,
        score=0.1,
        payload={
            "id": str(id_),
            "document_name": "source.md",
            "chunk_index": chunk_index,
            "text": text,
        },
    )


def test_rrf_and_alias_boost_promote_canonical_page():
    mining = _result(_text("Mining"))
    planetary = _result(_text("Planetary Industry", aliases=["Planetary Interaction"]))
    lexical = [
        (planetary.payload, 4.0),
        (mining.payload, 3.0),
    ]

    results = fuse_chunk_results(
        "planetary interaction",
        [mining, planetary],
        lexical,
        top_k=2,
    )

    assert parse_json_front_matter(results[0].payload["text"])["title"] == "Planetary Industry"


def test_primary_slots_are_canonical_page_diverse_then_retain_extra_chunks():
    trading_url = "https://wiki.eveuniversity.org/Trading"
    chunks = [
        _result(_text("Trading", url=trading_url, section=[f"Trading {index}"]))
        for index in range(3)
    ]
    chunks.extend([_result(_text("Mining")), _result(_text("Manufacturing"))])

    diverse = fuse_chunk_results("trading", chunks, [], top_k=3)
    diverse_pages = [
        parse_json_front_matter(result.payload["text"])["canonical_url"] for result in diverse
    ]
    assert len(diverse_pages) == len(set(diverse_pages)) == 3

    answer_context = fuse_chunk_results(
        "trading",
        chunks,
        [],
        top_k=5,
        page_limit=3,
        max_chunks_per_page=2,
    )
    answer_pages = [
        parse_json_front_matter(result.payload["text"])["canonical_url"]
        for result in answer_context
    ]
    assert answer_pages[:3] == diverse_pages
    assert answer_pages.count(trading_url) == 2


def test_historical_body_is_excluded_but_status_warning_remains_discoverable():
    status = _result(_text("Reprocessing", active=False, kind="status"))
    historical = _result(
        _text("Archived Reprocessing", active=False, body="Historical body must not leak.")
    )
    current = _result(_text("Compression"))

    results = fuse_chunk_results(
        "reprocessing",
        [current, historical, status],
        [(status.payload, 5.0), (current.payload, 4.0)],
        top_k=3,
    )

    texts = [result.payload["text"] for result in results]
    assert all("Historical body must not leak" not in text for text in texts)
    assert any(parse_json_front_matter(text)["chunk_kind"] == "status" for text in texts)


def test_context_is_bounded_without_losing_chunk_identity_or_provenance_header():
    first = _result(_text("Trading", body="A" * 2_000))
    second = _result(_text("Mining", body="B" * 2_000))

    results = fuse_chunk_results(
        "industry",
        [first, second],
        [],
        top_k=2,
        max_context_chars=1_200,
    )

    assert len(results) == 1
    assert len(results[0].payload["text"]) <= 1_200
    assert results[0].payload["text"].startswith("---\n")
    assert results[0].id == first.id


def test_canonical_front_matter_is_preserved_in_evidence():
    chunk = _result(
        _text(
            "Trading",
            section=["Taxes", "Sales tax"],
            body="Sales tax is currently 7.5 percent.",
        ),
        chunk_index=6,
    )

    evidence = format_chunk_references(
        [chunk],
        answer="Sales tax is currently 7.5 percent.",
    )

    assert "Trading: https://wiki.eveuniversity.org/Trading" in evidence
    assert "chunk 7" in evidence
