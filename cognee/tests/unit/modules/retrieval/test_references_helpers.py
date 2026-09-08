"""Unit tests for the LLM-free reference (Evidence) helpers.

Covers ``format_chunk_references`` (sync, payload-driven, answer-grounded) and
``build_answer_grounded_chunk_references`` (async, vector-engine-driven),
including the old-data graceful-degradation cases and backend-failure cases.
"""

from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.modules.retrieval.utils.references import (
    EVIDENCE_HEADER,
    build_answer_grounded_chunk_references,
    format_chunk_references,
)


# ---------------------------------------------------------------------------
# format_chunk_references (no answer: legacy retrieval-order behavior)
# ---------------------------------------------------------------------------


def _payload(**overrides):
    base = {
        "document_name": "annual_report.pdf",
        "chunk_index": 4,  # 1-based display number -> 5
        "text": "Revenue grew 12 percent year over year.",
    }
    base.update(overrides)
    return base


def test_format_chunk_references_renders_one_based_number_from_chunk_index():
    """chunk_index + 1 is rendered as the display number."""
    result = format_chunk_references([_payload(chunk_index=4)])

    assert result.startswith(EVIDENCE_HEADER + "\n")
    assert "- chunk 5 of document annual_report.pdf:" in result
    assert "Revenue grew 12 percent" in result


def test_format_chunk_references_prefers_explicit_chunk_number():
    """An explicit chunk_number wins over chunk_index when both present."""
    result = format_chunk_references([_payload(chunk_number=9, chunk_index=4)])

    assert "- chunk 9 of document annual_report.pdf:" in result
    assert "chunk 5" not in result


def test_format_chunk_references_reads_scored_result_like_objects():
    """ScoredResult-like objects expose .payload and .id; they are read correctly."""

    class FakeScored:
        def __init__(self, payload, id_):
            self.payload = payload
            self.id = id_

    objs = [FakeScored(_payload(), "id-1")]
    result = format_chunk_references(objs)

    assert "- chunk 5 of document annual_report.pdf (chunk_id: id-1):" in result


def test_format_chunk_references_includes_data_id_and_chunk_id():
    """document_id (== ingested data id) and chunk id are surfaced in the bullet."""
    result = format_chunk_references([_payload(document_id="data-123", id="chunk-9")])

    assert (
        "- chunk 5 of document annual_report.pdf (data_id: data-123, chunk_id: chunk-9):" in result
    )


def test_format_chunk_references_empty_when_document_name_missing():
    """Old-data case: missing document_name -> entry skipped -> empty string."""
    payload = _payload()
    del payload["document_name"]

    assert format_chunk_references([payload]) == ""


def test_format_chunk_references_empty_when_document_name_null():
    """Null document_name is unusable -> empty string."""
    assert format_chunk_references([_payload(document_name=None)]) == ""


def test_format_chunk_references_empty_when_no_chunk_number():
    """Missing both chunk_number and chunk_index -> empty string."""
    payload = _payload()
    del payload["chunk_index"]

    assert format_chunk_references([payload]) == ""


def test_format_chunk_references_empty_when_text_missing():
    """Missing text -> no usable snippet -> empty string."""
    payload = _payload()
    del payload["text"]

    assert format_chunk_references([payload]) == ""


def test_format_chunk_references_empty_for_empty_input():
    assert format_chunk_references([]) == ""
    assert format_chunk_references(None) == ""


def test_format_chunk_references_dedups_by_id():
    """Two payloads with the same object id collapse into one bullet."""

    class FakeScored:
        def __init__(self, payload, id_):
            self.payload = payload
            self.id = id_

    objs = [
        FakeScored(_payload(text="first"), "same-id"),
        FakeScored(_payload(text="second"), "same-id"),
    ]
    result = format_chunk_references(objs)

    assert result.count("- chunk 5 of document annual_report.pdf (chunk_id: same-id):") == 1


def test_format_chunk_references_caps_and_clamps_limit():
    """Limit is clamped into the 3-5 range."""
    payloads = [
        _payload(document_name=f"doc_{i}.pdf", chunk_index=i, text=f"text {i}", id=str(i))
        for i in range(10)
    ]
    # limit below the floor is clamped up to 3
    low = format_chunk_references(payloads, limit=1)
    assert low.count("- chunk ") == 3
    # limit above the ceiling is clamped down to 5
    high = format_chunk_references(payloads, limit=99)
    assert high.count("- chunk ") == 5


def test_format_chunk_references_snippet_truncated():
    """Long text is truncated with an ellipsis."""
    long_text = "word " * 300
    result = format_chunk_references([_payload(text=long_text)])
    # The bullet line contains a truncation ellipsis.
    assert "…" in result


def test_front_matter_title_and_canonical_url_override_generic_document_name():
    text = (
        '---\ntitle: "Trading"\n'
        'canonical_url: "https://docs.example.org/Trading"\n---\n\n'
        "Sales tax is 7.5 percent."
    )

    result = format_chunk_references([_payload(text=text, chunk_index=6)])

    assert "- Trading: https://docs.example.org/Trading (chunk 7)" in result


def test_front_matter_preserves_chunk_index_and_document_id():
    text = (
        '---\ndocument_id: "mediawiki:131:9001:chunk:0007"\n'
        'title: "Trading"\n'
        'canonical_url: "https://docs.example.org/Trading"\n'
        "chunk_index: 7\n---\n\n"
        "Sales tax is 7.5 percent."
    )
    payload = _payload(text=text)
    del payload["chunk_index"]

    result = format_chunk_references([payload])

    assert "Trading: https://docs.example.org/Trading (source chunk 0007)" in result
    assert "source_id: mediawiki:131:9001:chunk:0007" in result


def test_front_matter_is_removed_from_overlap_and_supporting_snippet():
    text = (
        '---\ndocument_id: "mediawiki:131:9001:chunk:0006"\n'
        'title: "Trading"\n'
        'canonical_url: "https://docs.example.org/Trading"\n'
        "chunk_index: 6\n---\n\n"
        "> source provenance: source_item_id=item\n\n"
        "# Trading\n\nSales tax is 7.5% and Accounting reduces it to 3.37%."
    )
    result = format_chunk_references(
        [_payload(text=text, document_id="data-123", id="chunk-123")],
        answer="Sales tax is 7.5% and Accounting reduces it to 3.37%.",
    )

    assert "source chunk 0006" in result
    assert "source_id: mediawiki:131:9001:chunk:0006" in result
    assert "data_id: data-123" in result
    assert "chunk_id: chunk-123" in result
    assert '"# Trading Sales tax is 7.5% and Accounting reduces it to 3.37%."' in result
    assert '"--- document_id:' not in result


def test_reference_includes_request_dataset_and_exact_node_sets():
    dataset_id = UUID("4b4d2964-0333-5057-bb0e-723d35f90810")
    text = (
        '---\ndocument_id: "mediawiki:131:9001:chunk:0006"\n'
        'title: "Trading"\n'
        'canonical_url: "https://docs.example.org/Trading"\n'
        "chunk_index: 6\n---\n\n"
        "Sales tax is 7.5 percent."
    )
    token = current_dataset_id.set(dataset_id)
    try:
        result = format_chunk_references(
            [
                _payload(
                    text=text,
                    document_id="data-123",
                    id="chunk-123",
                    belongs_to_set=["view:current", "site:candidate", "source:wiki"],
                )
            ]
        )
    finally:
        current_dataset_id.reset(token)

    assert f"dataset_id: {dataset_id}" in result
    assert "node_sets: site:candidate|source:wiki|view:current" in result


def test_supporting_snippet_focuses_on_late_answer_terms():
    text = (
        '---\ndocument_id: "mediawiki:1:2:chunk:0005"\n'
        'title: "Insurgency"\n'
        'canonical_url: "https://docs.example.org/Insurgency"\n'
        "chunk_index: 5\n---\n\n"
        + "Background mechanics. " * 80
        + "Winning pilots receive ISK and loyalty points at the conclusion."
    )

    result = format_chunk_references(
        [_payload(text=text, document_id="data-1", id="chunk-1")],
        answer="Pilots receive loyalty points.",
    )

    assert "loyalty points" in result
    assert "Background mechanics" in result
    assert '"--- document_id:' not in result


def test_supporting_snippet_prefers_distinctive_answer_phrase():
    text = (
        '---\ndocument_id: "mediawiki:10289:1:chunk:0004"\n'
        'title: "Moving your items"\n'
        'canonical_url: "https://docs.example.org/Moving_your_items"\n'
        "chunk_index: 4\n---\n\n"
        + "Cargo routes, hauler time, reward, collateral, and risk. " * 35
        + "Janice gives a Total Sell Value for the cargo. "
        + "Your collateral should be at least this value."
    )

    result = format_chunk_references(
        [_payload(text=text, document_id="data-1", id="chunk-1")],
        answer=(
            "Use Janice's Total Sell Value and set collateral to at least the cargo's "
            "replacement value."
        ),
    )

    assert "Total Sell Value" in result
    assert "collateral should be at least this value" in result


def test_supporting_snippet_retains_leading_identity_for_multihop_claim():
    text = (
        '---\ndocument_id: "mediawiki:3981:1:chunk:0002"\n'
        'title: "Sin"\ncanonical_url: "https://docs.example.org/Sin"\n'
        "chunk_index: 2\n---\n\n"
        "# Sin | Sin | Gallente Federation | Black Ops | "
        + "General ship description. " * 80
        + "The Sin has a jump drive."
    )

    result = format_chunk_references(
        [_payload(text=text, document_id="data-1", id="chunk-1")],
        answer="The Sin uses Oxygen Isotopes for its jump drive.",
    )

    assert "Sin | Sin | Gallente Federation" in result
    assert "The Sin has a jump drive" in result


def test_supporting_snippet_prioritizes_distinctive_alphanumeric_claim_code():
    text = (
        '---\ndocument_id: "mediawiki:7611:1:chunk:0010"\n'
        'title: "Wormhole attributes"\n'
        'canonical_url: "https://docs.example.org/Wormhole_attributes"\n'
        "chunk_index: 10\n---\n\n"
        + "Thera null security wormhole space background. " * 50
        + "E587 connects C12 Thera to C9 null security space. "
        + "Visual wormhole identification notes. " * 50
    )

    result = format_chunk_references(
        [_payload(text=text, document_id="data-1", id="chunk-1")],
        answer="E587 connects Thera to null security space.",
    )

    assert "E587 connects C12 Thera to C9" in result


def test_supporting_snippet_keeps_complete_early_numeric_claim():
    text = (
        '---\ntitle: "Compression"\nchunk_index: 6\n---\n\n'
        "# Compression ## Batch compressed ore "
        + "Historical compression background. " * 7
        + "100 units of ore compressed to 1 unit of compressed ore and are no longer creatable. "
        + "The current system uses 1 to 1 compression. "
        + "Volume details by ore type. " * 60
    )

    result = format_chunk_references(
        [_payload(text=text, document_id="data-1", id="chunk-1")],
        answer=(
            "Batch Compressed Ore turned 100 units into 1 unit, is no longer creatable, "
            "and current ore uses 1 to 1 compression."
        ),
    )

    assert "100 units of ore compressed to 1 unit" in result
    assert "no longer creatable" in result
    assert "1 to 1 compression" in result


def test_structured_answer_snippet_centers_claim_entities_not_json_fields():
    text = (
        '---\ndocument_id: "mediawiki:6281:231494:chunk:0002"\n'
        'title: "Jump drives"\n'
        'canonical_url: "https://docs.example.org/Jump_drives"\n'
        "chunk_index: 2\n---\n\n"
        + "Jump drives require fuel and have operational constraints. " * 30
        + "Gallente ships use Oxygen Isotopes for their jump drives."
    )
    structured_answer = """{
      "nodes": [
        {"id": "Sin", "type": "Ship", "name": "Sin"},
        {"id": "Oxygen Isotopes", "type": "Material", "name": "Oxygen Isotopes"}
      ],
      "edges": [
        {"source": "Sin", "target": "Oxygen Isotopes", "relationship": "uses_fuel",
         "description": "The Sin uses Oxygen Isotopes as jump-drive fuel."}
      ]
    }"""

    result = format_chunk_references(
        [_payload(text=text, document_id="data-1", id="chunk-1")],
        answer=structured_answer,
    )

    assert "Gallente ships use Oxygen Isotopes" in result
    assert '"nodes"' not in result


# ---------------------------------------------------------------------------
# format_chunk_references (answer-grounded filtering and ranking)
# ---------------------------------------------------------------------------


def test_answer_filtering_drops_chunks_without_overlap():
    """Chunks sharing no significant terms with the answer are not cited."""
    matching = _payload(document_name="report.pdf", chunk_index=0, text="Revenue grew 12 percent.")
    unrelated = _payload(
        document_name="other.pdf", chunk_index=1, text="Penguins live in Antarctica."
    )

    result = format_chunk_references([matching, unrelated], answer="Revenue grew 12 percent.")

    assert "report.pdf" in result
    assert "other.pdf" not in result


def test_answer_filtering_drops_weak_single_generic_term_overlap():
    relevant = _payload(document_name="Sin", chunk_index=0, text="The Sin is Gallente.")
    distractor = _payload(
        document_name="Mission", chunk_index=1, text="Deliver activist fuel as cargo."
    )

    result = format_chunk_references(
        [distractor, relevant], answer="The Sin uses Oxygen Isotopes as jump fuel."
    )

    assert "Sin" in result
    assert "Mission" not in result


def test_answer_filtering_empty_when_nothing_overlaps():
    """No candidate overlaps the answer -> Evidence omitted entirely."""
    unrelated = _payload(text="Penguins live in Antarctica.")

    assert format_chunk_references([unrelated], answer="Quarterly revenue increased.") == ""


def test_answer_filtering_ranks_by_overlap():
    """Higher answer-term overlap is cited before lower overlap."""
    weak = _payload(document_name="weak.pdf", chunk_index=0, text="Revenue is mentioned once.")
    strong = _payload(
        document_name="strong.pdf",
        chunk_index=1,
        text="Revenue grew twelve percent in the fourth quarter.",
    )

    result = format_chunk_references(
        [weak, strong], answer="Revenue grew twelve percent in the fourth quarter."
    )

    assert result.index("strong.pdf") < result.index("weak.pdf")


def test_answer_filtering_keeps_summary_link_for_multihop_citation():
    summary = _payload(
        document_name="sin-summary.md",
        chunk_index=6,
        text=(
            '---\ntitle: "Sin"\nsection_path: ["Summary"]\n'
            'chunk_kind: "prose"\n---\n\nThe Sin is a Gallente Black Ops.'
        ),
    )
    table = _payload(
        document_name="sin-table.md",
        chunk_index=4,
        text=(
            '---\ntitle: "Sin"\nsection_path: ["Statistics"]\n'
            'chunk_kind: "table"\n---\n\nThe Sin has a jump fuel capacity.'
        ),
    )

    result = format_chunk_references(
        [table, summary], answer="The Sin uses Oxygen Isotopes as jump fuel."
    )

    assert result.index("Gallente") < result.index("capacity")


def test_answer_filtering_keeps_table_with_distinctive_claim_code():
    exact = _payload(
        document_name="wormhole-table.md",
        chunk_index=10,
        text=(
            '---\ntitle: "Wormhole attributes"\nsection_path: ["Connections"]\n'
            'chunk_kind: "table"\n'
            'aliases: ["Wormhole Information", "Wormhole Types", "C12", "C9", '
            '"E587", "Thera to nullsec"]\n'
            '---\n\n| Code | Source | Destination | Class |\n'
            '| E587 | Thera | Nullsec (0.0) | 9 |'
        ),
    )
    generic = _payload(
        document_name="wormholes.md",
        chunk_index=2,
        text="Thera has many wormhole connections to null security space.",
    )

    result = format_chunk_references(
        [generic, exact], answer="E587 connects Thera to C9 null security space."
    )

    assert result.index("Wormhole attributes") < result.index("wormholes.md")
    assert "E587 | Thera | Nullsec" in result
    assert "Indexed search aliases" not in result


def test_flat_aliases_are_not_rendered_as_factual_evidence():
    alias_only = _payload(
        document_name="wormhole-visuals.md",
        chunk_index=9,
        text=(
            '---\ntitle: "Wormhole attributes"\nchunk_kind: "prose"\n'
            'aliases: ["C12", "C9", "E587", "Thera to nullsec"]\n'
            '---\n\nWormhole colors help pilots identify a destination skybox.'
        ),
    )

    result = format_chunk_references(
        [alias_only], answer="E587 connects Thera to C9 null security space."
    )

    assert result == ""


def test_answer_filtering_keeps_matching_status_warning():
    status = _payload(
        document_name="reprocessing-status.md",
        chunk_index=0,
        text=(
            '---\ntitle: "Reprocessing"\nsection_path: ["Status"]\n'
            'chunk_kind: "status"\n---\n\nAnalytical content is excluded because this page is outdated.'
        ),
    )
    prose = [
        _payload(
            document_name=f"compression-{index}.md",
            chunk_index=index,
            text=f"Compression and reprocessing yield details number {index}.",
        )
        for index in range(6)
    ]

    result = format_chunk_references(
        [*prose, status],
        answer="The Reprocessing analytical content is outdated and excluded.",
    )

    assert "document Reprocessing" in result


def test_answer_with_no_significant_terms_yields_no_evidence():
    """An answer made of stopwords/stubs cannot be grounded -> empty string."""
    assert format_chunk_references([_payload()], answer="It is.") == ""


def test_answer_none_keeps_all_usable_candidates():
    """answer=None preserves the unfiltered retrieval-order behavior."""
    unrelated = _payload(text="Penguins live in Antarctica.")

    assert "Penguins" in format_chunk_references([unrelated], answer=None)


# ---------------------------------------------------------------------------
# build_answer_grounded_chunk_references
# ---------------------------------------------------------------------------


def _scored(payload, id_):
    class FakeScored:
        def __init__(self, payload, id_):
            self.payload = payload
            self.id = id_

    return FakeScored(payload, id_)


@pytest.mark.asyncio
async def test_answer_grounded_references_query_chunk_index_with_answer():
    """The answer text is run as the vector query and grounds the bullets."""
    engine = AsyncMock()
    engine.search.return_value = [
        _scored(
            _payload(document_name="report.pdf", chunk_index=2, text="Revenue grew 12 percent."),
            "chunk-1",
        )
    ]

    result = await build_answer_grounded_chunk_references("Revenue grew 12 percent.", engine)

    assert result.startswith(EVIDENCE_HEADER + "\n")
    assert "- chunk 3 of document report.pdf (chunk_id: chunk-1):" in result
    engine.search.assert_awaited_once()
    assert engine.search.await_args.args[0] == "DocumentChunk_text"
    assert engine.search.await_args.args[1] == "Revenue grew 12 percent."


@pytest.mark.asyncio
async def test_answer_grounded_references_compact_oversized_structured_answer():
    engine = AsyncMock()
    engine.search.return_value = [
        _scored(
            _payload(text="Gallente jump drives use Oxygen Isotopes."),
            "chunk-1",
        )
    ]
    answer = (
        '{"nodes":[{"name":"Sin"}],"edges":['
        '{"description":"'
        + "Background mechanics. " * 600
        + '"},{"description":"The Sin uses Oxygen Isotopes."}]}'
    )

    result = await build_answer_grounded_chunk_references(answer, engine)

    query = engine.search.await_args.args[1]
    assert len(query) <= 8_001
    assert "Oxygen Isotopes" in query
    assert '"nodes"' not in query
    assert "Oxygen Isotopes" in result


@pytest.mark.asyncio
async def test_answer_grounded_references_drop_unrelated_results():
    """Vector hits that share no terms with the answer are filtered out."""
    engine = AsyncMock()
    engine.search.return_value = [
        _scored(_payload(text="Penguins live in Antarctica."), "chunk-1"),
    ]

    result = await build_answer_grounded_chunk_references("Quarterly revenue increased.", engine)

    assert result == ""


@pytest.mark.asyncio
async def test_answer_grounded_references_empty_on_search_failure():
    """A missing collection or backend failure degrades to no Evidence, no raise."""
    engine = AsyncMock()
    engine.search.side_effect = RuntimeError("collection not found")

    result = await build_answer_grounded_chunk_references("Revenue grew.", engine)

    assert result == ""


@pytest.mark.asyncio
async def test_answer_grounded_references_empty_for_blank_answer_or_engine():
    assert await build_answer_grounded_chunk_references("", AsyncMock()) == ""
    assert await build_answer_grounded_chunk_references("   ", AsyncMock()) == ""
    assert await build_answer_grounded_chunk_references("Revenue grew.", None) == ""
