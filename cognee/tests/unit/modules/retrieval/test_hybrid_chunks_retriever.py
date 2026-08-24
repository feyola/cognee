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
    alias_relations: dict[str, list[str]] | None = None,
    section: list[str] | None = None,
    active: bool = True,
    kind: str = "prose",
    body: str = "Current answer-bearing content.",
) -> str:
    metadata = {
        "title": title,
        "canonical_url": url or f"https://wiki.eveuniversity.org/{title.replace(' ', '_')}",
        "aliases": aliases or [],
        "alias_relations": alias_relations or {},
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


def test_vector_and_lexical_copies_fuse_by_stable_document_id():
    chunk = _result(_text("Trading"))
    chunk.payload.pop("id")
    chunk.payload["text"] = chunk.payload["text"].replace(
        "---\n", '---\ndocument_id: "mediawiki:131:1:chunk:0001"\n', 1
    )

    results = fuse_chunk_results(
        "trading",
        [chunk],
        [(chunk.payload, 10.0)],
        top_k=2,
        page_limit=1,
        max_chunks_per_page=2,
    )

    assert len(results) == 1
    assert results[0].id == chunk.id


def test_alias_boost_requires_a_contiguous_phrase():
    trading = _result(_text("Trading", aliases=["Station trading"]))
    skill_trading = _result(_text("Skill trading", aliases=["Skills trading"]))

    results = fuse_chunk_results(
        "How does station trading profitability change and which skills reduce fees?",
        [skill_trading, trading],
        [(skill_trading.payload, 10.0), (trading.payload, 9.0)],
        top_k=2,
    )

    assert parse_json_front_matter(results[0].payload["text"])["title"] == "Trading"


def test_related_alias_can_match_noncontiguous_query_terms():
    mission = _result(_text("Activist Fuel"))
    jump_drives = _result(
        _text(
            "Jump drives",
            aliases=["Sin fuel"],
            alias_relations={"Sin fuel": ["Oxygen Isotopes"]},
            body="Gallente jump ships use Oxygen Isotopes.",
        )
    )

    results = fuse_chunk_results(
        "What kind of fuel is used by Sin?",
        [mission, jump_drives],
        [(mission.payload, 10.0), (jump_drives.payload, 9.0)],
        top_k=2,
    )

    assert parse_json_front_matter(results[0].payload["text"])["title"] == "Jump drives"


def test_shared_supply_alias_promotes_distinct_answer_pages():
    blueprints = _result(_text("Blueprints", aliases=["Blueprint Originals"]))
    skillbooks = _result(
        _text(
            "Skills and learning",
            aliases=["NPC sell orders"],
            alias_relations={"NPC sell orders": ["skillbooks", "sold by NPC corporations"]},
            body="Most skillbooks are sold by NPC corporations for a fixed price.",
        )
    )
    command_centers = _result(
        _text(
            "Setting up a planetary colony",
            aliases=["NPC sell orders"],
            alias_relations={
                "NPC sell orders": ["Planetary Command Center", "sold by NPC merchants"]
            },
            body="Planetary Command Centers are sold by NPC merchants.",
        )
    )
    noise = [_result(_text(title)) for title in ("Trading", "Industry", "Research")]

    results = fuse_chunk_results(
        "blueprint originals and other market items supplied by NPC sell orders",
        noise + [blueprints, skillbooks, command_centers],
        [(result.payload, 10.0 - rank) for rank, result in enumerate(noise)],
        top_k=3,
    )

    assert {parse_json_front_matter(result.payload["text"])["title"] for result in results} == {
        "Blueprints",
        "Skills and learning",
        "Setting up a planetary colony",
    }


def test_related_alias_selects_answer_bearing_primary_chunk():
    url = "https://wiki.eveuniversity.org/Skills_and_learning"
    relations = {
        "NPC sell orders": [
            "skillbooks",
            "sold by NPC corporations",
            "fixed price",
        ]
    }
    overview = _result(
        _text(
            "Skills and learning",
            url=url,
            aliases=["NPC sell orders"],
            alias_relations=relations,
            body="Skills train character abilities over time.",
        )
    )
    acquiring = _result(
        _text(
            "Skills and learning",
            url=url,
            aliases=["NPC sell orders"],
            alias_relations=relations,
            body="Most skillbooks are sold by NPC corporations for a fixed price.",
        )
    )

    [result] = fuse_chunk_results(
        "Which items are supplied by NPC sell orders?",
        [overview, acquiring],
        [(overview.payload, 10.0), (acquiring.payload, 9.0)],
        top_k=1,
    )

    assert "sold by NPC corporations" in result.payload["text"]


def test_answer_body_overlap_selects_the_supporting_chunk_within_a_page():
    url = "https://wiki.eveuniversity.org/Insurgency"
    overview = _result(
        _text(
            "Insurgency",
            url=url,
            aliases=["Faction warfare"],
            body="Systems become corrupted or suppressed during an insurgency.",
        )
    )
    rewards = _result(
        _text(
            "Insurgency",
            url=url,
            aliases=["Faction warfare"],
            body="Winning pilots receive ISK and loyalty points.",
        )
    )

    [result] = fuse_chunk_results(
        "faction warfare loyalty point demand",
        [overview, rewards],
        [],
        top_k=1,
    )

    assert "loyalty points" in result.payload["text"]


def test_page_reranking_prefers_summary_over_table_and_notes_chunks():
    url = "https://wiki.eveuniversity.org/Sin"
    table = _result(
        _text("Sin", url=url, section=["Overview"], kind="table", body="Ship statistics."),
        chunk_index=2,
    )
    notes = _result(
        _text("Sin", url=url, section=["Notes"], body="Additional notes for Sin."),
        chunk_index=9,
    )
    summary = _result(
        _text("Sin", url=url, section=["Summary"], body="The Sin is a Gallente Black Ops."),
        chunk_index=6,
    )

    [result] = fuse_chunk_results(
        "What kind of fuel is used by Sin?",
        [table, notes, summary],
        [(table.payload, 10.0), (notes.payload, 9.0), (summary.payload, 8.0)],
        top_k=1,
    )

    assert "The Sin is a Gallente Black Ops" in result.payload["text"]


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


def test_secondary_chunk_adds_uncovered_query_aspect_within_selected_page():
    url = "https://wiki.eveuniversity.org/Insurgency"
    primary = _result(
        _text(
            "Insurgency",
            url=url,
            aliases=["Faction warfare"],
            body="Current faction warfare activity lets militias stage ships.",
        )
    )
    repetitive = _result(
        _text(
            "Insurgency",
            url=url,
            aliases=["Faction warfare"],
            body="Faction warfare activity changes a current system.",
        )
    )
    rewards = _result(
        _text(
            "Insurgency",
            url=url,
            aliases=["Faction warfare"],
            body="Winning pilots receive ISK and loyalty points.",
        )
    )

    results = fuse_chunk_results(
        "current faction warfare activity create loyalty point ship demand",
        [primary, repetitive, rewards],
        [],
        top_k=2,
        page_limit=1,
        max_chunks_per_page=2,
    )

    bodies = [result.payload["text"] for result in results]
    assert any("loyalty points" in body for body in bodies)
    assert all("changes a current system" not in body for body in bodies)


def test_secondary_chunk_prefers_more_uncovered_query_aspects():
    url = "https://wiki.eveuniversity.org/Trading"
    primary = _result(
        _text("Trading", url=url, body="Station trading profitability depends on margins.")
    )
    skills = _result(
        _text("Trading", url=url, body="Skills reduce trading overhead."),
    )
    taxes = _result(
        _text(
            "Trading",
            url=url,
            body="Sales tax is 7.5% and broker fees are reduced by Accounting.",
        )
    )

    results = fuse_chunk_results(
        "sales tax broker fees affect station trading profitability skills reduce",
        [primary, skills, taxes],
        [],
        top_k=2,
        page_limit=1,
        max_chunks_per_page=2,
    )

    bodies = [result.payload["text"] for result in results]
    assert any("Sales tax is 7.5%" in body for body in bodies)
    assert all("Skills reduce trading overhead" not in body for body in bodies)


def test_named_section_beats_broader_higher_ranked_chunk_on_same_page():
    url = "https://wiki.eveuniversity.org/Trading"
    primary = _result(
        _text(
            "Trading",
            url=url,
            section=["Station Trading"],
            body="Station trading profitability depends on margins.",
        )
    )
    broad = _result(
        _text(
            "Trading",
            url=url,
            section=["Skills"],
            body="Accounting and Broker Relations reduce trading overhead.",
        )
    )
    exact = _result(
        _text(
            "Trading",
            url=url,
            section=["Taxes", "Sales tax"],
            body="Sales tax is 7.5% and 3.37% at Accounting V.",
        )
    )

    results = fuse_chunk_results(
        "How does sales tax affect station trading?",
        [primary, broad, exact],
        [],
        top_k=2,
        page_limit=1,
        max_chunks_per_page=2,
    )

    bodies = [result.payload["text"] for result in results]
    assert any("3.37%" in body for body in bodies)
    assert all("reduce trading overhead" not in body for body in bodies)


def test_secondary_chunk_uses_related_alias_to_ground_answer_chain():
    url = "https://wiki.eveuniversity.org/Jump_drives"
    aliases = ["Sin fuel", "Oxygen Isotopes"]
    relations = {"Sin fuel": ["Oxygen Isotopes"]}
    formula = _result(
        _text(
            "Jump drives",
            url=url,
            aliases=aliases,
            alias_relations=relations,
            body="The isotope fuel-used formula depends on distance.",
        )
    )
    overview = _result(
        _text(
            "Jump drives",
            url=url,
            aliases=aliases,
            alias_relations=relations,
            body="A general overview of jump-capable ships.",
        )
    )
    isotope_table = _result(
        _text(
            "Jump drives",
            url=url,
            aliases=aliases,
            alias_relations=relations,
            body="The Isotope type table maps Gallente ships to Oxygen.",
        )
    )

    results = fuse_chunk_results(
        "What kind of fuel is used by Sin?",
        [formula, overview, isotope_table],
        [
            (formula.payload, 10.0),
            (overview.payload, 9.0),
            (isotope_table.payload, 8.0),
        ],
        top_k=2,
        page_limit=1,
        max_chunks_per_page=2,
    )

    bodies = [result.payload["text"] for result in results]
    assert any("maps Gallente ships to Oxygen" in body for body in bodies)
    assert all("general overview" not in body for body in bodies)


def test_exact_identifier_selects_answer_bearing_table_over_flat_alias_overview():
    url = "https://wiki.eveuniversity.org/Wormhole_attributes"
    aliases = ["C12", "C9", "E587", "Thera to nullsec", "wormhole connection"]
    overview = _result(
        _text(
            "Wormhole attributes",
            url=url,
            aliases=aliases,
            body="Wormhole colors can reveal a destination skybox.",
        )
    )
    relation_table = _result(
        _text(
            "Wormhole attributes",
            url=url,
            aliases=aliases,
            body="| Code | Source | Destination | Class |\n| E587 | Thera | Nullsec | 9 |",
        )
    )

    [result] = fuse_chunk_results(
        "What connection leads from C12 Thera to C9 null security space? E587",
        [overview, relation_table],
        [(overview.payload, 10.0), (relation_table.payload, 9.0)],
        top_k=1,
    )

    assert "E587 | Thera | Nullsec" in result.payload["text"]


def test_body_gated_alias_relations_reserve_distinct_support_pages():
    blueprint = _result(_text("Blueprints", body="NPC sellers offer Tech 1 BPOs."))
    skills = _result(
        _text(
            "Skills and learning",
            alias_relations={"NPC sell orders": ["skillbooks", "fixed price"]},
            body="Most skillbooks are sold by NPC corporations at a fixed price.",
        )
    )
    colony = _result(
        _text(
            "Setting up a planetary colony",
            alias_relations={"NPC sell orders": ["Planetary Command Center", "NPC merchants"]},
            body="A Planetary Command Center is sold by NPC merchants.",
        )
    )
    generic_market = _result(
        _text("Trading", body="Market buy and sell orders have regional prices.")
    )

    results = fuse_chunk_results(
        "Which items come from NPC sell orders?",
        [blueprint, generic_market, skills, colony],
        [
            (blueprint.payload, 12.0),
            (generic_market.payload, 11.0),
            (skills.payload, 5.0),
            (colony.payload, 4.0),
        ],
        top_k=3,
        page_limit=3,
    )

    titles = [parse_json_front_matter(result.payload["text"])["title"] for result in results]
    assert titles[:2] == ["Skills and learning", "Setting up a planetary colony"]
    assert titles[2] == "Blueprints"


def test_related_alias_evidence_outranks_unrelated_flat_alias():
    url = "https://wiki.eveuniversity.org/Jump_drives"
    distractor = _result(
        _text(
            "Jump drives",
            url=url,
            aliases=["Sin fuel", "Cynosural Field"],
            body="A Cynosural Field permits capital travel.",
        )
    )
    answer = _result(
        _text(
            "Jump drives",
            url=url,
            aliases=["Sin fuel", "Oxygen Isotopes"],
            alias_relations={"Sin fuel": ["Oxygen Isotopes"]},
            body="Gallente ships use Oxygen Isotopes.",
        )
    )

    results = fuse_chunk_results(
        "What kind of fuel is used by Sin?",
        [distractor, answer],
        [(distractor.payload, 10.0), (answer.payload, 9.0)],
        top_k=2,
        page_limit=1,
        max_chunks_per_page=2,
    )

    assert "Oxygen Isotopes" in results[0].payload["text"]
    assert "Cynosural Field" in results[1].payload["text"]


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
    assert parse_json_front_matter(results[0].payload["text"])["title"] == "Trading"
    assert results[0].id == first.id


def test_context_bound_omits_chunk_when_complete_front_matter_cannot_fit():
    first = _result(_text("Trading", body="A" * 2_000))

    results = fuse_chunk_results(
        "trading",
        [first],
        [],
        top_k=1,
        max_context_chars=20,
    )

    assert results == []


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


def test_personalization_reorders_hybrid_candidates_before_page_selection():
    first = _result("neutral candidate", chunk_index=0)
    preferred = _result("preferred candidate", chunk_index=1)

    results = fuse_chunk_results(
        "unmatched query",
        [first, preferred],
        [],
        top_k=1,
        preference_weights={
            str(first.payload["id"]): 0.0,
            str(preferred.payload["id"]): 1.0,
        },
        personalization_influence=1.0,
    )

    assert results[0].id == preferred.id


def test_personalization_uses_result_id_when_payload_has_no_id():
    first = _result("neutral candidate", chunk_index=0)
    preferred_id = uuid4()
    preferred = ScoredResult(
        id=preferred_id,
        score=0.1,
        payload={
            "document_name": "other.md",
            "chunk_index": 0,
            "text": "preferred candidate",
        },
    )

    results = fuse_chunk_results(
        "unmatched query",
        [first, preferred],
        [],
        top_k=1,
        preference_weights={
            str(first.payload["id"]): 0.0,
            str(preferred_id): 1.0,
        },
        personalization_influence=1.0,
    )

    assert results[0].id == preferred_id
