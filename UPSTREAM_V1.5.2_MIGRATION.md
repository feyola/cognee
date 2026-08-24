# Cognee v1.5.2 fork migration

This branch merges upstream `v1.5.2` (`df6b549594ef4e422a00702c8b97ee88424513fa`)
into the deployed fork lineage (`d69f64199229e3c6313ee4a7a5ddb684e74863ea`).

## Production patch disposition

All 50 production-only commits remain in first-parent ancestry. The table records the
semantic disposition after conflict resolution; `retained` means the behavior remains on
its original or auto-merged path, while `ported` means it moved to a new upstream path.

| Commit | Disposition | Verification |
| --- | --- | --- |
| `bab033af2` | retained | Hybrid lexical/vector retrieval and page-diverse selection tests |
| `c01259bb1` | retained | Front-matter identity and citation tests |
| `96129be84` | retained | Scoped lexical retrieval and evidence tests |
| `df6b1fdcf` | retained | Answer-body overlap tests |
| `091d3bf14` | retained | Supporting-passage evidence tests |
| `9177d6116` | retained | Context-bound citation identity tests |
| `d9a2a1694` | retained | Within-page answer-aspect tests |
| `de289630b` | retained | Trading-query retrieval regressions |
| `1b740cd4a` | retained | Concurrent endpoint-repair unit tests |
| `850bbb4b9` | retained | Cognify rollback input-status tests |
| `02489b44b` | retained | Alias-linked answer tests |
| `83bf6c0f3` | retained | Declared-alias evidence tests |
| `403756b70` | retained | Supported-claim evidence tests |
| `b2c8233f3` | retained | Bounded reference-embedding tests |
| `bf3a984d7` | retained | Oversized structured-answer embedding tests |
| `cff96912e` | ported | Temporal methods moved to `PostgresDemoAdapter`; focused SQL/shape tests |
| `a6c9b1e03` | retained | Per-page answer grouping tests |
| `a3403912c` | retained | Explicit alias-relation tests |
| `59ad42c61` | retained | Stable chunk-identity fusion tests |
| `e575952e1` | retained | Indexed citation UUID tests |
| `bddcbf19e` | retained | Complete citation-support tests |
| `d4e87f8ec` | retained | Answer-bearing page-overview tests |
| `d66817051` | retained | Multihop summary citation tests |
| `578d4af26` | retained | Weak-overlap rejection tests |
| `226ad3b5e` | retained | Live chunk-identity API tests |
| `71c802e19` | retained | Dataset authorization tests for identity verification |
| `6b1200342` | retained | Exact-section and claim-snippet tests |
| `73e898fce` | retained | Specific-section ranking tests |
| `1bc378141` | retained | Decisive table/status evidence tests |
| `d5a4da858` | retained | Indexed-alias evidence tests |
| `bc9f8d8c7` | retained | Bounded decisive-alias tests |
| `c1bffb18f` | retained | Complete early-claim tests |
| `9293f295a` | retained | Cross-page shared-alias tests |
| `1bc554965` | retained | Broad answer-page coverage tests |
| `b6f19e8e4` | retained | Alias-linked chunk-ranking tests |
| `a67393956` | retained | Post-page-selection capacity tests |
| `ee87a4941` | retained | Preserved in ancestry; intentionally reverted by `f4c9d470c` |
| `e555ca5e4` | retained | Preserved in ancestry; intentionally reverted by `0bba9983d` |
| `c8b13534a` | retained | Preserved in ancestry; intentionally reverted by `d9952351c` |
| `d9952351c` | retained | Revert remains effective |
| `0bba9983d` | retained | Revert remains effective |
| `f4c9d470c` | retained | Revert remains effective |
| `577a8e0e3` | retained | Answer-phrase citation anchoring tests |
| `f9699a0c8` | retained | Citation-window answer-phrase tests |
| `514c7c22a` | retained | Source-body evidence gate tests |
| `4f9b8636a` | retained | Multipart support-page reservation tests |
| `a19161519` | retained | Global-context failure propagation tests |
| `ed776c798` | retained | Global-context neighbor-prefetch bound tests |
| `c4449a917` | retained | Context-search embedding-slot tests |
| `d69f64199` | retained | Errored global-context result rejection tests |

## Conflict decisions

- The upstream data model using `Data.dataset_id` supersedes the removed `DatasetData` table.
- OpenAI-compatible embeddings accept both the upstream `input_type` and the fork's explicit
  Hugging Face tokenizer override.
- RAG completion keeps page-diverse hybrid retrieval, bounded currentness guidance, stable
  citations, and aliases. Upstream preference weights are applied in hybrid score space before
  page selection; a wider candidate pool is used only when chunk weights are present.
- Temporal PostgreSQL recall moved from the deleted `graph/postgres/adapter.py` to
  `graph/postgres_demo/adapter.py`; the `postgres` provider alias remains unchanged.
- The Docker image keeps the curated `api`, `postgres`, `llama-index`, and `ollama` extras while
  adopting upstream's non-root runtime, writable storage roots, healthcheck, and removal of the
  deleted `distributed` package.
- Instructor/Luna streaming, reasoning-content materialization, strict-schema normalization,
  and concurrency limiting remain explicit. Upstream's native structured-output fallback is
  retained separately; switching Family from Instructor is a deployment qualification decision.

## Validation before deployment

This ledger covers source integration only. A production-shaped database/file snapshot migration,
real Qwen embedding canary, real Luna/OmniRoute canary, API/dashboard/MCP checks, and rollback drill
remain required before Family promotion.
