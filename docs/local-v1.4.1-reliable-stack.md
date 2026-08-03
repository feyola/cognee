# Local v1.4.1 reliable-stack patch disposition

This branch starts from the immutable upstream Cognee `v1.4.1` tag at
`82bc3de9062af26ebcac3d61343d7e1a4f577586`. It carries the self-hosted
behaviors previously maintained on `codex/v1.3.0-reliable-stack` without
changing Cognee's public member-facing API or selecting new optional backends.

The deployment remains authenticated by default. PostgreSQL graph and PGVector
remain supported deployment choices, Instructor remains the structured-output
framework, and local llama.cpp/Ollama models remain external HTTP services.

## Disposition of the 18 v1.3 local commits

| Old commit | Disposition on v1.4.1 | Evidence or integration note |
| --- | --- | --- |
| `bf2beeb` local structured output and concurrency | Ported | The generic Instructor adapter retains its hard concurrency cap and reasoning-content fallback around upstream's overload-aware rate-limit context. Focused tests cover the cap. |
| `8523793` packaged extras, migration, visualization auth | Rewritten during port | The PostgreSQL enum migration uses Alembic's active bind and is tested for present/missing enum states. Visualization keeps upstream's bounded-subgraph parameters and remains authenticated unless its explicit local opt-in is enabled. |
| `023c79c` trusted shared deployments | Ported, secure default retained | Explicit trusted mode remains available, while backend access control still implies authentication when `REQUIRE_AUTHENTICATION` is unset. The template sets authentication on, and malformed boolean values fail closed. |
| `702aab7` shared `/users/me` | Ported | The posture-aware route is registered before the stock FastAPI Users route; authenticated mode still resolves the real caller. |
| `365d4dc` fetched visualization rendering | Rewritten during conflict resolution | `srcDoc` replaces browser blob URLs. Upstream's centralized dataset-status polling was retained instead of restoring the old duplicate poller. |
| `1122bc2` d3 force types | Ported | The local declaration removes the untyped dynamic-import suppression. |
| `40166cf` embedded history limits | Ported | Story view bounds history replacement defensively. |
| `2b70ca8` authenticated dashboard defaults | Ported and composed with upstream UI | Hard-coded default credentials and unauthenticated server-side login fallbacks remain removed; current upstream styling is retained. |
| `ca56d82` same-origin API proxy | Ported | Next rewrites `/cognee-api` to the internal API and logout uses the public same-origin path. |
| `8141e97` proxy in container config | Ported | The production container copies the rewrite-bearing config. |
| `a0f6b2f` streamed structured responses | Ported | Async LiteLLM streams are materialized before Instructor parses them. |
| `39add10` strict JSON schemas | Ported | Strict schemas close object properties without mutating Instructor input. |
| `6fde5d9` external tokenizer override | Ported | `HUGGINGFACE_TOKENIZER` reaches the OpenAI-compatible embedding engine; Qwen aliases can use `Qwen/Qwen3-Embedding-4B`. |
| `217cb14` discriminated unions | Ported | Pydantic discriminator `oneOf` schemas are normalized to OpenAI-supported `anyOf` and remain Pydantic-validated after generation. |
| `5cc1eea` SQL cache UUID normalization | Upstream-equivalent; not cherry-picked | Upstream v1.4.1 PR #4182 uses `StringKey` bind processors to stringify UUID and other identifiers across all relevant cache columns and adds broader PostgreSQL regression coverage. |
| `ce6bd6b` production API extras | Ported | The production image installs only API, PostgreSQL, llama-index, and Ollama extras required by the selected deployment. v1.4.1 core dependencies, including `cryptography`, still come from the lock. |
| `c650ec2` external llama.cpp server | Ported | In-process `llama-cpp-python` and its compiler toolchain remain excluded; compatible servers are reached over HTTP. |
| `a18dd63` stored PGVector embeddings for visualization | Ported and composed with upstream bounded visualization | PGVector can return stored vectors, and both cached leases and the runtime re-resolving vector handle expose the live adapter for capability detection. Upstream's seed/depth/node bounds remain unchanged. |

## Concurrency and rate behavior

The generic OpenAI-compatible Instructor adapter applies two independent
controls in this order:

1. `COGNEE_LLM_MAX_CONCURRENCY` bounds in-flight requests. Zero or unset means
   no hard concurrency cap.
2. Upstream's `LLM_RATE_LIMIT_*` and `AUTO_RATE_LIMIT` policy paces each dispatch
   attempt when configured or when overload evidence starts a cooldown.

Embedding indexing uses upstream's `EMBEDDING_MAX_CONCURRENT_DATA_POINTS` in
combination with `EMBEDDING_BATCH_SIZE`. Deployments should set these values
explicitly; this branch does not encode production throughput defaults.

Upstream v1.4.1 also changes the default cognify `chunks_per_batch` from 100 to
2000. Administrative callers should send an explicit tested value rather than
depending on that default.

## Open-source frontend integration

The v1.4.1 frontend sync contained several SaaS-only references without their
cloud modules. The local build now makes those boundaries explicit: local mode
does not provision or cache cloud tenants, the local user-state stub owns its
small return type, and the unused management-client export is absent. The
tenant pod HTTP client is included with contract tests for URL anchoring,
non-overridable API-key injection, cookie omission, and browser-failure
forwarding. Jest setup is checked in so these TypeScript frontend contracts can
run rather than only being type-checked by Next.js.

## Deliberately not adopted

This port does not select Turso, Slack OAuth, MCP sampling, upstream video
ingestion, code-graph ingestion, or a different graph/vector backend. Those are
independent product decisions. It also does not alter datasets, migrations in a
live deployment, or deployment configuration.
