# Local v1.4.2 upstream integration

This fork merges the immutable upstream Cognee `v1.4.2` tag at
`b0ea53e95caa7bfebf373ffabfbafb44f9fff922` into the self-hosted reliable-stack
baseline merged by fork PR #1. The merge preserves upstream ancestry so later
release comparisons do not replay or duplicate the local patches.

## Conflict disposition

Only four fork-owned paths overlapped the upstream release. Two merged without
manual resolution. The remaining conflicts were resolved as follows:

- `Dockerfile` retains the deliberately narrow production profile: API,
  PostgreSQL, llama-index, and Ollama extras. The upstream default image added
  FastEmbed and several optional providers, but this deployment uses an
  OpenAI-compatible external embedding service and does not select those
  backends. Keeping them out avoids silently expanding the production image.
- `get_visualize_router.py` composes upstream's JSON, semantic, brains, and
  live-event endpoints with the fork's fail-closed authentication posture.
  Every visualization route uses the same posture-aware dependency: normal
  deployments require an authenticated user, while the explicit
  `ALLOW_UNAUTHENTICATED_LOCAL_VISUALIZE=true` opt-in resolves the local default
  user. A route-level regression test covers new endpoints as well as the
  pre-existing HTML and multi-user routes.

The `.env.template` and PostgreSQL/PGVector adapter changes merged directly.

## Relevant upstream behavior

The integration adopts the v1.4.2 database session, pool, cancellation, and
deadlock fixes; dataset queue/cache lifecycle changes; inherited graph-schema
field preservation; default `improve()` projection skip; visualization JSON
APIs; permissions changes; and the new integration-credential migration.

`SUBPROCESS_IDLE_TTL_SECONDS` is retained for deployments using subprocess
Ladybug/LanceDB engines. The family deployment uses PostgreSQL and PGVector, so
that setting does not alter its current runtime behavior.

This baseline update does not claim the release added retrieval ranking
features. The final v1.4.2 tag has no net search/retrieval implementation change;
fork retrieval work remains separately reviewable.

## Upgrade and rollback notes

Before deployment, back up PostgreSQL and record the current image digest. Run
Alembic against a restored copy first because v1.4.2 adds the
`workspace_id` dimension to integration credentials. Build and validate the API
and frontend images from the reviewed merge commit, then run authenticated API,
visualization, ingestion, search, and migration smoke tests. Roll back both the
image and database together if the migration or compatibility checks fail.
