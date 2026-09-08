# Cognee v1.5.4 integration

This branch merges upstream tag `v1.5.4` (`20e0bd887`) into the fork's
v1.5.3 production baseline (`53437f101`). The upstream tag comparison contains
982 changed files and 470 reachable commits including merge commits. This is
a substantive upgrade, unlike the small v1.5.2-to-v1.5.3 change.

## Conflict decisions

- Preserve hybrid retrieval, citation grounding, live chunk identity checks,
  PostgreSQL temporal queries, stored-vector visualization, embedding context
  fallback, and bounded local LLM concurrency.
- Combine upstream streaming token sinks with local streaming-response
  materialization and reasoning-content compatibility.
- Preserve upstream graph-before-vector ordering and chunk ownership groups.
  Foreign-key endpoint repair retries the node ownership writer, rather than
  assigning an edge group's ownership to every node in the batch.
- Retain authenticated visualization by default for every HTTP route and the
  new WebSocket route. Anonymous visualization still needs the explicit
  `ALLOW_UNAUTHENTICATED_LOCAL_VISUALIZE=true` opt-in.
- Keep nullable legacy activity normalization. Adopt the new business graph
  page while retaining the fork's separate schema page.
- Adopt the standalone frontend image and canonical TypeScript Next config.
  Preserve same-origin `/cognee-api` rewrites, build-time URL/email arguments,
  and server-only `COGNEE_INTERNAL_API_URL` routing. Remove obsolete duplicate
  Next/Jest configs and unused cloud tenant helpers.
- Keep the slim self-hosted API extras, add upstream's `dlt` extra, and retain
  `COGNEE_EXTRAS` for optional provider packages.
- Refresh the upstream MCP lock from Cognee 1.5.3 to 1.5.4 (the release tag
  still carried the old lock). The lock changes only the Cognee package entry.

## Deployment compatibility

The standalone frontend starts with `node server.js`; deployments that override
the image command with `npm run start` must update that override. Set `PORT` to
the existing container port. A same-origin deployment should retain its
`NEXT_PUBLIC_LOCAL_API_URL=/cognee-api` build argument and configure the
server-only internal API URL.

The add API now accepts multipart `raw_data`; multipart uploaded `data` remains
supported. Local server-file access is disabled unless explicitly configured.
Relational migrations include a frozen-schema reconciliation and session agent
attribution. A production backup and restored-copy qualification remain required
before promotion; unit tests and a synthetic schema upgrade cannot substitute
for that gate.

## Verification and safety

Verified on Windows with Python 3.12: focused fork regression suites, PostgreSQL
adapter/storage/API tests, SQLite migration/model lockstep, and frontend tests,
type checking, and production build. Linux Docker API and frontend builds also
pass. A separate private deployment report records exact commands, results,
image identifiers, and the disposable PostgreSQL upgrade probe.

Do not run `test_ladybug_subprocess_lock_race.py::test_close_waits_for_worker_process_exit`
or any shard containing it under a native Windows harness. Qualification uses
explicitly selected test paths excluding that file.
