# Vulcan continuation roadmap

This is the execution plan for the next phases of Vulcan, written to be carried
out session-by-session by an AI coding agent (or a human) without re-deriving
decisions. It builds directly on the merged multi-provider foundation
(schema v2, PR #5) and the vision recorded in `docs/ARCHITECTURE.md`.

Read order before any work: `README.md` → `docs/ARCHITECTURE.md` → this file →
`config/vulcan.example.toml` → the test file nearest your phase.

---

## 1. How to work this plan (agent operating instructions)

0. **Next phase: operator-directed work or the §8 standing item.** Phases
   1–4 and the Road-to-1.0 gates (§9) are complete and merged.
1. **One phase per pull request.** Complete phases strictly in order. Do not
   start phase N+1 in the same PR as phase N. Small preparatory refactors
   belong in the phase PR that needs them.
2. **Session ritual.** At the start of every session:
   `uv sync --all-groups --locked`, then run the full quality gate (below) on a
   clean checkout of `main` to confirm a green baseline before changing code.
3. **Quality gate** — must pass before every push, no exceptions:
   ```bash
   uv run ruff format --check .
   uv run ruff check .
   uv run pytest
   uv run python scripts/smoke.py
   ```
4. **Tests are extended, never replaced.** Existing assertions may only change
   when the contract they pin intentionally changes in your phase, and the PR
   description must call out every such change. Every new upstream surface
   uses `httpx.MockTransport` — nothing in the suite may contact a real API.
5. **Every new surface gets leak tests.** Any new endpoint, adapter, stream,
   or CLI output needs sentinel tests proving API keys, auth headers, prompts,
   and upstream bodies cannot appear in responses, errors, or logs. Copy the
   sentinel pattern from `tests/test_routing.py` and `tests/test_anthropic.py`.
6. **Verify the invariants checklist (§3) in every PR description.**
7. **When this plan and observed reality disagree** (an API changed, a listed
   endpoint is gone), stop, write down the discrepancy in the PR, and choose
   the smallest design consistent with the invariants — do not silently
   improvise a bigger feature.

## 2. Vision (unchanged)

Vulcan is a **local-first, single-user AI gateway for explicitly configured
local and BYOK models**. Local tools speak to one stable loopback API; every
public alias routes to exactly one named provider (local Ollama or a hosted
API used with the operator's own key). Vulcan is infrastructure — the value is
predictability, safety, and honesty, not feature breadth.

## 3. Non-negotiable invariants (re-verify every PR)

- [ ] Listener binds loopback only; `Host`-header allowlist intact.
- [ ] Exactly one provider per alias; **no fallback, no retries, no
      auto-routing**; at most one upstream inference call per client request.
- [ ] Credentials only via `api_key_env` environment references; never in
      TOML, responses, errors, logs, or persisted state.
- [ ] Content-safe logs: fixed event names, recursive redaction, no prompt or
      response text, no native model names in readiness logs.
- [ ] Upstream bodies are classification-only; never surfaced verbatim.
- [ ] Hosted providers are never probed automatically (no billable calls to
      render `/healthz` or `/v1/models`); explicit operator CLI actions may.
- [ ] Strict TOML: unknown keys fail startup; no env-var config overrides.
- [ ] No telemetry, analytics, model downloads, or catalogue discovery.
- [ ] Ollama endpoints stay loopback-only, no path; hosted endpoints stay
      explicit HTTPS (HTTP loopback-only for mocks/proxies).
- [ ] Schema policy: **additive optional config fields keep
      `schema_version = 2`** (old configs must keep validating unchanged);
      bump to 3 only for a breaking rename/removal, with a v2→v3 migration
      message exactly as loud and specific as the current v1→v2 one.

## 4. Foundation summary (what already exists)

| Concern | Where |
| --- | --- |
| Strict v2 config, URL/env-name validation, v1 rejection | `src/vulcan/config.py` |
| Provider protocol (`provider_id`, `provider_type`, `chat`, `discover_runtime`, `aclose`) | `src/vulcan/providers/base.py` |
| Hardened client builder, credential resolution, hosted status mapping | `src/vulcan/providers/http.py` |
| Adapters: Ollama (native), Anthropic (Messages), OpenAI-compatible, deterministic | `src/vulcan/providers/*.py` |
| Exact routing, per-provider probe cache (post-probe TTL), preflight, error annotation | `src/vulcan/gateway.py` |
| Per-provider readiness reconciliation | `src/vulcan/readiness.py` |
| v1 HTTP contract (`/healthz`, `/v1/models[/{id}]`, `/v1/capabilities`, `/v1/chat/completions`) | `src/vulcan/api.py`, `src/vulcan/schemas.py` |
| Error taxonomy incl. `missing_credential`, `provider_auth_failed`, `provider_rate_limited` | `src/vulcan/errors.py` |
| Safe JSON logging + redaction | `src/vulcan/observability.py` |
| CLI `serve` + `check` (credential presence without values) | `src/vulcan/cli.py` |
| Real-process smoke test (all five endpoints) | `scripts/smoke.py` |
| 628 tests, all upstream traffic mocked | `tests/` |

---

## 5. Phase 1 — Streaming chat (SSE) — ✅ DONE

Shipped: SSE chunks on `stream: true`, `chat_stream` on all four adapters, the
pre-stream/mid-stream error split, cancellation handling, and streaming
coverage in `tests/test_streaming.py` plus `scripts/smoke.py`. The contract as
built is documented in the README ("Streaming") and `docs/ARCHITECTURE.md`
("Streaming (added after v2)"). The specification below is retained as the
design record.

**Why first:** most local tools assume `stream: true` works on an
OpenAI-style endpoint; it is the largest remaining gap between Vulcan and the
"one stable local API" goal.

### Contract

- `POST /v1/chat/completions` with `stream: true` returns
  `Content-Type: text/event-stream` with OpenAI-style chunks:
  `data: {"id", "object": "chat.completion.chunk", "created", "model",
  "provider", "choices": [{"index": 0, "delta": {"role"?: "assistant",
  "content"?: str}, "finish_reason": "stop"|"length"|null}]}` — first chunk
  carries `delta.role`, subsequent chunks carry `delta.content`, the final
  chunk carries `finish_reason` (and `usage` if the upstream supplied both
  counts; never invented) — terminated by `data: [DONE]`.
- `stream: false` (and the whole non-streaming path) must remain
  byte-for-byte unchanged.
- Errors **before** the first byte is sent use the normal JSON error envelope
  and status codes. Errors **mid-stream** (HTTP 200 already committed) emit
  one final SSE event `data: {"error": {code, message, retryable,
  details}}` — the same normalized `ErrorBody` shape, never upstream bytes —
  then close the stream without `[DONE]`. Document this shape in the README.
- `/v1/capabilities` reports `chat_completions.streaming: true`.

### Provider layer

Add to the `Provider` protocol:
`chat_stream(request) -> AsyncIterator[ProviderStreamEvent]` where
`ProviderStreamEvent` is a small frozen dataclass union:
`StreamDelta(text: str)` | `StreamEnd(finish_reason, usage | None)`.

- **openai_compatible**: send `"stream": true` (plus
  `{"stream_options": {"include_usage": true}}` only if trivially safe across
  vendors — if any doubt, omit and take usage when the final chunk has it);
  parse SSE `data:` lines; ignore unknown fields; map `finish_reason` as in
  the non-streaming path.
- **anthropic**: send `"stream": true`; translate SSE events —
  `message_start` (capture `usage.input_tokens`), `content_block_delta` with
  `text_delta` → `StreamDelta`, `message_delta` (capture `stop_reason`,
  `usage.output_tokens`), `message_stop` → `StreamEnd`. Any non-text content
  block type mid-stream is a protocol error. Local guards (temperature > 1,
  assistant-first) apply before any I/O, exactly as non-streaming.
- **ollama**: send `"stream": true`; parse NDJSON lines; final line has
  `done: true` plus optional eval counts → usage.
- **deterministic**: yield the configured text as one `StreamDelta` then
  `StreamEnd("stop", None)` — this powers smoke/contract tests.

### Rules

- Preflight, routing, `metadata` logging, and readiness behavior identical to
  the non-streaming path; count `output_chars` by summing delta lengths.
- Malformed frames/lines → normalized `provider_protocol_error` (mid-stream
  rules above). Timeouts between chunks → `provider_timeout` (httpx read
  timeout already applies per-read).
- Client disconnect must cancel the upstream request and close the provider
  response cleanly (use `httpx` streaming context managers; test with a
  cancelled request).
- No buffering of the whole reply; forward as received.

### Acceptance criteria

- All four adapters stream through mocked transports, with tests for: chunk
  translation, finish reasons, usage propagation, malformed frame → error
  event, mid-stream disconnect, pre-stream failures (missing credential, 401,
  model_not_found) still returning normal JSON envelopes, and sentinel leak
  tests on stream output and logs.
- `scripts/smoke.py` gains a deterministic streaming request asserting the
  exact chunk sequence and `[DONE]`.
- README + ARCHITECTURE updated (streaming contract, mid-stream error shape,
  capability flag). Quality gate green.

---

## 6. Phase 2 — Embeddings endpoint — ✅ DONE

Shipped: `POST /v1/embeddings` on ollama/openai_compatible/deterministic,
config-time rejection of embeddings on anthropic providers, finite-vector
and ordering validation, and coverage in `tests/test_embeddings.py` plus
`scripts/smoke.py`. The contract as built is documented in the README
("Embeddings") and `docs/ARCHITECTURE.md` ("Embeddings (added after v2)").
The specification below is retained as the design record.

**Why:** `Capability.EMBEDDINGS` already exists in config but is not
callable; local RAG tools need it.

### Contract

- `POST /v1/embeddings`: `{"model": alias, "input": str | [str, ...]}` —
  1–64 inputs, each 1–8192 chars, combined ≤ 65536, strict schema, blank
  inputs rejected.
- Response: `{"object": "list", "model": alias, "provider": provider_id,
  "data": [{"object": "embedding", "index": i, "embedding": [float, ...]}],
  "usage": {"prompt_tokens", "total_tokens"} | null}` — order matches input
  order; usage only when upstream supplies it.
- Routing: alias must declare the `embeddings` capability
  (`unsupported_capability` otherwise); same exact-provider, no-fallback
  rules; same error taxonomy.

### Adapters

- **ollama**: `POST /api/embed` `{"model", "input": [...]}` → `embeddings`
  list-of-lists.
- **openai_compatible**: `POST {base_url}/embeddings` with Bearer auth →
  standard shape.
- **anthropic**: does not offer embeddings — reject at **config load** with a
  dedicated reason code (`anthropic_embeddings_unsupported`) when a model on
  an anthropic-typed provider declares `embeddings`; never fail at request
  time for this.
- **deterministic**: fixed vector (e.g. eight `0.125`s) per input, for smoke
  and contract tests.
- Response validation: embedding entries must be finite floats — set
  `allow_inf_nan=False` on the response models (Python's JSON parser accepts
  `NaN`/`Infinity` by default; a malformed upstream must map to
  `provider_protocol_error`, not propagate).
- `/v1/capabilities` gains an `embeddings` block; keep "at least one chat
  model" as the startup rule.

### Acceptance criteria

Adapter translation tests (single + batch, order preservation), bounds
rejection tests, anthropic config rejection test, non-finite float rejection,
leak tests, README/ARCHITECTURE/capabilities/smoke updates, quality gate
green.

---

## 7. Phase 3 — Operator tooling and hardening — ✅ DONE

Shipped: per-provider single-flight probe locks, `vulcan check --verify-credentials`
(operator-invoked only), and streaming socket-hygiene coverage, all in
`tests/test_operator_tooling.py`. Documented in the README (credential
handling) and `docs/ARCHITECTURE.md`. The specification below is retained as
the design record.

Three small, independent items; one PR.

1. **Single-flight probes.** `Gateway._probe_provider` currently allows
   concurrent requests to trigger duplicate `/api/tags` probes for the same
   provider (bounded, but wasteful under burst). Add a per-provider
   `asyncio.Lock`; re-check the cache after acquiring. Test with two
   concurrent readiness calls asserting one upstream probe.
2. **`vulcan check --verify-credentials`.** Explicit, operator-invoked (never
   automatic) live verification: for each hosted provider make one metadata
   call — `GET {base_url}/models` (Bearer) for openai_compatible,
   `GET {base_url}/v1/models` (x-api-key + version header) for anthropic —
   and report per provider `verified | auth_failed | unreachable | error`
   without ever printing bodies or values. Timeout: the provider's configured
   timeout. Exit codes unchanged in spirit: any non-`verified` hosted
   provider ⇒ exit 1. Without the flag, `check` behavior is byte-identical to
   today. Tests mock the transports; include a test that the flag is required
   for any network attempt.
3. **Ollama keep-alive/socket hygiene pass.** Verify streaming (Phase 1) left
   no unclosed responses under error paths (aclose coverage tests); nothing
   speculative beyond that.

Update README (`check` flag docs) and ARCHITECTURE ("explicit operator
actions may call authenticated endpoints; automatic surfaces never do").

---

## 8. Phase 4 — Maintenance and small conveniences (as-needed backlog)

**Every listed item is done as of 2026-07; the roadmap through Phase 4 is
complete.** The entries stay here because each one records how to redo that
kind of work, not just that it happened. A session with nothing else pending
should default to the standing item at the end rather than inventing scope —
new features come from the operator, and §9 says what to refuse outright.

Do these only when a session has no higher phase pending, one PR per item:

- ~~**`/v1/usage`**~~ — ✅ DONE. In-memory, process-lifetime counters per alias
  and per provider, recorded on success only, with `requests_with_usage` making
  the token totals interpretable. Documented in the README ("Usage counters")
  and `docs/ARCHITECTURE.md`; covered by `tests/test_usage.py` and
  `scripts/smoke.py`.
- ~~**DeepSeek `reasoning_content`**~~ — ✅ DONE. Still ignored: never
  forwarded, never substituted for a missing `content`, and its token
  breakdown never added to usage. Pinned by `tests/test_reasoning_content.py`
  (buffered, streamed, and over HTTP) and documented in the README
  ("Vendor extension fields").
- ~~**Dependency bumps**~~ — ✅ DONE (2026-07). `uv lock --upgrade` moved
  annotated-types, certifi, fastapi, httpcore2, httpx2, and ruff; no package
  was added or removed and every bump stayed inside the existing conservative
  ranges, so `pyproject.toml` is untouched. Repeat the same way: upgrade the
  lock, run the four-command gate, and only widen a range in `pyproject.toml`
  when a bump actually needs it.
- ~~**CI matrix**~~ — ✅ DONE (2026-07). `quality` runs on 3.12 and 3.13 with
  `fail-fast: false`; `UV_PYTHON` pins each leg to its matrix interpreter so a
  job cannot silently test the wrong one. Verified on a real 3.13 before the
  matrix was added — lock resolves, ruff and 519 tests pass, smoke green.
  Add the next version the same way: prove the gate locally first, then widen
  the matrix.
- Keep the suite fast (< ~10s); parallelize only if it grows past that.

---

## 9. Road-to-1.0 gates (operator-directed, 2026-08-16) — ✅ DONE

After Phase 4, the operator directed a "Road-to-1.0" push focused on usage
visibility and spend control for the multi-seat mickey deployment. All of it
shipped on 2026-08-16 across six PRs; gates 4 and 5 are labeled as such in
their PRs, while gates 1–3 were named only in the fleet log discussion
(`shared/logs/mickey-actions.log`, 2026-08-16) and are not labeled in the
repo record. What landed, in order:

- **Seat usage attribution** (PR #14). Optional `seat` caller label on chat
  and embedding requests and a `by_seat` view in `/v1/usage`, so several
  local tools sharing one gateway can see who spent what. Attribution before
  enforcement: no budgets, no auth semantics; the label never leaves the
  process (pinned by `tests/test_seat.py` sentinels).
- **Operator read subcommands** (PR #15). `vulcan usage` / `vulcan models`
  GET the running gateway's JSON verbatim — curl+jq replacement for the two
  questions an operator actually asks on a headless box.
- **Gate 4 — durable usage ledger** (PRs #16, #17). Opt-in
  `[usage] ledger_path`: append-only JSONL, one line per completed request
  (never content), replayed into counters at boot so `/v1/usage` survives
  restarts. Asymmetric failure policy: an unopenable ledger kills startup
  loudly; a failed append is counted and logged but never fails a completed
  request. PR #17 hardened the trust boundary post-merge — replay enforces
  the write-side contract per line, violations become `skipped_lines`, and a
  poisoned ledger cannot reach HTTP responses (seven-class sentinel).
- **Gate 5 — per-seat daily budgets, hosted only** (PRs #18, #19). When
  `[budgets.seats.*]` exists, hosted requests are gated pre-flight: seat
  required, fail-closed resolution (own entry or `default`), UTC-day
  headroom. Refusals are typed (`seat_required` / `budget_unconfigured` /
  `budget_exhausted` + reset time) and never rerouted — the caller owns any
  fallback. PR #19 made the reservation lifecycle cancellation-safe
  (finally-guaranteed release on error, disconnect, or abandoned stream) and
  closed the replay-side cardinality cap.

Process lesson, recorded so the next ladder goes smoother: PRs #16 and #18
were each merged at their round-1 review heads, and later review rounds had
to land as separate fix-forward PRs (#17, #19). **Land all review rounds on
the branch before merging** — a fix-forward PR is the tax for merging early.

The mickey deployment these gates serve is documented in `deploy/` (systemd
unit + layout convention).

## 10. Out of scope until the operator explicitly asks

Chat UI, agents/tool-calling, images/multimodal, model
download/pull/management, auto-routing or "best model" selection, retries and
fallback chains, circuit breakers, load balancing, multi-user auth/state,
billing/cost tracking beyond `/v1/usage` counters, credential storage,
hosted-provider auto-probing, per-vendor adapters, external telemetry, and a
client SDK (revisit when at least two consumers exist). If a change seems to
require one of these, stop and ask instead of building it.

---

## 11. Phase 5 — Headless-harness hardening (operator-directed, 2026-08-23)

The operator directed a "best headless harness for headless mini PCs" push,
planned from mickey's live layout (unattended unified-memory box; full plan:
`~/ai-workspace/kimi/notes/2026-08-23_vulcan-headless-harness-plan.md`).
Items land one phase per PR in this order:

- ~~**CI runs the full gate**~~ — ✅ DONE (2026-08-23). CI now runs
  `uv run python scripts/smoke.py` after pytest on both matrix legs, closing
  the drift with the §1 four-command gate. Smoke needs only stdlib + `ss`
  (iproute2 is preinstalled on `ubuntu-latest`).
- ~~**`usage_reporter.py` test harness**~~ — ✅ DONE (2026-08-23).
  `tests/test_usage_reporter.py` (29 tests) loads the script by path
  (scripts/ is not a package; the `sys.modules` registration before
  `exec_module` is required for `@dataclass`), fakes the urllib opener —
  the same no-real-network guarantee MockTransport gives the gateway suite —
  and pins baseline/reset/delta honesty, HMAC signature recomputation, exit
  codes 0/1/2, and sentinel leak rules: the forge secret never reaches the
  wire body, headers, stdout, stderr, or the state file, and prompt-shaped
  payload fields never propagate into the digest. Repeat for any new
  script surface: fake the transport, recompute the signature, sentinel the
  secret.
- ~~**Truthfulness fixes**~~ — ✅ DONE (2026-08-23). The dead
  `provider_failed` event (allowlisted, never emitted) left `_SAFE_EVENTS`;
  its formatter test moved to `chat_failed` and a demotion pin proves the old
  name now renders as `external_log`. `/v1/capabilities` is derived from
  `registry.list()`: `callable_capabilities` is the union of configured
  capabilities and the `embeddings` block appears only when an alias declares
  it (`response_model_exclude_none`, so the key is absent, not null). The
  stale "592 tests" in §4 was corrected. Intentional contract changes flagged
  per §1.4: safe-event allowlist shrink; capabilities response shape for
  embeddings-less configs.
- ~~**Ollama `keep_alive` passthrough**~~ — ✅ DONE (2026-08-23). Optional
  per-alias `keep_alive` on `ModelConfig`: a strict validator accepts only
  `-1`, `0`, or whole-number Go durations (`(\d+(ns|us|ms|s|m|h))+` — no
  fractions, no signs beyond the bare pin), and the GatewayConfig cross-check
  rejects it on non-Ollama providers at load (`keep_alive_ollama_only`,
  mirroring `anthropic_embeddings_unsupported`). The value rides
  `ProviderChatRequest`/`ProviderEmbeddingRequest` into the Ollama chat and
  embed payloads only when set; unset aliases are byte-identical on the wire
  (pinned by the pre-existing exact-payload tests). Other adapters can
  provably never receive a non-None value, so they are unchanged. Tests:
  duration grammar accept/reject, cross-provider rejection, payload pins for
  chat + embed, and an end-to-end routing test proving the value reaches the
  Ollama payload and never the log stream. The core unified-memory knob:
  pin workhorses, TTL the rest.
- ~~**Operator memory-lifecycle CLI**~~ — ✅ DONE (2026-08-23). `vulcan ps` /
  `vulcan unload <alias>` / `vulcan warmup <alias>`, CLI-direct-to-Ollama with
  no new gateway HTTP surface: `ps` maps `/api/ps` residents back to aliases
  (unmatched residents flagged `unmanaged`; non-Ollama providers listed as
  `skipped`, never contacted); `unload` sends one `keep_alive = 0` load
  request; `warmup` sends one empty load request carrying the alias's
  configured `keep_alive` when set — chat aliases via `/api/generate`,
  embedding-only aliases via `/api/embed` (embedding models refuse
  `/api/generate`; caught live 2026-08-23, fixed same day). Hosted/deterministic aliases are refused
  (exit 2) before any client is built, response bodies are never read
  (classification by status alone), and native model names appear only in
  `ps` unmanaged rows — the documented operator-terminal exception. Tests
  (`tests/test_cli_ollama_ops.py`, 10): MockTransport client factory injected
  by monkeypatching `cli._ollama_client`; pins the refusal-before-network
  guarantee (factory never invoked for refused aliases) and the
  native-name/exception-text leak rules on success and error paths alike.
- ~~**Gateway concurrency bound**~~ — ✅ DONE (2026-08-23). `[server]
  max_concurrent_requests` (strict positive int; unset = unbounded, today's
  behavior) puts one `asyncio.Semaphore` across `chat`/`chat_stream`/`embed`.
  Admission is non-blocking (`locked()` then `acquire()`, no suspension point
  between — exact on the event loop); saturation raises `gateway_overloaded`
  (503, retryable) before provider selection, so refused requests make zero
  upstream calls and consume no budget slot. Slots are held for the full
  stream lifetime and released in the same `finally` as unsettled budget
  reservations; liveness/discovery endpoints are never gated. Tests
  (`tests/test_concurrency.py`, 7): saturation rejects chat+embed without
  reaching the provider, slot release on completion/mid-stream
  abandon/provider error, unbounded default, constructor validation, and the
  HTTP envelope + ungated-liveness pin; config grammar in `test_config.py`.
- ~~**systemd watchdog**~~ — ✅ DONE (2026-08-23). `src/vulcan/notify.py` is
  a raw-socket sd_notify client (~70 owned lines, no dependency — at a 4-dep
  posture, one datagram does not justify `sdnotify`): `NOTIFY_SOCKET` unset
  ⇒ complete no-op; set ⇒ `READY=1` at lifespan start plus `WATCHDOG=1`
  heartbeats on the uvicorn loop at `WATCHDOG_USEC`/2 (a wedged loop stops
  beating — that is the detection semantics); set-but-undeliverable ⇒
  startup fails loud. `@`→NUL abstract-namespace translation included. The
  unit gains `Type=notify` + `WatchdogSec=30` and must deploy together with
  the code (a watchdog without the heartbeat kills a healthy process).
  Tests (`tests/test_notify.py`, 10) use real AF_UNIX datagram receivers on
  tmp paths — no mocking, no systemd — plus a lifespan integration pin that
  heartbeats stop when the app shuts down.

Trigger-gated, design pre-agreed: `GET /metrics` when an actual scraper
exists on mickey (hand-rolled exposition, no prometheus-client dep); ledger
boot-time size-cap rotation when the ledger reaches tens of MB (never
logrotate — rename mode never rotates a held fd, copytruncate poisons
replay). Refused: SIGHUP config reload (`vulcan check` + restart runbook
covers the hazard), in-app log streaming (journald already streams), runtime
routing mutation, hosted lifecycle calls, new dependencies for notify or
metrics.
