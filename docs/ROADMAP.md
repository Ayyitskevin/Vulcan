# Vulcan continuation roadmap

This is the execution plan for the next phases of Vulcan, written to be carried
out session-by-session by an AI coding agent (or a human) without re-deriving
decisions. It builds directly on the merged multi-provider foundation
(schema v2, PR #5) and the vision recorded in `docs/ARCHITECTURE.md`.

Read order before any work: `README.md` → `docs/ARCHITECTURE.md` → this file →
`config/vulcan.example.toml` → the test file nearest your phase.

---

## 1. How to work this plan (agent operating instructions)

0. **Next phase: Phase 4 (maintenance backlog, as needed).** Phases 1-3 are
   complete and merged.
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
| 364 tests, all upstream traffic mocked | `tests/` |

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
- **CI matrix:** add Python 3.13 alongside 3.12 if the gate passes.
- Keep the suite fast (< ~10s); parallelize only if it grows past that.

## 9. Out of scope until the operator explicitly asks

Chat UI, agents/tool-calling, images/multimodal, model
download/pull/management, auto-routing or "best model" selection, retries and
fallback chains, circuit breakers, load balancing, multi-user auth/state,
billing/cost tracking beyond `/v1/usage` counters, credential storage,
hosted-provider auto-probing, per-vendor adapters, external telemetry, and a
client SDK (revisit when at least two consumers exist). If a change seems to
require one of these, stop and ask instead of building it.
