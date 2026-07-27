# Vulcan v2 architecture: local-first multi-provider gateway

This document explains the migration from the single-provider v1 gateway to the
multi-provider v2 gateway: the provider taxonomy, the configuration schema, the
safety model, and the compatibility decisions. It is the design record for
`schema_version = 2`.

## Goal

Vulcan is a local-first, single-user AI gateway for explicitly configured local
and BYOK (bring-your-own-key) models. Local tools talk to one stable loopback
API; every configured public model alias routes to exactly one named provider
and one provider-native model name. Vulcan remains infrastructure: no chat UI,
no agents, no autonomous routing, no model downloads, no multi-user state, no
billing, and no cloud credential management beyond reading operator-supplied
environment variables at request time.

## Provider taxonomy

Vulcan v2 has exactly four provider *types*. Operators declare any number of
named provider *instances* of those types.

| Type | Purpose | Network | Credential |
| --- | --- | --- | --- |
| `ollama` | Native adapter for a local Ollama runtime (`/api/chat`, `/api/tags`). | Loopback HTTP(S) only. | None. |
| `anthropic` | Native Anthropic Messages API (`/v1/messages`). | HTTPS (or loopback HTTP for local proxies/tests). | `api_key_env` → `x-api-key`. |
| `openai_compatible` | One adapter for OpenAI and every OpenAI-compatible chat-completions endpoint: OpenAI, xAI/Grok, Moonshot/Kimi, Z.AI/GLM, DeepSeek, and future compatible vendors. | HTTPS (or loopback HTTP for local proxies/tests). | `api_key_env` → `Authorization: Bearer`. |
| `deterministic` | In-process, no-I/O canned response for tests and smoke checks. | None. | None. |

There is deliberately **no per-vendor adapter** for OpenAI-compatible
companies. Vendor differences that matter (base URL, credential variable, and
the token-limit field name) are configuration, not code, so new compatible
vendors need zero application changes. The two safe per-instance knobs are:

- `base_url` — explicit, validated, per instance (e.g. `https://api.x.ai/v1`,
  `https://api.z.ai/api/paas/v4`). Vulcan appends the fixed path
  `/chat/completions`; it never builds URLs from request data.
- `max_tokens_field` — `"max_tokens"` (default) or `"max_completion_tokens"`
  (required by newer OpenAI models). Nothing else about the wire contract is
  configurable.

## Configuration schema v2

Configuration remains strict TOML: unknown keys fail startup, values are
type-strict (no boolean/number coercion), and validation errors report field
paths and reason codes only — never raw values.

```toml
schema_version = 2

[server]                      # unchanged from v1; loopback only
host = "127.0.0.1"
port = 8140
log_level = "INFO"

[readiness]                   # unchanged from v1
probe_ttl_seconds = 5.0

[providers.local-ollama]
type = "ollama"
base_url = "http://127.0.0.1:11434"
timeout_seconds = 60.0

[providers.openai]
type = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key_env = "VULCAN_OPENAI_API_KEY"
timeout_seconds = 60.0

[providers.anthropic]
type = "anthropic"
api_key_env = "VULCAN_ANTHROPIC_API_KEY"   # base_url defaults to https://api.anthropic.com
timeout_seconds = 60.0
default_max_tokens = 4096

[[models]]
id = "local-chat"             # public alias, the only ID clients ever see
provider = "local-ollama"     # exact provider instance ID; no fallback
provider_model = "llama3.1:8b"  # provider-native name, never exposed by the API
capabilities = ["chat"]
```

Key rules:

- **Provider IDs** match `^[a-z0-9][a-z0-9_-]{0,63}$`. They are operator-chosen,
  safe to log, and returned in API metadata (`provider` fields).
- **Every model names exactly one provider.** `models[].provider` must
  reference a configured `[providers.*]` table; a dangling reference fails
  startup. Public alias → (provider ID, provider-native model) is a total,
  unambiguous function.
- **Public IDs stay separate from native names.** `id` is the only value in
  API responses; `provider_model` (v1's `runtime_name`) never leaves the
  process.
- **`schema_version = 1` fails loudly** with a migration message before field
  validation runs (see “Migration from v1” below). There is no silent
  auto-migration: credentials and endpoints must be reviewed by a human.

## Secure BYOK model

- **Credentials never live in TOML.** Hosted providers declare
  `api_key_env = "SOME_ENV_VAR"`; the variable *name* must match
  `^[A-Z][A-Z0-9_]{0,127}$` and is safe metadata. The *value* is read from the
  process environment at request time, used to build one request header, and
  never stored, logged, echoed, or persisted.
- A missing/empty/non-printable variable value raises `missing_credential`
  (HTTP 503, non-retryable) naming only the variable and provider — never the
  value. The non-printable check also prevents header-injection via a
  malformed environment value.
- `vulcan check --config vulcan.toml` validates the file and reports, per
  provider, whether the referenced variable is currently set (`present` /
  `missing`) without revealing values. Exit codes: 0 = valid and all
  credentials present, 1 = valid but credentials missing, 2 = invalid config.
  It performs no network I/O.
- `vulcan check --verify-credentials` adds one metadata call per hosted
  provider (`GET {base_url}/models` with Bearer for openai_compatible,
  `GET {base_url}/v1/models` with `x-api-key` + version header for anthropic)
  and reports `verified` / `auth_failed` / `unreachable` / `error` / `missing`.
  Verdicts derive from status codes alone; the body is never read and the
  credential never appears in output. **Explicit operator actions may call
  authenticated endpoints; automatic surfaces never do** — health, models, and
  readiness stay free and unprobed for hosted providers.
- **Log redaction is unchanged and audited.** The safe JSON formatter renders
  only fixed event names and explicitly supplied metadata, and recursively
  redacts keys containing api-key/authorization/token/prompt/response/body/…
  markers. Tests assert that keys, bearer headers, prompts, and upstream
  response bodies never appear in logs, error envelopes, or `vulcan check`
  output.
- **Error chains stay safe.** Adapters raise typed Vulcan errors from httpx
  exceptions; the formatter emits only the exception class name, and HTTP
  handlers never render tracebacks or upstream bodies.

## Network safety model

- The **listener stays loopback-only** and the `Host`-header allowlist
  (localhost / literal loopback) is unchanged.
- **Ollama endpoints keep v1 protections**: loopback host, no path/query/
  fragment, no URL credentials.
- **Hosted endpoints must be explicitly configured** and are validated at
  startup: `https` to any host, or `http` to loopback only (for local mocks
  and proxies); no URL credentials, query, or fragment; an explicit path is
  allowed because several vendors require one. Endpoints come only from the
  operator's TOML — request data can never select or construct a URL, so there
  is no SSRF-style fetch surface.
- Every HTTP client is built by one hardened helper: finite configured
  timeout, `follow_redirects=False`, `trust_env=False` (no proxy/CA
  inheritance), fixed `User-Agent`/`Accept` headers, and clean shutdown on
  application exit.
- **No external telemetry, no analytics, no automatic provider or model
  discovery.** Vulcan calls exactly two kinds of upstream endpoints: the chat
  endpoint for a request the client made, and Ollama's local `/api/tags` for
  readiness.

## Routing and runtime semantics

- `model = "alias"` resolves via the immutable registry to one provider ID and
  one native model name. An unknown alias is `model_not_found`. A provider
  failure is returned as that provider's normalized error — **never** a retry,
  a fallback, another provider, or a fabricated response. Vulcan issues at
  most one upstream chat call per client request, so it can never create
  duplicate charges.
- **Readiness stays honest and free.** Ollama providers are probed via
  `/api/tags` (TTL-cached per provider, `?refresh=true` supported) exactly as
  in v1. Hosted providers are reported as `unchecked` — configured but
  deliberately unprobed, because probing would call authenticated/billable
  endpoints just to render `/healthz`. A hosted model's availability is
  learned only when a real request uses it. The deterministic provider
  remains `available` in-process. Chat preflight probes **only the routed
  provider** (a no-op for hosted/deterministic types), so a hung local
  runtime can never stall requests routed elsewhere, and it short-circuits
  (`model_unavailable`) only when a *live* Ollama inventory proved the native
  name absent. Probe reuse windows are computed from probe *completion* (a
  probe slower than the TTL still earns one full window), and a
  `model_unavailable` outcome invalidates only that provider's cached probe.
  Probes are single-flight per provider: concurrent requests wait on one
  in-flight probe and re-check the cache after acquiring the lock, so a
  burst costs one upstream call rather than one per request.
- `/healthz` reports one entry per configured provider (`id`, `type`,
  `availability`); `/v1/models` and `/v1/chat/completions` report the selected
  provider ID per model/request.

## Streaming (added after v2)

`stream: true` on `/v1/chat/completions` returns Server-Sent Events carrying
OpenAI-style `chat.completion.chunk` frames, terminated by `data: [DONE]`.
Routing, preflight, per-provider probing, error normalization, and logging are
identical to the buffered path — streaming changes the transport, not the
policy, and still issues exactly one upstream call per request.

- **Provider boundary.** `Provider.chat_stream()` returns an async iterator of
  `StreamDelta(text)` events followed by exactly one `StreamEnd(finish_reason,
  usage)`. A stream that ends without `StreamEnd` is a truncated reply and maps
  to `provider_protocol_error`; usage is still reported only when the upstream
  supplied both counts.
- **Error boundary.** Adapters open the upstream connection and classify its
  status *before* yielding, so pre-stream failures still produce an ordinary
  JSON error envelope with its normal status code. After the response headers
  are committed the gateway can no longer change the status, so a failure is
  emitted as one terminal SSE frame containing the same normalized error body
  (plus `request_id`) and the stream closes without `[DONE]`. Upstream bodies
  are never forwarded in either case.
- **Cancellation.** Adapters send with `stream=True` and close the response in
  a `finally`, rather than using httpx's `stream()` context manager: closing an
  async generator suspended inside that context manager violates contextlib's
  athrow protocol, whereas an explicit close releases the upstream connection
  cleanly when a client disconnects mid-stream.
- **Translation.** OpenAI-compatible: SSE `data:` frames, `delta.content`
  accumulated, `[DONE]` ends the stream, usage taken from whichever chunk
  provides it (`stream_options.include_usage` is deliberately not sent — vendor
  support varies). Anthropic: `message_start` (input tokens),
  `content_block_delta`/`text_delta` (text), `message_delta` (stop reason,
  output tokens), `message_stop` (end); any non-text block or delta type is a
  protocol error, and `error` events are classified from their type alone.
  Ollama: newline-delimited JSON, terminal `done: true` chunk carries the
  reason and eval counts.

## Embeddings (added after v2)

`POST /v1/embeddings` embeds one bounded batch through the alias's configured
provider. Routing, preflight, error normalization, and logging mirror the chat
paths; only the payload differs.

- **Provider boundary.** `Provider.embed()` takes
  `ProviderEmbeddingRequest(provider_model, inputs)` and returns
  `ProviderEmbeddingResult(vectors, usage)` with one vector per input, in input
  order. The gateway rejects a vector/input count mismatch as
  `provider_protocol_error` rather than returning records a client cannot align.
- **Ordering.** OpenAI-compatible responses may arrive out of order, so `index`
  is authoritative when present and must form exactly 0..n-1; a gap, duplicate,
  or partially-indexed batch is a protocol error. Ollama's `/api/embed` returns
  vectors in input order.
- **Finite vectors.** Response models set `allow_inf_nan=False`: Python's JSON
  parser accepts `NaN`/`Infinity`, and such a value must never reach a client
  as a "vector".
- **Usage honesty.** Reported only when the upstream supplies counts.
  Embeddings have no completion tokens, so `total_tokens` mirrors
  `prompt_tokens` unless the upstream reports its own total.
- **Anthropic.** No embeddings API exists, so a model declaring `embeddings` on
  an anthropic-typed provider is rejected at **config load**
  (`anthropic_embeddings_unsupported`) rather than at request time. The adapter
  still raises `unsupported_capability` as defence in depth.
- **Capabilities.** `/v1/capabilities` reports `callable_capabilities`
  `["chat", "embeddings"]` and an `embeddings` block carrying the input bounds.
  Startup still requires at least one chat-capable model.

## Usage counters (added after v2)

`GET /v1/usage` exposes an in-memory `UsageRecorder` held by the gateway:
counts of completed requests plus the token counts upstreams reported, keyed
by public alias and configured provider ID.

- **Recorded on success only.** Chat (buffered and streaming) and embeddings
  record after the response is fully assembled; a failure records nothing,
  because Vulcan cannot know whether a failed upstream call consumed tokens.
- **`requests_with_usage` is the honesty field.** Upstreams that omit token
  counts contribute a request and zero tokens, so token totals are only
  interpretable against that counter. Embeddings contribute prompt tokens only
  (there are no completion tokens).
- **Process-scoped, never persisted.** Counters reset on restart, carry no
  costs or currencies, and are exposed only over the loopback API. Aliases and
  provider IDs are already public metadata; no prompt, native model name, or
  credential is involved.
- **No locking needed.** Increments happen on one event loop with no await
  between read and write.

## Error normalization

All upstream failures map to stable Vulcan codes; upstream bodies are parsed
for classification but never surfaced. Additions in v2:

| Code | HTTP | Retryable | Meaning |
| --- | --- | --- | --- |
| `missing_credential` | 503 | no | The provider's `api_key_env` variable is unset/empty/unusable. |
| `provider_auth_failed` | 502 | no | Upstream 401/403: the configured credential was rejected. |
| `provider_rate_limited` | 429 | yes | Upstream 429. |
| `provider_unavailable` | 503 | yes | Connection failure, or upstream 503/529. |
| `provider_timeout` | 504 | yes | The finite configured timeout expired. |
| `model_unavailable` | 503 | no | Upstream 404 (unknown native model), or a live Ollama list proved absence. |
| `provider_protocol_error` | 502 | no | Malformed/incomplete upstream payload. |
| `provider_error` | 502 | varies | Any other upstream non-success (5xx and 408 retryable, other 4xx not). |

Provider-side errors carry `details.provider` (the safe configured ID) so a
multi-provider operator can tell which upstream failed.

## Translation decisions

**Anthropic Messages API** (text-only system/user/assistant contract):

- All `system` messages (any position) are concatenated with blank lines into
  the top-level `system` parameter, preserving order.
- Consecutive same-role user/assistant messages are merged with blank lines so
  the transmitted conversation alternates strictly.
- The first non-system message must be `user`; an assistant-led conversation
  is rejected locally (422 `unsupported_capability`) instead of guessing at
  upstream behavior.
- `max_tokens` is mandatory upstream; requests without it use the provider's
  explicit `default_max_tokens`.
- Anthropic accepts `temperature` 0–1 while the Vulcan contract allows 0–2;
  values above 1 are rejected locally (422 `unsupported_capability`) rather
  than silently clamped.
- Response: `content` must contain only `text` blocks (Vulcan requests no
  tools); any other block type is a protocol error. `end_turn`/`stop_sequence`
  → `stop`, `max_tokens` → `length`, anything else → `null`. Usage is reported
  only when both `input_tokens` and `output_tokens` are present.
- `anthropic-version: 2023-06-01` is pinned in code; it is a wire-format
  version, not a model choice.

**OpenAI-compatible** (POST `{base_url}/chat/completions`):

- Messages pass through role/content verbatim; `stream: false` is always sent.
- The first choice's `message.content` must be a string; `finish_reason`
  `stop`/`length` map through, anything else → `null`. Usage requires both
  `prompt_tokens` and `completion_tokens`, non-negative.
- Unknown response fields are ignored (vendors extend the shape freely);
  type violations on required fields are protocol errors, matching the
  strictness of the v1 Ollama adapter.

**Ollama** keeps its v1 adapter behavior byte-for-byte on the wire (chat
payload, `/api/tags` probing, tagged-name matching, 404 classification).

## Migration from v1

Breaking changes are confined to configuration and discovery metadata; the
chat request contract is unchanged.

| v1 | v2 |
| --- | --- |
| `schema_version = 1` | `schema_version = 2` |
| Single `[provider]` table | Named `[providers.<id>]` tables |
| `provider.kind` | `providers.<id>.type` |
| `models[].runtime_name` | `models[].provider_model` |
| — | `models[].provider = "<id>"` (required) |
| `/healthz` `provider: {kind, availability}` | `providers: [{id, type, availability}, …]` |
| `/v1/models` `discovery: {source, live, availability}` | `discovery: {source}`; readiness is per-model + per-provider in `/healthz` |
| Response `provider: "ollama"` (type) | Response `provider: "<configured id>"` |

A v1 file fails startup with a `configuration_error` that names the migration
explicitly. The README carries the operator-facing step-by-step guide.

## Deliberately deferred

- Tools, images, agents, and any UI.
- Retries/backoff (risk of duplicate charges), circuit breakers, fallback
  chains, load balancing, and “best model” selection.
- Cost or price computation, quotas, and persisted usage history.
- Hosted-provider readiness probing (billable), model catalogue discovery,
  and credential storage of any kind.
- Per-vendor adapters or vendor-specific request extensions (e.g. DeepSeek
  `reasoning_content` passthrough, OpenAI `response_format`).
- A shared client SDK (revisit when at least two consumers exist).
