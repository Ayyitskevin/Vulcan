# Vulcan

Vulcan is a local-first, single-user AI gateway for explicitly configured local and
BYOK (bring-your-own-key) models. Same-machine clients get one stable loopback API to
discover configured model aliases and submit guarded chat requests (buffered or
streaming) and embedding batches; each alias routes to exactly one named provider —
a local Ollama runtime or a hosted API used with your own key.

Vulcan is infrastructure. It is not a chat UI, agent framework, autonomous router,
model downloader, training system, credential manager, billing platform, or
multi-user service. Every model alias is pinned to one provider; Vulcan never falls
back to another provider, never retries upstream calls, and never fabricates a
response.

## What Vulcan includes

- A typed FastAPI service with a versioned `/v1` contract.
- An immutable, configuration-driven model registry: public aliases map to exactly
  one provider ID and provider-native model name.
- Named providers of four types:
  - `ollama` — native adapter for a local Ollama runtime.
  - `anthropic` — native Anthropic Messages API.
  - `openai_compatible` — one adapter for OpenAI, xAI/Grok, Moonshot/Kimi,
    Z.AI/GLM, DeepSeek, and any future OpenAI-compatible endpoint (new vendors
    are configuration, not code).
  - `deterministic` — explicit no-I/O adapter for tests and smoke checks.
- Secure BYOK: hosted providers reference credentials by environment-variable name
  (`api_key_env`); values are read at request time and never stored, logged, or
  echoed.
- Stable structured errors, content-safe operational JSON logs, and a
  `vulcan check` command that reports credential availability without revealing
  values.
- Loopback-only server binding, HTTP `Host` validation, and hardened upstream HTTP
  clients (finite timeouts, no redirects, no proxy inheritance).

Model discovery remains configuration-owned: Vulcan never invents public IDs from a
runtime inventory. Ollama providers are probed via `/api/tags` and each configured
alias is annotated `available` / `unavailable` / `unchecked`. Hosted providers are
deliberately **not** probed — that would call authenticated (often billable)
endpoints just to render metadata — so their models are reported as configured but
`unchecked` until a real request uses them.

The design record for the multi-provider architecture is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the phased continuation plan
(operator tooling and maintenance) is in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Local setup

Vulcan requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). No model
runtime is installed or downloaded by these commands.

```bash
uv sync --all-groups --locked
cp config/vulcan.example.toml vulcan.toml
# Edit vulcan.toml: keep only the providers you use and point each model alias
# at an installed Ollama model or a hosted provider-native model name.
export VULCAN_OPENAI_API_KEY="..."   # only for the hosted providers you kept
uv run vulcan check --config vulcan.toml
uv run vulcan serve --config vulcan.toml
```

The example binds `127.0.0.1:8140`. Vulcan rejects non-loopback server hosts,
URL credentials, redirects, and environment proxy inheritance. `localhost`, IPv4
loopback, and IPv6 loopback are allowed.

## Running as a service

For a supervised, boot-persistent install (recommended once Vulcan is part of
your daily tooling), `deploy/` ships a reference systemd unit and the deploy
convention it assumes — code checkout and state directory kept separate,
restart-on-failure, structured logs to the journal. See
[`deploy/README.md`](deploy/README.md).

## Configuration

Configuration is strict TOML; unknown fields fail startup. `schema_version = 2` is
required. See [`config/vulcan.example.toml`](config/vulcan.example.toml) for a
complete redacted multi-provider example.

```toml
schema_version = 2

[server]
host = "127.0.0.1" # default; loopback only
port = 8140         # default; 1..65535
log_level = "INFO" # DEBUG, INFO, WARNING, ERROR, or CRITICAL

[readiness]
probe_ttl_seconds = 5.0  # optional; 0..60, default 5; 0 = never reuse

[providers.local-ollama]
type = "ollama"
base_url = "http://127.0.0.1:11434" # loopback only, no path
timeout_seconds = 60.0

[providers.openai]
type = "openai_compatible"
base_url = "https://api.openai.com/v1"     # HTTPS required off-loopback
api_key_env = "VULCAN_OPENAI_API_KEY"      # variable NAME only, never a value
timeout_seconds = 60.0
# max_tokens_field = "max_completion_tokens"  # for newer OpenAI models

[providers.anthropic]
type = "anthropic"
api_key_env = "VULCAN_ANTHROPIC_API_KEY"
timeout_seconds = 60.0
default_max_tokens = 4096  # Anthropic requires max_tokens; used when unset

[[models]]
id = "local-chat"              # public alias, the only ID clients see
provider = "local-ollama"      # exact provider ID; no fallback
provider_model = "an-installed-model"  # provider-native name, never exposed
capabilities = ["chat"]
description = "Optional description"
```

Provider IDs are operator-chosen, match `[a-z0-9][a-z0-9_-]*`, and appear in API
metadata and logs (they are safe, non-secret values). Startup requires at least one
provider, at least one model with the callable `chat` capability, and every
`models[].provider` to name a configured provider. Declare
`capabilities = ["chat", "embeddings"]` on an alias that should serve both.

### Adding a provider and model alias

1. Add a `[providers.<id>]` table: `type`, `base_url` (explicit HTTPS endpoint for
   hosted providers), `api_key_env`, and `timeout_seconds`.
2. Export the referenced environment variable in the shell that runs Vulcan.
3. Add a `[[models]]` entry pointing a new public alias at the provider ID and the
   provider-native model name.
4. Run `uv run vulcan check --config vulcan.toml`, then restart `vulcan serve`.

A new OpenAI-compatible vendor needs no application code: add another
`openai_compatible` provider table with its base URL and key variable.

### Credential handling

- Raw credentials never appear in TOML, API responses, errors, or logs. Hosted
  provider tables reference an environment variable by name
  (`api_key_env = "VULCAN_OPENAI_API_KEY"`); names must match `[A-Z][A-Z0-9_]*`.
- Values are read from the process environment per request, used for exactly one
  upstream `Authorization: Bearer` (OpenAI-compatible) or `x-api-key` (Anthropic)
  header, and never persisted.
- A missing/empty/unusable variable fails that request with `missing_credential`
  (503); other aliases and providers keep working.
- `uv run vulcan check --config vulcan.toml` validates the file and prints, per
  provider, whether the referenced variable is set (`present`/`missing`) without
  revealing values. Exit codes: 0 = valid + all credentials present, 1 = valid but
  some missing, 2 = invalid configuration. This makes no network calls.
- `uv run vulcan check --config vulcan.toml --verify-credentials` additionally
  makes **one** metadata call per hosted provider to confirm the credential is
  actually accepted, reporting `verified` / `auth_failed` / `unreachable` /
  `error` / `missing` per provider. Values and upstream bodies are never printed;
  verdicts come from status codes alone. Any non-`verified` hosted provider exits
  1. This is the only place Vulcan calls an authenticated endpoint outside serving
  a client request, and it happens solely because an operator asked for it —
  `/healthz`, `/v1/models`, and readiness never do.
- `uv run vulcan usage --config vulcan.toml` and
  `uv run vulcan models --config vulcan.toml` read `/v1/usage` and `/v1/models`
  from the **running** gateway named by the config and print its JSON verbatim
  (already content-safe). Exit codes: 0 = success, 1 = gateway unreachable or a
  non-success response (the sanitized error names the fix), 2 = invalid
  configuration. Loopback only, like everything else.

### Migrating from schema v1

Vulcan rejects `schema_version = 1` files at startup with a `configuration_error`
naming this migration. The chat request contract is unchanged; configuration and
discovery metadata changed:

1. Set `schema_version = 2`.
2. Replace the single `[provider]` table with a named `[providers.<id>]` table and
   rename its `kind` key to `type` (e.g. `[provider] kind = "ollama"` becomes
   `[providers.local-ollama] type = "ollama"`).
3. In every `[[models]]` entry, add `provider = "<id>"` and rename `runtime_name`
   to `provider_model`.
4. Response/discovery metadata now reports the configured provider ID: `/healthz`
   returns a `providers` array (`{id, type, availability}`), `/v1/models` discovery
   metadata is `{"source": "configuration"}` with per-model `provider`,
   `provider_type`, and `availability`, and chat responses' `provider` field is the
   configured provider ID rather than the adapter type.

## API contract

This is a documented subset of the OpenAI chat-completions shape, not full OpenAI
compatibility. Tools, images, logprobs, response formats, and other fields are
rejected or unsupported.

The machine-readable schema is available at `/openapi.json`; CDN-backed docs UIs are
disabled.

| Method | Path | Contract |
| --- | --- | --- |
| `GET` | `/healthz` | Gateway liveness (`status: ok`), API version, one entry per configured provider (`id`, `type`, honest `availability`), and model count. Optional `?refresh=true` forces a new probe. |
| `GET` | `/v1/models` | Configured public aliases only: description, declared capabilities, selected provider ID/type, and readiness annotation. Optional `?refresh=true`. |
| `GET` | `/v1/models/{id}` | One configured public model with the same annotation; `model_not_found` if the alias is not configured. Optional `?refresh=true`. |
| `GET` | `/v1/capabilities` | Callable v1 gateway features: chat (buffered and streaming), embeddings and their bounds, supported roles, and configuration-driven discovery. |
| `POST` | `/v1/chat/completions` | One selected-alias chat request routed to exactly one provider; buffered JSON by default, Server-Sent Events when `stream: true`. |
| `POST` | `/v1/embeddings` | One selected-alias embedding batch routed to exactly one provider. |
| `GET` | `/v1/usage` | Completed-request counters per alias, per provider, and per seat label; in-memory by default, durable across restarts with `[usage] ledger_path`; includes per-seat budget state when `[budgets]` is configured. |

Chat request fields:

```json
{
  "model": "local-chat",
  "messages": [{"role": "user", "content": "Reply with one word."}],
  "temperature": 0.2,
  "max_tokens": 16,
  "stream": false
}
```

`messages` must contain 1–64 entries and at least one `user` message. Roles are
`system`, `user`, or `assistant`; each content field must be nonblank. `temperature`
is 0–2, `max_tokens` is 1–32768, and combined message content is capped at 65536
characters. Unknown fields are rejected.

Anthropic-specific constraints (rejected locally with `unsupported_capability`
instead of guessing at upstream behavior): `temperature` above 1, and conversations
whose first non-system message is not from the user. Requests without `max_tokens`
use the provider's configured `default_max_tokens`.

A successful response contains one assistant choice and optional token usage. Usage
is omitted unless the upstream supplies both prompt and completion counts; Vulcan
does not invent counts. `provider` is the configured provider ID that served the
request.

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1784550000,
  "model": "local-chat",
  "provider": "local-ollama",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Ready."},
      "finish_reason": "stop"
    }
  ],
  "usage": null
}
```

### Streaming

`stream: true` returns `Content-Type: text/event-stream` with OpenAI-style chunks:
one `data:` frame per event, terminated by `data: [DONE]`. The first chunk carries
`delta.role`, later chunks carry `delta.content`, and the final chunk carries
`finish_reason` (plus `usage` only when the upstream reported both counts). Absent
delta fields are omitted; `finish_reason` is always present (null until the end).

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":1784550000,"model":"local-chat","provider":"local-ollama","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":1784550000,"model":"local-chat","provider":"local-ollama","choices":[{"index":0,"delta":{"content":"Ready."},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":1784550000,"model":"local-chat","provider":"local-ollama","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Failures **before** the first byte (unknown alias, missing credential, upstream
auth failure, …) return the ordinary JSON error envelope with its normal status
code — a streaming request that fails early never becomes a 200. Once headers are
committed, a failure can only be reported inside the stream: Vulcan emits one
terminal error frame carrying the same normalized error body and then closes
**without** `[DONE]`. Upstream bodies are never forwarded.

```
data: {"error":{"code":"provider_protocol_error","message":"The selected provider returned an invalid response.","retryable":false,"details":{"provider":"local-ollama"},"validation":null},"request_id":"…"}
```

Clients that disconnect mid-stream cancel the upstream request; Vulcan closes the
provider response rather than draining it.

### Vendor extension fields (including reasoning content)

OpenAI-compatible vendors extend the response shape freely, so Vulcan ignores
fields outside the documented contract instead of failing on them. The fields it
does rely on are still parsed strictly and never coerced.

The notable case is **DeepSeek's `reasoning_content`**, which carries a reasoner
model's chain of thought alongside the answer. Vulcan drops it:

- It is never forwarded — neither in `choices[].message` nor in a stream
  `delta`. During the reasoning phase DeepSeek sends chunks whose `content` is
  null; those produce no `delta.content` frames, so a stream may legitimately
  contain no text before `finish_reason`.
- It is never substituted for the answer. A response whose `content` is missing
  or not a string is a `provider_protocol_error`, even when reasoning is present.
- Its tokens are not added to usage. `usage.completion_tokens` is authoritative;
  the `completion_tokens_details.reasoning_tokens` breakdown is ignored, so
  reasoning tokens are counted once, not twice.

Exposing reasoning content would need a deliberate contract change (a new
response field and its own privacy review), not a parser tweak. `tests/test_reasoning_content.py`
pins the current behavior.

### Embeddings

`POST /v1/embeddings` embeds one batch through the alias's configured provider.
The alias must declare the `embeddings` capability, otherwise the request fails
with `unsupported_capability`.

```json
{
  "model": "local-embed",
  "input": ["first document", "second document"]
}
```

`input` is one string or 1–64 strings; each is nonblank and at most 8192
characters, and the combined length is capped at 65536. Unknown fields are
rejected.

The response returns one record per input, in input order. `usage` appears only
when the upstream reported token counts; embeddings have no completion tokens, so
`total_tokens` equals `prompt_tokens` unless the upstream says otherwise.

```json
{
  "object": "list",
  "model": "local-embed",
  "provider": "local-ollama",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.01, -0.02]},
    {"object": "embedding", "index": 1, "embedding": [0.03, -0.04]}
  ],
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

Embeddings are supported on `ollama`, `openai_compatible`, and `deterministic`
providers. **Anthropic publishes no embeddings API**, so declaring the
`embeddings` capability on an `anthropic` provider fails at startup rather than
leaving an alias that could only fail at request time. Vectors must be finite
numbers: a non-finite value from an upstream is a `provider_protocol_error`, as
is a response whose vector count does not match the input count.

### Usage counters

`GET /v1/usage` reports completed request counts and the token counts
upstreams actually reported, broken down by public alias, by provider, and —
for requests that carry the optional `seat` label — by caller. Scope is the
current process by default, or the durable ledger when enabled (below).

```json
{
  "object": "usage",
  "scope": "process",
  "started_at": 1784550000,
  "totals": {
    "requests": 12,
    "requests_with_usage": 9,
    "prompt_tokens": 4210,
    "completion_tokens": 880,
    "total_tokens": 5090
  },
  "by_model": [
    {"model": "local-chat", "provider": "local-ollama", "totals": {"...": 0}}
  ],
  "by_provider": [
    {"provider": "local-ollama", "totals": {"...": 0}}
  ],
  "by_seat": [
    {"seat": "claude", "totals": {"...": 0}}
  ]
}
```

Chat and embedding requests accept an optional `seat` field — an
operator-chosen caller label (`^[a-z0-9][a-z0-9_-]{0,63}$`, same shape as a
provider ID) so several local tools sharing one gateway can see who spent
what. Attribution only: unlabeled requests still count in every other view,
an unknown label is never rejected against a roster, and the label is never
forwarded upstream or logged with content. Attribution enforces nothing by
itself; the opt-in per-seat budgets documented below are a separate, explicit
gate the operator turns on in configuration.

Deliberate limits, so this stays infrastructure rather than a billing system:

- **In-memory and process-scoped by default.** Counters reset on restart and
  nothing is persisted — unless the operator opts into the durable ledger:

  ```toml
  [usage]
  ledger_path = "/absolute/path/usage-ledger.jsonl"
  ```

  With the section present, every completed request appends one JSON line
  (timestamp, alias, provider, optional seat, reported token counts — never
  message content, never provider-native model names) and the counters replay
  the file at startup, so `/v1/usage` reports `"scope": "ledger"` and survives
  restarts. A `ledger` object carries honesty counters (`replayed_requests`,
  `skipped_lines`, `write_failures`); torn lines are skipped and counted, an
  unopenable ledger fails startup loudly (`ledger_error`, exit 2) rather than
  silently falling back to memory, and a failed append is counted and logged
  but never fails the already-completed request. Writes are flushed per line,
  not fsynced: a hard power cut may lose the tail. One gateway per ledger
  file.
- **Completed requests only.** A failed request is never counted as usage —
  Vulcan cannot know whether a failed upstream call consumed tokens.
- **No invented tokens.** Providers that omit token counts contribute a request
  but no tokens, so `requests_with_usage` is what makes the token totals
  interpretable. Embeddings report prompt tokens only.
- **No costs, prices, or currencies.**
- **Seats are labels, not identities.** `seat` is voluntary caller metadata
  for attribution — it is not authentication. Budgets (below) therefore guard
  against accidents, not adversaries.

### Per-seat daily budgets (hosted only)

```toml
[budgets.seats.default]
hosted_requests_per_day = 500     # REQUIRED — bounds in-flight concurrency
hosted_tokens_per_day = 100000    # optional
[budgets.seats.fable]
hosted_requests_per_day = 2000
hosted_tokens_per_day = 500000
```

When the `[budgets]` section exists, requests to **hosted** providers are
gated before the upstream call: they must carry a `seat`, the seat must
resolve to an entry (its own or `default` — fail-closed otherwise), and the
seat must have headroom in the current UTC day. Over budget → `429
budget_exhausted` (retryable) with `details.window_resets_at`; unlabeled →
`400 seat_required`; unbudgeted seat → `403 budget_unconfigured`. **Vulcan
never reroutes an over-budget request to another provider** — one alias, one
provider, always; the caller owns any fallback decision. Local providers are
never budgeted, and without the section behavior is unchanged.

Budgets **require** the durable ledger (`[usage].ledger_path` — startup
refuses the combination without it): spend replays at boot, so a restart
never resets an allowance. The request slot is reserved atomically inside the
pre-flight check, so concurrent requests can never all pass the last slot of
a request cap; a failure, cancellation, or client disconnect releases its
slot via a finally-guaranteed path (failures are never spend). The request
cap is REQUIRED on every entry — it is what bounds in-flight concurrency and
therefore token overshoot. A request straddling UTC midnight counts in its
completion day, exactly as ledger replay will later reconstruct it. Known, documented imperfections: token overshoot is bounded by the
number of concurrently in-flight requests per seat (token counts arrive after
completion; the request cap bounds the in-flight count), and providers that
omit counts under-meter — the request cap is the backstop. Distinct tracked
seats are capped at 4096 per window as a state-growth guard. `/v1/usage`
gains a `budgets` array with per-seat limits/spend/reset when enabled.

All request failures use the same envelope and include a generated `X-Request-ID`
response header. Validation details contain only field paths and reason codes;
request values and provider bodies are never echoed. Provider-side failures carry
`details.provider` (the configured ID) so you can tell which upstream failed.

```json
{
  "error": {
    "code": "provider_unavailable",
    "message": "The selected provider is unavailable.",
    "retryable": true,
    "details": {"provider": "local-ollama"},
    "validation": null
  },
  "request_id": "..."
}
```

| HTTP | Code | Meaning |
| --- | --- | --- |
| 400 | `invalid_host` | The HTTP `Host` header is missing, ambiguous, or not loopback. |
| 400 | `seat_required` | Budgets are enabled and the hosted request carried no seat label. |
| 403 | `budget_unconfigured` | The seat has no budget entry (its own or `default`) and budgets are fail-closed. |
| 404 | `model_not_found` | The public alias is not configured. |
| 422 | `invalid_request` | JSON or fields do not match the v1 contract. |
| 422 | `unsupported_capability` | Streaming, a model without `chat`, or a provider-specific constraint (see above). |
| 429 | `provider_rate_limited` | The upstream provider rate limited the request. |
| 429 | `budget_exhausted` | The seat's daily hosted budget is spent; retryable after `details.window_resets_at`. |
| 502 | `provider_auth_failed` | The upstream provider rejected the configured credential. |
| 502 | `provider_error` | The provider returned another non-success status. |
| 502 | `provider_protocol_error` | The provider returned malformed or incomplete data. |
| 503 | `missing_credential` | The provider's `api_key_env` variable is unset or unusable. |
| 503 | `provider_unavailable` | The provider cannot be reached (or upstream 503/529). |
| 503 | `model_unavailable` | The alias is configured but the native model is absent upstream. |
| 504 | `provider_timeout` | The finite provider timeout expired. |
| 500 | `configuration_error` / `internal_error` | A safe gateway-side failure. Invalid startup configuration exits with a JSON `configuration_error` on stderr. |

## Routing: strict and boring by design

A request for `model = "alias"` resolves to exactly one configured provider and
native model name. There is no fallback, auto-routing, "best model" selection,
retry, load balancing, or cross-provider traffic of any kind: if the selected
provider fails, the request fails with that provider's normalized error. Vulcan
issues at most one upstream chat call per client request, so it can never create
duplicate charges.

## Readiness semantics

Ollama providers are probed via `/api/tags` and each provider's result is
reused for `[readiness].probe_ttl_seconds` (default 5s, max 60, 0 = never
reuse) across health, models, model retrieve, and chat preflight. Chat
preflight probes only the provider the alias routes to — hosted and
deterministic aliases never wait on a local runtime probe. Operators may
force a re-probe with `?refresh=true`. Safe operational logs emit `readiness_probed` /
`readiness_reused` with counts only (no native model names). A configured model is
available when its `provider_model` matches a live name exactly, or (untagged
config only) matches exactly one `name:tag` live entry; multi-tag collisions stay
unavailable. Chat fails loud with `model_unavailable` when a successful live list
has already proven absence (no `/api/chat` call), or when the provider returns
that outcome (the readiness cache is then invalidated). When the probe is
unchecked/unavailable, chat still reaches the provider rather than inventing a
success. Deterministic providers are non-network and report known-ready without a
live inventory; hosted providers always report `unchecked` without any probe.

## Security and privacy defaults

- Both the listener and Ollama runtime connections are restricted to loopback.
  Incoming `Host` headers must name `localhost` or a literal loopback address.
- Hosted endpoints must be explicitly configured `https` URLs (cleartext `http` is
  allowed to loopback only, for local proxies and tests); URL credentials, query
  strings, and fragments are rejected. Request data can never select or construct
  an upstream URL.
- All upstream clients use finite configured timeouts, do not follow redirects, do
  not inherit proxy settings, never retry automatically, never pull a model, and
  never fall back.
- No external telemetry, analytics, credential persistence, or automatic
  provider/model discovery. Vulcan only calls configured chat endpoints (for client
  requests) and local Ollama `/api/tags` (for readiness).
- The JSON log formatter emits only fixed application event names; unknown log
  messages and their arguments are not rendered. Keys containing prompt, message,
  content, response, body, authorization, cookie, credential, token, password,
  API-key, or secret markers are redacted recursively. Application events contain
  safe operational metadata only: request ID, method, route, status, latency,
  public alias, provider ID/type, turn/character counts, and error code. Uvicorn
  access logging is disabled.
- Upstream response bodies are parsed for classification only and never surfaced
  in errors or logs; upstream failures map to the stable error codes above.

## What Vulcan does not do

No auth layer, multi-user state, telemetry, billing, cost tracking, quotas, model
management or downloads, tools, images, agents,
UI, deployment tooling, retries, fallback, or credential storage. Hosted providers
are never probed for health or model catalogues; a hosted model's availability is
learned when a request uses it. A shared SDK should wait until at least two
consumers exist.

## Quality commands

```bash
uv sync --all-groups --locked
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run python scripts/smoke.py
```

The smoke script launches a real Uvicorn process on an ephemeral `127.0.0.1` port,
checks the listener and all endpoints using the deterministic provider, asserts the
exact chat reply, checks logs for prompt/response sentinels, and shuts the process
down. Tests exercise hosted providers only through mocked transports; nothing in
the suite contacts a real API.

CI runs the first four commands on both supported interpreters (3.12 and 3.13)
from the same lock file, and does not cancel one version's job when the other
fails.

## License

Vulcan is licensed under the GNU Affero General Public License v3.0 only. See `LICENSE`.
