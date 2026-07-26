# Vulcan

Vulcan is a local-first, single-user AI gateway for explicitly configured local and
BYOK (bring-your-own-key) models. Same-machine clients get one stable loopback API to
discover configured model aliases and submit guarded, non-streaming chat requests;
each alias routes to exactly one named provider — a local Ollama runtime or a hosted
API used with your own key.

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
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Local setup

Vulcan requires Python 3.12 and [uv](https://docs.astral.sh/uv/). No model runtime is
installed or downloaded by these commands.

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
`models[].provider` to name a configured provider.

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
  some missing, 2 = invalid configuration.

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
compatibility. Tools, images, streaming, logprobs, response formats, and other
fields are rejected or unsupported.

The machine-readable schema is available at `/openapi.json`; CDN-backed docs UIs are
disabled.

| Method | Path | Contract |
| --- | --- | --- |
| `GET` | `/healthz` | Gateway liveness (`status: ok`), API version, one entry per configured provider (`id`, `type`, honest `availability`), and model count. Optional `?refresh=true` forces a new probe. |
| `GET` | `/v1/models` | Configured public aliases only: description, declared capabilities, selected provider ID/type, and readiness annotation. Optional `?refresh=true`. |
| `GET` | `/v1/models/{id}` | One configured public model with the same annotation; `model_not_found` if the alias is not configured. Optional `?refresh=true`. |
| `GET` | `/v1/capabilities` | Callable v1 gateway features: non-streaming chat, supported roles, and configuration-driven discovery. |
| `POST` | `/v1/chat/completions` | One selected-alias, non-streaming chat request routed to exactly one provider. |

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
characters. Unknown fields are rejected. `stream: true` returns
`unsupported_capability`.

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
| 404 | `model_not_found` | The public alias is not configured. |
| 422 | `invalid_request` | JSON or fields do not match the v1 contract. |
| 422 | `unsupported_capability` | Streaming, a model without `chat`, or a provider-specific constraint (see above). |
| 429 | `provider_rate_limited` | The upstream provider rate limited the request. |
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

Ollama providers are probed via `/api/tags` and one result is reused for
`[readiness].probe_ttl_seconds` (default 5s, max 60, 0 = never reuse) across
health, models, model retrieve, and chat preflight. Operators may force a
re-probe with `?refresh=true`. Safe operational logs emit `readiness_probed` /
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

No auth layer, multi-user state, telemetry, billing, quota tracking, model
management or downloads, streaming, tools, images, embeddings endpoints, agents,
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

## License

Vulcan is licensed under the GNU Affero General Public License v3.0 only. See `LICENSE`.
