# Vulcan

Vulcan is a small, local-first AI inference gateway. It gives same-machine clients a
stable way to discover configured models, inspect declared capabilities, and submit one
guarded, non-streaming chat request without coupling those clients to a local runtime.

Vulcan is infrastructure. It is not a chat UI, agent framework, model router, downloader,
training system, cloud credential manager, or multi-user service. The first release exposes
one explicitly selected provider and never falls back to another provider or a fake response.

## What the first slice includes

- A typed FastAPI service with a versioned `/v1` contract.
- An immutable, configuration-driven model registry.
- A native Ollama `/api/chat` adapter for an explicitly configured local runtime.
- An explicit deterministic no-I/O adapter for tests and contract smoke checks.
- Stable structured errors and content-safe operational JSON logs.
- Loopback-only server, provider URL, and HTTP `Host` validation.

Model discovery remains configuration-owned: Vulcan never invents public IDs from a
runtime inventory. When the selected provider is Ollama, `/healthz` and `/v1/models`
probe `/api/tags` with the configured finite timeout and annotate each *configured*
public ID as `available`, `unavailable`, or `unchecked`. Probe failures never claim
loaded state. The deterministic provider is non-network and reports known-ready
without a live inventory (`live: false`).

## Local setup

Vulcan requires Python 3.12 and [uv](https://docs.astral.sh/uv/). No local model runtime is
installed or downloaded by these commands.

```bash
uv sync --all-groups --locked
cp config/vulcan.example.toml vulcan.toml
# Edit runtime_name to an Ollama model already installed on this machine.
uv run vulcan serve --config vulcan.toml
```

The example binds `127.0.0.1:8140` and targets `http://127.0.0.1:11434`. Vulcan rejects
non-loopback server hosts, non-loopback provider URLs, provider URL credentials, redirects,
and environment proxy inheritance. `localhost`, IPv4 loopback, and IPv6 loopback are allowed.

Configuration is strict TOML; unknown fields fail startup. `schema_version` and the Ollama
timeout are required explicitly. There are no environment-variable overrides or hidden defaults
beyond the documented `server` field defaults below.

```toml
schema_version = 1

[server]
host = "127.0.0.1" # default; loopback only
port = 8140         # default; 1..65535
log_level = "INFO" # DEBUG, INFO, WARNING, ERROR, or CRITICAL

[provider]
kind = "ollama"
base_url = "http://127.0.0.1:11434"
timeout_seconds = 60.0

[[models]]
id = "local-chat"                  # public client-facing ID
runtime_name = "an-installed-model" # provider-specific name, never returned by the API
capabilities = ["chat"]
description = "Optional description"
```

`kind = "deterministic"` must be selected explicitly and requires `response_text`. It performs
no network I/O and is visibly reported as provider `deterministic`; Vulcan never selects it as
a fallback. `config/vulcan.deterministic.toml` is test-only.

Startup requires at least one configured model that declares the callable `chat` capability.

## API contract

This is a documented subset of the OpenAI chat-completions shape, not full OpenAI
compatibility. Tools, images, streaming, logprobs, response formats, and other fields are
rejected or unsupported.

The machine-readable schema is available at `/openapi.json`; CDN-backed docs UIs are disabled.

| Method | Path | Contract |
| --- | --- | --- |
| `GET` | `/healthz` | Gateway liveness (`status: ok`), API version, configured provider kind, model count, and honest provider readiness (`available` / `unavailable` / `unchecked`). |
| `GET` | `/v1/models` | Configured public IDs only, descriptions, declared capabilities, provider kind, and discovery metadata from a live probe when possible. |
| `GET` | `/v1/capabilities` | Callable v1 gateway features: non-streaming chat, supported roles, and configuration-driven discovery. |
| `POST` | `/v1/chat/completions` | One selected-model, non-streaming chat request. |

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

`messages` must contain 1–64 entries and at least one `user` message. Roles are `system`,
`user`, or `assistant`; each content field must be nonblank. `temperature` is 0–2,
`max_tokens` is 1–32768, and combined message content is capped at 65536 characters.
Unknown fields are rejected. `stream: true` returns `unsupported_capability`.

A successful response contains one assistant choice and optional token usage. Usage is omitted
unless the runtime supplies both prompt and completion counts; Vulcan does not invent counts.

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1784550000,
  "model": "local-chat",
  "provider": "ollama",
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

All request failures use the same envelope and include a generated `X-Request-ID` response
header. Validation details contain only field paths and reason codes; request values and
provider bodies are never echoed.

```json
{
  "error": {
    "code": "provider_unavailable",
    "message": "The configured local provider is unavailable.",
    "retryable": true,
    "details": null,
    "validation": null
  },
  "request_id": "..."
}
```

| HTTP | Code | Meaning |
| --- | --- | --- |
| 400 | `invalid_host` | The HTTP `Host` header is missing, ambiguous, or not loopback. |
| 404 | `model_not_found` | The public model ID is not configured. |
| 422 | `invalid_request` | JSON or fields do not match the v1 contract. |
| 422 | `unsupported_capability` | Streaming was requested or the model lacks `chat`. |
| 503 | `provider_unavailable` | The local runtime cannot be reached. |
| 503 | `model_unavailable` | The model is configured but absent in the runtime. |
| 504 | `provider_timeout` | The finite provider timeout expired. |
| 502 | `provider_error` | The provider returned another non-success status. |
| 502 | `provider_protocol_error` | The provider returned malformed or incomplete data. |
| 500 | `configuration_error` / `internal_error` | A safe gateway-side failure. Invalid startup configuration exits with a JSON `configuration_error` on stderr. |

## Security and privacy defaults

`provider_error.retryable` is false for deterministic redirects and client rejections, except
timeout/rate-limit statuses 408 and 429; server-side provider failures are marked retryable.

Vulcan accepts no credentials and supports no cloud provider. Both listening and runtime
connections are restricted to loopback. Incoming `Host` headers must name `localhost` or a
literal loopback address, preventing a browser from reaching Vulcan through a rebinding domain.
Ollama calls use finite timeouts, do not follow redirects, do not inherit proxy settings, never
retry automatically, never pull a model, and never fall back.

The JSON formatter emits only fixed application event names; unknown log messages and their
arguments are not rendered. Keys containing prompt, message, content, response, body,
authorization, cookie, credential, token, password, API-key, or secret markers are redacted
recursively. Application events contain safe operational metadata such as request ID, method,
declared route, status, latency, public model ID, provider kind, turn/character counts, and error
code. Chat events share the HTTP request ID for correlation. Uvicorn access logging is disabled.

## Quality commands

```bash
uv sync --all-groups --locked
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run python scripts/smoke.py
```

The smoke script launches a real Uvicorn process on an ephemeral `127.0.0.1` port, checks the
listener and all four endpoints using the deterministic provider, asserts the exact chat reply,
checks logs for prompt/response sentinels, and shuts the process down.

## Deliberate limitations

Vulcan has one provider per process, text-only messages, non-streaming chat, and configuration-
owned discovery (runtime probes only annotate configured IDs). It has no auth, multi-user state,
telemetry, billing, model management, auto-routing, fallback, model pull/download, agents, UI,
deployment, or direct Athena/Icarus integration.

Readiness probes Ollama via `/api/tags` and reuses one result for
`READINESS_PROBE_TTL_SECONDS` (5s) across health, models, and chat preflight.
Chat fails loud with `model_unavailable` when a successful live list has already
proven the configured runtime name is absent — without calling `/api/chat`.
When the probe is unchecked/unavailable, chat still reaches the provider and
fails at that boundary rather than inventing a success. A shared SDK should wait
until at least two consumers exist.

## License

Vulcan is licensed under the GNU Affero General Public License v3.0 only. See `LICENSE`.
