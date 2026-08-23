# Changelog

Releases are tagged on `main`. The phase-by-phase engineering history —
including every contract change and its rationale — lives in
`docs/ROADMAP.md`; this file records what each tagged release contains.

## 1.0.0 — 2026-08-23

First tagged release. Everything below is deployed and live-verified on the
reference deployment (see `deploy/README.md`).

The gateway: a loopback-only, single-operator inference gateway with
OpenAI-compatible `/v1/chat/completions` (buffered + SSE) and
`/v1/embeddings`, routing configured public aliases each to exactly one
provider (Ollama local, Anthropic, or any OpenAI-compatible vendor via BYOK).
No fallback, no retries, no auto-routing, ever — refusals are typed, loud,
and content-safe.

- **Discovery that never lies**: `/v1/models` (+ optional per-alias `class`
  label), `/v1/capabilities` derived from configured reality, `/healthz`
  with honest per-provider availability.
- **Usage accounting**: process-lifetime counters and an optional durable
  append-only ledger (flock-enforced single writer, replay on restart);
  per-seat attribution; fail-closed per-seat UTC-day budgets for hosted
  providers (429/400/403, never reroute).
- **Operations**: reference systemd unit (`Type=notify`, watchdog heartbeats
  via built-in raw-socket sd_notify), an in-flight concurrency bound
  (immediate typed 503, never a queue), per-alias Ollama `keep_alive`
  passthrough, and the operator memory-lifecycle CLI (`ps`/`unload`/
  `warmup`) with refusal-before-network guarantees.
- **Reporting**: a hardened oneshot usage reporter posting a daily
  content-safe digest into Athena's signed forge ingest.
- **Verification**: a five-command quality gate — format, lint, the pytest
  suite (mock-transport only, leak sentinels throughout), a real-process
  smoke test, and an operator-run live contract check against real Ollama
  (`scripts/live_check.py`); `deploy/update.sh` makes the deploy flow
  mechanical with config validation before every restart.
