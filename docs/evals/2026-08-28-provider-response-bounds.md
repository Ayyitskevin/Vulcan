# Provider response-bound evaluation — 2026-08-28

## Decision under review

Cap every decoded upstream body a serving adapter reads at **16 MiB
(16,777,216 bytes)**, whether the response is buffered JSON or a streaming
SSE/NDJSON feed. Keep the public failure contract unchanged: crossing the
ceiling is a non-retryable `502 provider_protocol_error`, closes the upstream
response, and never retries or falls back.

This is a security-sensitive candidate. It requires independent review and
Kevin's exact-candidate approval before merge or deployment.

## Assumptions checked

- Vulcan's public chat contract already caps `max_tokens` at 32,768 and an
  embedding batch at 64 inputs. The ceiling therefore allows 512 decoded bytes
  per maximum requested completion token.
- The deployed fleet is the primary workload. A future provider that
  legitimately returns more than 16 MiB must be admitted by a deliberate,
  measured contract change rather than silently making the gateway unbounded.
- The cap counts bytes after HTTP content decoding, so compressed and chunked
  bodies do not bypass the cumulative limit. HTTPX may transiently allocate the
  current decoded chunk before Vulcan can reject it; the gateway does not retain
  or concatenate that chunk after refusal.
- This closes byte-growth and line-buffer growth. It does not add a whole-stream
  wall-clock deadline; the configured HTTPX read timeout remains per received
  chunk.

## Pre-change live measurement

No prompt or response content was retained. Measurements record only status,
latency, wire size, dimensions, and provider-reported token counts.

| Measurement | Result |
| --- | --- |
| Existing durable ledger | 364 records with completion counts; p50 1,200, p95/p99/max 8,192 tokens |
| 12 realistic local `code` chat cases, `max_tokens=32` | 12/12 HTTP 200; each envelope 302 bytes; latency 1,761–2,906 ms; max completion 32 tokens |
| Maximum-count local embedding batch | 64/64 vectors; 768 dimensions; 611,650-byte response; 512 prompt tokens; 810 ms |
| Safety margin over measured maximum response | 27.4× (16,777,216 / 611,650) |

The sample intentionally avoided hosted aliases: no paid request was needed to
measure the gateway's byte-shape and cleanup behavior.

## Implementation evidence

- Pinned runtime: HTTPX 0.28.1 (`uv.lock`). Its documented streaming API exposes
  `AsyncClient.send(..., stream=True)` plus `Response.aiter_bytes()`; inspection
  of the installed 0.28.1 source confirmed `stream=False` calls `aread()` before
  returning.
- HTTPX API: <https://www.python-httpx.org/api/#asyncclient>
- HTTPX streaming guidance: <https://www.python-httpx.org/quickstart/#streaming-responses>
- Network JSON is UTF-8: <https://www.rfc-editor.org/rfc/rfc8259.html#section-8.1>
- SSE is UTF-8 and permits LF, CRLF, or CR framing:
  <https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream>
- Synthetic boundary matrix: 21 cases covering exact-limit acceptance,
  first-byte-over refusal, fragmentation, UTF-8 byte accounting, LF/CRLF/CR,
  split BOM, invalid UTF-8, status-only body avoidance, all three buffered
  adapters, all three streaming adapters, and upstream cleanup. Focused result:
  `176 passed` with coverage disabled only to avoid applying the repository-wide
  floor to a partial suite.

## Ship criterion

The complete repository gate must pass after the final edit, followed by an
independent non-author review of the exact commit. No live deployment or hosted
provider probe is part of this evaluation.
