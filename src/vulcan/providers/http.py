"""Centralized HTTP hardening and safe upstream-failure mapping.

Every outbound client Vulcan creates goes through :func:`build_client` so the
same protections apply to all providers: finite configured timeouts, redirects
disabled, no environment proxy/CA inheritance, and fixed request headers.
Credentials are resolved per request by :func:`resolve_api_key` and are never
stored on a client, logged, or attached to an error.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx

from vulcan import __version__
from vulcan.errors import (
    MissingCredentialError,
    ModelUnavailableError,
    ProviderAuthError,
    ProviderError,
    ProviderProtocolError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)

_USER_AGENT = f"vulcan/{__version__}"

# One ceiling for every decoded upstream response, buffered or streamed. This
# is deliberately far above Vulcan's measured largest fleet response while
# making a compressed, chunked, or unterminated provider response finite.
MAX_PROVIDER_RESPONSE_BYTES = 16 * 1024 * 1024

# Wire-format version of the Anthropic Messages API, not a model choice. It
# lives here (rather than in the adapter) so credential verification can send
# it without importing the adapter, which imports this module.
ANTHROPIC_VERSION = "2023-06-01"


def build_client(*, base_url: str, timeout_seconds: float) -> httpx.AsyncClient:
    """Build a hardened AsyncClient for one explicitly configured endpoint."""

    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )


async def send_response(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json_body: Any | None = None,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """Open one upstream response without pre-buffering it.

    HTTPX's convenience request methods read the entire response before they
    return. Building a request and sending it with ``stream=True`` is the
    documented route that lets Vulcan enforce its own decoded-byte ceiling.

    Source: https://www.python-httpx.org/api/#asyncclient
    """

    request = client.build_request(method, url, json=json_body, headers=headers)
    return await client.send(request, stream=True)


@asynccontextmanager
async def open_response(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json_body: Any | None = None,
    headers: Mapping[str, str] | None = None,
) -> AsyncIterator[httpx.Response]:
    """Open and always close one bounded or status-only response."""

    response = await send_response(
        client,
        method,
        url,
        json_body=json_body,
        headers=headers,
    )
    try:
        yield response
    finally:
        await response.aclose()


async def iter_bounded_bytes(
    response: httpx.Response,
    *,
    max_bytes: int | None = None,
) -> AsyncIterator[bytes]:
    """Yield decoded response chunks up to one finite total-byte ceiling."""

    max_bytes = MAX_PROVIDER_RESPONSE_BYTES if max_bytes is None else max_bytes
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    seen = 0
    async for chunk in response.aiter_bytes():
        seen += len(chunk)
        if seen > max_bytes:
            raise ProviderProtocolError
        yield chunk


async def read_bounded_json(
    response: httpx.Response,
    *,
    max_bytes: int | None = None,
) -> Any:
    """Read one bounded upstream JSON document.

    Invalid JSON remains a ``ValueError`` for adapters to classify alongside
    their schema-validation failures; crossing the byte ceiling is already a
    stable ``ProviderProtocolError``.
    """

    chunks = [chunk async for chunk in iter_bounded_bytes(response, max_bytes=max_bytes)]
    return json.loads(b"".join(chunks))


async def iter_bounded_lines(
    response: httpx.Response,
    *,
    max_bytes: int | None = None,
) -> AsyncIterator[str]:
    """Yield UTF-8 lines while bounding the entire decoded response.

    SSE permits LF, CRLF, and bare CR separators and strips one leading UTF-8
    BOM. Ollama's NDJSON is a strict subset of the same framing. Parsing here
    avoids HTTPX's otherwise unbounded line accumulator.

    Sources:
    - https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream
    - https://www.rfc-editor.org/rfc/rfc8259.html#section-8.1
    """

    pending = bytearray()
    skip_lf = False
    first_line = True

    async for chunk in iter_bounded_bytes(response, max_bytes=max_bytes):
        cursor = 0
        while cursor < len(chunk):
            if skip_lf:
                skip_lf = False
                if chunk[cursor] == 0x0A:
                    cursor += 1
                    if cursor == len(chunk):
                        break

            lf = chunk.find(b"\n", cursor)
            cr = chunk.find(b"\r", cursor)
            separators = [index for index in (lf, cr) if index >= 0]
            if not separators:
                pending.extend(chunk[cursor:])
                break

            end = min(separators)
            pending.extend(chunk[cursor:end])
            raw = bytes(pending)
            pending.clear()
            if first_line and raw.startswith(b"\xef\xbb\xbf"):
                raw = raw[3:]
            first_line = False
            try:
                yield raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ProviderProtocolError from exc
            skip_lf = chunk[end] == 0x0D
            cursor = end + 1

    if pending:
        raw = bytes(pending)
        if first_line and raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        try:
            yield raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProviderProtocolError from exc


def _usable_credential(api_key_env: str) -> str | None:
    """The referenced variable's usable value, or None; callers must not log it."""

    value = (os.environ.get(api_key_env) or "").strip()
    if value and all(0x21 <= ord(char) <= 0x7E for char in value):
        return value
    return None


def credential_available(api_key_env: str) -> bool:
    """Whether the referenced variable holds a usable value, without exposing it."""

    return _usable_credential(api_key_env) is not None


def resolve_api_key(api_key_env: str) -> str:
    """Read a credential from the environment at request time.

    Raises :class:`MissingCredentialError` (naming only the variable) when the
    variable is unset, blank, or contains non-printable characters — the last
    check keeps a malformed value from ever reaching header encoding, where a
    library error could echo it.
    """

    value = _usable_credential(api_key_env)
    if value is None:
        raise MissingCredentialError(api_key_env)
    return value


async def iter_sse_payloads(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the payload of each ``data:`` field of a Server-Sent Events stream.

    Comments (``:`` lines), blank lines, and every other SSE field (``event:``,
    ``id:``, ``retry:``) are ignored: adapters classify events from the JSON
    payload itself, which both supported vendors provide.
    """

    async for line in iter_bounded_lines(response):
        if line.startswith("data:"):
            yield line[len("data:") :].strip()


CredentialVerdict = Literal["verified", "auth_failed", "unreachable", "error", "missing"]


async def verify_hosted_credential(
    config: object,
    *,
    client: httpx.AsyncClient | None = None,
) -> CredentialVerdict:
    """Make ONE metadata call to confirm a hosted credential is accepted.

    Operator-invoked only (``vulcan check --verify-credentials``); no automatic
    surface ever calls this. Returns a verdict derived from the status code
    alone — the response body is never read, and the credential value never
    appears in the result.
    """

    api_key_env = getattr(config, "api_key_env", None)
    base_url = getattr(config, "base_url", None)
    timeout_seconds = getattr(config, "timeout_seconds", None)
    provider_type = getattr(config, "type", None)
    if api_key_env is None or base_url is None or timeout_seconds is None:
        return "error"
    if _usable_credential(api_key_env) is None:
        return "missing"

    if provider_type == "anthropic":
        path = "/v1/models"
        headers = {
            "x-api-key": resolve_api_key(api_key_env),
            "anthropic-version": ANTHROPIC_VERSION,
        }
    else:
        path = "/models"
        headers = {"Authorization": f"Bearer {resolve_api_key(api_key_env)}"}

    verifier = client or build_client(base_url=base_url, timeout_seconds=timeout_seconds)
    try:
        async with open_response(verifier, "GET", path, headers=headers) as response:
            status_code = response.status_code
    except httpx.RequestError:
        # Timeouts are a kind of RequestError: both mean "could not confirm".
        return "unreachable"
    finally:
        await verifier.aclose()

    if 200 <= status_code < 300:
        return "verified"
    if status_code in {401, 403}:
        return "auth_failed"
    return "error"


def raise_for_hosted_status(status_code: int) -> None:
    """Map a hosted provider's non-success status to one stable Vulcan error.

    The response body is deliberately not consulted: classification uses only
    the status code, so upstream text can never leak through an error.
    """

    if status_code in {401, 403}:
        raise ProviderAuthError
    if status_code == 404:
        # The configured provider_model (or endpoint) does not exist upstream.
        raise ModelUnavailableError
    if status_code == 429:
        raise ProviderRateLimitError
    if status_code in {503, 529}:
        raise ProviderUnavailableError
    raise ProviderError(retryable=status_code >= 500 or status_code == 408)
