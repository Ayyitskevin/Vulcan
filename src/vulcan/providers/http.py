"""Centralized HTTP hardening and safe upstream-failure mapping.

Every outbound client Vulcan creates goes through :func:`build_client` so the
same protections apply to all providers: finite configured timeouts, redirects
disabled, no environment proxy/CA inheritance, and fixed request headers.
Credentials are resolved per request by :func:`resolve_api_key` and are never
stored on a client, logged, or attached to an error.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Literal

import httpx

from vulcan import __version__
from vulcan.errors import (
    MissingCredentialError,
    ModelUnavailableError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)

_USER_AGENT = f"vulcan/{__version__}"

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

    async for line in response.aiter_lines():
        field = line.rstrip("\r")
        if field.startswith("data:"):
            yield field[len("data:") :].strip()


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
        response = await verifier.get(path, headers=headers)
    except httpx.RequestError:
        # Timeouts are a kind of RequestError: both mean "could not confirm".
        return "unreachable"
    finally:
        await verifier.aclose()

    if response.is_success:
        return "verified"
    if response.status_code in {401, 403}:
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
