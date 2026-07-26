"""Centralized HTTP hardening and safe upstream-failure mapping.

Every outbound client Vulcan creates goes through :func:`build_client` so the
same protections apply to all providers: finite configured timeouts, redirects
disabled, no environment proxy/CA inheritance, and fixed request headers.
Credentials are resolved per request by :func:`resolve_api_key` and are never
stored on a client, logged, or attached to an error.
"""

from __future__ import annotations

import os

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
