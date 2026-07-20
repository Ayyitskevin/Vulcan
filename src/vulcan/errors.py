"""Stable domain failures exposed by the HTTP boundary."""

from __future__ import annotations

from collections.abc import Mapping

ErrorDetails = Mapping[str, str | int | bool]


class VulcanError(Exception):
    code = "internal_error"
    status_code = 500
    retryable = False
    message = "Vulcan could not complete the request."

    def __init__(self, *, details: ErrorDetails | None = None) -> None:
        super().__init__(self.code)
        self.details = dict(details) if details is not None else None


class ModelNotFoundError(VulcanError):
    code = "model_not_found"
    status_code = 404
    message = "The selected model is not configured."

    def __init__(self, model_id: str) -> None:
        super().__init__(details={"model": model_id})


class UnsupportedCapabilityError(VulcanError):
    code = "unsupported_capability"
    status_code = 422
    message = "The selected model or endpoint does not support this capability."

    def __init__(self, capability: str, model_id: str | None = None) -> None:
        details: dict[str, str | int | bool] = {"capability": capability}
        if model_id is not None:
            details["model"] = model_id
        super().__init__(details=details)


class ProviderUnavailableError(VulcanError):
    code = "provider_unavailable"
    status_code = 503
    retryable = True
    message = "The configured local provider is unavailable."


class ModelUnavailableError(VulcanError):
    code = "model_unavailable"
    status_code = 503
    retryable = False
    message = "The configured model is unavailable in the local provider."

    def __init__(self, model_id: str | None = None) -> None:
        super().__init__(details={"model": model_id} if model_id is not None else None)


class ProviderTimeoutError(VulcanError):
    code = "provider_timeout"
    status_code = 504
    retryable = True
    message = "The configured local provider timed out."


class ProviderError(VulcanError):
    code = "provider_error"
    status_code = 502
    retryable = True
    message = "The configured local provider rejected or failed the request."

    def __init__(self, *, retryable: bool = True) -> None:
        super().__init__()
        self.retryable = retryable


class ProviderProtocolError(VulcanError):
    code = "provider_protocol_error"
    status_code = 502
    retryable = False
    message = "The configured local provider returned an invalid response."


class ConfigurationError(VulcanError):
    code = "configuration_error"
    status_code = 500
    message = "Vulcan is not configured to complete this request."
