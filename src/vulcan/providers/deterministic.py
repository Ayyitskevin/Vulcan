"""Explicit no-I/O provider for tests and local contract checks."""

from typing import Literal

from vulcan.config import DeterministicProviderConfig
from vulcan.providers.base import ProviderChatRequest, ProviderChatResult


class DeterministicProvider:
    kind: Literal["deterministic"] = "deterministic"

    def __init__(self, config: DeterministicProviderConfig) -> None:
        self._response_text = config.response_text

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        del request
        return ProviderChatResult(content=self._response_text, finish_reason="stop")

    async def aclose(self) -> None:
        return None
