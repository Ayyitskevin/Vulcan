"""Tool calling: the contract, the Ollama wire, and the shape callers already had.

Vulcan brokers tool calls; it does not execute them and does not interpret the
caller's contract with its own tools. These tests pin what crosses the boundary
and, just as importantly, that a request without tools produces byte-identical
output to what callers received before tool calling existed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from vulcan.config import OllamaProviderConfig
from vulcan.providers.base import ProviderChatRequest, ProviderMessage, ProviderTool
from vulcan.providers.ollama import OllamaProvider
from vulcan.schemas import AssistantMessage, ChatCompletionRequest, ChunkDelta

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up current weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": "public-chat",
        "messages": [{"role": "user", "content": "weather in Oslo?"}],
    }
    base.update(overrides)
    return base


# --- the request contract -------------------------------------------------


def test_tools_are_accepted_and_carried() -> None:
    request = ChatCompletionRequest.model_validate(_payload(tools=[WEATHER_TOOL]))

    assert request.tools is not None
    assert request.tools[0].function.name == "get_weather"
    assert request.tools[0].function.parameters == WEATHER_TOOL["function"]["parameters"]


def test_a_tool_result_must_name_the_call_it_answers() -> None:
    """An unattributed tool result is not something a model can reconcile."""

    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            _payload(messages=[{"role": "tool", "content": "12C"}])
        )


def test_tool_call_id_is_rejected_on_a_non_tool_message() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            _payload(messages=[{"role": "user", "content": "hi", "tool_call_id": "call_0"}])
        )


def test_tool_choice_without_tools_is_refused() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_payload(tool_choice="auto"))


def test_duplicate_tool_names_are_refused() -> None:
    """Two tools with one name make the model's answer unresolvable."""

    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_payload(tools=[WEATHER_TOOL, WEATHER_TOOL]))


def test_unknown_top_level_fields_are_still_forbidden() -> None:
    """Adding tools must not have loosened the strict boundary."""

    with pytest.raises(ValidationError) as raised:
        ChatCompletionRequest.model_validate(_payload(frequency_penalty=0.5))
    assert any(item["type"] == "extra_forbidden" for item in raised.value.errors())


# --- the shape callers already had ---------------------------------------


def test_a_reply_without_tool_calls_is_byte_identical_to_before() -> None:
    """The regression that matters: existing callers must see no new field."""

    assert AssistantMessage(content="hi").model_dump(mode="json") == {
        "role": "assistant",
        "content": "hi",
    }
    assert ChunkDelta(role="assistant").model_dump(mode="json") == {
        "role": "assistant",
        "content": None,
    }


# --- the Ollama wire ------------------------------------------------------


def _ollama(handler: Any) -> OllamaProvider:
    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    config = OllamaProviderConfig(
        type="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0
    )
    return OllamaProvider("local-ollama", config, client=client)


def _run(handler: Any, request: ProviderChatRequest) -> Any:
    provider = _ollama(handler)
    try:
        return asyncio.run(provider.chat(request))
    finally:
        asyncio.run(provider.aclose())


def _request(**overrides: Any) -> ProviderChatRequest:
    fields: dict[str, Any] = {
        "provider_model": "qwen",
        "messages": (ProviderMessage(role="user", content="weather?"),),
        "temperature": None,
        "max_tokens": None,
    }
    fields.update(overrides)
    return ProviderChatRequest(**fields)


def _reply(**message: Any) -> Any:
    body = {"role": "assistant", "content": "", **message}
    return httpx.Response(200, json={"message": body, "done": True, "done_reason": "stop"})


def test_tools_reach_ollama_in_its_own_wire_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _reply(content="ok")

    _run(
        handler,
        _request(
            tools=(
                ProviderTool(
                    name="get_weather",
                    description="Look up weather",
                    parameters={"type": "object"},
                ),
            )
        ),
    )

    assert seen["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_ollama_tool_calls_come_back_normalized() -> None:
    """Ollama returns decoded arguments and no call id; both are normalized."""

    def handler(_: httpx.Request) -> httpx.Response:
        return _reply(
            tool_calls=[{"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]
        )

    result = _run(handler, _request())

    assert result.tool_calls is not None
    call = result.tool_calls[0]
    assert call.name == "get_weather"
    assert json.loads(call.arguments) == {"city": "Oslo"}
    # A requested call is not a finished answer.
    assert result.finish_reason == "tool_calls"


def test_no_tools_means_no_tools_key_on_the_wire() -> None:
    """A request without tools must reach Ollama exactly as it did before."""

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _reply(content="hi")

    result = _run(handler, _request())

    assert "tools" not in seen
    assert result.tool_calls is None
    assert result.finish_reason == "stop"
