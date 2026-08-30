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


# --- the OpenAI-compatible wire ------------------------------------------


def test_openai_compatible_carries_tools_and_returns_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already-OpenAI-shaped: an id and a JSON-string argument pass through."""

    from vulcan.config import OpenAICompatibleProviderConfig
    from vulcan.providers.openai_compatible import OpenAICompatibleProvider

    monkeypatch.setenv("VENDOR_KEY", "sk-test-sentinel")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Oslo"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.example-vendor.com",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    config = OpenAICompatibleProviderConfig(
        type="openai_compatible",
        base_url="https://api.example-vendor.com",
        api_key_env="VENDOR_KEY",
        timeout_seconds=1.0,
    )
    provider = OpenAICompatibleProvider("vendor", config, client=client)
    try:
        result = asyncio.run(
            provider.chat(
                _request(
                    tools=(ProviderTool(name="get_weather", description=None, parameters=None),),
                    tool_choice="auto",
                )
            )
        )
    finally:
        asyncio.run(provider.aclose())

    assert seen["tools"][0]["function"]["name"] == "get_weather"
    assert seen["tool_choice"] == "auto"
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls is not None
    assert result.tool_calls[0].id == "call_abc"
    assert json.loads(result.tool_calls[0].arguments) == {"city": "Oslo"}
    # A tool-only reply has no prose, and that is not an error.
    assert result.content == ""


# --- the Anthropic wire ---------------------------------------------------


def test_anthropic_uses_its_own_tool_vocabulary() -> None:
    """`input_schema` not `parameters`; a tool result is a user-turn block."""

    from vulcan.providers.anthropic import _translate_messages

    system, turns = _translate_messages(
        _request(
            messages=(
                ProviderMessage(role="user", content="weather?"),
                ProviderMessage(role="assistant", content="calling"),
                ProviderMessage(role="tool", content="12C", tool_call_id="toolu_1"),
            )
        )
    )

    assert system is None
    assert turns[-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "12C"}],
    }


def test_anthropic_lifts_tool_use_blocks_out_of_content() -> None:
    from vulcan.providers.anthropic import _AnthropicContentBlock, _tool_calls

    calls = _tool_calls(
        [
            _AnthropicContentBlock(type="text", text="let me check"),
            _AnthropicContentBlock(
                type="tool_use", id="toolu_1", name="get_weather", input={"city": "Oslo"}
            ),
        ]
    )

    assert calls is not None
    assert calls[0].id == "toolu_1"
    assert json.loads(calls[0].arguments) == {"city": "Oslo"}


def test_anthropic_chat_accepts_a_tool_use_block_but_not_an_unknown_one() -> None:
    """Caught live, not by a unit test: `chat()` rejected every non-text block.

    The original guard refused anything but text because Vulcan requested no
    tools. Its intent — never silently drop a block and misreport partial
    content as a complete answer — still holds for genuinely unknown types.
    """

    from vulcan.config import AnthropicProviderConfig
    from vulcan.errors import ProviderProtocolError
    from vulcan.providers.anthropic import AnthropicProvider

    def _provider(blocks: list[dict[str, Any]], monkeypatch: Any) -> Any:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"role": "assistant", "content": blocks, "stop_reason": "tool_use"}
            )

        client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            transport=httpx.MockTransport(handler),
            trust_env=False,
        )
        config = AnthropicProviderConfig(
            type="anthropic",
            base_url="https://api.anthropic.com",
            api_key_env="ANTHROPIC_KEY_TEST",
            timeout_seconds=1.0,
            default_max_tokens=64,
        )
        return AnthropicProvider("anthropic", config, client=client)

    import os

    os.environ["ANTHROPIC_KEY_TEST"] = "sk-test-sentinel"

    provider = _provider(
        [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}},
        ],
        None,
    )
    try:
        result = asyncio.run(provider.chat(_request()))
    finally:
        asyncio.run(provider.aclose())

    assert result.content == "checking"
    assert result.tool_calls is not None
    assert result.tool_calls[0].id == "toolu_1"
    assert result.finish_reason == "tool_calls"

    unknown = _provider([{"type": "a_block_type_anthropic_does_not_send"}], None)
    try:
        with pytest.raises(ProviderProtocolError):
            asyncio.run(unknown.chat(_request()))
    finally:
        asyncio.run(unknown.aclose())


def test_anthropic_thinking_block_beside_a_tool_call_is_discarded_not_joined() -> None:
    """Observed live at roughly 1 in 3: `thinking` arrives before `tool_use`.

    Rejecting it made real tool calls fail intermittently. Joining it into
    `content` would be worse — reasoning is not an answer, which is the rule
    the reasoning-content tests already pin for the other adapter.
    """

    import os

    from vulcan.config import AnthropicProviderConfig
    from vulcan.providers.anthropic import AnthropicProvider

    os.environ["ANTHROPIC_KEY_TEST"] = "sk-test-sentinel"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "role": "assistant",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "thinking", "thinking": "the user wants Oslo weather"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Oslo"},
                    },
                ],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = AnthropicProvider(
        "anthropic",
        AnthropicProviderConfig(
            type="anthropic",
            base_url="https://api.anthropic.com",
            api_key_env="ANTHROPIC_KEY_TEST",
            timeout_seconds=1.0,
            default_max_tokens=64,
        ),
        client=client,
    )
    try:
        result = asyncio.run(provider.chat(_request()))
    finally:
        asyncio.run(provider.aclose())

    assert result.tool_calls is not None
    assert result.tool_calls[0].name == "get_weather"
    # The reasoning text must not have become the answer.
    assert result.content == ""
