from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vulcan.schemas import ChatCompletionRequest, ChatMessage, MessageRole


def _chat_payload(**overrides: object) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "chat-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.5,
        "max_tokens": 64,
        "stream": False,
    }
    payload.update(overrides)
    return payload


def test_chat_request_accepts_documented_boundaries_and_preserves_content() -> None:
    prompt = "  preserve intentional surrounding whitespace  "
    request = ChatCompletionRequest.model_validate(
        _chat_payload(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1,
        )
    )

    assert request.model == "chat-model"
    assert isinstance(request.messages, tuple)
    assert request.messages == (ChatMessage(role=MessageRole.USER, content=prompt),)
    assert request.temperature == 0.0
    assert request.max_tokens == 1
    assert request.stream is False


@pytest.mark.parametrize(
    "model",
    [
        "",
        " contains-space",
        "contains space",
        "!invalid-prefix",
        "x" * 129,
    ],
)
def test_chat_request_rejects_invalid_public_model_ids(model: str) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_chat_payload(model=model))


def test_chat_request_rejects_unknown_top_level_fields() -> None:
    payload = _chat_payload()
    payload["tools"] = []

    with pytest.raises(ValidationError) as raised:
        ChatCompletionRequest.model_validate(payload)

    assert any(item["type"] == "extra_forbidden" for item in raised.value.errors())


def test_chat_message_rejects_unknown_fields() -> None:
    payload = _chat_payload(
        messages=[{"role": "user", "content": "hello", "name": "not-supported"}]
    )

    with pytest.raises(ValidationError) as raised:
        ChatCompletionRequest.model_validate(payload)

    assert any(item["type"] == "extra_forbidden" for item in raised.value.errors())


@pytest.mark.parametrize("messages", [[], None, "not-an-array"])
def test_chat_request_requires_a_nonempty_message_array(messages: object) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_chat_payload(messages=messages))


def test_chat_request_rejects_more_than_64_messages() -> None:
    messages = [{"role": "user", "content": "x"} for _ in range(65)]

    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_chat_payload(messages=messages))


def test_chat_request_accepts_exactly_64_messages() -> None:
    messages = [{"role": "user", "content": "x"} for _ in range(64)]

    request = ChatCompletionRequest.model_validate(_chat_payload(messages=messages))

    assert len(request.messages) == 64


def test_chat_request_requires_at_least_one_user_message() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "assistant"},
    ]

    with pytest.raises(ValidationError) as raised:
        ChatCompletionRequest.model_validate(_chat_payload(messages=messages))

    assert any("at least one user message" in item["msg"] for item in raised.value.errors())


@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": "hello"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "   \t\n"},
        {"role": "user", "content": 42},
    ],
)
def test_chat_request_rejects_invalid_message_role_or_content(
    message: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_chat_payload(messages=[message]))


def test_chat_request_accepts_exact_per_message_and_combined_content_limits() -> None:
    first = "a" * 32768
    second = "b" * 32768

    request = ChatCompletionRequest.model_validate(
        _chat_payload(
            messages=[
                {"role": "user", "content": first},
                {"role": "system", "content": second},
            ]
        )
    )

    assert sum(len(message.content) for message in request.messages) == 65536


def test_chat_request_rejects_content_over_per_message_limit() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            _chat_payload(messages=[{"role": "user", "content": "x" * 32769}])
        )


def test_chat_request_rejects_content_over_combined_limit() -> None:
    messages = [
        {"role": "user", "content": "a" * 32768},
        {"role": "system", "content": "b" * 32768},
        {"role": "assistant", "content": "c"},
    ]

    with pytest.raises(ValidationError) as raised:
        ChatCompletionRequest.model_validate(_chat_payload(messages=messages))

    assert any("combined message content" in item["msg"] for item in raised.value.errors())


@pytest.mark.parametrize("temperature", [-0.001, 2.001, float("nan"), float("inf")])
def test_chat_request_rejects_temperature_outside_finite_range(temperature: float) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_chat_payload(temperature=temperature))


@pytest.mark.parametrize("max_tokens", [0, -1, 32769])
def test_chat_request_rejects_max_tokens_outside_range(max_tokens: int) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_chat_payload(max_tokens=max_tokens))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", "0.5"),
        ("max_tokens", "64"),
        ("max_tokens", 64.0),
        ("stream", "false"),
        ("stream", 0),
    ],
)
def test_chat_request_rejects_json_primitive_type_coercion(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_chat_payload(**{field: value}))


def test_stream_true_is_preserved_for_explicit_gateway_capability_rejection() -> None:
    request = ChatCompletionRequest.model_validate(_chat_payload(stream=True))

    assert request.stream is True


def test_public_validation_error_never_echoes_prompt_or_invalid_input(
    client: TestClient,
) -> None:
    prompt_sentinel = "PROMPT_SENTINEL_MUST_NOT_BE_ECHOED"
    invalid_value_sentinel = "INVALID_VALUE_SENTINEL_MUST_NOT_BE_ECHOED"
    payload = _chat_payload(
        messages=[{"role": "user", "content": prompt_sentinel}],
    )
    payload["undocumented"] = invalid_value_sentinel

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422
    assert prompt_sentinel not in response.text
    assert invalid_value_sentinel not in response.text
    body = response.json()
    assert body["error"] == {
        "code": "invalid_request",
        "message": "The request does not match the Vulcan v1 contract.",
        "retryable": False,
        "details": None,
        "validation": [
            {
                "path": "body.undocumented",
                "reason": "extra_forbidden",
            }
        ],
    }
    assert body["request_id"] == response.headers["X-Request-ID"]
