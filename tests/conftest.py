from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.config import GatewayConfig


@pytest.fixture
def valid_config_document() -> dict[str, Any]:
    """Return a fresh, fully explicit configuration document for each test."""

    return {
        "schema_version": 2,
        "server": {
            "host": "127.0.0.1",
            "port": 8140,
            "log_level": "INFO",
        },
        "providers": {
            "det": {
                "type": "deterministic",
                "response_text": "deterministic-test-response",
            }
        },
        "models": [
            {
                "id": "chat-model",
                "provider": "det",
                "provider_model": "runtime/chat-model",
                "capabilities": ["chat"],
                "description": "Test-only chat model",
            }
        ],
    }


@pytest.fixture
def gateway_config(valid_config_document: dict[str, Any]) -> GatewayConfig:
    return GatewayConfig.model_validate(valid_config_document)


@pytest.fixture
def client(gateway_config: GatewayConfig) -> Iterator[TestClient]:
    with TestClient(create_app(gateway_config), base_url="http://127.0.0.1") as test_client:
        yield test_client
