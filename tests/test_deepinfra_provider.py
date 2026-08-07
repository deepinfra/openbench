"""Unit tests for the DeepInfra provider's request timeout and service tier."""

from unittest.mock import patch

import httpx
import pytest
from inspect_ai.model import GenerateConfig

from openbench.model._providers.deepinfra import (
    DEFAULT_REQUEST_TIMEOUT_SECS,
    DeepInfraAPI,
)


def _timeout_passed_to_client(monkeypatch, **kwargs) -> httpx.Timeout:
    """Build the provider and return the timeout it handed to AsyncOpenAI."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    with patch(
        "inspect_ai.model._providers.openai_compatible.AsyncOpenAI"
    ) as mock_client:
        DeepInfraAPI(model_name="deepinfra/org/model", **kwargs)
    assert mock_client.call_count == 1
    return mock_client.call_args[1]["timeout"]


class TestRequestTimeout:
    def test_default_is_applied(self, monkeypatch) -> None:
        timeout = _timeout_passed_to_client(monkeypatch)
        assert timeout.read == DEFAULT_REQUEST_TIMEOUT_SECS

    def test_connect_stays_short(self, monkeypatch) -> None:
        """A bare float would raise every phase, so an unreachable endpoint
        would hang for the whole budget instead of failing in seconds."""
        timeout = _timeout_passed_to_client(monkeypatch)
        assert timeout.connect is not None and timeout.connect <= 60.0

    def test_config_timeout_is_not_the_request_cap(self, monkeypatch) -> None:
        """inspect spends config.timeout as the retry budget. Reusing it here
        would let one attempt consume the budget, leaving nothing to retry."""
        timeout = _timeout_passed_to_client(
            monkeypatch, config=GenerateConfig(timeout=600)
        )
        assert timeout.read == DEFAULT_REQUEST_TIMEOUT_SECS

    def test_explicit_model_arg_wins(self, monkeypatch) -> None:
        timeout = _timeout_passed_to_client(
            monkeypatch, timeout=httpx.Timeout(99.0, connect=5.0)
        )
        assert timeout.read == 99.0


class TestServiceTier:
    def _config_after_generate(self, monkeypatch, config: GenerateConfig):
        monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
        with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
            api = DeepInfraAPI(model_name="deepinfra/org/model")
        seen = {}

        async def fake_generate(input, tools, tool_choice, config):
            seen["config"] = config
            return ModelOutputStub()

        class ModelOutputStub:
            pass

        with patch.object(
            type(api).__mro__[1], "generate", side_effect=fake_generate
        ):
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                api.generate([], [], "auto", config)
            )
        return seen["config"]

    def test_env_var_sets_the_tier(self, monkeypatch) -> None:
        monkeypatch.setenv("DEEPINFRA_SERVICE_TIER", "flex")
        config = self._config_after_generate(monkeypatch, GenerateConfig())
        assert config.extra_body["service_tier"] == "flex"

    def test_absent_env_var_leaves_config_alone(self, monkeypatch) -> None:
        monkeypatch.delenv("DEEPINFRA_SERVICE_TIER", raising=False)
        config = self._config_after_generate(monkeypatch, GenerateConfig())
        assert config.extra_body is None

    def test_explicit_extra_body_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("DEEPINFRA_SERVICE_TIER", "flex")
        config = self._config_after_generate(
            monkeypatch, GenerateConfig(extra_body={"service_tier": "default"})
        )
        assert config.extra_body["service_tier"] == "default"


def test_missing_api_key_is_refused(monkeypatch) -> None:
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key not found"):
        DeepInfraAPI(model_name="deepinfra/org/model")
