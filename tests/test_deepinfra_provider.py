"""Unit tests for the DeepInfra provider's request timeout and service tier."""

from unittest.mock import patch

import httpx
import pytest
from inspect_ai.model import GenerateConfig

from openbench.model._providers.deepinfra import (
    DEFAULT_REQUEST_TIMEOUT_SECS,
    REQUEST_TIMEOUT_ENV_VAR,
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


class TestDeepInfraRequestTimeout:
    """The per-request cap must be set explicitly.

    Left alone, the OpenAI SDK applies read=600s and nothing upstream overrides
    it: openai_compatible.py builds AsyncOpenAI without a timeout, and inspect
    spends GenerateConfig.timeout on tenacity's stop_after_delay, which bounds
    retrying rather than any single request. A generation running past ten
    minutes was therefore cut off by the client mid-request.
    """

    def test_default_is_applied_without_configuration(self, monkeypatch) -> None:
        monkeypatch.delenv(REQUEST_TIMEOUT_ENV_VAR, raising=False)
        timeout = _timeout_passed_to_client(monkeypatch)
        assert timeout.read == DEFAULT_REQUEST_TIMEOUT_SECS

    def test_env_var_overrides_the_default(self, monkeypatch) -> None:
        monkeypatch.setenv(REQUEST_TIMEOUT_ENV_VAR, "1200")
        timeout = _timeout_passed_to_client(monkeypatch)
        assert timeout.read == 1200.0

    def test_connect_stays_short(self, monkeypatch) -> None:
        """A bare float would raise every phase, so an unreachable endpoint
        would hang for the whole budget instead of failing in seconds."""
        monkeypatch.setenv(REQUEST_TIMEOUT_ENV_VAR, "3600")
        timeout = _timeout_passed_to_client(monkeypatch)
        assert timeout.connect is not None and timeout.connect <= 60.0
        assert timeout.read == 3600.0

    def test_config_timeout_is_not_reused_as_the_request_cap(
        self, monkeypatch
    ) -> None:
        """inspect already spends config.timeout as the retry budget. Reusing it
        here would let one attempt consume the budget, leaving no retry -- the
        original bug, where a 600s cap and a 600s budget coincided."""
        monkeypatch.delenv(REQUEST_TIMEOUT_ENV_VAR, raising=False)
        timeout = _timeout_passed_to_client(
            monkeypatch, config=GenerateConfig(timeout=600)
        )
        assert timeout.read == DEFAULT_REQUEST_TIMEOUT_SECS

    def test_explicit_model_arg_wins(self, monkeypatch) -> None:
        monkeypatch.setenv(REQUEST_TIMEOUT_ENV_VAR, "1200")
        timeout = _timeout_passed_to_client(
            monkeypatch, timeout=httpx.Timeout(99.0, connect=5.0)
        )
        assert timeout.read == 99.0

    def test_missing_api_key_is_refused(self, monkeypatch) -> None:
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key not found"):
            DeepInfraAPI(model_name="deepinfra/org/model")
