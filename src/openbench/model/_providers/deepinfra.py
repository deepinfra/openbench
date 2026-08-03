"""DeepInfra AI provider implementation."""

import os
from typing import Any

import httpx

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import ChatMessage, GenerateConfig, ModelCall, ModelOutput
from inspect_ai.tool import ToolChoice, ToolInfo

# The OpenAI SDK defaults to read=600s, and nothing upstream overrides it for
# an OpenAI-compatible provider: `openai_compatible.py` builds its AsyncOpenAI
# without a `timeout=`, and inspect spends GenerateConfig.timeout solely on
# tenacity's stop_after_delay, which bounds *retrying* and never reaches the
# socket. So a generation that legitimately runs past ten minutes is cut off
# mid-request by the client, not by the server.
#
# That is not hypothetical: benchmarking Qwen3.5-9B on mbpp, which leaves
# max_tokens unset, 19-24% of requests died at exactly 599-602s while the API
# had admitted every one of them in about a second and refused none. Successful
# requests in the same window tailed out to 479s, so the 600s cut lands inside
# the live part of the latency distribution -- and it removes the *slowest*
# samples, which biases the score rather than merely shrinking it.
#
# An hour is far above any single completion we have measured, and it is a
# backstop rather than a target: a run that needs it has a problem worth seeing.
#
# Deliberately NOT read from config.timeout, which is what the Groq provider in
# this repo does. inspect already spends config.timeout as the retry budget
# (stop_after_delay), so using the same number for a single request lets one
# attempt consume the entire budget and leaves nothing for a retry -- which is
# the failure this exists to fix: at --timeout 600 the per-request cap and the
# retry budget coincided, so the first timeout killed the sample outright. The
# two are different quantities and a caller needs to be able to set the
# per-request cap *below* the retry budget.
DEFAULT_REQUEST_TIMEOUT_SECS = 3600.0
CONNECT_TIMEOUT_SECS = 30.0
REQUEST_TIMEOUT_ENV_VAR = "DEEPINFRA_REQUEST_TIMEOUT"


class DeepInfraAPI(OpenAICompatibleAPI):
    """DeepInfra AI provider - scalable inference infrastructure.

    Uses OpenAI-compatible API with DeepInfra-specific optimizations.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # Extract model name without service prefix
        model_name_clean = model_name.replace("deepinfra/", "", 1)

        # Set defaults for DeepInfra
        base_url = base_url or os.environ.get(
            "DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"
        )
        api_key = api_key or os.environ.get("DEEPINFRA_API_KEY")

        if not api_key:
            raise ValueError(
                "DeepInfra API key not found. Set DEEPINFRA_API_KEY environment variable."
            )

        # Read from the environment for the same reason the service tier is: a
        # harness can set it without editing eval code. An explicit `timeout` in
        # model_args still wins, and it reaches AsyncOpenAI through the base's
        # **model_args, where it takes precedence over the http client's own
        # default.
        timeout_env = os.environ.get(REQUEST_TIMEOUT_ENV_VAR)
        budget = float(timeout_env) if timeout_env else DEFAULT_REQUEST_TIMEOUT_SECS
        # An httpx.Timeout rather than a bare float: a float would raise every
        # phase to the budget, including connect, so an unreachable endpoint
        # would hang for the full hour instead of failing in seconds. Only
        # waiting for generated tokens deserves the long budget.
        model_args.setdefault(
            "timeout",
            httpx.Timeout(budget, connect=CONNECT_TIMEOUT_SECS),
        )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="deepinfra",
            service_base_url="https://api.deepinfra.com/v1/openai",
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        # Optional service tier (e.g. "flex" rides DeepInfra's spare-capacity
        # tier), set via env so benchmark harnesses can control it without
        # touching eval code. Explicit extra_body settings win.
        service_tier = os.environ.get("DEEPINFRA_SERVICE_TIER")
        if service_tier:
            config = config.model_copy()
            if config.extra_body is None:
                config.extra_body = {}
            if "service_tier" not in config.extra_body:
                config.extra_body["service_tier"] = service_tier
        return await super().generate(input, tools, tool_choice, config)
