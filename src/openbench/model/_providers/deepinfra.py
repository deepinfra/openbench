"""DeepInfra AI provider implementation."""

import os
from typing import Any

import httpx

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import ChatMessage, GenerateConfig, ModelCall, ModelOutput
from inspect_ai.tool import ToolChoice, ToolInfo

# The OpenAI SDK's read timeout defaults to 600s and nothing upstream raises
# it, so a long generation is cut off client-side mid-request. An hour is a
# backstop, not a target. Not taken from config.timeout, which inspect already
# spends as the retry budget -- the two are different quantities.
DEFAULT_REQUEST_TIMEOUT_SECS = 3600.0
# Only waiting for tokens deserves the long budget; a bare float would raise
# every phase, so an unreachable endpoint would hang for the whole hour.
CONNECT_TIMEOUT_SECS = 30.0


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

        model_args.setdefault(
            "timeout",
            httpx.Timeout(DEFAULT_REQUEST_TIMEOUT_SECS,
                          connect=CONNECT_TIMEOUT_SECS),
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
        # DeepInfra accepts an optional service_tier ("flex" rides spare
        # capacity). Set via env so a harness can control it without touching
        # eval code; an explicit extra_body setting wins.
        service_tier = os.environ.get("DEEPINFRA_SERVICE_TIER")
        if service_tier:
            config = config.model_copy()
            if config.extra_body is None:
                config.extra_body = {}
            config.extra_body.setdefault("service_tier", service_tier)
        return await super().generate(input, tools, tool_choice, config)
