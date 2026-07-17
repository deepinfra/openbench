"""DeepInfra AI provider implementation."""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import ChatMessage, GenerateConfig, ModelCall, ModelOutput
from inspect_ai.tool import ToolChoice, ToolInfo


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
