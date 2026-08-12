"""DeepInfra AI provider implementation."""

import os
from typing import Any, Literal

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig
from openai._types import NOT_GIVEN
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletion


def parse_stream_arg(value: Any) -> bool | Literal["auto"]:
    """Parse the ``stream`` model arg / env var into True, False, or "auto"."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    return text in ("1", "true", "yes")


class DeepInfraAPI(OpenAICompatibleAPI):
    """DeepInfra AI provider - scalable inference infrastructure.

    Uses OpenAI-compatible API with DeepInfra-specific optimizations.

    Responses can be streamed over SSE and reassembled locally, which avoids
    proxy idle timeouts on long generations while returning the exact same
    ``ChatCompletion`` object as a non-streaming call. Controlled by the
    ``stream`` model arg (``-M stream=true|false|auto``) or the
    ``DEEPINFRA_STREAM`` environment variable. The default ``"auto"`` streams
    when reasoning is requested or ``max_tokens >= 8192``.
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

        # Pop before super().__init__ - leftover model_args are forwarded to
        # the AsyncOpenAI constructor, which rejects unknown kwargs
        stream_arg = model_args.pop("stream", None)
        if stream_arg is None:
            stream_arg = os.environ.get("DEEPINFRA_STREAM", "auto")
        self.streaming: bool | Literal["auto"] = parse_stream_arg(stream_arg)

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

    def should_stream(self, config: GenerateConfig) -> bool:
        """Decide whether this request goes over SSE."""
        if self.streaming != "auto":
            return self.streaming
        # Stream when reasoning is in play or the response is large enough
        # to risk proxy idle/read timeouts
        return (
            config.reasoning_effort is not None
            or config.reasoning_tokens is not None
            or (config.max_tokens is not None and config.max_tokens >= 8192)
        )

    async def _generate_completion(
        self, request: dict[str, Any], config: GenerateConfig
    ) -> ChatCompletion:
        if not self.should_stream(config):
            return await super()._generate_completion(request, config)

        tools = request.get("tools")
        state: ChatCompletionStreamState[object] = ChatCompletionStreamState(
            input_tools=tools if tools and tools is not NOT_GIVEN else NOT_GIVEN
        )
        stream = await self.client.chat.completions.create(
            **request,
            stream=True,
            stream_options={"include_usage": True},  # usage rides the final chunk
        )
        async for chunk in stream:
            state.handle_chunk(chunk)
        try:
            return state.get_final_completion()
        except Exception:
            # get_final_completion() validates more aggressively than the
            # non-streaming path (e.g. malformed tool-call JSON); fall back to
            # the raw accumulated completion rather than losing the response
            return state.current_completion_snapshot
