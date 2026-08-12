"""Unit tests for the DeepInfra provider (streaming config and accumulation)."""

import asyncio
import os
from typing import Any, Literal, Optional

import pytest
from unittest.mock import AsyncMock

from inspect_ai.model import GenerateConfig
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice as CompletionChoice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from openbench.model._providers.deepinfra import DeepInfraAPI, parse_stream_arg


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Keep ambient DeepInfra env vars from leaking into tests."""
    monkeypatch.delenv("DEEPINFRA_STREAM", raising=False)
    monkeypatch.delenv("DEEPINFRA_BASE_URL", raising=False)


def make_api(**model_args: Any) -> DeepInfraAPI:
    return DeepInfraAPI(
        model_name="deepinfra/test-org/test-model",
        api_key="test-key",
        **model_args,
    )


def make_chunk(
    text: Optional[str] = None,
    role: Optional[Literal["developer", "system", "user", "assistant", "tool"]] = None,
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    ] = None,
    usage: Optional[CompletionUsage] = None,
    tool_calls: Optional[list] = None,
    usage_only: bool = False,
) -> ChatCompletionChunk:
    choices = (
        []
        if usage_only
        else [
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(content=text, role=role, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )
    return ChatCompletionChunk(
        id="chatcmpl-test",
        choices=choices,
        created=1,
        model="test-model",
        object="chat.completion.chunk",
        usage=usage,
    )


def mock_streamed_chunks(api: DeepInfraAPI, chunks: list) -> AsyncMock:
    async def chunk_iterator():
        for chunk in chunks:
            yield chunk

    mock_create = AsyncMock(return_value=chunk_iterator())
    api.client.chat.completions.create = mock_create  # type: ignore[method-assign]
    return mock_create


REQUEST: dict[str, Any] = {
    "model": "test-org/test-model",
    "messages": [{"role": "user", "content": "hi"}],
}


class TestStreamArgParsing:
    def test_parse_values(self):
        assert parse_stream_arg(True) is True
        assert parse_stream_arg(False) is False
        assert parse_stream_arg("true") is True
        assert parse_stream_arg("True") is True
        assert parse_stream_arg("1") is True
        assert parse_stream_arg("yes") is True
        assert parse_stream_arg("false") is False
        assert parse_stream_arg("0") is False
        assert parse_stream_arg("auto") == "auto"
        assert parse_stream_arg("AUTO") == "auto"

    def test_default_is_auto(self):
        api = make_api()
        assert api.streaming == "auto"

    def test_model_arg_overrides_default(self):
        assert make_api(stream=True).streaming is True
        assert make_api(stream=False).streaming is False
        assert make_api(stream="auto").streaming == "auto"

    def test_env_var_used_when_no_model_arg(self, monkeypatch):
        monkeypatch.setenv("DEEPINFRA_STREAM", "true")
        assert make_api().streaming is True

    def test_model_arg_beats_env_var(self, monkeypatch):
        monkeypatch.setenv("DEEPINFRA_STREAM", "true")
        assert make_api(stream=False).streaming is False

    def test_stream_arg_not_forwarded_to_openai_client(self):
        # AsyncOpenAI rejects unknown kwargs, so constructing with a stream
        # model arg must not raise (i.e. the arg was popped before super())
        api = make_api(stream=True)
        assert api.client is not None


class TestShouldStream:
    def test_explicit_true_always_streams(self):
        api = make_api(stream=True)
        assert api.should_stream(GenerateConfig()) is True
        assert api.should_stream(GenerateConfig(max_tokens=16)) is True

    def test_explicit_false_never_streams(self):
        api = make_api(stream=False)
        assert api.should_stream(GenerateConfig(max_tokens=128000)) is False
        assert api.should_stream(GenerateConfig(reasoning_effort="high")) is False

    def test_auto_max_tokens_threshold(self):
        api = make_api()
        assert api.should_stream(GenerateConfig(max_tokens=8192)) is True
        assert api.should_stream(GenerateConfig(max_tokens=8191)) is False
        assert api.should_stream(GenerateConfig(max_tokens=4096)) is False
        assert api.should_stream(GenerateConfig()) is False  # max_tokens unset

    def test_auto_streams_for_reasoning(self):
        api = make_api()
        assert api.should_stream(GenerateConfig(reasoning_effort="high")) is True
        assert api.should_stream(GenerateConfig(reasoning_tokens=1024)) is True


class TestStreamingGeneration:
    def test_accumulates_content_and_usage(self):
        api = make_api(stream=True)
        mock_create = mock_streamed_chunks(
            api,
            [
                make_chunk(role="assistant", text="Hel"),
                make_chunk(text="lo"),
                make_chunk(finish_reason="stop"),
                make_chunk(
                    usage_only=True,
                    usage=CompletionUsage(
                        completion_tokens=2, prompt_tokens=3, total_tokens=5
                    ),
                ),
            ],
        )

        completion = asyncio.run(
            api._generate_completion(dict(REQUEST), GenerateConfig())
        )

        assert completion.choices[0].message.content == "Hello"
        assert completion.choices[0].finish_reason == "stop"
        assert completion.usage is not None
        assert completion.usage.total_tokens == 5

        _, kwargs = mock_create.call_args
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}

    def test_accumulates_tool_calls(self):
        api = make_api(stream=True)
        mock_streamed_chunks(
            api,
            [
                make_chunk(
                    role="assistant",
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_1",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name="get_weather", arguments='{"cit'
                            ),
                        )
                    ],
                ),
                make_chunk(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            function=ChoiceDeltaToolCallFunction(
                                arguments='y": "Paris"}'
                            ),
                        )
                    ],
                ),
                make_chunk(finish_reason="tool_calls"),
            ],
        )

        completion = asyncio.run(
            api._generate_completion(dict(REQUEST), GenerateConfig())
        )

        tool_calls = completion.choices[0].message.tool_calls
        assert tool_calls is not None and len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments == '{"city": "Paris"}'
        assert completion.choices[0].finish_reason == "tool_calls"

    def test_parse_failure_falls_back_to_snapshot(self, monkeypatch):
        api = make_api(stream=True)
        mock_streamed_chunks(
            api,
            [
                make_chunk(role="assistant", text="Hello"),
                make_chunk(finish_reason="stop"),
            ],
        )

        def boom(self):
            raise ValueError("parse failed")

        monkeypatch.setattr(ChatCompletionStreamState, "get_final_completion", boom)

        completion = asyncio.run(
            api._generate_completion(dict(REQUEST), GenerateConfig())
        )

        assert completion.choices[0].message.content == "Hello"
        assert completion.choices[0].finish_reason == "stop"

    def test_non_streaming_path_unchanged(self):
        api = make_api(stream=False)
        response = ChatCompletion(
            id="chatcmpl-test",
            choices=[
                CompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hi"),
                    finish_reason="stop",
                )
            ],
            created=1,
            model="test-org/test-model",
            object="chat.completion",
        )
        mock_create = AsyncMock(return_value=response)
        api.client.chat.completions.create = mock_create  # type: ignore[method-assign]

        completion = asyncio.run(
            api._generate_completion(dict(REQUEST), GenerateConfig(max_tokens=128000))
        )

        assert completion is response
        _, kwargs = mock_create.call_args
        assert "stream" not in kwargs
        assert "stream_options" not in kwargs


class TestProviderBasics:
    def test_service_prefix_stripped_once(self):
        api = make_api()
        assert api.service_model_name() == "test-org/test-model"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPINFRA_API_KEY"):
            DeepInfraAPI(model_name="deepinfra/test-model")


@pytest.mark.integration
def test_streaming_generation_against_api():
    """Streamed and non-streamed generations return the same output shape."""
    if not os.environ.get("DEEPINFRA_API_KEY"):
        pytest.skip("DEEPINFRA_API_KEY not set")

    from inspect_ai.model import get_model

    async def generate(stream: str) -> Any:
        model = get_model(
            "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct", stream=stream
        )
        return await model.generate("Reply with exactly the word: hello")

    streamed = asyncio.run(generate("true"))
    plain = asyncio.run(generate("false"))

    for output in (streamed, plain):
        assert output.completion.strip()
        assert output.usage is not None
        assert output.usage.total_tokens > 0
