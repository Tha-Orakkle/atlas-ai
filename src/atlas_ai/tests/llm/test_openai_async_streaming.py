import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import atlas_ai.llm.open_ai as openai_module
from atlas_ai.errors import (
    LLMIncompleteResponseError,
    LLMRateLimitError,
    LLMResponseConsistencyError,
    LLMTimeoutError,
)
from atlas_ai.llm.streaming import (
    ResponseCompleted,
    StreamError,
    TextDelta,
    ToolCallCompleted,
)
from atlas_ai.models import (
    AssistantMessage,
    TokenUsage,
    ToolCall,
    UserMessage,
)
from atlas_ai.observability import RequestContext


def make_response(
    *,
    status="completed",
    output_text="",
    output=None,
    usage=None,
):
    return SimpleNamespace(
        status=status,
        output_text=output_text,
        output=output or [],
        usage=usage,
    )


def configure_async_stream(client, events, response):
    stream = MagicMock()
    stream.__aiter__.return_value = events
    stream.get_final_response = AsyncMock(return_value=response)

    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=stream)
    manager.__aexit__ = AsyncMock(return_value=False)

    client.async_client.responses.stream.return_value = manager
    return stream


def collect(async_iterator):
    async def _collect():
        return [event async for event in async_iterator]

    return asyncio.run(_collect())


def test_astream_yields_text_deltas_and_completion(openai_client):
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        input_tokens_details=SimpleNamespace(cached_tokens=30),
        output_tokens_details=SimpleNamespace(reasoning_tokens=5)
    )

    response = make_response(
        output_text="Hello world.",
        usage=usage
    )

    configure_async_stream(
        openai_client,
        events=[
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Hello ",
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="world.",
            ),
        ],
        response=response,
    )

    events = collect(
        openai_client.astream([UserMessage(content="Say hello.")])
    )

    assert events == [
        TextDelta(text="Hello "),
        TextDelta(text="world."),
        ResponseCompleted(
            message=AssistantMessage(text="Hello world."),
            usage=TokenUsage(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                cached_input_tokens=30,
                reasoning_tokens=5
            )
        ),
    ]

    openai_client.async_client.responses.stream.assert_called_once_with(
        model="gpt-test",
        input=[{"role": "user", "content": "Say hello."}],
        tools=openai_client.tools_schema
    )


def test_astram_yields_completed_tool_call_with_parsed_arguments(
    openai_client
):
    function_call = SimpleNamespace(
        type="function_call",
        name="calculate",
        call_id="call_123",
        arguments='{"expression": "2 + 2"}'
    )

    configure_async_stream(
        openai_client,
        events=[
            SimpleNamespace(
                type="response.output_item.done",
                item=function_call
            )
        ],
        response=SimpleNamespace(
            status="completed",
            output_text="",
            output=function_call
        )
    )

    events = collect(
        openai_client.astream([UserMessage(content="Calculate 2 + 2?")])
    )

    expected_tool_call = ToolCall(
        call_id="call_123",
        name="calculate",
        arguments={"expression": "2 + 2"}
    )

    assert events == [
        ToolCallCompleted(tool_call=expected_tool_call),
        ResponseCompleted(message=AssistantMessage(
            tool_calls=[expected_tool_call]
        ))
    ]


def test_astream_emits_error_for_incomplete_response(openai_client):
    configure_async_stream(
        openai_client,
        events=[],
        response=SimpleNamespace(
            status="incomplete",
            output_text="",
            output=[],
        ),
    )

    events = collect(
        openai_client.astream([UserMessage(content="Say hello.")])
    )

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, LLMIncompleteResponseError)
    assert not any(
        isinstance(event, ResponseCompleted)
        for event in events
    )


def test_astream_awaits_final_response(openai_client):
    stream = configure_async_stream(
        openai_client,
        events=[
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Hello.",
            ),
        ],
        response=SimpleNamespace(
            status="completed",
            output_text="Hello.",
            output=[],
        ),
    )

    collect(openai_client.astream([UserMessage(content="Hello.")]))

    stream.get_final_response.assert_awaited_once()


def test_astream_emits_rate_limit_error(monkeypatch, openai_client):

    class FakeLLMRateLimitError(Exception):
        def __init__(self):
            self.response = SimpleNamespace(
                headers={"Retry-After": 3}
            )

    def falling_event():
        raise FakeLLMRateLimitError()
        yield

    configure_async_stream(
        openai_client,
        events=falling_event(),
        response=make_response()
    )

    monkeypatch.setattr(
        openai_module,
        "RateLimitError",
        FakeLLMRateLimitError
    )

    events = collect(
        openai_client.astream([UserMessage(content="Hello")])
    )
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, LLMRateLimitError)
    assert events[0].error.retry_after == 3


def test_astream_emits_timeout_error(monkeypatch, openai_client):
    class FakeTimeoutError(Exception):
        pass

    def failing_event():
        raise FakeTimeoutError()
        yield

    configure_async_stream(
        client=openai_client,
        events=failing_event(),
        response=make_response()
    )

    monkeypatch.setattr(
        openai_module,
        "APITimeoutError",
        FakeTimeoutError
    )

    events = collect(
        openai_client.astream([UserMessage(content="Hello")])
    )

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, LLMTimeoutError)


def test_astream_emits_consistency_error_for_mismatched_response(
    openai_client
):
    configure_async_stream(
        client=openai_client,
        events=[
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Hello "
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="country."
            ),
        ],
        response=make_response(
            output_text="Hello world."
        ),
    )

    events = collect(
        openai_client.astream([UserMessage(content="Hello.")])
    )

    assert isinstance(events[-1], StreamError)
    error = events[-1].error
    assert isinstance(error, LLMResponseConsistencyError)
    assert "text did not match" in str(error)


def test_astream_does_not_use_retry_wrapper(
    monkeypatch,
    openai_client
):
    configure_async_stream(
        openai_client,
        events=[
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Hello."
            ),
        ],
        response=make_response(output_text="Hello.")
    )

    retry_mock = MagicMock(
        side_effect=AssertionError("Retry must not be called.")
    )

    monkeypatch.setattr(openai_module, "retry", retry_mock)

    events = collect(
        openai_client.astream([UserMessage(content="Say hello.")])
    )

    assert isinstance(events[-1], ResponseCompleted)
    retry_mock.assert_not_called()


def test_astream_logs_calling_llm(caplog, openai_client):
    configure_async_stream(
        openai_client,
        events=[],
        response=make_response()
    )

    request_context = RequestContext(
        interaction_id="interaction_id",
        request_id="request_id"
    )

    with caplog.at_level(logging.INFO, logger="atlas_ai.llm.open_ai"):
        collect(
            openai_client.astream(
                [UserMessage(content="Hello")],
                request_context=request_context
            )
        )

    assert (
        f"LLM request started (async-streaming) | "
        f"provider={openai_client.provider} "
        f"interaction_id={request_context.interaction_id} "
        f"request_id={request_context.request_id}"
    ) in caplog.text

    assert (
        f"LLM request completed (async-streaming) | "
        f"provider={openai_client.provider} "
        f"interaction_id={request_context.interaction_id} "
        f"request_id={request_context.request_id} provider_request_id="
    ) in caplog.text


def test_astream_logs_provider_request_id(caplog, openai_client):
    stream = configure_async_stream(
        openai_client,
        events=[],
        response=make_response(output_text=""),
    )
    stream._response = SimpleNamespace(
        headers={"x-request-id": "provider-async-stream-123"}
    )
    context = RequestContext(
        interaction_id="interaction-123",
        request_id="request-456",
    )

    with caplog.at_level(logging.INFO, logger="atlas_ai.llm.open_ai"):
        events = collect(
            openai_client.astream(
                [UserMessage(content="Hello.")],
                request_context=context,
            )
        )

    assert isinstance(events[-1], ResponseCompleted)
    assert "provider_request_id=provider-async-stream-123" in caplog.text


def test_astream_logs_missing_provider_request_id(caplog, openai_client):
    stream = configure_async_stream(
        openai_client,
        events=[],
        response=make_response(output_text=""),
    )
    stream._response = SimpleNamespace()
    context = RequestContext(
        interaction_id="interaction-123",
        request_id="request-456",
    )

    with caplog.at_level(logging.INFO, logger="atlas_ai.llm.open_ai"):
        events = collect(
            openai_client.astream(
                [UserMessage(content="Hello.")],
                request_context=context,
            )
        )

    assert isinstance(events[-1], ResponseCompleted)
    assert "provider_request_id=None" in caplog.text


def test_astream_does_not_log_prompt_or_raw_tool_arguments(
    caplog, openai_client
):
    prompt_secret = "PRIVATE_PROMPT_SECRET"
    argument_secret = "RAW_TOOL_ARGUMENT_SECRET"

    stream = configure_async_stream(
        openai_client,
        events=[
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Safe response.",
            )
        ],
        response=make_response(output_text="Safe response."),
    )
    stream._response = SimpleNamespace(headers={})

    context = [
        UserMessage(content=prompt_secret),
        AssistantMessage(
            tool_calls=[
                ToolCall(
                    call_id="call_123",
                    name="sensitive_tool",
                    arguments={"secret": argument_secret},
                )
            ]
        ),
    ]

    with caplog.at_level(logging.INFO, logger="atlas_ai.llm.open_ai"):
        events = collect(openai_client.astream(context))

    assert isinstance(events[-1], ResponseCompleted)
    assert prompt_secret not in caplog.text
    assert argument_secret not in caplog.text
    assert "provider_request_id=None" in caplog.text
