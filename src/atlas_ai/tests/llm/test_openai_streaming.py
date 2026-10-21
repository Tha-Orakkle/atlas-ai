import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import atlas_ai.llm.open_ai as openai_module
from atlas_ai.errors import (
    LLMBadResponseError,
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
    ToolCall,
    UserMessage,
)


def configure_stream(client, events, response):
    """Configure the mocked OpenAI stream context manager."""
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.__exit__.return_value = False
    stream.__iter__.return_value = iter(events)
    stream.get_final_response.return_value = response

    client.client.responses.stream.return_value = stream
    return stream


def make_response(
    *,
    status="completed",
    output_text="",
    output=None
):
    return SimpleNamespace(
        status=status,
        output_text=output_text,
        output=output or []
    )


def test_stream_yields_text_deltas_and_completed_response(openai_client):
    response = make_response(output_text="Hello world.")

    configure_stream(
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

    events = list(openai_client.stream([UserMessage(content="Say hello.")]))

    assert events == [
        TextDelta(text="Hello "),
        TextDelta(text="world."),
        ResponseCompleted(
            message=AssistantMessage(
                text="Hello world."
            )
        )
    ]

    openai_client.client.responses.stream.assert_called_once_with(
        model="gpt-test",
        input=[{"role": "user", "content": "Say hello."}],
        tools=openai_client.tools_schema
    )


def test_stream_yields_completed_tool_call_with_parsed_arguments(
    openai_client
):
    function_call = SimpleNamespace(
        type="function_call",
        call_id="call_123",
        name="calculate",
        arguments='{"expression": "2 + 2"}'
    )

    response = make_response(
        output=function_call
    )

    configure_stream(
        client=openai_client,
        events=[
            SimpleNamespace(
                type="response.output_item.done",
                item=function_call
            )
        ],
        response=response
    )

    events = list(openai_client.stream([
        UserMessage(content="Calculate 2 + 2")
    ]))

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


def test_stream_emits_error_for_invalid_tool_call_json(
    openai_client
):
    function_call = SimpleNamespace(
        type="function_call",
        name="calculate",
        call_id="call_123",
        arguments='{"expression": }'
    )

    configure_stream(
        client=openai_client,
        events=[
            SimpleNamespace(
                type="response.output_item.done",
                item=function_call
            )
        ],
        response=make_response()
    )

    events = list(openai_client.stream([UserMessage("calculate something")]))

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, LLMBadResponseError)


def test_stream_emits_rate_limit_error(monkeypatch, openai_client):

    class FakeLLMRateLimitError(Exception):
        def __init__(self):
            self.response = SimpleNamespace(
                headers={"Retry-After": 3}
            )

    def failing_event():
        raise FakeLLMRateLimitError()
        yield

    configure_stream(
        client=openai_client,
        events=failing_event(),
        response=make_response()
    )

    monkeypatch.setattr(
        openai_module,
        "RateLimitError",
        FakeLLMRateLimitError
    )

    events = list(openai_client.stream([UserMessage(content="Hello")]))

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, LLMRateLimitError)
    assert events[0].error.retry_after == 3


def test_stream_emits_timeout_error(monkeypatch, openai_client):
    class FakeTimeoutError(Exception):
        pass

    def failing_event():
        raise FakeTimeoutError()
        yield

    configure_stream(
        client=openai_client,
        events=failing_event(),
        response=make_response()
    )

    monkeypatch.setattr(
        openai_module,
        "APITimeoutError",
        FakeTimeoutError
    )

    events = list(openai_client.stream([UserMessage(content="Hello")]))

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, LLMTimeoutError)


def test_stream_emits_error_when_response_is_incomplete(
    openai_client
):
    configure_stream(
        openai_client,
        events=[],
        response=make_response(status="incomplete")
    )

    events = list(openai_client.stream([UserMessage(content="Hello.")]))
    assert isinstance(events[-1], StreamError)

    error = events[-1].error
    assert isinstance(error, LLMIncompleteResponseError)
    assert "did not complete successfully" in str(error)


def test_stream_emits_consistency_error_for_mismatched_response(
    openai_client
):
    configure_stream(
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

    events = list(openai_client.stream([UserMessage(content="Hello.")]))

    assert isinstance(events[-1], StreamError)
    error = events[-1].error
    assert isinstance(error, LLMResponseConsistencyError)
    assert "text did not match" in str(error)


def test_stream_does_not_use_retry_wrapper(
    monkeypatch,
    openai_client
):
    configure_stream(
        client=openai_client,
        events=[
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Hello."
            ),
        ],
        response=make_response(output_text="Hello.")
    )

    retry_mock = Mock(side_effect=AssertionError("Retry must not be called"))
    monkeypatch.setattr(openai_module, "retry", retry_mock)

    events = list(openai_client.stream([UserMessage(content="Hello.")]))

    assert isinstance(events[-1], ResponseCompleted)
    retry_mock.assert_not_called()


def test_stream_logs_calling_llm(caplog, openai_client):
    configure_stream(
        openai_client,
        events=[],
        response=make_response()
    )

    with caplog.at_level(logging.INFO, logger="atlas_ai.llm.open_ai"):
        list(openai_client.stream([UserMessage(content="Hello")]))

    assert "Starting OpenAI streaming request" in caplog.text
