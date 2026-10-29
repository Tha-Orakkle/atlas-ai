import asyncio
import logging

import pytest

from atlas_ai.errors import (
    LLMAuthenticationError,
    LLMRateLimitError,
)
from atlas_ai.llm.streaming import (
    ResponseCompleted,
    StreamError,
    TextDelta,
    ToolCallCompleted,
)
from atlas_ai.models import AssistantMessage, ToolCall
from atlas_ai.observability import RequestContext
from atlas_ai.reliability.retry import RetryPolicy
from atlas_ai.reliability.stream_retry import (
    retry_astream,
    retry_stream,
)


def collect(async_iterator):
    async def run():
        return [event async for event in async_iterator]

    return asyncio.run(run())


def test_sync_retries_failure_before_text_and_then_succeeds(
    monkeypatch,
):
    calls = 0
    sleeps = []
    context = RequestContext("interaction-1", "request-1")

    def operation():
        nonlocal calls
        calls += 1

        if calls == 1:
            yield StreamError(error=LLMRateLimitError("limited"))
        else:
            yield TextDelta("Hello")
            yield ResponseCompleted(
                message=AssistantMessage(text="Hello")
            )

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        sleeps.append,
    )

    events = list(
        retry_stream(
            operation,
            policy=RetryPolicy(
                max_attempts=2,
                base_delay=1,
                max_delay=1,
            ),
            request_context=context,
        )
    )

    assert calls == 2
    assert len(sleeps) == 1
    assert events == [
        TextDelta("Hello"),
        ResponseCompleted(message=AssistantMessage(text="Hello")),
    ]


def test_sync_does_not_retry_after_text_was_emitted(monkeypatch):
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        yield TextDelta("Partial answer")
        yield StreamError(error=LLMRateLimitError("limited"))

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        lambda _: pytest.fail("Must not retry after text"),
    )

    events = list(
        retry_stream(
            operation,
            policy=RetryPolicy(max_attempts=3),
        )
    )

    assert calls == 1
    assert events[0] == TextDelta("Partial answer")
    assert isinstance(events[1], StreamError)
    assert isinstance(events[1].error, LLMRateLimitError)


def test_sync_does_not_retry_non_retryable_error(monkeypatch):
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        yield StreamError(
            error=LLMAuthenticationError("invalid credentials")
        )

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        lambda _: pytest.fail("Must not retry non-retryable error"),
    )

    events = list(
        retry_stream(
            operation,
            policy=RetryPolicy(max_attempts=3),
        )
    )

    assert calls == 1
    assert len(events) == 1
    assert isinstance(events[0].error, LLMAuthenticationError)


def test_sync_stops_at_max_attempts(monkeypatch):
    calls = 0
    sleeps = []

    def operation():
        nonlocal calls
        calls += 1
        yield StreamError(error=LLMRateLimitError("limited"))

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        sleeps.append,
    )

    events = list(
        retry_stream(
            operation,
            policy=RetryPolicy(max_attempts=3),
        )
    )

    assert calls == 3
    assert len(sleeps) == 2
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, LLMRateLimitError)


def test_sync_respects_retry_after(monkeypatch):
    calls = 0
    sleeps = []

    def operation():
        nonlocal calls
        calls += 1

        if calls == 1:
            yield StreamError(
                error=LLMRateLimitError(
                    "limited",
                    retry_after=2.5,
                )
            )
        else:
            yield ResponseCompleted(
                message=AssistantMessage(text="Done")
            )

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        sleeps.append,
    )

    events = list(
        retry_stream(
            operation,
            policy=RetryPolicy(max_attempts=2),
        )
    )

    assert calls == 2
    assert sleeps == [2.5]
    assert events == [
        ResponseCompleted(message=AssistantMessage(text="Done"))
    ]


def test_sync_retry_logs_correlation_without_exception_message(
    monkeypatch, caplog,
):
    secret = "PRIVATE_PROMPT_SECRET"

    def operation():
        yield StreamError(error=LLMRateLimitError(secret))

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        lambda _: None,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="atlas_ai.reliability.stream_retry",
    ):
        list(
            retry_stream(
                operation,
                policy=RetryPolicy(max_attempts=2),
                request_context=RequestContext(
                    "interaction-1",
                    "request-1",
                ),
            )
        )

    assert "interaction_id=interaction-1" in caplog.text
    assert "request_id=request-1" in caplog.text
    assert "attempt=1" in caplog.text
    assert "error_type=LLMRateLimitError" in caplog.text
    assert secret not in caplog.text


def test_async_retries_failure_before_text_and_then_succeeds(
    monkeypatch,
):
    calls = 0
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def operation():
        nonlocal calls
        calls += 1

        if calls == 1:
            yield StreamError(error=LLMRateLimitError("limited"))
        else:
            yield TextDelta("Hello")
            yield ResponseCompleted(
                message=AssistantMessage(text="Hello")
            )

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        fake_sleep,
    )

    events = collect(
        retry_astream(
            operation,
            policy=RetryPolicy(
                max_attempts=2,
                base_delay=1,
                max_delay=1,
            ),
            request_context=RequestContext(
                "interaction-1",
                "request-1",
            ),
        )
    )

    assert calls == 2
    assert len(sleeps) == 1
    assert events == [
        TextDelta("Hello"),
        ResponseCompleted(message=AssistantMessage(text="Hello")),
    ]


def test_async_does_not_retry_after_text_was_emitted(monkeypatch):
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        yield TextDelta("Partial answer")
        yield StreamError(error=LLMRateLimitError("limited"))

    async def unexpected_sleep(_):
        pytest.fail("Must not retry after text")

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        unexpected_sleep,
    )

    events = collect(
        retry_astream(
            operation,
            policy=RetryPolicy(max_attempts=3),
        )
    )

    assert calls == 1
    assert events[0] == TextDelta("Partial answer")
    assert isinstance(events[1], StreamError)
    assert isinstance(events[1].error, LLMRateLimitError)


def test_async_does_not_retry_non_retryable_error(monkeypatch):
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        yield StreamError(
            error=LLMAuthenticationError("invalid credentials")
        )

    async def unexpected_sleep(_):
        pytest.fail("Must not retry non-retryable error")

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        unexpected_sleep,
    )

    events = collect(
        retry_astream(
            operation,
            policy=RetryPolicy(max_attempts=3),
        )
    )

    assert calls == 1
    assert len(events) == 1
    assert isinstance(events[0].error, LLMAuthenticationError)


def test_async_stops_at_max_attempts(monkeypatch):
    calls = 0
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def operation():
        nonlocal calls
        calls += 1
        yield StreamError(error=LLMRateLimitError("limited"))

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        fake_sleep,
    )

    events = collect(
        retry_astream(
            operation,
            policy=RetryPolicy(max_attempts=3),
        )
    )

    assert calls == 3
    assert len(sleeps) == 2
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, LLMRateLimitError)


def test_async_respects_retry_after(monkeypatch):
    calls = 0
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def operation():
        nonlocal calls
        calls += 1

        if calls == 1:
            yield StreamError(
                error=LLMRateLimitError(
                    "limited",
                    retry_after=2.5,
                )
            )
        else:
            yield ResponseCompleted(
                message=AssistantMessage(text="Done")
            )

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        fake_sleep,
    )

    events = collect(
        retry_astream(
            operation,
            policy=RetryPolicy(max_attempts=2),
        )
    )

    assert calls == 2
    assert sleeps == [2.5]
    assert events == [
        ResponseCompleted(message=AssistantMessage(text="Done"))
    ]


def test_sync_logs_non_retryable_stream_failure(
    monkeypatch, caplog,
):
    secret = "PRIVATE_ERROR_MESSAGE"
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        yield StreamError(
            error=LLMAuthenticationError(secret)
        )

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        lambda _: pytest.fail("Must not retry"),
    )

    with caplog.at_level(
        logging.WARNING,
        logger="atlas_ai.reliability.stream_retry",
    ):
        events = list(
            retry_stream(
                operation,
                policy=RetryPolicy(max_attempts=3),
                request_context=RequestContext(
                    "interaction-1", "request-1"
                ),
            )
        )

    assert calls == 1
    assert len(events) == 1
    assert isinstance(events[0], StreamError)

    assert "Streaming request attempt failed" in caplog.text
    assert "interaction_id=interaction-1" in caplog.text
    assert "request_id=request-1" in caplog.text
    assert "attempt=1" in caplog.text
    assert "error_type=LLMAuthenticationError" in caplog.text
    assert "retryable=False" in caplog.text
    assert secret not in caplog.text
    assert "Retrying streaming request" not in caplog.text
    assert "safe_to_retry=True" in caplog.text


def test_async_logs_non_retryable_stream_failure(
    monkeypatch, caplog,
):
    secret = "PRIVATE_ERROR_MESSAGE"
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        yield StreamError(
            error=LLMAuthenticationError(secret)
        )

    async def unexpected_sleep(_):
        pytest.fail("Must not retry")

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        unexpected_sleep,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="atlas_ai.reliability.stream_retry",
    ):
        events = collect(
            retry_astream(
                operation,
                policy=RetryPolicy(max_attempts=3),
                request_context=RequestContext(
                    "interaction-1", "request-1"
                ),
            )
        )

    assert calls == 1
    assert len(events) == 1
    assert isinstance(events[0], StreamError)

    assert "Streaming request attempt failed" in caplog.text
    assert "interaction_id=interaction-1" in caplog.text
    assert "request_id=request-1" in caplog.text
    assert "attempt=1" in caplog.text
    assert "error_type=LLMAuthenticationError" in caplog.text
    assert "retryable=False" in caplog.text
    assert secret not in caplog.text
    assert "Retrying streaming request" not in caplog.text
    assert "safe_to_retry=True" in caplog.text


def test_sync_does_not_retry_after_tool_call_was_emitted(monkeypatch):
    calls = 0
    tool_event = ToolCallCompleted(
        tool_call=ToolCall(
            call_id="call-1",
            name="calculate",
            arguments={"expression": "2 + 2"},
        )
    )

    def operation():
        nonlocal calls
        calls += 1
        yield tool_event
        yield StreamError(error=LLMRateLimitError("limited"))

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        lambda _: pytest.fail("Must not retry after tool call"),
    )

    events = list(
        retry_stream(operation, policy=RetryPolicy(max_attempts=3))
    )

    assert calls == 1
    assert events[0] == tool_event
    assert isinstance(events[1], StreamError)
    assert isinstance(events[1].error, LLMRateLimitError)


def test_async_does_not_retry_after_tool_call_was_emitted(monkeypatch):
    calls = 0
    tool_event = ToolCallCompleted(
        tool_call=ToolCall(
            call_id="call-1",
            name="calculate",
            arguments={"expression": "2 + 2"},
        )
    )

    async def operation():
        nonlocal calls
        calls += 1
        yield tool_event
        yield StreamError(error=LLMRateLimitError("limited"))

    async def unexpected_sleep(_):
        pytest.fail("Must not retry after tool call")

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        unexpected_sleep,
    )

    events = collect(
        retry_astream(operation, policy=RetryPolicy(max_attempts=3))
    )

    assert calls == 1
    assert events[0] == tool_event
    assert isinstance(events[1], StreamError)
    assert isinstance(events[1].error, LLMRateLimitError)


def test_sync_logs_retryable_failure_and_retry_decision(
    monkeypatch, caplog,
):
    secret = "PRIVATE_ERROR_MESSAGE"
    calls = 0
    context = RequestContext("interaction-1", "request-1")

    def operation():
        nonlocal calls
        calls += 1

        if calls == 1:
            yield StreamError(error=LLMRateLimitError(secret))
        else:
            yield ResponseCompleted(
                message=AssistantMessage(text="Recovered")
            )

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        lambda _: None,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="atlas_ai.reliability.stream_retry",
    ):
        events = list(
            retry_stream(
                operation,
                policy=RetryPolicy(max_attempts=2),
                request_context=context,
            )
        )

    assert calls == 2
    assert isinstance(events[-1], ResponseCompleted)
    assert "Streaming request attempt failed" in caplog.text
    assert "Retrying streaming request" in caplog.text
    assert "interaction_id=interaction-1" in caplog.text
    assert "request_id=request-1" in caplog.text
    assert "attempt=1" in caplog.text
    assert "error_type=LLMRateLimitError" in caplog.text
    assert "retryable=True" in caplog.text
    assert "next_attempt=2" in caplog.text
    assert secret not in caplog.text


def test_async_logs_retryable_failure_and_retry_decision(
    monkeypatch, caplog,
):
    secret = "PRIVATE_ERROR_MESSAGE"
    calls = 0
    context = RequestContext("interaction-1", "request-1")

    async def operation():
        nonlocal calls
        calls += 1

        if calls == 1:
            yield StreamError(error=LLMRateLimitError(secret))
        else:
            yield ResponseCompleted(
                message=AssistantMessage(text="Recovered")
            )

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        fake_sleep,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="atlas_ai.reliability.stream_retry",
    ):
        events = collect(
            retry_astream(
                operation,
                policy=RetryPolicy(max_attempts=2),
                request_context=context,
            )
        )

    assert calls == 2
    assert isinstance(events[-1], ResponseCompleted)
    assert "Streaming request attempt failed" in caplog.text
    assert "Retrying streaming request" in caplog.text
    assert "interaction_id=interaction-1" in caplog.text
    assert "request_id=request-1" in caplog.text
    assert "attempt=1" in caplog.text
    assert "error_type=LLMRateLimitError" in caplog.text
    assert "retryable=True" in caplog.text
    assert "next_attempt=2" in caplog.text
    assert secret not in caplog.text


def test_sync_retries_retryable_exception_during_iteration(monkeypatch):
    calls = 0

    def operation():
        nonlocal calls
        calls += 1

        if calls == 1:
            raise LLMRateLimitError("limited")

        yield ResponseCompleted(
            message=AssistantMessage(text="Recovered")
        )

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        lambda _: None,
    )

    events = list(
        retry_stream(operation, policy=RetryPolicy(max_attempts=2))
    )

    assert calls == 2
    assert events == [
        ResponseCompleted(message=AssistantMessage(text="Recovered"))
    ]


def test_async_retries_retryable_exception_during_iteration(monkeypatch):
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1

        if calls == 1:
            raise LLMRateLimitError("limited")

        yield ResponseCompleted(
            message=AssistantMessage(text="Recovered")
        )

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        fake_sleep,
    )

    events = collect(
        retry_astream(operation, policy=RetryPolicy(max_attempts=2))
    )

    assert calls == 2
    assert events == [
        ResponseCompleted(message=AssistantMessage(text="Recovered"))
    ]


def test_sync_does_not_retry_exception_after_text(monkeypatch):
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        yield TextDelta("Partial answer")
        raise LLMRateLimitError("limited")

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        lambda _: pytest.fail("Must not retry after text"),
    )

    events = list(
        retry_stream(operation, policy=RetryPolicy(max_attempts=3))
    )

    assert calls == 1
    assert events[0] == TextDelta("Partial answer")
    assert isinstance(events[1], StreamError)
    assert isinstance(events[1].error, LLMRateLimitError)


def test_async_does_not_retry_exception_after_text(monkeypatch):
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        yield TextDelta("Partial answer")
        raise LLMRateLimitError("limited")

    async def unexpected_sleep(_):
        pytest.fail("Must not retry after text")

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        unexpected_sleep,
    )

    events = collect(
        retry_astream(operation, policy=RetryPolicy(max_attempts=3))
    )

    assert calls == 1
    assert events[0] == TextDelta("Partial answer")
    assert isinstance(events[1], StreamError)
    assert isinstance(events[1].error, LLMRateLimitError)


def test_sync_logs_non_retryable_exception_during_iteration(
    monkeypatch, caplog,
):
    secret = "PRIVATE_AUTH_ERROR"
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise LLMAuthenticationError(secret)
        yield  # Makes this a generator.

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.time.sleep",
        lambda _: pytest.fail("Must not retry"),
    )

    with caplog.at_level(
        logging.WARNING,
        logger="atlas_ai.reliability.stream_retry",
    ):
        events = list(
            retry_stream(
                operation,
                policy=RetryPolicy(max_attempts=3),
                request_context=RequestContext(
                    "interaction-1", "request-1"
                ),
            )
        )

    assert calls == 1
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, LLMAuthenticationError)
    assert "Streaming request attempt failed" in caplog.text
    assert "interaction_id=interaction-1" in caplog.text
    assert "request_id=request-1" in caplog.text
    assert "error_type=LLMAuthenticationError" in caplog.text
    assert "retryable=False" in caplog.text
    assert secret not in caplog.text
    assert "Retrying streaming request" not in caplog.text


def test_async_logs_non_retryable_exception_during_iteration(
    monkeypatch, caplog,
):
    secret = "PRIVATE_AUTH_ERROR"
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        raise LLMAuthenticationError(secret)
        yield  # Makes this an async generator.

    async def unexpected_sleep(_):
        pytest.fail("Must not retry")

    monkeypatch.setattr(
        "atlas_ai.reliability.stream_retry.asyncio.sleep",
        unexpected_sleep,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="atlas_ai.reliability.stream_retry",
    ):
        events = collect(
            retry_astream(
                operation,
                policy=RetryPolicy(max_attempts=3),
                request_context=RequestContext(
                    "interaction-1", "request-1"
                ),
            )
        )

    assert calls == 1
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, LLMAuthenticationError)
    assert "Streaming request attempt failed" in caplog.text
    assert "interaction_id=interaction-1" in caplog.text
    assert "request_id=request-1" in caplog.text
    assert "error_type=LLMAuthenticationError" in caplog.text
    assert "retryable=False" in caplog.text
    assert secret not in caplog.text
    assert "Retrying streaming request" not in caplog.text
