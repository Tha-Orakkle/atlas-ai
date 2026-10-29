import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator

from atlas_ai.errors import AtlasError
from atlas_ai.llm.streaming import (
    StreamError,
    StreamEvent,
    TextDelta,
)
from atlas_ai.observability import RequestContext
from atlas_ai.reliability.retry import (
    RetryPolicy,
    calculate_retry_delay,
)

logger = logging.getLogger(__name__)


def _can_retry(
    error: Exception,
    *,
    emitted_text: bool,
    attempt: int,
    policy: RetryPolicy,
) -> bool:
    return (
        isinstance(error, AtlasError)
        and error.retryable
        and not emitted_text
        and policy.safe_to_retry
        and attempt < policy.max_attempts
    )


def _log_retry(
    error: AtlasError,
    *,
    attempt: int,
    policy: RetryPolicy,
    request_context: RequestContext | None
) -> float:
    delay = calculate_retry_delay(error, policy, attempt)

    logger.warning(
        "Retrying streaming request | interaction_id=%s"
        "request_id=%s next_attempt=%d max_attempts=%d "
        "delay_seconds=%.2f",
        request_context.interaction_id if request_context else None,
        request_context.request_id if request_context else None,
        attempt + 1,
        policy.max_attempts,
        delay
    )

    return delay


def _log_stream_failure(
    error: Exception,
    *,
    attempt: int,
    policy: RetryPolicy,
    request_context: RequestContext | None,
) -> None:
    logger.warning(
        "Streaming request attempt failed | interaction_id=%s "
        "request_id=%s attempt=%d max_attempt=%d "
        "error_type=%s retryable=%s safe_to_retry=%s",
        request_context.interaction_id if request_context else None,
        request_context.request_id if request_context else None,
        attempt,
        policy.max_attempts,
        type(error).__name__,
        getattr(error, "retryable", False),
        getattr(policy, "safe_to _retry", False)
    )


def retry_stream(
    operation: Callable[[], Iterator[StreamEvent]],
    *,
    policy: RetryPolicy | None = None,
    request_context: RequestContext | None = None
) -> Iterator[StreamEvent]:
    """Retry a stream only until user-visible text has been emitted."""
    policy = policy or RetryPolicy()

    if policy.max_attempts < 1:
        raise ValueError("max_attempts must be at leaset 1")

    for attempt in range(1, policy.max_attempts + 1):
        emitted_text = False
        retry_error: AtlasError | None = None

        stream = None

        try:
            stream = operation()

            for event in stream:
                if isinstance(event, StreamError):
                    error = event.error

                    _log_stream_failure(
                        error,
                        attempt=attempt,
                        policy=policy,
                        request_context=request_context
                    )

                    if _can_retry(
                        error,
                        emitted_text=emitted_text,
                        attempt=attempt,
                        policy=policy,
                    ):
                        retry_error = error
                        break

                    yield event
                    return

                if isinstance(event, TextDelta):
                    emitted_text = True

                yield event

        except AtlasError as error:
            _log_stream_failure(
                error,
                attempt=attempt,
                policy=policy,
                request_context=request_context
            )

            if _can_retry(
                error,
                emitted_text=emitted_text,
                attempt=attempt,
                policy=policy,
            ):
                retry_error = error

            else:
                yield StreamError(error=error)
                return

        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

        if retry_error is None:
            return

        delay = _log_retry(
            retry_error,
            attempt=attempt,
            policy=policy,
            request_context=request_context
        )
        time.sleep(delay)


async def retry_astream(
    operation: Callable[[], AsyncIterator[StreamEvent]],
    *,
    policy: RetryPolicy | None = None,
    request_context: RequestContext | None = None
) -> AsyncIterator[StreamEvent]:
    """Async equivalent of retry_stream()."""
    policy = policy or RetryPolicy()

    if policy.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, policy.max_attempts + 1):
        emitted_text = False
        retry_error: AtlasError | None = None

        stream = None

        try:
            stream = operation()

            async for event in stream:
                if isinstance(event, StreamError):
                    error = event.error

                    _log_stream_failure(
                        error,
                        attempt=attempt,
                        policy=policy,
                        request_context=request_context
                    )

                    if _can_retry(
                        error,
                        emitted_text=emitted_text,
                        attempt=attempt,
                        policy=policy,
                    ):
                        retry_error = error
                        break

                    yield event
                    return

                if isinstance(event, TextDelta):
                    emitted_text = True

                yield event

        except AtlasError as error:
            _log_stream_failure(
                error,
                attempt=attempt,
                policy=policy,
                request_context=request_context
            )

            if _can_retry(
                error,
                emitted_text=emitted_text,
                attempt=attempt,
                policy=policy
            ):
                retry_error = error

            else:
                yield StreamError(error=error)
                return

        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

        if retry_error is None:
            return

        delay = _log_retry(
            retry_error,
            attempt=attempt,
            policy=policy,
            request_context=request_context
        )

        await asyncio.sleep(delay)
