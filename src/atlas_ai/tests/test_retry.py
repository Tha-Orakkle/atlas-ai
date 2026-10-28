import logging

import pytest

from atlas_ai.errors import AtlasError, LLMRateLimitError, LLMTimeoutError
from atlas_ai.observability import RequestContext
from atlas_ai.reliability.retry import RetryPolicy, retry


def test_successful_operation_runs_once():
    """
    Test that the reliability policy does not retry a
    successful operation.
    """
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        return "success"

    result = retry(operation)

    assert result == "success"
    assert calls == 1


def test_retryable_failure_retries_and_succeeds(monkeypatch):
    """
    Test that a retryable error is retried and succeeds on the
    third try.
    """
    calls = 0

    def operation():
        nonlocal calls
        calls += 1

        if calls < 3:
            raise LLMRateLimitError("Rate limited.")

        return "success"

    monkeypatch.setattr(
        "atlas_ai.reliability.retry.time.sleep",
        lambda _: None
    )

    result = retry(operation)

    assert calls == 3
    assert result == "success"


def test_non_retryable_error_does_not_retry():
    """
    Test that non-retryable errors do not get retried.
    """
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise AtlasError("Permanent error.")

    with pytest.raises(AtlasError):
        retry(operation)

    assert calls == 1


def test_max_attempts_are_respected(monkeypatch):
    """
    Test that reliability policy respects the max
    attempts limit.
    """
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise LLMRateLimitError("rate limited")

    monkeypatch.setattr(
        "atlas_ai.reliability.retry.time.sleep",
        lambda _: None
    )

    policy = RetryPolicy(max_attempts=5)

    with pytest.raises(LLMRateLimitError):
        retry(operation, policy=policy)

    assert calls == 5


def test_unsafe_operation_is_not_retried(monkeypatch):
    """
    Test that unsafe operations are not retried.
    """
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise LLMTimeoutError("timeout")

    monkeypatch.setattr(
        "atlas_ai.reliability.retry.time.sleep",
        lambda _: None
    )

    policy = RetryPolicy(safe_to_retry=False)

    with pytest.raises(LLMTimeoutError):
        retry(operation, policy=policy)

    assert calls == 1


def test_retry_after_is_respected(monkeypatch):
    """
    Test that a Retry-After value returned by the
    provided server is respected rather than using
    an exponential backoff with full jitter.
    """
    calls = 0
    sleep_times = []

    def operation():
        nonlocal calls
        calls += 1

        if calls == 1:
            raise LLMRateLimitError(
                "rate limited",
                retry_after=7.5
            )

        return "success"

    monkeypatch.setattr(
        "atlas_ai.reliability.retry.time.sleep",
        sleep_times.append
    )

    result = retry(operation)

    assert result == "success"
    assert calls == 2
    assert sleep_times == [7.5]


def test_backoff_is_used_when_retry_after_is_not_provided(monkeypatch):
    """
    Test that when the server does not provide the Retry-After value,
    the delay falls backoff and jitter.
    """
    calls = 0
    sleep_times = []

    def operation():
        nonlocal calls
        calls += 1

        if calls < 2:
            raise LLMRateLimitError("rate limited.")

        return "success"

    monkeypatch.setattr(
        "atlas_ai.reliability.retry.time.sleep",
        sleep_times.append
    )

    # make the jitter deterministic
    monkeypatch.setattr(
        "atlas_ai.reliability.retry.random.uniform",
        lambda mininmum, maximum: maximum
    )

    policy = RetryPolicy(
        base_delay=2.0,
        max_delay=30.0
    )

    result = retry(operation, policy=policy)

    assert result == "success"
    assert calls == 2

    # First retry:
    # base_delay * 2 ** (attempt - 1)
    # = 2 * 2 ** 0
    # = 2
    assert sleep_times == [2]


def test_backoff_is_capped_by_max_delay(monkeypatch):
    """
    Test that the delay does not exceed the set maximum delay.
    """
    calls = 0
    sleep_times = []

    def operation():
        nonlocal calls
        calls += 1
        raise LLMRateLimitError("rate limited.")

    monkeypatch.setattr(
        "atlas_ai.reliability.retry.time.sleep",
        sleep_times.append
    )

    monkeypatch.setattr(
        "atlas_ai.reliability.retry.random.uniform",
        lambda minimum, maximum: maximum
    )

    policy = RetryPolicy(
        max_attempts=3,
        base_delay=10,
        max_delay=15
    )

    with pytest.raises(LLMRateLimitError):
        retry(operation, policy=policy)

    assert calls == 3

    # Attempt 1:
    # 10 * 2**0 = 10
    #
    # Attempt 2:
    # 10 * 2**1 = 20 -> capped at 15
    assert sleep_times == [10, 15]


def test_retry_attempts_remain_logged(monkeypatch, caplog):
    """Test that each failed retry attempt is logged."""
    calls = 0
    request_context = RequestContext(
        interaction_id="interaction_id",
        request_id="request_id"
    )

    def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise LLMRateLimitError("rate limited")
        return "success"

    monkeypatch.setattr(
        "atlas_ai.reliability.retry.time.sleep",
        lambda _: None,
    )

    with caplog.at_level(logging.INFO, logger="atlas_ai.reliability.retry"):
        result = retry(
            operation,
            policy=RetryPolicy(max_attempts=3),
            request_context=request_context
        )

    assert result == "success"

    def _log(n):
        return (
            f"LLM request attempt failed | interaction_id=interaction_id "
            f"request_id=request_id attempt={n} max_attempts=3 "
            "error_type=LLMRateLimitError"
        )
    assert _log(1) in caplog.text
    assert _log(2) in caplog.text


def test_retry_failure_log_does_not_include_exception_message(
    caplog, monkeypatch
):
    secret = "PRIVATE_PROMPT_AND_TOOL_ARGUMENTS"
    context = RequestContext(
        interaction_id="interaction-123",
        request_id="request-456",
    )

    def operation():
        raise LLMRateLimitError(secret)

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.reliability.retry",
    ), pytest.raises(LLMRateLimitError):
        retry(
            operation,
            policy=RetryPolicy(max_attempts=1),
            request_context=context,
        )

    assert secret not in caplog.text
    assert "interaction_id=interaction-123" in caplog.text
    assert "request_id=request-456" in caplog.text
    assert "error_type=LLMRateLimitError" in caplog.text
