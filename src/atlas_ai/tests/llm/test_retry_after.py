from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from atlas_ai.llm.open_ai import _parse_retry_after


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2", 2.0),
        ("2.5", 2.5),
        ("0", 0.0),
    ],
)
def test_parses_numeric_retry_after(value, expected):
    assert _parse_retry_after(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "invalid",
        "-1",
        "inf",
        "-inf",
        "nan",
    ],
)
def test_invalid_retry_after_returns_none(value):
    assert _parse_retry_after(value) is None


def test_parses_future_http_date():
    retry_at = datetime.now(UTC) + timedelta(seconds=60)
    value = format_datetime(retry_at, usegmt=True)

    delay = _parse_retry_after(value)

    assert delay is not None
    assert 0 < delay <= 60


def test_past_http_date_returns_none():
    retry_at = datetime.now(UTC) - timedelta(seconds=60)
    value = format_datetime(retry_at, usegmt=True)

    assert _parse_retry_after(value) is None
