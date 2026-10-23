from decimal import Decimal

import pytest

from atlas_ai.cost import CostCalculator, ModelPricing
from atlas_ai.models import TokenUsage


@pytest.fixture
def calculator():
    return CostCalculator(
        {
            "example-model": ModelPricing(
                input_per_million=Decimal("2.00"),
                output_per_million=Decimal("8.00"),
                cached_input_per_million=Decimal("0.50"),
            )
        }
    )


def test_estimates_cost_with_cached_input_tokens(calculator):
    usage = TokenUsage(
        input_tokens=1000,
        output_tokens=300,
        total_tokens=1300,
        cached_input_tokens=200,
    )

    result = calculator.estimate("example-model", usage)

    assert result == Decimal("0.0041")


def test_estimates_cost_without_cached_input_tokens(calculator):
    usage = TokenUsage(
        input_tokens=1000,
        output_tokens=300,
        total_tokens=1300,
    )

    result = calculator.estimate("example-model", usage)

    assert result == Decimal("0.0044")


def test_returns_none_for_unknown_model(calculator):
    usage = TokenUsage(1000, 300, 1300)

    assert calculator.estimate("unconfigured-model", usage) is None


def test_returns_none_when_usage_is_unavailable(calculator):
    assert calculator.estimate("example-model", None) is None


def test_returns_none_when_cached_token_price_is_unknown():
    calculator = CostCalculator(
        {
            "example-model": ModelPricing(
                input_per_million=Decimal("2.00"),
                output_per_million=Decimal("8.00"),
            )
        }
    )
    usage = TokenUsage(
        input_tokens=1000,
        output_tokens=300,
        total_tokens=1300,
        cached_input_tokens=200,
    )

    assert calculator.estimate("example-model", usage) is None


def test_rejects_cached_tokens_exceeding_input_tokens(calculator):
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        cached_input_tokens=200,
    )

    with pytest.raises(ValueError, match="Cached input tokens"):
        calculator.estimate("example-model", usage)
