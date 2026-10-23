from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from atlas_ai.models import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    """Prices in USD per one million tokens."""
    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None


class CostCalculator:
    def __init__(self, pricing: Mapping[str, ModelPricing]):
        self._pricing = dict(pricing)

    def estimate(
        self,
        model: str,
        usage: TokenUsage | None
    ) -> Decimal | None:
        """Estimate interaction cost in USD, or None if unknown."""
        if usage is None:
            return None

        model_pricing = self._pricing.get(model)
        if not model_pricing:
            return None

        if usage.cached_input_tokens > usage.input_tokens:
            raise ValueError(
                "Cached input tokens cannot exceed total input tokens."
            )

        ordinary_input_tokens = (
            usage.input_tokens - usage.cached_input_tokens
        )

        if (
            usage.cached_input_tokens > 0
            and model_pricing.cached_input_per_million is None
        ):
            return None

        million = Decimal(1000000)

        input_cost = (
            Decimal(ordinary_input_tokens)
            * model_pricing.input_per_million
            / million
        )

        output_cost = (
            Decimal(usage.output_tokens)
            * model_pricing.output_per_million
            / million
        )

        cached_input_cost = Decimal(0)

        if usage.cached_input_tokens:
            cached_input_cost = (
                Decimal(usage.cached_input_tokens)
                * model_pricing.cached_input_per_million
                / million
            )

        return input_cost + cached_input_cost + output_cost
