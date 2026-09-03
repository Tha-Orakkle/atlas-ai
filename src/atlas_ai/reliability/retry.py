import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Any

from atlas_ai import config
from atlas_ai.errors import AtlasError, LLMTimeoutError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = config.RETRY_MAX_ATTEMPTS
    base_delay: float = config.RETRY_BASE_DELAY
    max_delay: float = config.RETRY_MAX_DELAY

    # Whether the operation can be safely be repeated
    safe_to_retry: bool = True


def retry(
    operation: Callable[[], Any],
    *,
    policy: RetryPolicy | None = None
) -> Any:
    if policy is None:
        policy = RetryPolicy()

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()

        except AtlasError as exc:
            logger.info(
                "Attempt %d/%d failed: %s",
                attempt,
                policy.max_attempts,
                exc
            )

            if not exc.retryable:
                logger.error("Error is not retryable. Aborting")
                raise

            if attempt == policy.max_attempts:
                logger.error("Maximum retry attempts reached.")
                raise

            # Only retry operations that are safe to retry
            if not policy.safe_to_retry:
                logger.warning(
                    "Operation is not safe to retry. Aborting."
                )
                raise

            # Check for explicitly provided delay
            if exc.retry_after is not None:
                wait_time = exc.retry_after
            else:
                exponential_delay = (
                    policy.base_delay * (2 ** (attempt - 1))
                )
                wait_time = random.uniform(
                    0,
                    min(
                        exponential_delay,
                        policy.max_delay
                    )
                )
            logger.warning("Waiting ~%.2f seconds...", wait_time)

            time.sleep(wait_time)
