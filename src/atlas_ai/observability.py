from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    interaction_id: str
    request_id: str
