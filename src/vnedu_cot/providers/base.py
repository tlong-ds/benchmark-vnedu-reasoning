"""Transport protocol and normalised response types shared by the clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RawTransportResponse:
    status_code: int
    body: dict[str, Any]


class Transport(Protocol):
    def post(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> RawTransportResponse: ...


@dataclass(frozen=True)
class NormalizedResponse:
    model: str
    provider: str
    content: str
    reasoning: str
    reasoning_details: tuple[dict[str, Any], ...]
    usage: dict[str, Any]
    cost: float | None
    request_id: str | None
    finish_reason: str | None
    raw: dict[str, Any]

    @property
    def reasoning_tokens(self) -> int:
        direct = self.usage.get("reasoning_tokens")
        if isinstance(direct, int):
            return direct
        details = self.usage.get("completion_tokens_details", {})
        if isinstance(details, dict):
            value = details.get("reasoning_tokens")
            if isinstance(value, int):
                return value
        return 0

