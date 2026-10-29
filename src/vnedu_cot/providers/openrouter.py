"""OpenRouter client for the intervention: retries, usage and response normalisation."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from vnedu_cot.providers.base import (
    NormalizedResponse,
    RawTransportResponse,
    Transport,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE = {429, 500, 502, 503, 504}


class HttpxTransport:
    def post(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> RawTransportResponse:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        body: dict[str, Any] = response.json()
        return RawTransportResponse(response.status_code, body)


AttemptSink = Callable[[dict[str, Any]], None]


class OpenRouterClient:
    def __init__(
        self,
        *,
        transport: Transport | None = None,
        api_key: str | None = None,
        attempt_sink: AttemptSink | None = None,
        max_attempts: int = 5,
    ) -> None:
        self.transport = transport or HttpxTransport()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.attempt_sink = attempt_sink
        self.max_attempts = max_attempts

    @staticmethod
    def build_payload(
        *,
        model: str,
        provider: str,
        allow_fallbacks: bool,
        system_prompt: str,
        user_prompt: str,
        assistant_prefill: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        reasoning_effort: str,
        seed: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prefill},
            ],
            "provider": {"order": [provider], "allow_fallbacks": allow_fallbacks},
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_output_tokens,
            "seed": seed,
            "reasoning": {"effort": reasoning_effort},
            "include_reasoning": True,
        }
        forbidden = {"prefix", "reasoning_content", "continue_final_message"}
        if forbidden & payload.keys():
            raise AssertionError("forbidden native-continuation request field")
        return payload

    def complete(self, payload: dict[str, Any]) -> NormalizedResponse:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tlong-ds/benchmark-vnedu-reasoning",
            "X-Title": "VNEDU-CoT Research",
        }
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                delay = 2 ** (attempt - 2)
                time.sleep(delay + random.uniform(0, delay * 0.25))
            try:
                response = self.transport.post(
                    url=OPENROUTER_URL,
                    payload=payload,
                    headers=headers,
                    timeout=300.0,
                )
                event = {
                    "attempt": attempt,
                    "request": payload,
                    "status_code": response.status_code,
                    "response": response.body,
                }
                if self.attempt_sink:
                    self.attempt_sink(event)
                if response.status_code in RETRYABLE and attempt < self.max_attempts:
                    last_error = f"retryable HTTP {response.status_code}"
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"OpenRouter HTTP {response.status_code}: "
                        f"{json.dumps(response.body, ensure_ascii=False)[:500]}"
                    )
                return normalize_openrouter_response(response.body)
            except (httpx.TransportError, TimeoutError) as exc:
                last_error = str(exc)
                if self.attempt_sink:
                    self.attempt_sink(
                        {"attempt": attempt, "request": payload, "error": last_error}
                    )
                if attempt == self.max_attempts:
                    break
        raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")


def normalize_openrouter_response(data: dict[str, Any]) -> NormalizedResponse:
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected OpenRouter response: {data!r}") from exc
    usage = data.get("usage") or {}
    provider = data.get("provider") or usage.get("provider") or ""
    cost = usage.get("cost", data.get("cost"))
    return NormalizedResponse(
        model=str(data.get("model") or ""),
        provider=str(provider),
        content=str(message.get("content") or ""),
        reasoning=str(message.get("reasoning") or ""),
        reasoning_details=tuple(message.get("reasoning_details") or ()),
        usage=dict(usage),
        cost=float(cost) if isinstance(cost, int | float) else None,
        request_id=data.get("id") or data.get("request_id"),
        finish_reason=choice.get("finish_reason"),
        raw=data,
    )
