"""Model configs (``configs/models/*.toml``), per-draw seeds and the chat client.

Needs ``OPENROUTER_API_KEY`` in the environment or ``.env``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import Field

from vnedu_cot.schemas.core import (
    ApiGateway,
    MeasurementMode,
    ModelAccessTier,
    RecordModel,
)

# Loaded once at import time rather than on every call_chat_model() call --
# a multi-thousand-call pilot run would otherwise redo this file I/O on
# every single request.
load_dotenv()


def draw_seed(item_id: str, model: str, condition: str, draw_index: int) -> int:
    """Deterministic per-draw seed, distinct for every (item, model,
    condition, draw index). Some gateways
    decode deterministically regardless of temperature unless an explicit
    seed varies *per draw*. ``draw_index`` must therefore map to a
    distinct seed, never a constant shared across a whole request batch.

    Deliberately not stored on ``RunRecord`` -- callers/auditors re-derive
    the expected seed from this function rather than trusting a stored
    value, matching the codebase's "nothing a validator can't re-derive"
    style.
    """

    digest = hashlib.sha256(
        f"{item_id}|{model}|{condition}|{draw_index}".encode()
    ).hexdigest()
    return int(digest, 16) % 2_147_483_647


# --- model configuration --------------------------------------------------


class ModelConfig(RecordModel):
    """Identity and fixed generation settings for a benchmark model.

    ``access_tier`` (Brief SS3) determines which conditions are even
    runnable for this model — corruption and truncation require Tier 1 or
    Tier 2; Tier 3 models take hint injection only. ``measurement_mode``
    determines which of a :class:`~runs.RunRecord`'s ``sampled_answers`` /
    ``logprob_answers`` is populated. ``api_gateway`` is a separate axis
    from ``access_tier`` -- it says which HTTP API actually serves this
    model (currently only OpenRouter),
    not how much control the harness has over its reasoning.
    """

    model_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    access_tier: ModelAccessTier = ModelAccessTier.FULL_CONTROL
    api_gateway: ApiGateway = ApiGateway.OPENROUTER
    measurement_mode: MeasurementMode = MeasurementMode.SAMPLING
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: int = Field(default=8192, ge=1)
    seed: int | None = None
    allow_fallbacks: bool = False


def load_model_configs(path: Path) -> list[ModelConfig]:
    """Read a model roster: all ``*.toml`` files in a directory, or one file."""
    if path.is_dir():
        files = sorted(path.glob("*.toml"))
        if not files:
            raise FileNotFoundError(f"no .toml model config files found in {path}")
    else:
        if path.suffix != ".toml":
            raise ValueError(f"expected a .toml model config file, got {path.name}")
        files = [path]
    configs: list[ModelConfig] = []
    for f in files:
        data = tomllib.loads(f.read_text(encoding="utf-8"))
        if "models" in data and isinstance(data["models"], list):
            for item in data["models"]:
                configs.append(ModelConfig.model_validate_json(json.dumps(item)))
        else:
            configs.append(ModelConfig.model_validate_json(json.dumps(data)))
    return configs


# --- chat-completion client --------------------------------------------
#
# Call whichever gateway a model sits behind and return the raw response,
# including reasoning traces where the gateway exposes them.

_GATEWAY_URL: dict[ApiGateway, str] = {
    ApiGateway.OPENROUTER: "https://openrouter.ai/api/v1/chat/completions",
}
_GATEWAY_API_KEY_ENV: dict[ApiGateway, str] = {
    ApiGateway.OPENROUTER: "OPENROUTER_API_KEY",
}
DEFAULT_MODEL = "deepseek/deepseek-r1-0528"
DEFAULT_GATEWAY = ApiGateway.OPENROUTER

# gpt-oss's Harmony chat template needs its own reasoning-toggle branch
# (see call_chat_model) regardless of which vLLM-backed gateway serves it.
_GPT_OSS_PREFIX = "openai/gpt-oss"

# Retried on 429 (rate limit) and 5xx (upstream flakiness), plus read
# timeouts/connect errors -- a several-thousand-call pilot run cannot die
# on one transient failure.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 6
_BACKOFF_BASE_SECONDS = 2.0

# Proactive floor on top of the retry backoff above -- a per-gateway
# minimum spacing between request *starts*, enforced regardless of how
# concurrently a caller (e.g. baseline.py's --concurrency fan-out) fires
# requests. Add an entry to _GATEWAY_MIN_INTERVAL_SECONDS for a gateway whose
# rate limit is tight enough that the reactive backoff alone burns through
# _MAX_ATTEMPTS before its window clears.
_DEFAULT_MIN_INTERVAL_SECONDS = 1.5
_GATEWAY_MIN_INTERVAL_SECONDS: dict[ApiGateway, float] = {}
_rate_locks: dict[ApiGateway, threading.Lock] = {
    gw: threading.Lock() for gw in _GATEWAY_URL
}
_last_call_at: dict[ApiGateway, float] = {}


def _throttle(gateway: ApiGateway) -> None:
    """Block until at least the gateway's minimum interval has elapsed
    since the last request start to that same gateway."""

    min_interval = _GATEWAY_MIN_INTERVAL_SECONDS.get(
        gateway, _DEFAULT_MIN_INTERVAL_SECONDS
    )
    with _rate_locks[gateway]:
        now = time.monotonic()
        last = _last_call_at.get(gateway)
        wait = 0.0 if last is None else last + min_interval - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _last_call_at[gateway] = now


class ModelCallError(RuntimeError):
    """A malformed or unexpected response body from a chat-completion API."""


@dataclass(frozen=True)
class ModelResponse:
    """Complete raw response from a chat-completion API, including any
    reasoning trace.

    Attributes:
        model: Model slug that actually served the request.
        content: The assistant's final answer text.
        reasoning: Raw reasoning / chain-of-thought trace (None if the
            model doesn't support it or it was excluded).
        reasoning_details: Structured reasoning steps array, if provided.
        usage: Token usage breakdown (includes reasoning_tokens).
        raw_json: The entire HTTP response body as a dict — nothing lost.
        request_id: Gateway request identifier, when the response supplies
            one. ``raw_json`` remains available for provider-specific fields.
        finish_reason: Why generation stopped ("stop", "length", etc.).
    """

    model: str
    content: str
    reasoning: str | None
    reasoning_details: list[dict[str, Any]] | None
    usage: dict[str, Any]
    raw_json: dict[str, Any]
    request_id: str | None
    finish_reason: str | None


def call_chat_model(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    gateway: ApiGateway = DEFAULT_GATEWAY,
    system_prompt: str | None = None,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    reasoning_effort: str | None = "high",
    assistant_prefill: str | None = None,
    seed: int | None = None,
    top_p: float | None = None,
    response_format: dict[str, Any] | None = None,
    provider: str | None = None,
    allow_fallbacks: bool = False,
) -> ModelResponse:
    """Send a prompt to a model's gateway and return the full raw response.

    The response preserves the reasoning trace (chain-of-thought tokens)
    when the model supports it (e.g. DeepSeek R1) and ``reasoning_effort``
    requests one. Pass ``reasoning_effort=None`` to suppress reasoning
    entirely -- required by the no-CoT difficulty pilot (Control 5), which
    measures direct answering with no chain at all. The toggle is
    gateway-gated: OpenRouter's ``reasoning``/``include_reasoning`` fields
    are its own extension, not part of the plain OpenAI-compatible surface.

    Retries on 429/5xx and network errors with exponential backoff, since a
    multi-thousand-call pilot run should not die on one transient failure.
    """
    env_var = _GATEWAY_API_KEY_ENV[gateway]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(f"{env_var} not set. Add it to .env or export it.")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if assistant_prefill is not None:
        messages.append({"role": "assistant", "content": assistant_prefill})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if top_p is not None:
        payload["top_p"] = top_p
    if response_format is not None:
        payload["response_format"] = response_format
    if model.startswith(_GPT_OSS_PREFIX):
        # gpt-oss's Harmony chat template has no boolean off-switch --
        # ``enable_thinking`` is a Qwen/Gemma template key it silently
        # ignores, reasoning through its own channel regardless. Only a
        # qualitative ``reasoning_effort`` (low/medium/high) is honoured,
        # and even that channel is never fully empty: "low" and "none" both
        # still emitted ~185 reasoning tokens before content, on par with
        # each other (confirmed empirically against openai/gpt-oss-120b on
        # NVIDIA NIM) -- "low" is the practical floor. Checked
        # before the gateway branches below since gpt-oss needs this
        # Harmony-specific key regardless of which gateway routes to it
        # (e.g. openai/gpt-oss-120b is configured with api_gateway=openrouter,
        # so the OpenRouter branch would otherwise shadow this one).
        payload["chat_template_kwargs"] = {
            "reasoning_effort": "low" if reasoning_effort is None else reasoning_effort
        }
    elif gateway is ApiGateway.OPENROUTER:
        # Per OpenRouter's docs (openrouter.ai/docs/use-cases/reasoning-tokens):
        # "none" is the documented way to disable reasoning; there is no
        # documented enabled=false. Once effort is set, reasoning:{} is used.
        if reasoning_effort is None:
            # Endpoints where reasoning cannot be disabled (e.g. DeepSeek R1)
            # return HTTP 400 if effort: "none" is sent.
            if not model.startswith("deepseek/deepseek-r1"):
                payload["reasoning"] = {"effort": "none"}
        else:
            payload["reasoning"] = {"effort": reasoning_effort}

        pinned_provider = provider or (
            "DeepInfra" if model.startswith("deepseek/deepseek-r1") else None
        )
        if pinned_provider:
            payload["provider"] = {
                "order": [pinned_provider],
                "allow_fallbacks": allow_fallbacks,
            }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if gateway is ApiGateway.OPENROUTER:
        headers["HTTP-Referer"] = "https://github.com/tlong-ds/benchmark-vnedu-reasoning"
        headers["X-Title"] = "VNEDU-CoT Research"

    _throttle(gateway)
    data = _post_with_retry(payload, headers, url=_GATEWAY_URL[gateway])

    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelCallError(
            f"unexpected {gateway.value} response shape: {data!r}"
        ) from exc

    return ModelResponse(
        model=data.get("model", model),
        # A ``content`` key present but null (seen from gpt-oss on NVIDIA
        # when it's still mid-reasoning at cutoff) isn't caught by dict.get's
        # default -- only a missing key is -- so normalize both to "".
        content=message.get("content") or "",
        # OpenRouter uses "reasoning"; NIM's vLLM-backed models use
        # "reasoning_content" instead (confirmed against gemma-4-31b-it).
        reasoning=message.get("reasoning") or message.get("reasoning_content"),
        reasoning_details=message.get("reasoning_details"),
        usage=data.get("usage", {}),
        raw_json=data,
        request_id=_request_id(data),
        finish_reason=choice.get("finish_reason"),
    )


def _request_id(data: dict[str, Any]) -> str | None:
    """Extract a gateway-supplied request identifier without narrowing raw JSON."""

    for key in ("id", "request_id"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None


def _post_with_retry(
    payload: dict[str, Any], headers: dict[str, str], *, url: str
) -> dict[str, Any]:
    """POST to a gateway, retrying transient failures with backoff."""

    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        if attempt > 0:
            delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            time.sleep(delay + random.uniform(0, delay * 0.25))
        try:
            resp = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=300.0,
            )
        except httpx.TransportError as exc:
            last_exc = exc
            continue

        if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
            last_exc = httpx.HTTPStatusError(
                f"retryable status {resp.status_code}",
                request=resp.request,
                response=resp,
            )
            continue

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ModelCallError(
                f"request to {url} failed with status {resp.status_code}: "
                f"{resp.text[:500]}"
            ) from exc
        data: dict[str, Any] = resp.json()
        return data

    assert last_exc is not None  # loop always sets it before exhausting
    raise ModelCallError(
        f"request to {url} failed after {_MAX_ATTEMPTS} attempts"
    ) from last_exc
