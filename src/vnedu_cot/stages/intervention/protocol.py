"""Frozen intervention protocol, loaded from a TOML file in ``configs/experiments/``."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from vnedu_cot.schemas.base import RecordModel
from vnedu_cot.schemas.intervention import Arm, GenerationSettings


class InterventionProtocol(RecordModel):
    protocol_id: str = Field(min_length=1)
    schema_version: str = "2.0"
    model: str
    gateway: str
    provider: str
    allow_fallbacks: bool
    samples_per_arm: int = Field(ge=1)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    reasoning_effort: str
    max_output_tokens: int = Field(ge=1)
    arms: tuple[Arm, ...]
    concurrency: int = Field(ge=1)
    items_in_flight: int = Field(default=1, ge=1)
    rating_key: Path
    corrupted_chains: Path
    items: Path
    source_path: Path

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        if self.gateway != "openrouter":
            raise ValueError("protocol requires gateway='openrouter'")
        if self.allow_fallbacks:
            raise ValueError("protocol forbids provider fallback")
        if set(self.arms) != set(Arm) or len(self.arms) != 3:
            raise ValueError("protocol requires clean, placebo, corrupted arms")
        required_workers = len(self.arms) * self.items_in_flight
        if self.concurrency < required_workers:
            raise ValueError(
                "concurrency must cover every arm of every in-flight item "
                f"(need at least {required_workers})"
            )
        return self

    @property
    def settings(self) -> GenerationSettings:
        return GenerationSettings(
            temperature=self.temperature,
            top_p=self.top_p,
            reasoning_effort=self.reasoning_effort,
            max_output_tokens=self.max_output_tokens,
        )

    @property
    def digest(self) -> str:
        data = self.model_dump(mode="json", exclude={"source_path"})
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def load(cls, path: Path, *, repo_root: Path | None = None) -> Self:
        resolved = path.resolve()
        data = tomllib.loads(resolved.read_text(encoding="utf-8"))
        root = repo_root.resolve() if repo_root else resolved.parents[2]
        data["arms"] = tuple(Arm(value) for value in data["arms"])
        for key in ("rating_key", "corrupted_chains", "items"):
            value = Path(data[key])
            data[key] = value if value.is_absolute() else root / value
        data["source_path"] = resolved
        return cls.model_validate(data)

