"""Records written by the intervention runner: draws, calibration, reports."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from vnedu_cot.schemas.base import RecordModel


class Arm(StrEnum):
    CLEAN = "clean"
    PLACEBO = "placebo"
    CORRUPTED = "corrupted"


class AnswerToken(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    FLAG = "flag"
    UNPARSEABLE = "unparseable"


class GenerationSettings(RecordModel):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    reasoning_effort: str = Field(min_length=1)
    max_output_tokens: int = Field(ge=1)


class DrawDiagnostics(RecordModel):
    reasoning_channel_present: bool
    positive_reasoning_tokens: bool
    prefix_echo: bool
    apparent_step_4_continuation: bool


class InterventionDrawRecord(RecordModel):
    """One immutable normalized response for one scheduled intervention draw."""

    schema_version: str
    protocol_id: str
    protocol_hash: str
    run_id: str
    calibration_id: str
    draw_id: str
    item_id: str
    item_version: int = Field(ge=1)
    chain_id: str
    source_base_chain_id: str
    gold_answer: str
    propagation_answer: str
    arm: Arm
    draw_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    settings: GenerationSettings
    requested_model: str
    served_model: str
    requested_provider: str
    served_provider: str
    reasoning_prefix: str
    assistant_prefill: str
    reasoning_continuation: str
    full_reasoning: str
    # The provider's unmodified visible response.  ``final_response`` may be
    # normalized for presentation after a deliberately auditable migration.
    raw_final_response: str | None = None
    final_response: str
    parsed_answer: AnswerToken
    reasoning_details: tuple[dict[str, Any], ...] = ()
    usage: dict[str, Any]
    cost: float | None = Field(default=None, ge=0)
    request_id: str | None = None
    finish_reason: str | None = None
    started_at: dt.datetime
    finished_at: dt.datetime
    latency_ms: int = Field(ge=0)
    prompt_hash: str
    prefix_hash: str
    request_hash: str
    source_record_hash: str
    diagnostics: DrawDiagnostics
    protocol_valid: bool
    protocol_failures: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_reconstruction(self) -> Self:
        if self.assistant_prefill != "<think>\n" + self.reasoning_prefix:
            raise ValueError("assistant_prefill must be unfinished <think> plus prefix")
        if self.full_reasoning != self.reasoning_prefix + self.reasoning_continuation:
            raise ValueError("full_reasoning must reconstruct prefix plus continuation")
        if "<think>" in self.reasoning_prefix or "</think>" in self.reasoning_prefix:
            raise ValueError("reasoning_prefix contains reserved control tokens")
        if self.protocol_valid == bool(self.protocol_failures):
            raise ValueError("protocol_valid and protocol_failures disagree")
        return self


class CalibrationRecord(RecordModel):
    schema_version: str
    protocol_id: str
    protocol_hash: str
    run_id: str
    calibration_id: str
    draw_id: str
    requested_model: str
    served_model: str
    requested_provider: str
    served_provider: str
    reasoning_prefix: str
    assistant_prefill: str
    reasoning_continuation: str
    raw_final_response: str | None = None
    final_response: str
    parsed_answer: AnswerToken
    reasoning_tokens: int
    request_id: str | None = None
    diagnostics: DrawDiagnostics
    passed: bool
    failures: tuple[str, ...]
    timestamp: dt.datetime


class InterventionReport(RecordModel):
    schema_version: str = "2.0"
    protocol_id: str
    run_id: str
    item_id: str
    chain_id: str
    model: str
    n_per_arm: int
    p_propagation_clean: float
    p_propagation_placebo: float
    p_propagation_corrupted: float
    p_gold_clean: float
    p_gold_placebo: float
    p_gold_corrupted: float
    effect_corrupted_vs_placebo: float
    effect_placebo_vs_clean: float
    effect_corrupted_vs_clean: float
    corrupted_behavior_counts: dict[str, int]

