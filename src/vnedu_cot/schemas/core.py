"""Shared immutable value spaces for the maintained package."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RecordModel(BaseModel):
    """Immutable, strict records that reject undocumented fields."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, str_strip_whitespace=True
    )


class Level(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    HIGH_SCHOOL = "high_school"
    UNIVERSITY = "university"


class Domain(StrEnum):
    COMPUTATION = "computation"
    LITERATURE = "literature"
    HISTORY = "history"
    THEORETICAL = "theoretical"


class OptionLetter(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class StepType(StrEnum):
    RETRIEVAL = "retrieval"
    INFERENCE = "inference"
    COMPUTATION = "computation"
    EVALUATION = "evaluation"


class ChainVariant(StrEnum):
    BASE = "base"
    CORRUPTED = "corrupted"
    PLACEBO = "placebo"


class RunCondition(StrEnum):
    BASE = "base"
    CORRUPTED = "corrupted"
    PLACEBO = "placebo"
    NO_COT = "no_cot"


class ModelAccessTier(StrEnum):
    FULL_CONTROL = "tier_1_full_control"
    PREFILL = "tier_2_prefill"
    BLOCKED = "tier_3_blocked"


class MeasurementMode(StrEnum):
    SAMPLING = "sampling"
    LOGPROB = "logprob"


class ApiGateway(StrEnum):
    OPENROUTER = "openrouter"


class AnswerToken(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    FLAG = "flag"
    UNPARSEABLE = "unparseable"
