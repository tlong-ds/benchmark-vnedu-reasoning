"""Benchmark-item records and their corpus invariants."""

import datetime as dt
from enum import StrEnum
from typing import Self

from pydantic import AliasChoices, Field, model_validator

from vnedu_cot.schemas.core import Domain, Level, OptionLetter, RecordModel

CHANCE_LEVEL = 0.25
NO_COT_BAND: tuple[float, float] = (0.2, 0.8)


class Authorship(StrEnum):
    HAND_WRITTEN = "hand_written"
    LLM_ASSISTED = "llm_assisted"
    LLM_DRAFTED_HUMAN_VERIFIED = "llm_drafted_human_verified"


class KnowledgeProbe(RecordModel):
    q: str = Field(min_length=1)
    a: str = Field(min_length=1)


class BenchmarkItem(RecordModel):
    item_id: str = Field(min_length=1)
    level: Level
    domain: Domain
    question_vi: str = Field(min_length=1)
    options: dict[OptionLetter, str]
    gold: OptionLetter
    answer_format: str = Field(min_length=1)
    no_cot_accuracy: dict[str, float] | None = None
    schema_id: str | None = None
    pair_id: str | None = None
    knowledge_probes: tuple[KnowledgeProbe, ...] = ()
    n_external_premises: int = Field(default=0, ge=0)
    authorship: Authorship
    verified_by: str | None = None
    verification_date: dt.date | None = Field(
        default=None, validation_alias=AliasChoices("verification_date", "date")
    )
    source: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_item(self) -> Self:
        if len(self.options) != 4:
            raise ValueError("expected exactly 4 options")
        if len({value.strip() for value in self.options.values()}) != 4:
            raise ValueError("option text must be distinct")
        if self.gold not in self.options:
            raise ValueError("gold must be one of the option letters")
        if self.n_external_premises != len(self.knowledge_probes):
            raise ValueError("n_external_premises must equal len(knowledge_probes)")
        if self.authorship in (
            Authorship.LLM_ASSISTED,
            Authorship.LLM_DRAFTED_HUMAN_VERIFIED,
        ) and (self.verified_by is None or self.verification_date is None):
            raise ValueError(
                "llm_assisted items require verified_by and verification_date (R4)"
            )
        return self
