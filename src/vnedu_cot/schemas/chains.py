"""Canonical and corrupted reasoning-chain records."""

from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from vnedu_cot.language_checks import (
    ReasoningLanguage,
    classify_reasoning_language,
    contains_meta_planning,
    discloses_planted_corruption,
)
from vnedu_cot.schemas.core import OptionLetter, RecordModel, StepType

STEPS_PER_CHAIN = 4
CORRUPTED_STEP_INDEX = 3
CORRUPTED_PREFIX_LENGTH = 3


class BaseStep(RecordModel):
    i: int = Field(ge=1)
    type: StepType
    text: str = Field(min_length=1)


class _IndexedStep(Protocol):
    i: int


def validate_step_indices(steps: tuple[_IndexedStep, ...]) -> None:
    if len(steps) != STEPS_PER_CHAIN:
        raise ValueError(f"every chain has exactly {STEPS_PER_CHAIN} steps (R5)")
    if [step.i for step in steps] != list(range(1, STEPS_PER_CHAIN + 1)):
        raise ValueError("step indices must be contiguous from 1 (R5)")


class BaseChain(RecordModel):
    chain_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    steps: tuple[BaseStep, ...] = ()
    base_answer: OptionLetter
    gold: OptionLetter
    reasoning_language: ReasoningLanguage = ReasoningLanguage.VI
    raw_reasoning: str | None = None
    response_text: str | None = None

    @model_validator(mode="after")
    def validate_steps(self) -> Self:
        if self.base_answer is not self.gold:
            raise ValueError("validity gate failed: base_answer must equal gold")
        if not self.steps:
            return self
        validate_step_indices(self.steps)
        text = "\n".join(step.text for step in self.steps)
        if contains_meta_planning(text):
            raise ValueError("validity gate failed: chain is meta-planning")
        observed = classify_reasoning_language(text)
        if self.reasoning_language is not observed:
            raise ValueError("reasoning_language must match classifier result")
        if observed is ReasoningLanguage.OTHER:
            raise ValueError("validity gate failed: unsupported chain language")
        return self


class CorruptedStep(RecordModel):
    i: int = Field(ge=1)
    type: StepType
    text: str = Field(min_length=1)
    corrupted: bool


class CorruptedChain(RecordModel):
    chain_id: str = Field(min_length=1)
    base_chain: BaseChain
    steps: tuple[CorruptedStep, ...]
    corrupted_step: int = Field(default=CORRUPTED_STEP_INDEX, ge=1)
    propagation_answer: OptionLetter
    propagation_unique: Literal[True]
    no_world_knowledge_shortcut: Literal[True]
    option_length_balanced: Literal[True]
    rationale_vi: str = Field(min_length=1)
    placebo_steps: tuple[CorruptedStep, ...] | None = None
    detectability: int | None = Field(default=None, ge=0, le=5)
    answer_relevance: bool | None = None

    @property
    def item_id(self) -> str:
        return self.base_chain.item_id

    @property
    def model(self) -> str:
        return self.base_chain.model

    @model_validator(mode="after")
    def validate_corruption(self) -> Self:
        if len(self.steps) != CORRUPTED_PREFIX_LENGTH:
            raise ValueError("corrupted chain must contain exactly three steps")
        if [step.i for step in self.steps] != [1, 2, 3]:
            raise ValueError("corrupted step indices must be 1, 2, 3")
        if [step.i for step in self.steps if step.corrupted] != [CORRUPTED_STEP_INDEX]:
            raise ValueError("only step 3 may be corrupted")
        if self.corrupted_step != CORRUPTED_STEP_INDEX:
            raise ValueError("corrupted_step is always 3")
        base = self.base_chain.steps
        if len(base) != STEPS_PER_CHAIN:
            raise ValueError("corrupted chain requires a four-step base chain")
        if tuple(step.text for step in self.steps[:2]) != tuple(
            step.text for step in base[:2]
        ):
            raise ValueError("corruption must preserve base steps 1 and 2")
        if self.steps[2].text == base[2].text:
            raise ValueError("corruption must replace base step 3")
        if classify_reasoning_language(self.steps[2].text) is not ReasoningLanguage.VI:
            raise ValueError("corrupted step 3 must be Vietnamese")
        if discloses_planted_corruption(self.steps[2].text):
            raise ValueError("corrupted step 3 must not disclose the manipulation")
        if classify_reasoning_language(self.rationale_vi) is not ReasoningLanguage.VI:
            raise ValueError("rationale_vi must be Vietnamese")
        if self.propagation_answer is self.base_chain.gold:
            raise ValueError("propagation_answer must differ from gold")
        if self.placebo_steps is not None:
            placebo = self.placebo_steps
            if (
                len(placebo) != CORRUPTED_PREFIX_LENGTH
                or [step.i for step in placebo] != [1, 2, 3]
            ):
                raise ValueError("placebo chain must contain steps 1, 2, 3")
            if tuple(step.text for step in placebo[:2]) != tuple(
                step.text for step in base[:2]
            ):
                raise ValueError("placebo must preserve base steps 1 and 2")
            if placebo[2].text in {base[2].text, self.steps[2].text}:
                raise ValueError(
                    "placebo step 3 must differ from base and corrupted text"
                )
            if classify_reasoning_language(placebo[2].text) is not ReasoningLanguage.VI:
                raise ValueError("placebo step 3 must be Vietnamese")
            if any(step.corrupted for step in placebo):
                raise ValueError("placebo steps cannot be marked corrupted")
        return self
