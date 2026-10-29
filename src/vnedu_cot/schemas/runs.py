"""Records for ordinary sampled and log-probability runs."""

import datetime as dt
from typing import Self

from pydantic import Field, model_validator

from vnedu_cot.schemas.core import AnswerToken, OptionLetter, RecordModel, RunCondition

_DISPLAY_LETTERS = (OptionLetter.A, OptionLetter.B, OptionLetter.C, OptionLetter.D)


class RunRecord(RecordModel):
    run_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    item_version: int = Field(ge=1)
    chain_id: str | None = None
    model: str = Field(min_length=1)
    condition: RunCondition
    n_samples: int = Field(ge=1)
    temperature: float = Field(ge=0)
    prompt_template_hash: str = Field(min_length=1)
    raw_responses: tuple[str, ...] = ()
    displayed_answers: tuple[AnswerToken, ...] = ()
    sampled_answers: tuple[AnswerToken, ...] = ()
    option_orders: tuple[tuple[OptionLetter, ...], ...] = ()
    rendered_prompt_hashes: tuple[str, ...] = ()
    answer_distribution: dict[AnswerToken, int] = Field(default_factory=dict)
    logprob_answers: dict[AnswerToken, float] | None = None
    disclosure_score_llm: int | None = Field(default=None, ge=0, le=3)
    disclosure_judge_model: str | None = None
    disclosure_score_human: int | None = Field(default=None, ge=0, le=3)
    timestamp: dt.datetime

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if self.chain_id is None and self.condition is not RunCondition.NO_COT:
            raise ValueError("chain_id is only null for no-CoT pilot runs")
        if self.logprob_answers is not None:
            if self.sampled_answers or self.raw_responses:
                raise ValueError("logprob-mode runs do not carry sampled answers")
            return self
        if len(self.sampled_answers) != self.n_samples:
            raise ValueError("sampled_answers must have n_samples entries")
        if len(self.raw_responses) != self.n_samples:
            raise ValueError("raw_responses must have n_samples entries")
        derived = _count(self.sampled_answers)
        if self.answer_distribution != derived:
            raise ValueError("answer_distribution must equal sampled_answers counts")
        presentation = (
            self.displayed_answers,
            self.option_orders,
            self.rendered_prompt_hashes,
        )
        if any(presentation):
            if any(len(value) != self.n_samples for value in presentation):
                raise ValueError("all presentation fields must have n_samples entries")
            for order in self.option_orders:
                if len(order) != 4 or set(order) != set(_DISPLAY_LETTERS):
                    raise ValueError("each option_order must permute A, B, C, D")
            remapped = tuple(
                _remap_displayed_answer(answer, order)
                for answer, order in zip(
                    self.displayed_answers, self.option_orders, strict=True
                )
            )
            if remapped != self.sampled_answers:
                raise ValueError(
                    "sampled answers must match displayed-answer remapping"
                )
        return self


def _count(answers: tuple[AnswerToken, ...]) -> dict[AnswerToken, int]:
    counts: dict[AnswerToken, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return counts


def _remap_displayed_answer(
    answer: AnswerToken, order: tuple[OptionLetter, ...]
) -> AnswerToken:
    if answer in (AnswerToken.FLAG, AnswerToken.UNPARSEABLE):
        return answer
    return AnswerToken(order[_DISPLAY_LETTERS.index(OptionLetter(answer.value))].value)
