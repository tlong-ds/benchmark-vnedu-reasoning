"""Blinded rater-sheet rows for the human severity rating.

A rater sees the question, options and steps 1-3 only: never step 4, the
catch-trial flag, the domain, the model or any corruption metadata.
``build_rater_sheet_row`` returns exactly ``rater_sheet_columns()``, and
``stages.validation`` checks that list never leaks a blinding field.
"""

from vnedu_cot.schemas.chains import BaseChain, CorruptedChain, CorruptedStep
from vnedu_cot.schemas.items import BenchmarkItem

_RATER_SHEET_COLUMNS: tuple[str, ...] = (
    "rating_id",
    "question",
    "options",
    "step_1",
    "step_2",
    "step_3",
    "detectability",
    "answer_relevance",
    "notes",
)


def rater_sheet_columns() -> tuple[str, ...]:
    """The exact, ordered column set a rater's sheet may contain."""

    return _RATER_SHEET_COLUMNS


def build_rater_sheet_row(
    item: BenchmarkItem,
    chain: BaseChain | CorruptedChain,
    rating_id: str,
    *,
    steps_override: tuple[CorruptedStep, ...] | None = None,
) -> dict[str, str]:
    """Build one blinded row for ``chain``, stopping at step 3.

    ``steps_override`` renders another step set of the same chain, such as a
    corrupted chain's ``placebo_steps``.
    """

    options_line = " | ".join(
        f"{letter.value}. {text}" for letter, text in item.options.items()
    )
    raw_steps = steps_override if steps_override is not None else chain.steps
    step_texts = [s.text for s in raw_steps[:3]]
    return {
        "rating_id": rating_id,
        "question": item.question_vi,
        "options": options_line,
        "step_1": step_texts[0],
        "step_2": step_texts[1],
        "step_3": step_texts[2],
        "detectability": "",
        "answer_relevance": "",
        "notes": "",
    }
