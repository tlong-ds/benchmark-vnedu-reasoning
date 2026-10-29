from vnedu_cot.schemas.chains import BaseChain, BaseStep
from vnedu_cot.schemas.core import OptionLetter, StepType
from vnedu_cot.schemas.items import BenchmarkItem
from vnedu_cot.stages.annotation import build_rater_sheet_row, rater_sheet_columns


def _item() -> BenchmarkItem:
    return BenchmarkItem.model_construct(
        item_id="ITEM-1",
        question_vi="Một cộng một bằng bao nhiêu?",
        options={
            OptionLetter.A: "2",
            OptionLetter.B: "3",
            OptionLetter.C: "4",
            OptionLetter.D: "5",
        },
    )


def _steps(step_three: str = "1 + 1 = 2.") -> tuple[BaseStep, ...]:
    return (
        BaseStep(i=1, type=StepType.RETRIEVAL, text="Ta có hai số 1."),
        BaseStep(i=2, type=StepType.INFERENCE, text="Cần cộng hai số."),
        BaseStep(i=3, type=StepType.COMPUTATION, text=step_three),
        BaseStep(i=4, type=StepType.EVALUATION, text="Chọn A."),
    )


def test_rater_row_firewall_has_only_permitted_columns() -> None:
    chain = BaseChain.model_construct(
        chain_id="ITEM-1::model::canonical", steps=_steps()
    )
    row = build_rater_sheet_row(_item(), chain, "STIM-A")
    assert tuple(row) == rater_sheet_columns()
    forbidden = {
        "model",
        "domain",
        "level",
        "is_catch_trial",
        "rationale_vi",
        "propagation_answer",
        "step_4",
    }
    assert not forbidden & row.keys()
    assert row["step_3"] == "1 + 1 = 2."


def test_placebo_override_uses_same_blinding_firewall() -> None:
    chain = BaseChain.model_construct(
        chain_id="ITEM-1::model::canonical", steps=_steps()
    )
    placebo_steps = _steps("Cộng 1 với 1 thu được 2.")[:3]
    row = build_rater_sheet_row(
        _item(), chain, "PL-A", steps_override=placebo_steps  # type: ignore[arg-type]
    )
    assert tuple(row) == rater_sheet_columns()
    assert row["step_3"] == "Cộng 1 với 1 thu được 2."
