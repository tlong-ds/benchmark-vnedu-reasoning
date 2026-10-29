"""A blank probe answer fails by rule and never reaches the LLM judge."""

from types import SimpleNamespace

import pytest

from vnedu_cot.schemas.items import KnowledgeProbe
from vnedu_cot.stages import probes

ITEM = SimpleNamespace(item_id="S1-HS-01-HIS")
MODEL = SimpleNamespace(
    model_id="deepseek/deepseek-r1-0528",
    api_gateway=None,
    temperature=0.0,
    max_output_tokens=None,
    provider=None,
    allow_fallbacks=False,
)
PROBE = KnowledgeProbe(q="Năm nào diễn ra Cách mạng tháng Tám?", a="Năm 1945.")


def _fake_chat(replies: list[str], calls: list[str]):
    def call(prompt: str, **kwargs: object) -> SimpleNamespace:
        calls.append(prompt)
        return SimpleNamespace(content=replies[len(calls) - 1])

    return call


@pytest.mark.parametrize("blank", ["", "   \n"])
def test_blank_answer_fails_without_judge(monkeypatch, blank: str) -> None:
    calls: list[str] = []
    monkeypatch.setattr(probes, "call_chat_model", _fake_chat([blank], calls))

    result, parsed = probes._probe_one(ITEM, 0, PROBE, MODEL, "judge/model")

    assert len(calls) == 1  # the model was asked; the judge was not
    assert parsed
    assert result.correct is False
    assert result.judge_model == probes.EMPTY_RESPONSE_JUDGE
    assert result.model_answer == probes.EMPTY_RESPONSE_ANSWER


def test_nonblank_answer_is_judged(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(probes, "call_chat_model", _fake_chat(["1945", "ĐÚNG"], calls))

    result, parsed = probes._probe_one(ITEM, 0, PROBE, MODEL, "judge/model")

    assert len(calls) == 2
    assert parsed
    assert result.correct is True
    assert result.judge_model == "judge/model"
    assert result.model_answer == "1945"
