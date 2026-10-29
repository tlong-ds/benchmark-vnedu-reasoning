from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vnedu_cot.analysis.intervention import (
    build_reports,
    classify_corrupted_behavior,
)
from vnedu_cot.providers.openrouter import (
    OpenRouterClient,
    normalize_openrouter_response,
)
from vnedu_cot.schemas.intervention import (
    AnswerToken,
    Arm,
    DrawDiagnostics,
    GenerationSettings,
    InterventionDrawRecord,
)
from vnedu_cot.stages.intervention.prompts import render_assistant_prefill
from vnedu_cot.stages.intervention.protocol import InterventionProtocol
from vnedu_cot.stages.intervention.runner import _calibrate, parse_answer
from vnedu_cot.stages.intervention.schedule import ScheduledDraw, build_schedule
from vnedu_cot.stages.intervention.sources import SourceChain, _render_steps


def _protocol(
    tmp_path: Path, *, samples: int = 2, items_in_flight: int = 1, concurrency: int = 3
) -> InterventionProtocol:
    protocol_file = tmp_path / "protocol.toml"
    protocol_file.write_text("protocol fixture\n", encoding="utf-8")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    return InterventionProtocol(
        protocol_id="test",
        schema_version="2.0",
        model="deepseek/deepseek-r1-0528",
        gateway="openrouter",
        provider="DeepInfra",
        allow_fallbacks=False,
        samples_per_arm=samples,
        temperature=0.7,
        top_p=1.0,
        reasoning_effort="high",
        max_output_tokens=8192,
        arms=(Arm.CLEAN, Arm.PLACEBO, Arm.CORRUPTED),
        concurrency=concurrency,
        items_in_flight=items_in_flight,
        rating_key=source,
        corrupted_chains=source,
        items=source,
        source_path=protocol_file,
    )


def _chain(item_id: str = "ITEM-1") -> SourceChain:
    return SourceChain(
        chain_id=f"{item_id}::model::corrupted",
        base_chain_id=f"{item_id}::model::canonical",
        item_id=item_id,
        item_version=1,
        model="deepseek/deepseek-r1-0528",
        gold="A",
        propagation_answer="C",
        question="Câu hỏi?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        prefixes={
            Arm.CLEAN: "1. clean one\n2. clean two\n3. clean three\n",
            Arm.PLACEBO: "1. clean one\n2. clean two\n3. placebo three\n",
            Arm.CORRUPTED: "1. clean one\n2. clean two\n3. corrupted three\n",
        },
        source_record_hash="source-hash",
    )


def _record(
    *, arm: Arm, answer: AnswerToken, continuation: str = "4. tiếp tục"
) -> InterventionDrawRecord:
    prefix = _chain().prefixes[arm]
    now = dt.datetime.now(dt.UTC)
    return InterventionDrawRecord(
        schema_version="2.0",
        protocol_id="test",
        protocol_hash="protocol-hash",
        run_id="run-1",
        calibration_id="cal-1",
        draw_id=f"{arm.value}-{answer.value}",
        item_id="ITEM-1",
        item_version=1,
        chain_id="ITEM-1::model::corrupted",
        source_base_chain_id="ITEM-1::model::canonical",
        gold_answer="A",
        propagation_answer="C",
        arm=arm,
        draw_index=0,
        seed=1,
        settings=GenerationSettings(
            temperature=0.7,
            top_p=1.0,
            reasoning_effort="high",
            max_output_tokens=8192,
        ),
        requested_model="deepseek/deepseek-r1-0528",
        served_model="deepseek/deepseek-r1-0528",
        requested_provider="DeepInfra",
        served_provider="DeepInfra",
        reasoning_prefix=prefix,
        assistant_prefill="<think>\n" + prefix,
        reasoning_continuation=continuation,
        full_reasoning=prefix + continuation,
        raw_final_response=f"Đáp án: {answer.value}",
        final_response=f"Đáp án: {answer.value}",
        parsed_answer=answer,
        reasoning_details=(),
        usage={"completion_tokens_details": {"reasoning_tokens": 2}},
        cost=0.1,
        request_id="req",
        finish_reason="stop",
        started_at=now,
        finished_at=now,
        latency_ms=1,
        prompt_hash="prompt",
        prefix_hash="prefix",
        request_hash="request",
        source_record_hash="source",
        diagnostics=DrawDiagnostics(
            reasoning_channel_present=True,
            positive_reasoning_tokens=True,
            prefix_echo=False,
            apparent_step_4_continuation=True,
        ),
        protocol_valid=True,
        protocol_failures=(),
    )


def test_exact_unfinished_think_rendering_for_all_arms() -> None:
    chain = _chain()
    for arm in Arm:
        rendered = render_assistant_prefill(chain.prefixes[arm])
        assert rendered == "<think>\n" + chain.prefixes[arm]
        assert not rendered.endswith("</think>")


def test_prefix_validation_rejects_shape_and_control_tokens() -> None:
    with pytest.raises(ValueError, match="steps 1, 2, and 3"):
        _render_steps([{"i": 1, "text": "one"}])
    with pytest.raises(ValueError, match="control token"):
        _render_steps(
            [
                {"i": 1, "text": "one"},
                {"i": 2, "text": "</think>"},
                {"i": 3, "text": "three"},
            ]
        )


def test_openrouter_payload_pins_provider_and_native_reasoning() -> None:
    payload = OpenRouterClient.build_payload(
        model="deepseek/deepseek-r1-0528",
        provider="DeepInfra",
        allow_fallbacks=False,
        system_prompt="system",
        user_prompt="user",
        assistant_prefill="<think>\n1. a\n2. b\n3. c\n",
        temperature=0.7,
        top_p=1.0,
        max_output_tokens=8192,
        reasoning_effort="high",
        seed=12,
    )
    assert payload["provider"] == {
        "order": ["DeepInfra"],
        "allow_fallbacks": False,
    }
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["include_reasoning"] is True
    assert not {
        "prefix",
        "reasoning_content",
        "continue_final_message",
    } & payload.keys()


def test_response_normalization_preserves_reasoning_and_provenance() -> None:
    fixture = json.loads(
        Path("tests/fixtures/openrouter_r1_step4_response.json").read_text()
    )
    response = normalize_openrouter_response(fixture)
    assert response.model == "deepseek/deepseek-r1-0528"
    assert response.provider == "DeepInfra"
    assert response.reasoning.startswith("4.")
    assert response.reasoning_details[0]["type"] == "reasoning.text"
    assert response.reasoning_tokens == 18
    assert response.request_id == "gen-test-step4"
    assert response.cost == 0.00042


def test_record_reconstruction_is_enforced() -> None:
    data = _record(arm=Arm.CLEAN, answer=AnswerToken.A).model_dump()
    data["full_reasoning"] = "wrong"
    with pytest.raises(ValidationError, match="reconstruct"):
        InterventionDrawRecord.model_validate(data)


@pytest.mark.parametrize(
    "text",
    (
        "Đáp án: A",
        "Đáp án: <A>",
        "### Đáp án:\nA",
        "**Đáp án**: **A**",
        "Đáp án: **A**.",
        "### Đáp án\nA",
        "→ Đáp án: A.",
        "**Đáp án: A** (giải thích thêm)",
    ),
)
def test_parse_answer_accepts_explicit_markdown_answer_variants(text: str) -> None:
    assert parse_answer(text) is AnswerToken.A


def test_schedule_is_stable_unique_and_complete_per_item(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    chains = [_chain("ITEM-1"), _chain("ITEM-2")]
    first = build_schedule(protocol, chains)
    second = build_schedule(protocol, chains)
    assert first == second
    assert len(first) == 12
    assert len({row.draw_id for row in first}) == 12
    assert len({row.seed for row in first}) == 12
    assert [(row.item_id, row.draw_index, row.arm) for row in first] == [
        ("ITEM-1", 0, Arm.CLEAN),
        ("ITEM-1", 0, Arm.PLACEBO),
        ("ITEM-1", 0, Arm.CORRUPTED),
        ("ITEM-1", 1, Arm.CLEAN),
        ("ITEM-1", 1, Arm.PLACEBO),
        ("ITEM-1", 1, Arm.CORRUPTED),
        ("ITEM-2", 0, Arm.CLEAN),
        ("ITEM-2", 0, Arm.PLACEBO),
        ("ITEM-2", 0, Arm.CORRUPTED),
        ("ITEM-2", 1, Arm.CLEAN),
        ("ITEM-2", 1, Arm.PLACEBO),
        ("ITEM-2", 1, Arm.CORRUPTED),
    ]


def test_schedule_runs_two_three_arm_item_bundles_concurrently(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path, items_in_flight=2, concurrency=6)
    chains = [_chain("ITEM-1"), _chain("ITEM-2"), _chain("ITEM-3")]
    schedule = build_schedule(protocol, chains)
    assert [(row.item_id, row.draw_index, row.arm) for row in schedule[:12]] == [
        ("ITEM-1", 0, Arm.CLEAN),
        ("ITEM-1", 0, Arm.PLACEBO),
        ("ITEM-1", 0, Arm.CORRUPTED),
        ("ITEM-2", 0, Arm.CLEAN),
        ("ITEM-2", 0, Arm.PLACEBO),
        ("ITEM-2", 0, Arm.CORRUPTED),
        ("ITEM-1", 1, Arm.CLEAN),
        ("ITEM-1", 1, Arm.PLACEBO),
        ("ITEM-1", 1, Arm.CORRUPTED),
        ("ITEM-2", 1, Arm.CLEAN),
        ("ITEM-2", 1, Arm.PLACEBO),
        ("ITEM-2", 1, Arm.CORRUPTED),
    ]


def test_unparseable_stays_in_denominator_and_primary_is_corrupted_vs_placebo() -> None:
    rows = [
        _record(arm=Arm.CLEAN, answer=AnswerToken.C),
        _record(arm=Arm.PLACEBO, answer=AnswerToken.UNPARSEABLE),
        _record(arm=Arm.CORRUPTED, answer=AnswerToken.C),
    ]
    report = build_reports(rows, expected_per_arm=1)[0]
    assert report.p_propagation_clean == 1.0  # no clean-arm filtering
    assert report.p_propagation_placebo == 0.0
    assert report.effect_corrupted_vs_placebo == 1.0
    assert report.effect_placebo_vs_clean == -1.0


def test_behavior_classification_excludes_supplied_prefix() -> None:
    record = _record(
        arm=Arm.CORRUPTED,
        answer=AnswerToken.C,
        continuation="4. Tiếp tục lập luận mà không nhận xét thêm.",
    )
    data = record.model_dump()
    prefix = "1. a\n2. b\n3. Đây là bước sai.\n"
    data.update(
        reasoning_prefix=prefix,
        assistant_prefill="<think>\n" + prefix,
        full_reasoning=prefix + record.reasoning_continuation,
    )
    assert (
        classify_corrupted_behavior(InterventionDrawRecord.model_validate(data))
        == "propagated"
    )


class _CalibrationClient:
    def __init__(self, response: object) -> None:
        self.response = response

    def complete(self, _payload: object) -> object:
        return self.response


@pytest.mark.parametrize(
    ("model", "provider", "reasoning", "tokens", "failure"),
    [
        ("other/model", "DeepInfra", "4. reasoning", 2, "model_mismatch"),
        ("deepseek/deepseek-r1-0528", "Other", "4. reasoning", 2, "provider_mismatch"),
        ("deepseek/deepseek-r1-0528", "DeepInfra", "", 2, "missing_reasoning"),
        (
            "deepseek/deepseek-r1-0528",
            "DeepInfra",
            "4. reasoning",
            0,
            "zero_reasoning_tokens",
        ),
    ],
)
def test_calibration_failures(
    tmp_path: Path,
    model: str,
    provider: str,
    reasoning: str,
    tokens: int,
    failure: str,
) -> None:
    from vnedu_cot.providers.base import NormalizedResponse

    response = NormalizedResponse(
        model=model,
        provider=provider,
        content="Đáp án: A",
        reasoning=reasoning,
        reasoning_details=(),
        usage={"completion_tokens_details": {"reasoning_tokens": tokens}},
        cost=None,
        request_id="req",
        finish_reason="stop",
        raw={},
    )
    draw = ScheduledDraw("draw", "ITEM-1", _chain().chain_id, Arm.CLEAN, 0, 1)
    result = _calibrate(
        protocol=_protocol(tmp_path),
        run_id="run",
        chain=_chain(),
        draw=draw,
        client=_CalibrationClient(response),  # type: ignore[arg-type]
    )
    assert not result.passed
    assert failure in result.failures

