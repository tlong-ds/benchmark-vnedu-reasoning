from __future__ import annotations

import json
from pathlib import Path

from vnedu_cot.providers.base import NormalizedResponse
from vnedu_cot.reporting.intervention import write_intervention_report
from vnedu_cot.schemas.intervention import Arm
from vnedu_cot.stages.intervention.protocol import InterventionProtocol
from vnedu_cot.stages.intervention.runner import run_protocol
from vnedu_cot.stages.intervention.validation import validate_run_directory


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    item = {
        "item_id": "ITEM-1",
        "version": 1,
        "question_vi": "Một cộng một bằng bao nhiêu?",
        "options": {"A": "2", "B": "3", "C": "4", "D": "5"},
    }
    base_steps = [
        {"i": 1, "type": "retrieval", "text": "Ta có hai số 1."},
        {"i": 2, "type": "inference", "text": "Cần cộng hai số."},
        {"i": 3, "type": "computation", "text": "1 + 1 = 2."},
        {"i": 4, "type": "evaluation", "text": "Chọn A."},
    ]
    corrupted = {
        "chain_id": "ITEM-1::deepseek/deepseek-r1-0528::corrupted",
        "base_chain": {
            "chain_id": "ITEM-1::deepseek/deepseek-r1-0528::canonical",
            "item_id": "ITEM-1",
            "model": "deepseek/deepseek-r1-0528",
            "steps": base_steps,
            "gold": "A",
        },
        "steps": [*base_steps[:2], {**base_steps[2], "text": "1 + 1 = 4."}],
        "placebo_steps": [
            *base_steps[:2],
            {**base_steps[2], "text": "Cộng 1 với 1 thu được 2."},
        ],
        "propagation_answer": "C",
    }
    key = {
        "rating_id": "R1",
        "chain_id": corrupted["chain_id"],
        "is_catch_trial": False,
    }
    paths = (
        tmp_path / "rating.jsonl",
        tmp_path / "chains.jsonl",
        tmp_path / "items.jsonl",
    )
    for path, value in zip(paths, (key, corrupted, item), strict=True):
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


def _protocol(tmp_path: Path) -> InterventionProtocol:
    rating, chains, items = _write_sources(tmp_path)
    protocol_path = tmp_path / "protocol.toml"
    protocol_path.write_text("fixture\n", encoding="utf-8")
    return InterventionProtocol(
        protocol_id="integration",
        schema_version="2.0",
        model="deepseek/deepseek-r1-0528",
        gateway="openrouter",
        provider="DeepInfra",
        allow_fallbacks=False,
        samples_per_arm=1,
        temperature=0.7,
        top_p=1.0,
        reasoning_effort="high",
        max_output_tokens=8192,
        arms=(Arm.CLEAN, Arm.PLACEBO, Arm.CORRUPTED),
        concurrency=3,
        rating_key=rating,
        corrupted_chains=chains,
        items=items,
        source_path=protocol_path,
    )


class _FakeClient:
    def __init__(self, *, fail_call: int | None = None) -> None:
        self.calls = 0
        self.fail_call = fail_call

    def complete(self, payload: dict[str, object]) -> NormalizedResponse:
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("simulated interruption")
        assert payload["reasoning"] == {"effort": "high"}
        assert payload["include_reasoning"] is True
        return NormalizedResponse(
            model="deepseek/deepseek-r1-0528",
            provider="DeepInfra",
            content="Lời giải cuối cùng.\nĐáp án: A",
            reasoning="4. Đối chiếu với các phương án và kết luận.",
            reasoning_details=({"type": "reasoning.text", "text": "4. kết luận"},),
            usage={"completion_tokens_details": {"reasoning_tokens": 8}},
            cost=0.001,
            request_id=f"req-{self.calls}",
            finish_reason="stop",
            raw={},
        )


def test_fake_transport_three_arms_interrupted_then_resumed(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    output = tmp_path / "outputs"
    first_client = _FakeClient(fail_call=2)
    first = run_protocol(
        protocol,
        run_id="resume-me",
        output_root=output,
        client_factory=lambda _sink: first_client,  # type: ignore[arg-type,return-value]
    )
    assert first.scheduled_draws == 3
    assert first.completed_draws == 2
    assert first.incomplete_draws == 1

    second_client = _FakeClient()
    second = run_protocol(
        protocol,
        run_id="resume-me",
        output_root=output,
        client_factory=lambda _sink: second_client,  # type: ignore[arg-type,return-value]
    )
    assert second.completed_draws == 3
    assert second.incomplete_draws == 0
    assert second_client.calls == 2  # mandatory calibration + one pending draw
    assert second.run_dir is not None
    lines = (second.run_dir / "draws.jsonl").read_text().splitlines()
    draw_ids = [json.loads(line)["draw_id"] for line in lines]
    assert len(draw_ids) == len(set(draw_ids)) == 3
    assert validate_run_directory(second.run_dir) == []
    assert write_intervention_report(second.run_dir, expected_per_arm=1) == 1
    assert validate_run_directory(second.run_dir) == []


def test_real_protocol_dry_run_schedules_analysed_set() -> None:
    protocol = InterventionProtocol.load(
        Path("configs/experiments/deepseek_r1_intervention.toml"),
        repo_root=Path.cwd(),
    )
    summary = run_protocol(protocol, dry_run=True)
    assert summary.chain_count == 162
    assert summary.scheduled_draws == 162 * 3 * 20 == 9720

