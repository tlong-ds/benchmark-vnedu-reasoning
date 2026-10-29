"""Knowledge probes: each item's factual premises, asked standalone.

A judge model grades the free-text answers. The gate is per (item, model):
an item is knowledge-eligible only if that model answers every probe
correctly. Run:

    uv run python -m vnedu_cot.stages.probes run --dry-run --limit 3
    uv run python -m vnedu_cot.stages.probes run --models configs/models/
"""

import argparse
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import Field

from vnedu_cot.providers import ModelConfig, call_chat_model, load_model_configs
from vnedu_cot.schemas.core import RecordModel
from vnedu_cot.schemas.items import BenchmarkItem, KnowledgeProbe
from vnedu_cot.storage import MODELS_DIR, PROBE_RUNS_PATH, load_items, read_jsonl


class KnowledgeProbeResult(RecordModel):
    """One model's answer to one probe, with a judge model's verdict.

    Answers are free-text Vietnamese, so ``correct`` comes from a judge
    (``judge_model``), not a string match.
    """

    item_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    probe_index: int = Field(ge=0)
    model_answer: str = Field(min_length=1)
    correct: bool
    judge_model: str | None = None


# --- prompts -----------------------------------------------------------------
# The reference answer reaches the judge only, never the probed model.

PROBE_SYSTEM_VI = (
    "Bạn đang trả lời một câu hỏi kiến thức ngắn. Hãy trả lời trực tiếp "
    "và ngắn gọn, chỉ nêu đúng thông tin được hỏi. Không giải thích dài "
    "dòng, không nhắc lại câu hỏi."
)

PROBE_TEMPLATE_VI = "Câu hỏi: {q}\n\nTrả lời ngắn gọn:"


def render_probe_prompt(probe: KnowledgeProbe) -> tuple[str, str]:
    """Render the (system, user) prompt pair for one standalone probe."""

    return PROBE_SYSTEM_VI, PROBE_TEMPLATE_VI.format(q=probe.q)


JUDGE_SYSTEM_VI = (
    "Bạn là người chấm bài. Bạn nhận một câu hỏi kiến thức, đáp án tham "
    "chiếu, và câu trả lời của một mô hình. Hãy quyết định câu trả lời đó "
    "có đúng về mặt nội dung hay không. Chấp nhận cách diễn đạt khác nhau "
    "miễn là nội dung đúng. Chỉ trả lời bằng một từ duy nhất: ĐÚNG hoặc SAI."
)

JUDGE_TEMPLATE_VI = (
    "Câu hỏi: {q}\n"
    "\n"
    "Đáp án tham chiếu: {reference}\n"
    "\n"
    "Câu trả lời của mô hình: {answer}\n"
    "\n"
    "Câu trả lời của mô hình có đúng nội dung không? "
    "Chỉ ghi ĐÚNG hoặc SAI.\n"
    "Kết luận:"
)

_JUDGE_CORRECT = re.compile(r"^\W*(ĐÚNG|DUNG|ĐÚNG\.)\W*$", re.IGNORECASE)
_JUDGE_INCORRECT = re.compile(r"^\W*(SAI|SAI\.)\W*$", re.IGNORECASE)


def render_judge_prompt(probe: KnowledgeProbe, model_answer: str) -> tuple[str, str]:
    """Render the (system, user) prompt pair grading one probe answer."""

    user = JUDGE_TEMPLATE_VI.format(
        q=probe.q, reference=probe.a, answer=model_answer.strip()
    )
    return JUDGE_SYSTEM_VI, user


def parse_judge_verdict(text: str) -> bool | None:
    """Parse a judge reply into pass/fail, or None if unreadable.

    An unreadable verdict is recorded as a fail, but None lets the caller
    count it: many of them mean the judge prompt is broken.
    """

    stripped = text.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for candidate in (stripped, lines[-1] if lines else ""):
        if _JUDGE_CORRECT.match(candidate):
            return True
        if _JUDGE_INCORRECT.match(candidate):
            return False
    return None


# --- aggregation -----------------------------------------------------------


def knowledge_scores(
    results: Iterable[KnowledgeProbeResult],
) -> dict[tuple[str, str], float]:
    """Fraction of probes answered correctly, per (item_id, model)."""

    buckets: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        buckets[(r.item_id, r.model)].append(r.correct)
    return {key: sum(flags) / len(flags) for key, flags in buckets.items()}


def knowledge_ok(
    results: Iterable[KnowledgeProbeResult],
) -> dict[tuple[str, str], bool]:
    """Whether the model passed every probe of the item (no tolerance).

    A pair missing from the map has no probe data and has not passed.
    """

    return {key: score == 1.0 for key, score in knowledge_scores(results).items()}


def knowledge_eligible(
    item: BenchmarkItem, model: str, results_ok: dict[tuple[str, str], bool]
) -> bool:
    """The knowledge gate for one pair; an item with no probes passes vacuously."""

    if len(item.knowledge_probes) == 0:
        return True
    return results_ok.get((item.item_id, model), False)


# --- run ---------------------------------------------------------------------


def _already_run(runs_path: Path) -> set[tuple[str, str, int]]:
    """(item_id, model, probe_index) triples already recorded on disk."""

    if not runs_path.exists():
        return set()
    return {
        (r.item_id, r.model, r.probe_index)
        for r in read_jsonl(runs_path, KnowledgeProbeResult)
    }


DEFAULT_JUDGE_MODEL = "deepseek/deepseek-chat"
# Recorded in place of a judge for a blank answer, which fails without judging.
EMPTY_RESPONSE_JUDGE = "empty_response_rule"
EMPTY_RESPONSE_ANSWER = "(empty response)"


def _blank_result(
    item: BenchmarkItem, probe_index: int, model: ModelConfig
) -> KnowledgeProbeResult:
    """A blank answer fails by rule: a judge shown only the reference passes it."""

    return KnowledgeProbeResult(
        item_id=item.item_id,
        model=model.model_id,
        probe_index=probe_index,
        model_answer=EMPTY_RESPONSE_ANSWER,
        correct=False,
        judge_model=EMPTY_RESPONSE_JUDGE,
    )


def _ask(probe: KnowledgeProbe, model: ModelConfig) -> str:
    system, user = render_probe_prompt(probe)
    return call_chat_model(
        user,
        model=model.model_id,
        gateway=model.api_gateway,
        system_prompt=system,
        temperature=model.temperature,
        max_tokens=model.max_output_tokens or 2048,
        reasoning_effort="low" if "r1" in model.model_id.lower() else None,
        provider=model.provider,
        allow_fallbacks=model.allow_fallbacks,
    ).content


def _judge(probe: KnowledgeProbe, answer: str, judge_model: str) -> bool | None:
    system, user = render_judge_prompt(probe, answer)
    reply = call_chat_model(
        user,
        model=judge_model,
        system_prompt=system,
        temperature=0.0,
        max_tokens=256,
        reasoning_effort="low" if "r1" in judge_model.lower() else None,
    ).content
    return parse_judge_verdict(reply)


def _probe_one(
    item: BenchmarkItem,
    probe_index: int,
    probe: KnowledgeProbe,
    model: ModelConfig,
    judge_model: str,
) -> tuple[KnowledgeProbeResult, bool]:
    """Ask one probe of one model and grade it. Returns (result, parsed)."""

    answer = _ask(probe, model)
    if not answer.strip():
        return _blank_result(item, probe_index, model), True
    verdict = _judge(probe, answer, judge_model)
    result = KnowledgeProbeResult(
        item_id=item.item_id,
        model=model.model_id,
        probe_index=probe_index,
        model_answer=answer.strip(),
        correct=verdict is True,
        judge_model=judge_model,
    )
    return result, verdict is not None


def _print_dry_run(items: Sequence[BenchmarkItem]) -> None:
    n_probes = 0
    for item in items:
        for probe_index, probe in enumerate(item.knowledge_probes):
            system, user = render_probe_prompt(probe)
            print(f"=== {item.item_id} probe {probe_index} ===")
            print("SYSTEM:", system)
            print("USER:")
            print(user)
            print(f"REFERENCE (judge only, never sent to the probed model): {probe.a}")
            print()
            n_probes += 1
    n_no_probes = sum(1 for it in items if not it.knowledge_probes)
    print(f"{n_probes} probe(s) across {len(items)} item(s).")
    if n_no_probes:
        print(
            f"note: {n_no_probes} item(s) have no knowledge_probes and will be "
            "skipped -- they cannot pass Control 7 until probes are written."
        )


def _pending_jobs(
    items: Sequence[BenchmarkItem], models: Sequence[ModelConfig]
) -> list[tuple[BenchmarkItem, int, KnowledgeProbe, ModelConfig]]:
    done = _already_run(PROBE_RUNS_PATH)
    return [
        (item, probe_index, probe, model)
        for item in items
        for probe_index, probe in enumerate(item.knowledge_probes)
        for model in models
        if (item.item_id, model.model_id, probe_index) not in done
    ]


def _run_jobs(
    jobs: Sequence[tuple[BenchmarkItem, int, KnowledgeProbe, ModelConfig]],
    judge_model: str,
    concurrency: int,
) -> tuple[int, int, int]:
    """Append results as they finish; return (done, correct, unreadable)."""

    n_done = n_correct = n_unreadable = 0
    PROBE_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (
        PROBE_RUNS_PATH.open("a", encoding="utf-8") as out,
        ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool,
    ):
        futures = {
            pool.submit(_probe_one, item, index, probe, model, judge_model): (
                item,
                index,
                model,
            )
            for item, index, probe, model in jobs
        }
        for future in as_completed(futures):
            item, index, model = futures[future]
            label = f"{item.item_id} probe {index} x {model.model_id}"
            try:
                result, parsed = future.result()
            except Exception as exc:
                print(f"{label}: FAILED, will retry next run ({exc})", file=sys.stderr)
                continue
            out.write(result.model_dump_json() + "\n")
            out.flush()
            n_done += 1
            n_correct += result.correct
            if not parsed:
                n_unreadable += 1
                print(
                    f"{label}: judge verdict unreadable, recorded as INCORRECT",
                    file=sys.stderr,
                )
            verdict = "correct" if result.correct else "incorrect"
            print(f"[{n_done}/{len(jobs)}] {label}: {verdict}")
    return n_done, n_correct, n_unreadable


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        items = load_items(args.triplet)[: args.limit]
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.dry_run:
        _print_dry_run(items)
        return 0

    if not args.models.exists():
        print(f"no such path: {args.models}", file=sys.stderr)
        return 2
    models = load_model_configs(args.models)
    if not models:
        print(f"no models in {args.models}", file=sys.stderr)
        return 2

    jobs = _pending_jobs(items, models)
    if not jobs:
        print("nothing to do: every (item, model, probe) is already run")
        return 0
    print(f"{len(jobs)} (item, probe, model) job(s), judged by {args.judge_model}")

    n_done, n_correct, n_unreadable = _run_jobs(
        jobs, args.judge_model, args.concurrency
    )
    rate = n_correct / n_done if n_done else 0.0
    print(f"done. {n_done} probe result(s), {rate:.1%} correct")
    if n_unreadable:
        print(
            f"warning: {n_unreadable} judge verdict(s) were unreadable and recorded "
            "as incorrect; check the judge model before trusting this run's gate"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="ask every probe of every model and grade it")
    run.add_argument(
        "--models",
        type=Path,
        default=MODELS_DIR,
        help="ModelConfig file or directory (default: configs/models/)",
    )
    run.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"model that grades probe answers (default: {DEFAULT_JUDGE_MODEL})",
    )
    run.add_argument(
        "--triplet",
        default=None,
        help="restrict to the items of one triplet, given its pair_id (e.g. S1-HS-01)",
    )
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--limit", type=int, default=None, help="cap item count")
    run.add_argument(
        "--dry-run", action="store_true", help="print prompts, call nothing"
    )
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
