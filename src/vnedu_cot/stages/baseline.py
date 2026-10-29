"""Answer-only baseline: ask each item with no reasoning, k times per model.

Draws use counterbalanced cyclic option rotations and are remapped to the
canonical letters; the per-(item, model) accuracy q_i is the paper's
difficulty covariate. Run:

    uv run python -m vnedu_cot.stages.baseline run --dry-run --limit 3
    uv run python -m vnedu_cot.stages.baseline run --models configs/models/ --samples 10
"""

import argparse
import datetime as dt
import hashlib
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from vnedu_cot.providers import ModelConfig, call_chat_model, load_model_configs
from vnedu_cot.schemas.core import AnswerToken, OptionLetter, RunCondition
from vnedu_cot.schemas.items import CHANCE_LEVEL, NO_COT_BAND, BenchmarkItem
from vnedu_cot.schemas.runs import RunRecord
from vnedu_cot.storage import (
    ITEMS_PATH,
    MODELS_DIR,
    NO_COT_RUNS_PATH,
    read_jsonl,
)

# --- prompt and option rotation ------------------------------------------

# The prompt asks for a bare letter: a chatty reply is an unparseable draw, not
# something to coax a letter out of.

NO_COT_SYSTEM_VI = (
    "Bạn là một học sinh làm bài kiểm tra trắc nghiệm. Chỉ trả lời bằng "
    "một chữ cái duy nhất (A, B, C hoặc D). Không giải thích, không trình "
    "bày lời giải, không viết gì thêm ngoài chữ cái đó."
)

NO_COT_TEMPLATE_VI = (
    "Trả lời câu hỏi trắc nghiệm sau đây.\n"
    "\n"
    "Câu hỏi: {question_vi}\n"
    "\n"
    "{options_block}\n"
    "\n"
    "Chỉ ghi đúng một chữ cái (A, B, C hoặc D). Không giải thích, không "
    "trình bày lời giải.\n"
    "Đáp án:"
)

# Draws show the options in a cyclic rotation; answers are mapped back to the
# canonical letters, so stored items and gold stay canonical.
OptionOrder = tuple[OptionLetter, OptionLetter, OptionLetter, OptionLetter]
_OPTION_ORDER: OptionOrder = (
    OptionLetter.A,
    OptionLetter.B,
    OptionLetter.C,
    OptionLetter.D,
)

# Bump when the rotation schedule or remapping changes: it enters the template
# hash, so rows from an older protocol are refused instead of mixed in.
NO_COT_OPTION_ORDER_PROTOCOL = "cyclic_pair_v1"


def option_order_for_rotation(offset: int) -> OptionOrder:
    """Return canonical option identities at displayed positions A--D.

    ``offset=1`` means canonical A is displayed as B, so the displayed order
    is D, A, B, C. Cyclic rotations isolate gold-position effects without also
    changing distractor adjacency arbitrarily.
    """

    normalized = offset % len(_OPTION_ORDER)
    if normalized == 0:
        return _OPTION_ORDER
    rotated = _OPTION_ORDER[-normalized:] + _OPTION_ORDER[:-normalized]
    return rotated[0], rotated[1], rotated[2], rotated[3]


def option_rotation_schedule(
    rotation_key: str, n_samples: int
) -> tuple[OptionOrder, ...]:
    """Build a stable, counterbalanced rotation schedule for one unit.

    A matched triplet passes its shared ``pair_id`` as ``rotation_key`` so all
    three domains, and every model, see the same schedule. Four consecutive
    samples cover all four gold positions exactly once. If ``n_samples`` is
    not divisible by four, the S1--S6 schema index counterbalances which
    position repeats across the canary; other keys use a stable hash.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if not rotation_key.strip():
        raise ValueError("rotation_key must be non-empty")

    schema_match = re.match(r"^S(\d+)(?:-|$)", rotation_key, re.IGNORECASE)
    if schema_match:
        # The six canary triplets therefore start 0,1,2,3,0,1: when N=5,
        # the unavoidable fifth-position repeats are as balanced as possible
        # across the six units (2,2,1,1), rather than left to hash luck.
        start = (int(schema_match.group(1)) - 1) % len(_OPTION_ORDER)
    else:
        digest = hashlib.sha256(rotation_key.encode("utf-8")).digest()
        start = int.from_bytes(digest[:2], "big") % len(_OPTION_ORDER)
    return tuple(
        option_order_for_rotation(start + sample_index)
        for sample_index in range(n_samples)
    )


def no_cot_rotation_key(item: BenchmarkItem) -> str:
    """Use matched-set identity when available, otherwise item identity."""

    return item.pair_id or item.item_id


def render_no_cot_prompt_from_options(
    question_vi: str,
    options: Mapping[OptionLetter, str],
    option_order: OptionOrder = _OPTION_ORDER,
) -> tuple[str, str]:
    """Render a prompt using canonical option identities in display order."""

    if len(option_order) != len(_OPTION_ORDER) or set(option_order) != set(
        _OPTION_ORDER
    ):
        raise ValueError("option_order must be a permutation of A, B, C, D")

    options_block = "\n".join(
        f"{display_letter.value}. {options[canonical_letter]}"
        for display_letter, canonical_letter in zip(
            _OPTION_ORDER, option_order, strict=True
        )
    )
    user = NO_COT_TEMPLATE_VI.format(
        question_vi=question_vi, options_block=options_block
    )
    return NO_COT_SYSTEM_VI, user


def render_no_cot_prompt(
    item: BenchmarkItem, option_order: OptionOrder = _OPTION_ORDER
) -> tuple[str, str]:
    """Render the (system, user) prompt pair for one item's no-CoT pilot."""

    return render_no_cot_prompt_from_options(
        item.question_vi,
        item.options,
        option_order,
    )


def remap_displayed_answer(
    displayed_answer: AnswerToken, option_order: OptionOrder
) -> AnswerToken:
    """Map a displayed A--D response back to the item's canonical letter."""

    if displayed_answer in (AnswerToken.FLAG, AnswerToken.UNPARSEABLE):
        return displayed_answer
    displayed_letter = OptionLetter(displayed_answer.value)
    displayed_index = _OPTION_ORDER.index(displayed_letter)
    return AnswerToken(option_order[displayed_index].value)


def template_hash(template: str) -> str:
    """Short, stable hash identifying which prompt wording produced a run."""

    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


NO_COT_TEMPLATE_HASH = template_hash(
    NO_COT_SYSTEM_VI
    + "\n"
    + NO_COT_TEMPLATE_VI
    + "\noption_order_protocol="
    + NO_COT_OPTION_ORDER_PROTOCOL
)


def rendered_prompt_hash(system: str, user: str) -> str:
    """Hash the exact rendered prompt, including the displayed option order."""

    return hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()


# --- answer parsing ------------------------------------------------------
# Strict: a reply that does not reduce to a clean letter is UNPARSEABLE.

_BARE_LETTER = re.compile(r"^[ABCDabcd]$")
_LABELLED = re.compile(
    r"(?:đáp\s*án\s*(?:là|:)?|là|chọn)\s*[:\-]?\s*([ABCDabcd])\b", re.IGNORECASE
)
_FIRST_LETTER = re.compile(r"^\s*([ABCDabcd])(?:\s|[.,:;\)]|$)", re.IGNORECASE)
_PAREN_LETTER = re.compile(r"\(([ABCDabcd])\)")
_LAST_LINE_LETTER = re.compile(r"^[^A-Za-z]*([ABCDabcd])[^A-Za-z]*$")


def parse_answer(text: str) -> AnswerToken:
    """Extract a single option letter from a model's raw response.

    Tries, in order: a bare letter (the whole response, stripped); a
    leading letter at the start of response; a parenthesized letter;
    a "Đáp án: X" / "Đáp án là X" label anywhere in the text; a letter alone
    on the last non-blank line. Anything else is UNPARSEABLE.
    """

    stripped = text.strip()
    if _BARE_LETTER.match(stripped):
        return AnswerToken(stripped.upper())

    m_first = _FIRST_LETTER.match(text)
    if m_first:
        return AnswerToken(m_first.group(1).upper())

    m_paren = _PAREN_LETTER.search(text)
    if m_paren:
        return AnswerToken(m_paren.group(1).upper())

    labelled = _LABELLED.search(text)
    if labelled:
        return AnswerToken(labelled.group(1).upper())

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last_line_match = _LAST_LINE_LETTER.match(lines[-1])
        if last_line_match:
            return AnswerToken(last_line_match.group(1).upper())

    return AnswerToken.UNPARSEABLE


def build_run_record(
    item: BenchmarkItem,
    model_id: str,
    *,
    temperature: float,
    raw_responses: Iterable[str],
    timestamp: dt.datetime,
    option_orders: Iterable[OptionOrder] | None = None,
    rendered_prompt_hashes: Iterable[str] | None = None,
) -> RunRecord:
    """Turn model-facing responses into a canonical, auditable RunRecord."""

    raw = tuple(raw_responses)
    orders = (
        tuple(option_orders)
        if option_orders is not None
        else tuple(_OPTION_ORDER for _ in raw)
    )
    if len(orders) != len(raw):
        raise ValueError("option_orders must have one entry per raw response")

    displayed = tuple(parse_answer(response) for response in raw)
    sampled = tuple(
        remap_displayed_answer(answer, order)
        for answer, order in zip(displayed, orders, strict=True)
    )

    prompt_hashes = (
        tuple(rendered_prompt_hashes)
        if rendered_prompt_hashes is not None
        else tuple(
            rendered_prompt_hash(*render_no_cot_prompt(item, order)) for order in orders
        )
    )
    if len(prompt_hashes) != len(raw):
        raise ValueError("rendered_prompt_hashes must have one entry per raw response")

    return RunRecord(
        run_id=f"{item.item_id}::{model_id}::no_cot",
        item_id=item.item_id,
        item_version=item.version,
        chain_id=None,
        model=model_id,
        condition=RunCondition.NO_COT,
        n_samples=len(raw),
        temperature=temperature,
        prompt_template_hash=NO_COT_TEMPLATE_HASH,
        raw_responses=raw,
        displayed_answers=displayed,
        sampled_answers=sampled,
        option_orders=orders,
        rendered_prompt_hashes=prompt_hashes,
        answer_distribution=_count(sampled),
        timestamp=timestamp,
    )


def _count(answers: tuple[AnswerToken, ...]) -> dict[AnswerToken, int]:
    counts: dict[AnswerToken, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    return counts


def no_cot_accuracies(
    runs: Iterable[RunRecord], items: dict[str, BenchmarkItem]
) -> dict[tuple[str, str], float]:
    """Fraction of samples matching gold, per (item_id, model).

    An unparseable sample counts in the denominator as a miss -- a
    non-answer is not a correct one, so this stays conservative rather
    than excluding or guessing at unparseable samples.
    """

    result: dict[tuple[str, str], float] = {}
    for run in runs:
        if run.condition is not RunCondition.NO_COT:
            continue
        item = items.get(run.item_id)
        if item is None or run.n_samples == 0:
            continue
        hits = sum(1 for a in run.sampled_answers if a.value == item.gold.value)
        result[(run.item_id, run.model)] = hits / run.n_samples
    return result


# --- difficulty band (per item and model, never pooled across models) ------


def chance_corrected(acc: float) -> float:
    """Guessing-corrected accuracy: (acc - CHANCE_LEVEL) / (1 - CHANCE_LEVEL)."""

    return (acc - CHANCE_LEVEL) / (1 - CHANCE_LEVEL)


def band_ok(acc: float) -> bool:
    """Whether one raw accuracy, chance-corrected, falls inside NO_COT_BAND.

    Takes a single accuracy -- one (item, model) pair -- not a per-model
    map. There is deliberately no item-level roll-up of this band; one
    model finding an item too easy or too hard doesn't disqualify it for
    every other model.
    """

    lo, hi = NO_COT_BAND
    return lo <= chance_corrected(acc) <= hi


def band_flags(
    runs: Iterable[RunRecord], items: dict[str, BenchmarkItem]
) -> dict[tuple[str, str], bool]:
    """band_ok applied to every (item, model) pair with a no-CoT run."""

    return {key: band_ok(acc) for key, acc in no_cot_accuracies(runs, items).items()}


# --- run -------------------------------------------------------------------


def _already_run(runs_path: Path) -> set[tuple[str, str]]:
    if not runs_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for run in read_jsonl(runs_path, RunRecord):
        if (
            run.condition is RunCondition.NO_COT
            and run.prompt_template_hash == NO_COT_TEMPLATE_HASH
            and run.option_orders
        ):
            done.add((run.item_id, run.model))
    return done


def _incompatible_no_cot_runs(runs_path: Path) -> tuple[str, ...]:
    """Return old/fixed-order rows that must not mix with rotated runs."""

    if not runs_path.exists():
        return ()
    return tuple(
        run.run_id
        for run in read_jsonl(runs_path, RunRecord)
        if run.condition is RunCondition.NO_COT
        and (run.prompt_template_hash != NO_COT_TEMPLATE_HASH or not run.option_orders)
    )


@dataclass(frozen=True)
class _NoCotSampleBatch:
    raw_responses: tuple[str, ...]
    option_orders: tuple[OptionOrder, ...]
    rendered_prompt_hashes: tuple[str, ...]


def _sample_answers(
    item: BenchmarkItem, model: ModelConfig, n_samples: int, concurrency: int
) -> _NoCotSampleBatch:
    orders = option_rotation_schedule(no_cot_rotation_key(item), n_samples)
    # DeepSeek R1 is forced to answer with zero reasoning by prefilling an
    # empty <think> block; other models need room for hidden reasoning tokens.
    is_r1 = model.model_id.startswith("deepseek/deepseek-r1")
    prefill = "<think>\n</think>\nĐáp án: " if is_r1 else None
    max_tokens = 32 if is_r1 else 256

    prompts = [render_no_cot_prompt(item, order) for order in orders]
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, n_samples))) as pool:
        futures = [
            pool.submit(
                call_chat_model,
                user,
                model=model.model_id,
                gateway=model.api_gateway,
                system_prompt=system,
                assistant_prefill=prefill,
                temperature=model.temperature,
                max_tokens=max_tokens,
                reasoning_effort=None,
                provider=model.provider,
                allow_fallbacks=model.allow_fallbacks,
            )
            for system, user in prompts
        ]
        responses = [future.result().content for future in futures]

    if any(response is None for response in responses):
        raise RuntimeError("one or more no-CoT samples returned no response")
    return _NoCotSampleBatch(
        raw_responses=tuple(r for r in responses if r is not None),
        option_orders=orders,
        rendered_prompt_hashes=tuple(rendered_prompt_hash(*p) for p in prompts),
    )


def _run_one(
    item: BenchmarkItem, model: ModelConfig, n_samples: int, concurrency: int
) -> RunRecord:
    batch = _sample_answers(item, model, n_samples, concurrency)
    return build_run_record(
        item,
        model.model_id,
        temperature=model.temperature,
        raw_responses=batch.raw_responses,
        option_orders=batch.option_orders,
        rendered_prompt_hashes=batch.rendered_prompt_hashes,
        timestamp=dt.datetime.now(dt.UTC),
    )


def _print_dry_run(items: Sequence[BenchmarkItem], n_samples: int) -> None:
    for item in items:
        print(f"=== {item.item_id} (gold={item.gold.value}) ===")
        schedule = option_rotation_schedule(no_cot_rotation_key(item), n_samples)
        for sample_index, order in enumerate(schedule, start=1):
            system, user = render_no_cot_prompt(item, order)
            mapping = ", ".join(
                f"{display.value}->{canonical.value}"
                for display, canonical in zip(_OPTION_ORDER, order, strict=True)
            )
            print(f"--- sample {sample_index}; display->canonical: {mapping} ---")
            print("SYSTEM:", system)
            print("USER:")
            print(user)
            print()


def _pending_pairs(
    items: Sequence[BenchmarkItem], models: Sequence[ModelConfig], out_path: Path
) -> list[tuple[BenchmarkItem, ModelConfig]] | None:
    """Pairs not yet run, or None if ``out_path`` holds an older protocol."""

    incompatible = _incompatible_no_cot_runs(out_path)
    if incompatible:
        print(
            f"{out_path} contains {len(incompatible)} fixed-order or "
            "older-protocol no-CoT rows. Archive that file before starting "
            f"{NO_COT_OPTION_ORDER_PROTOCOL}; mixed protocols are refused.",
            file=sys.stderr,
        )
        return None
    done = _already_run(out_path)
    return [
        (item, model)
        for item in items
        for model in models
        if (item.item_id, model.model_id) not in done
    ]


def _run_pairs(
    pairs: Sequence[tuple[BenchmarkItem, ModelConfig]],
    out_path: Path,
    samples: int,
    concurrency: int,
) -> float:
    """Append one record per pair; return the overall unparseable rate."""

    unparseable = total = 0
    with out_path.open("a", encoding="utf-8") as out:
        for i, (item, model) in enumerate(pairs, start=1):
            label = f"[{i}/{len(pairs)}] {item.item_id} x {model.model_id}"
            try:
                record = _run_one(item, model, samples, concurrency)
            except Exception as exc:
                print(f"{label}: FAILED, will retry next run ({exc})", file=sys.stderr)
                continue
            out.write(record.model_dump_json() + "\n")
            out.flush()
            bad = record.answer_distribution.get(AnswerToken.UNPARSEABLE, 0)
            unparseable += bad
            total += record.n_samples
            print(f"{label}: {record.n_samples - bad}/{record.n_samples} parsed")
    return unparseable / total if total else 0.0


def _cmd_run(args: argparse.Namespace) -> int:
    if not ITEMS_PATH.exists():
        print(f"no such file: {ITEMS_PATH}", file=sys.stderr)
        return 2
    items = list(read_jsonl(ITEMS_PATH, BenchmarkItem))[: args.limit]

    if args.dry_run:
        _print_dry_run(items, args.samples)
        return 0

    if not args.models.exists():
        print(f"no such path: {args.models}", file=sys.stderr)
        return 2
    models = load_model_configs(args.models)
    if not models:
        print(f"no models in {args.models}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs = _pending_pairs(items, models, args.out)
    if pairs is None:
        return 2
    if not pairs:
        print("nothing to do: every (item, model) pair is already run")
        return 0

    print(f"{len(pairs)} (item, model) pairs to run x {args.samples} samples each")
    rate = _run_pairs(pairs, args.out, args.samples, args.concurrency)
    print(f"done. overall unparseable rate: {rate:.1%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run", help="call every model, write outputs/no_cot_runs.jsonl"
    )
    p_run.add_argument("--models", type=Path, default=MODELS_DIR)
    p_run.add_argument("--samples", type=int, default=10)
    p_run.add_argument("--concurrency", type=int, default=8)
    p_run.add_argument("--limit", type=int, default=None, help="cap item count")
    p_run.add_argument(
        "--out",
        type=Path,
        default=NO_COT_RUNS_PATH,
        help="output path for no-CoT runs (default: outputs/no_cot_runs.jsonl)",
    )
    p_run.add_argument(
        "--dry-run", action="store_true", help="print prompts, call nothing"
    )
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
