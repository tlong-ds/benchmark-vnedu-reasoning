"""Generate one temperature-0 base chain per (item, model).

A chain that fails the validity gate in ``BaseChain`` (answer equals gold,
observed Vietnamese, no meta-planning) is dropped, never resampled. The
gate runs when the record is built. Every call carries its own deterministic
seed. Run:

    uv run python -m vnedu_cot.stages.chains generate --models configs/models/
"""

import argparse
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from vnedu_cot.language_checks import ReasoningLanguage, classify_reasoning_language
from vnedu_cot.providers import (
    ModelCallError,
    ModelConfig,
    call_chat_model,
    draw_seed,
    load_model_configs,
)
from vnedu_cot.schemas.chains import BaseChain
from vnedu_cot.schemas.core import OptionLetter
from vnedu_cot.schemas.items import BenchmarkItem
from vnedu_cot.storage import (
    MODELS_DIR,
    NEW_BASE_CHAINS_PATH,
    load_items,
    read_jsonl,
)

GENERATE_SYSTEM_VI = (
    "Bạn là một học sinh giỏi làm bài kiểm tra trắc nghiệm. "
    "Hãy suy nghĩ và giải bài tập từng bước một cách tự nhiên, chi tiết "
    "và logic.\n\n"
    "Khi trình bày lời giải, mạch suy nghĩ của bạn thường tự nhiên đi qua "
    "các giai đoạn sau -- đây chỉ là gợi ý về mạch suy nghĩ, không phải "
    "khuôn cứng, và bạn không cần đánh số hay gắn nhãn 'Bước 1/2/3/4':\n"
    "- Trước tiên, nhắc lại những dữ kiện đề bài cho và quy tắc/công thức "
    "liên quan.\n"
    "- Tiếp theo, nêu trạng thái hoặc điều kiện trung gian cần thiết để "
    "giải bài toán -- nhưng chưa vội kết luận đáp án ở đây.\n"
    "- Sau đó, hãy viết rõ ràng, thành một đoạn tách biệt, đúng đoạn lập "
    "luận hoặc phép tính thực sự quyết định đáp án nào đúng, đáp án nào "
    "sai -- đây là đoạn quan trọng nhất trong lời giải.\n"
    "- Cuối cùng, đối chiếu kết quả với các phương án A, B, C, D và đưa "
    "ra kết luận.\n\n"
    "Ở dòng cuối cùng, hãy ghi rõ đáp án lựa chọn theo đúng cú pháp:\n"
    "Đáp án: <A|B|C|D>"
)

GENERATE_TEMPLATE_VI = (
    "Giải bài trắc nghiệm sau từng bước, sau đó đưa ra kết luận và "
    "đáp án cuối cùng.\n\n"
    "Câu hỏi: {question_vi}\n\n{options_block}\n"
)

_ANSWER_PATTERNS = (
    r"(?i)(?:đáp án|đáp án là|chọn đáp án|chọn|kết luận|answer|final answer)"
    r"[:\s*]+([A-D])\b",
    r"(?i)\b([A-D])\s*là đáp án\b",
    r'"answer"\s*:\s*"?([A-Da-d])"?',
)

_DROP_ERRORS = (ValidationError, ValueError, KeyError, TypeError, ModelCallError)


def _render_generate_prompt(item: BenchmarkItem) -> tuple[str, str]:
    options_block = "\n".join(
        f"{letter.value}. {text}" for letter, text in item.options.items()
    )
    return GENERATE_SYSTEM_VI, GENERATE_TEMPLATE_VI.format(
        question_vi=item.question_vi, options_block=options_block
    )


def _parse_generated_answer(raw: str, default_gold: OptionLetter) -> OptionLetter:
    for pattern in _ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, raw))
        if matches:
            return OptionLetter(matches[-1].group(1).upper())
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    for line in reversed(lines[-3:]):
        if match := re.search(r"\b([A-D])\b", line):
            return OptionLetter(match.group(1).upper())
    return default_gold


def _generate_one(item: BenchmarkItem, model: ModelConfig) -> BaseChain:
    system, user = _render_generate_prompt(item)
    response = call_chat_model(
        user,
        model=model.model_id,
        gateway=model.api_gateway,
        system_prompt=system,
        temperature=0.0,
        max_tokens=model.max_output_tokens,
        seed=draw_seed(item.item_id, model.model_id, "base", 0),
        reasoning_effort="high",
        provider=model.provider,
        allow_fallbacks=model.allow_fallbacks,
    )
    content = response.content or ""
    base_answer = _parse_generated_answer(
        content or (response.reasoning or ""), item.gold
    )

    trace = response.reasoning or content
    observed = classify_reasoning_language(trace) if trace else ReasoningLanguage.VI
    if observed is ReasoningLanguage.OTHER:
        observed = ReasoningLanguage.VI

    return BaseChain(
        chain_id=f"{item.item_id}::{model.model_id}::base",
        item_id=item.item_id,
        model=model.model_id,
        steps=(),
        base_answer=base_answer,
        gold=item.gold,
        reasoning_language=observed,
        raw_reasoning=response.reasoning,
        response_text=content,
    )


def _store(chain: BaseChain, out: TextIO) -> None:
    out.write(chain.model_dump_json() + "\n")
    out.flush()
    status = (
        "MATCH_GOLD"
        if chain.base_answer == chain.gold
        else f"ANSWER_{chain.base_answer.value}"
    )
    print(f"STORED {chain.item_id} x {chain.model} [{status}]", flush=True)


def _select_items(args: argparse.Namespace) -> list[BenchmarkItem] | None:
    try:
        items = load_items(args.triplet)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return None
    return items[: args.limit]


def _run_sequential(
    jobs: list[tuple[BenchmarkItem, ModelConfig]], sleep: float, out: TextIO
) -> int:
    dropped = 0
    for idx, (item, model) in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {model.model_id} for {item.item_id}...", flush=True)
        try:
            _store(_generate_one(item, model), out)
        except _DROP_ERRORS as exc:
            dropped += 1
            print(f"DROP {item.item_id} x {model.model_id}: {exc}", file=sys.stderr)
        if sleep > 0:
            time.sleep(sleep)
    return dropped


def _run_parallel(
    jobs: list[tuple[BenchmarkItem, ModelConfig]], workers: int, out: TextIO
) -> int:
    dropped = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_generate_one, i, m): (i, m) for i, m in jobs}
        for future in as_completed(futures):
            item, model = futures[future]
            try:
                _store(future.result(), out)
            except _DROP_ERRORS as exc:
                dropped += 1
                print(f"DROP {item.item_id} x {model.model_id}: {exc}", file=sys.stderr)
    return dropped


def _cmd_generate(args: argparse.Namespace) -> int:
    items = _select_items(args)
    if items is None:
        return 2
    models = load_model_configs(args.models)
    if not models:
        print(f"no models in {args.models}", file=sys.stderr)
        return 2

    done: set[tuple[str, str]] = set()
    if NEW_BASE_CHAINS_PATH.exists():
        done = {
            (c.item_id, c.model) for c in read_jsonl(NEW_BASE_CHAINS_PATH, BaseChain)
        }
    jobs = [
        (item, model)
        for item in items
        for model in models
        if (item.item_id, model.model_id) not in done
    ]

    NEW_BASE_CHAINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NEW_BASE_CHAINS_PATH.open("a", encoding="utf-8") as out:
        run: Callable[[], int] = (
            (lambda: _run_sequential(jobs, args.sleep, out))
            if args.concurrency == 1 or args.sleep > 0
            else (lambda: _run_parallel(jobs, args.concurrency, out))
        )
        dropped = run()
    print(f"done: {len(jobs) - dropped} chains written, {dropped} dropped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="write outputs/base_chains.jsonl")
    generate.add_argument("--models", type=Path, default=MODELS_DIR)
    generate.add_argument("--concurrency", type=int, default=4)
    generate.add_argument("--sleep", type=float, default=0.0, help="seconds per call")
    source = generate.add_mutually_exclusive_group()
    source.add_argument("--limit", type=int, default=None)
    source.add_argument("--triplet", default=None, help="one pair_id, e.g. S5-HS-01")
    generate.set_defaults(func=_cmd_generate)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
