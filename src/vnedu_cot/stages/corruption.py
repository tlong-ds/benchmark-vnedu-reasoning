"""Step-3 corrupted and placebo edits, built from a curator worksheet.

``sheet`` writes one worksheet row per base chain (the canonical chains if
present). The curator fills in the corrupted step 3, the propagation answer,
the three attestations and the placebo step 3; ``import`` then builds the
corrupted chains. Run:

    uv run python -m vnedu_cot.stages.corruption sheet
    uv run python -m vnedu_cot.stages.corruption import

The worksheet defaults to ``outputs/corruption_worksheet.csv``.
"""

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from vnedu_cot.language_checks import states_conclusion_directly
from vnedu_cot.schemas.chains import (
    CORRUPTED_PREFIX_LENGTH,
    CORRUPTED_STEP_INDEX,
    BaseChain,
    CorruptedChain,
    CorruptedStep,
)
from vnedu_cot.schemas.core import OptionLetter
from vnedu_cot.storage import (
    NEW_BASE_CHAINS_PATH,
    NEW_CANONICAL_CHAINS_PATH,
    NEW_CORRUPTED_CHAINS_PATH,
    read_jsonl,
    write_jsonl,
)

DEFAULT_WORKSHEET = Path("outputs/corruption_worksheet.csv")

_WORKSHEET_FIELDS = [
    "item_id", "model",
    "step_1_text", "step_2_text", "step_3_text_original", "step_4_text",
    "step_3_corrupted", "propagation_answer",
    "propagation_unique", "no_world_knowledge_shortcut", "option_length_balanced",
    "rationale_vi", "step_3_placebo",
]  # fmt: skip


def _source_chains() -> list[BaseChain] | None:
    path = (
        NEW_CANONICAL_CHAINS_PATH
        if NEW_CANONICAL_CHAINS_PATH.exists()
        else NEW_BASE_CHAINS_PATH
    )
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return None
    return [c for c in read_jsonl(path, BaseChain) if c.steps]


def _attest(row: Mapping[str, str], field: str) -> Literal[True]:
    if row[field].strip().upper() != "TRUE":
        raise ValueError(f"attestation {field} is not TRUE")
    return True


def _prefix(
    base: BaseChain, step_3: str, *, corrupted: bool
) -> tuple[CorruptedStep, ...]:
    return tuple(
        CorruptedStep(
            i=i,
            type=base.steps[i - 1].type,
            text=step_3 if i == CORRUPTED_STEP_INDEX else base.steps[i - 1].text,
            corrupted=corrupted and i == CORRUPTED_STEP_INDEX,
        )
        for i in range(1, CORRUPTED_PREFIX_LENGTH + 1)
    )


def _chain_from_row(row: Mapping[str, str], base: BaseChain) -> CorruptedChain:
    placebo_text = row.get("step_3_placebo", "").strip()
    return CorruptedChain(
        chain_id=f"{base.item_id}::{base.model}::corrupted",
        base_chain=base,
        steps=_prefix(base, row["step_3_corrupted"], corrupted=True),
        propagation_answer=OptionLetter(row["propagation_answer"]),
        propagation_unique=_attest(row, "propagation_unique"),
        no_world_knowledge_shortcut=_attest(row, "no_world_knowledge_shortcut"),
        option_length_balanced=_attest(row, "option_length_balanced"),
        rationale_vi=row["rationale_vi"],
        placebo_steps=(
            _prefix(base, placebo_text, corrupted=False) if placebo_text else None
        ),
    )


def _cross_model_errors(chains: list[CorruptedChain]) -> list[str]:
    """Every model's chain for an item must share the same edits and answer."""

    by_item: dict[str, list[CorruptedChain]] = defaultdict(list)
    for chain in chains:
        by_item[chain.item_id].append(chain)

    errors: list[str] = []
    for item_id, group in by_item.items():
        variants = {
            "step_3_corrupted": {c.steps[2].text for c in group},
            "step_3_placebo": {
                c.placebo_steps[2].text for c in group if c.placebo_steps is not None
            },
            "propagation_answer": {c.propagation_answer.value for c in group},
        }
        errors += [
            f"{item_id}: {name} differs across models ({len(values)} variants)"
            for name, values in variants.items()
            if len(values) > 1
        ]
    return errors


def _cmd_sheet(args: argparse.Namespace) -> int:
    base_chains = _source_chains()
    if base_chains is None:
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_WORKSHEET_FIELDS)
        writer.writeheader()
        for chain in base_chains:
            writer.writerow(
                dict.fromkeys(_WORKSHEET_FIELDS, "")
                | {
                    "item_id": chain.item_id,
                    "model": chain.model,
                    "step_1_text": chain.steps[0].text,
                    "step_2_text": chain.steps[1].text,
                    "step_3_text_original": chain.steps[2].text,
                    "step_4_text": chain.steps[3].text,
                }
            )
    print(f"wrote {len(base_chains)} row(s) to {args.out}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    if not args.sheet.exists():
        print(f"no such file: {args.sheet}", file=sys.stderr)
        return 2
    base_chains = _source_chains()
    if base_chains is None:
        return 2
    base_by_key = {(c.item_id, c.model): c for c in base_chains}

    corrupted: list[CorruptedChain] = []
    skipped = failed = 0
    with args.sheet.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row.get("step_3_corrupted", "").strip():
                skipped += 1
                continue
            key = (row["item_id"], row["model"])
            base = base_by_key.get(key)
            if base is None:
                print(f"SKIP {key}: no matching base chain", file=sys.stderr)
                skipped += 1
                continue
            try:
                chain = _chain_from_row(row, base)
            except (ValidationError, ValueError, KeyError) as exc:
                failed += 1
                print(f"FAILED {key}: {exc}", file=sys.stderr)
                continue
            if states_conclusion_directly(chain.steps[2].text):
                print(
                    f"WARN {key}: corrupted step 3 already states its own "
                    "conclusion; a matching answer may reflect reading, not logic.",
                    file=sys.stderr,
                )
            corrupted.append(chain)

    errors = _cross_model_errors(corrupted)
    if errors:
        for message in errors:
            print(f"ERROR {message}", file=sys.stderr)
        print(f"refusing to write {NEW_CORRUPTED_CHAINS_PATH}", file=sys.stderr)
        return 1

    NEW_CORRUPTED_CHAINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(NEW_CORRUPTED_CHAINS_PATH, corrupted)
    print(f"wrote {len(corrupted)} chain(s), {skipped} skipped, {failed} failed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sheet = sub.add_parser("sheet", help="write a curator worksheet")
    sheet.add_argument("--out", type=Path, default=DEFAULT_WORKSHEET)
    sheet.set_defaults(func=_cmd_sheet)
    import_ = sub.add_parser("import", help="build corrupted chains from the worksheet")
    import_.add_argument("--sheet", type=Path, default=DEFAULT_WORKSHEET)
    import_.set_defaults(func=_cmd_import)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
