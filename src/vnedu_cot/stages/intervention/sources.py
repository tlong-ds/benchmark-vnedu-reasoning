"""Load the source chains and build their clean, placebo and corrupted prefixes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vnedu_cot.schemas.intervention import Arm
from vnedu_cot.storage import read_json_rows


@dataclass(frozen=True)
class SourceChain:
    chain_id: str
    base_chain_id: str
    item_id: str
    item_version: int
    model: str
    gold: str
    propagation_answer: str
    question: str
    options: dict[str, str]
    prefixes: dict[Arm, str]
    source_record_hash: str


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _render_steps(steps: list[dict[str, Any]]) -> str:
    if len(steps) != 3 or [step.get("i") for step in steps] != [1, 2, 3]:
        raise ValueError("intervention prefix must contain steps 1, 2, and 3")
    texts: list[str] = []
    for step in steps:
        text = str(step.get("text") or "")
        if not text.strip():
            raise ValueError("intervention step text must be non-empty")
        if "<think>" in text or "</think>" in text:
            raise ValueError("intervention step contains reserved think control token")
        texts.append(f"{step['i']}. {text}")
    return "\n".join(texts) + "\n"


def load_source_chains(
    *, rating_key: Path, corrupted_chains: Path, items: Path, model: str
) -> list[SourceChain]:
    key_rows = read_json_rows(rating_key)
    selected = [
        str(row["chain_id"])
        for row in key_rows
        if not bool(row.get("is_catch_trial"))
    ]
    if len(selected) != len(set(selected)):
        raise ValueError("rating key contains duplicate eligible chain IDs")

    corrupted_by_id = {
        str(row["chain_id"]): row for row in read_json_rows(corrupted_chains)
    }
    items_by_id = {str(row["item_id"]): row for row in read_json_rows(items)}
    missing = sorted(set(selected) - corrupted_by_id.keys())
    if missing:
        raise ValueError(
            f"rating-key chains missing from corrupted source: {missing[:5]}"
        )

    result: list[SourceChain] = []
    for chain_id in sorted(selected):
        row = corrupted_by_id[chain_id]
        base = row["base_chain"]
        if base["model"] != model:
            raise ValueError(f"{chain_id} belongs to {base['model']}, expected {model}")
        placebo = row.get("placebo_steps")
        if not placebo:
            raise ValueError(f"{chain_id} has no placebo prefix")
        item_id = str(base["item_id"])
        item = items_by_id.get(item_id)
        if item is None:
            raise ValueError(f"{item_id} missing from items source")
        prefixes = {
            Arm.CLEAN: _render_steps(list(base["steps"][:3])),
            Arm.PLACEBO: _render_steps(list(placebo)),
            Arm.CORRUPTED: _render_steps(list(row["steps"])),
        }
        result.append(
            SourceChain(
                chain_id=chain_id,
                base_chain_id=str(base["chain_id"]),
                item_id=item_id,
                item_version=int(item.get("version", 1)),
                model=str(base["model"]),
                gold=str(base["gold"]),
                propagation_answer=str(row["propagation_answer"]),
                question=str(item["question_vi"]),
                options={str(k): str(v) for k, v in item["options"].items()},
                prefixes=prefixes,
                source_record_hash=canonical_hash(row),
            )
        )
    return result

