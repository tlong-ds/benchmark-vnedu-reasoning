"""Create a canonical-chain template from the base chains.

Each base chain keeps the model's full reasoning (``raw_reasoning`` and
``response_text``) but no steps. The template turns every base chain into a
canonical-chain record with four empty typed steps (retrieval, inference,
computation, evaluation), keeping the reasoning alongside as the source to
condense. Fill in each step's ``text``; the filled file has the format of
``data/canonical_chains.jsonl``.

Run: uv run python scripts/make_canonical_template.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vnedu_cot.schemas.core import StepType

STEP_TYPES = (
    StepType.RETRIEVAL,
    StepType.INFERENCE,
    StepType.COMPUTATION,
    StepType.EVALUATION,
)


def to_template(base: dict[str, object]) -> dict[str, object]:
    chain_id = str(base["chain_id"]).removesuffix("::base") + "::canonical"
    steps = [
        {"i": i, "type": step_type.value, "text": ""}
        for i, step_type in enumerate(STEP_TYPES, start=1)
    ]
    # Canonical steps are written in Vietnamese, whatever the raw reasoning mixed in.
    return {**base, "chain_id": chain_id, "reasoning_language": "vi", "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=Path("data/base_chains.jsonl"))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/canonical_chains_template.jsonl")
    )
    args = parser.parse_args()

    with args.base.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_template(row), ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} template(s) to {args.out}")


if __name__ == "__main__":
    main()
