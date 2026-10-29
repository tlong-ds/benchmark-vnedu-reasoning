"""Typed JSONL I/O and the repository's fixed file locations.

``data/`` holds the shipped corpus and chains; stage commands write new runs
to ``outputs/``.
"""

import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vnedu_cot.schemas.core import RecordModel
from vnedu_cot.schemas.items import BenchmarkItem

DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR = Path("configs/models")
INTERVENTIONS_DIR = OUTPUTS_DIR / "interventions"

ITEMS_PATH = DATA_DIR / "items.jsonl"
BASE_CHAINS_PATH = DATA_DIR / "base_chains.jsonl"
CORRUPTED_CHAINS_PATH = DATA_DIR / "corrupted_chains.jsonl"

# Where the chain stages write when the pipeline is rerun.
NEW_BASE_CHAINS_PATH = OUTPUTS_DIR / "base_chains.jsonl"
NEW_CANONICAL_CHAINS_PATH = OUTPUTS_DIR / "canonical_chains.jsonl"
NEW_CORRUPTED_CHAINS_PATH = OUTPUTS_DIR / "corrupted_chains.jsonl"

NO_COT_RUNS_PATH = OUTPUTS_DIR / "no_cot_runs.jsonl"
PROBE_RUNS_PATH = OUTPUTS_DIR / "probe_runs.jsonl"


def read_jsonl[T: RecordModel](path: Path, model: type[T]) -> Iterator[T]:
    """Yield one validated record per non-blank line."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line := line.strip():
                yield model.model_validate_json(line)


def write_jsonl[T: RecordModel](path: Path, records: list[T]) -> None:
    """Write validated records in canonical JSONL form."""
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.model_dump_json())
            stream.write("\n")


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file (plain or ``.gz``) into plain dicts."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_items(pair_id: str | None = None) -> list[BenchmarkItem]:
    """Load the corpus, or only the items of one triplet (``pair_id``)."""
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(f"no such file: {ITEMS_PATH}")
    items = list(read_jsonl(ITEMS_PATH, BenchmarkItem))
    if pair_id is None:
        return items
    items = [item for item in items if item.pair_id == pair_id]
    if not items:
        raise ValueError(f"no items with pair_id {pair_id!r} in {ITEMS_PATH}")
    return items
