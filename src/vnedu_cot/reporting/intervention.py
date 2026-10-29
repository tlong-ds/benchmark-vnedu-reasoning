"""Write the per-item report for a finished run directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from vnedu_cot.analysis.intervention import build_reports
from vnedu_cot.schemas.intervention import InterventionDrawRecord
from vnedu_cot.stages.intervention.runner import _write_checksums


def _load_draws(path: Path) -> list[InterventionDrawRecord]:
    return [
        InterventionDrawRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_intervention_report(run_dir: Path, *, expected_per_arm: int) -> int:
    records = _load_draws(run_dir / "draws.jsonl")
    reports = build_reports(records, expected_per_arm=expected_per_arm)
    report_path = run_dir / "report.jsonl"
    report_path.write_text(
        "".join(report.model_dump_json() + "\n" for report in reports),
        encoding="utf-8",
    )
    csv_path = run_dir / "analysis_ready.csv"
    rows = [report.model_dump(mode="json") for report in reports]
    fields = [
        "protocol_id",
        "run_id",
        "item_id",
        "chain_id",
        "model",
        "n_per_arm",
        "p_propagation_clean",
        "p_propagation_placebo",
        "p_propagation_corrupted",
        "p_gold_clean",
        "p_gold_placebo",
        "p_gold_corrupted",
        "effect_corrupted_vs_placebo",
        "effect_placebo_vs_clean",
        "effect_corrupted_vs_clean",
        "corrupted_behavior_counts",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row["corrupted_behavior_counts"] = json.dumps(
                row["corrupted_behavior_counts"], ensure_ascii=False, sort_keys=True
            )
            writer.writerow({field: row[field] for field in fields})
    _write_checksums(run_dir)
    return len(reports)

