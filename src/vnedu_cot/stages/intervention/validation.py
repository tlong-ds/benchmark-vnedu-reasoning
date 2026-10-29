"""Check a finished run directory: files, draw counts and checksums."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from vnedu_cot.schemas.intervention import CalibrationRecord, InterventionDrawRecord
from vnedu_cot.storage import sha256


def validate_run_directory(run_dir: Path) -> list[str]:
    errors: list[str] = []
    for name in ("manifest.json", "calibration.json", "draws.jsonl"):
        if not (run_dir / name).exists():
            errors.append(f"missing {name}")
    if errors:
        return errors
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    try:
        calibration_data = json.loads(
            (run_dir / "calibration.json").read_text(encoding="utf-8")
        )
        raw_calibrations = calibration_data.get("calibrations", [calibration_data])
        calibrations = [
            CalibrationRecord.model_validate_json(json.dumps(value))
            for value in raw_calibrations
        ]
        latest_id = calibration_data.get(
            "latest_calibration_id", calibrations[-1].calibration_id
        )
        calibration = next(
            record for record in calibrations if record.calibration_id == latest_id
        )
    except Exception as exc:
        errors.append(f"invalid calibration.json: {exc}")
        return errors
    calibration_ids = {record.calibration_id for record in calibrations}
    if not calibration.passed:
        errors.append(f"calibration failed: {', '.join(calibration.failures)}")
    if calibration.protocol_hash != manifest.get("protocol_hash"):
        errors.append("calibration protocol hash does not match manifest")
    if calibration.run_id != manifest.get("run_id"):
        errors.append("calibration run ID does not match manifest")
    scheduled = {row["draw_id"] for row in manifest["schedule"]}
    records: list[InterventionDrawRecord] = []
    for line_number, line in enumerate(
        (run_dir / "draws.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            records.append(InterventionDrawRecord.model_validate_json(line))
        except Exception as exc:
            errors.append(f"draws.jsonl:{line_number}: {exc}")
    ids = [record.draw_id for record in records]
    duplicates = [draw_id for draw_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate completed draw IDs: {duplicates[:5]}")
    outside = set(ids) - scheduled
    if outside:
        errors.append(f"completed draws outside frozen schedule: {sorted(outside)[:5]}")
    if manifest.get("completed_draws") != len(set(ids)):
        errors.append("manifest completed_draws does not match draws.jsonl")
    for record in records:
        if record.run_id != manifest["run_id"]:
            errors.append(f"{record.draw_id}: run ID mismatch")
        if record.protocol_hash != manifest["protocol_hash"]:
            errors.append(f"{record.draw_id}: protocol hash mismatch")
        if record.calibration_id not in calibration_ids:
            errors.append(f"{record.draw_id}: calibration evidence is missing")
        if not record.protocol_valid:
            errors.append(
                f"{record.draw_id}: protocol failure "
                f"{', '.join(record.protocol_failures)}"
            )
    if manifest.get("state") == "complete" and set(ids) != scheduled:
        errors.append("complete manifest does not contain every scheduled draw")
    sums = run_dir / "SHA256SUMS"
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, name = line.split(maxsplit=1)
            target = run_dir / name.strip()
            if not target.exists():
                errors.append(f"checksum target missing: {name.strip()}")
            elif sha256(target) != expected:
                errors.append(f"checksum mismatch: {name.strip()}")
    return errors


__all__ = ["validate_run_directory"]
