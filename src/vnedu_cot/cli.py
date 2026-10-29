"""``vnedu-cot intervention run|validate|report``: the three-arm intervention.

    uv run vnedu-cot intervention run --dry-run

The default protocol is ``configs/experiments/deepseek_r1_intervention.toml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vnedu_cot.reporting.intervention import write_intervention_report
from vnedu_cot.stages.intervention.protocol import InterventionProtocol
from vnedu_cot.stages.intervention.runner import run_protocol
from vnedu_cot.stages.intervention.validation import validate_run_directory
from vnedu_cot.storage import INTERVENTIONS_DIR

DEFAULT_PROTOCOL = Path("configs/experiments/deepseek_r1_intervention.toml")


def _find_run(run_id: str, output_root: Path) -> Path:
    matches = [path for path in output_root.glob(f"*/{run_id}") if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one run directory for {run_id!r}, found {len(matches)}"
        )
    return matches[0]


def _samples_from_manifest(run_dir: Path) -> int:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    counts: dict[tuple[str, str], int] = {}
    for row in manifest["schedule"]:
        key = (str(row["chain_id"]), str(row["arm"]))
        counts[key] = counts.get(key, 0) + 1
    values = set(counts.values())
    if len(values) != 1:
        raise ValueError("manifest schedule has inconsistent per-arm sample counts")
    return next(iter(values))


def _run(args: argparse.Namespace) -> int:
    protocol = InterventionProtocol.load(args.protocol, repo_root=Path.cwd())
    summary = run_protocol(
        protocol,
        run_id=args.run_id,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "protocol_id": protocol.protocol_id,
                "run_id": summary.run_id,
                "eligible_chains": summary.chain_count,
                "arms": [arm.value for arm in protocol.arms],
                "samples_per_arm": protocol.samples_per_arm,
                "scheduled_draws": summary.scheduled_draws,
                "completed_draws": summary.completed_draws,
                "incomplete_draws": summary.incomplete_draws,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    run_dir = _find_run(args.run_id, args.output_root)
    errors = validate_run_directory(run_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"valid intervention run: {run_dir}")
    return 0


def _report(args: argparse.Namespace) -> int:
    run_dir = _find_run(args.run_id, args.output_root)
    errors = validate_run_directory(run_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    count = write_intervention_report(
        run_dir, expected_per_arm=_samples_from_manifest(run_dir)
    )
    print(f"wrote {count} confirmatory report rows to {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vnedu-cot")
    commands = parser.add_subparsers(dest="stage", required=True)
    intervention = commands.add_parser("intervention")
    sub = intervention.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    run.add_argument("--run-id")
    run.add_argument("--output-root", type=Path, default=INTERVENTIONS_DIR)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=_run)

    validate = sub.add_parser("validate")
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--output-root", type=Path, default=INTERVENTIONS_DIR)
    validate.set_defaults(func=_validate)

    report = sub.add_parser("report")
    report.add_argument("--run-id", required=True)
    report.add_argument("--output-root", type=Path, default=INTERVENTIONS_DIR)
    report.set_defaults(func=_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

