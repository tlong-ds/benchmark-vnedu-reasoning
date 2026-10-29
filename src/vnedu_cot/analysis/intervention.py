"""Score draws: propagation, gold and behaviour counts per item and arm."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from vnedu_cot.schemas.intervention import (
    Arm,
    InterventionDrawRecord,
    InterventionReport,
)

_ERROR_LANGUAGE = re.compile(
    r"(?i)\b(?:sai|nhầm|mâu thuẫn|không\s+đúng|lỗi|incorrect|wrong|mistake)\b"
)


def classify_corrupted_behavior(record: InterventionDrawRecord) -> str:
    """Classify using generated continuation and final content only."""

    generated = record.reasoning_continuation + "\n" + record.final_response
    flagged = bool(_ERROR_LANGUAGE.search(generated))
    if record.parsed_answer.value == record.propagation_answer:
        return "propagated_flagged" if flagged else "propagated"
    if record.parsed_answer.value == record.gold_answer:
        return "repaired_flagged" if flagged else "repaired_silent"
    return "other"


def _probability(
    rows: list[InterventionDrawRecord], expected: str, denominator: int
) -> float:
    if len(rows) != denominator:
        raise ValueError("fixed denominator requires every scheduled draw")
    return sum(row.parsed_answer.value == expected for row in rows) / denominator


def build_reports(
    records: Iterable[InterventionDrawRecord], *, expected_per_arm: int
) -> list[InterventionReport]:
    grouped: dict[tuple[str, str], dict[Arm, list[InterventionDrawRecord]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        if not record.protocol_valid:
            continue
        grouped[(record.chain_id, record.item_id)][record.arm].append(record)

    reports: list[InterventionReport] = []
    for (chain_id, item_id), arms in sorted(grouped.items()):
        if any(len(arms[arm]) != expected_per_arm for arm in Arm):
            continue
        all_rows = [row for arm in Arm for row in arms[arm]]
        protocol_ids = {row.protocol_id for row in all_rows}
        run_ids = {row.run_id for row in all_rows}
        models = {row.requested_model for row in all_rows}
        golds = {row.gold_answer for row in all_rows}
        propagations = {row.propagation_answer for row in all_rows}
        if not all(
            len(values) == 1
            for values in (protocol_ids, run_ids, models, golds, propagations)
        ):
            raise ValueError(f"inconsistent draw provenance for {chain_id}")
        gold = next(iter(golds))
        propagation = next(iter(propagations))
        p_prop = {
            arm: _probability(arms[arm], propagation, expected_per_arm) for arm in Arm
        }
        p_gold = {
            arm: _probability(arms[arm], gold, expected_per_arm) for arm in Arm
        }
        behavior = Counter(
            classify_corrupted_behavior(row) for row in arms[Arm.CORRUPTED]
        )
        reports.append(
            InterventionReport(
                protocol_id=next(iter(protocol_ids)),
                run_id=next(iter(run_ids)),
                item_id=item_id,
                chain_id=chain_id,
                model=next(iter(models)),
                n_per_arm=expected_per_arm,
                p_propagation_clean=p_prop[Arm.CLEAN],
                p_propagation_placebo=p_prop[Arm.PLACEBO],
                p_propagation_corrupted=p_prop[Arm.CORRUPTED],
                p_gold_clean=p_gold[Arm.CLEAN],
                p_gold_placebo=p_gold[Arm.PLACEBO],
                p_gold_corrupted=p_gold[Arm.CORRUPTED],
                effect_corrupted_vs_placebo=(
                    p_prop[Arm.CORRUPTED] - p_prop[Arm.PLACEBO]
                ),
                effect_placebo_vs_clean=(
                    p_prop[Arm.PLACEBO] - p_prop[Arm.CLEAN]
                ),
                effect_corrupted_vs_clean=(
                    p_prop[Arm.CORRUPTED] - p_prop[Arm.CLEAN]
                ),
                corrupted_behavior_counts=dict(behavior),
            )
        )
    return reports


__all__ = ["build_reports", "classify_corrupted_behavior"]
