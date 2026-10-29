"""Corpus-level checks that no single record's schema can express.

Gold-letter balance, matched-set completeness, cross-file references, the
answer-only pilot and the knowledge-probe gate, over the shipped corpus. Only
structural problems are errors; pilot-gate findings are warnings.
Run: ``uv run python -m vnedu_cot.stages.validation``
"""

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from vnedu_cot.schemas.chains import BaseChain, CorruptedChain
from vnedu_cot.schemas.core import Domain, Level, RecordModel, RunCondition
from vnedu_cot.schemas.items import BenchmarkItem
from vnedu_cot.schemas.runs import RunRecord
from vnedu_cot.stages.annotation import rater_sheet_columns
from vnedu_cot.stages.baseline import band_flags, no_cot_accuracies
from vnedu_cot.stages.probes import (
    KnowledgeProbeResult,
    knowledge_eligible,
    knowledge_ok,
)
from vnedu_cot.storage import (
    BASE_CHAINS_PATH,
    CORRUPTED_CHAINS_PATH,
    ITEMS_PATH,
    NO_COT_RUNS_PATH,
    PROBE_RUNS_PATH,
    read_jsonl,
)

# Fields that must never reach a rater (R6 / Brief SS8.1) -- the sheet's
# column list must be disjoint from all of these, regardless of what the
# item/chain records carry.
_FORBIDDEN_RATER_FIELDS = {
    "is_catch_trial",
    "rationale_vi",
    "domain",
    "level",
    "model",
    "propagation_answer",
    "corrupted_step",
    "propagation_unique",
    "no_world_knowledge_shortcut",
    "option_length_balanced",
}

# R1: no gold letter should exceed this share of a domain x level cell.
_GOLD_BALANCE_THRESHOLD = 0.30


@dataclass
class ValidationReport:
    """Accumulated errors and warnings from a corpus scan."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage: dict[tuple[Level, Domain], int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, rule: str, message: str) -> None:
        self.errors.append(f"[{rule}] {message}")

    def warn(self, rule: str, message: str) -> None:
        self.warnings.append(f"[{rule}] {message}")


@dataclass
class Corpus:
    """The shipped corpus, loaded once and keyed for cross-file checks."""

    items: dict[str, BenchmarkItem]
    base_chains: list[BaseChain]
    corrupted_chains: list[CorruptedChain]
    probe_results: list[KnowledgeProbeResult]
    no_cot_runs: list[RunRecord]


def _read_all[T: RecordModel](path: Path, model: type[T]) -> list[T]:
    return list(read_jsonl(path, model)) if path.exists() else []


def load_corpus() -> Corpus:
    """Read the fixed layout in ``data/`` and ``outputs/``; absent files are empty."""

    return Corpus(
        items={i.item_id: i for i in _read_all(ITEMS_PATH, BenchmarkItem)},
        base_chains=_read_all(BASE_CHAINS_PATH, BaseChain),
        corrupted_chains=_read_all(CORRUPTED_CHAINS_PATH, CorruptedChain),
        probe_results=_read_all(PROBE_RUNS_PATH, KnowledgeProbeResult),
        no_cot_runs=_read_all(NO_COT_RUNS_PATH, RunRecord),
    )


def check_r6_firewall(report: ValidationReport) -> None:
    """R6: the rater sheet's column set must never include a blinding field.

    This does not depend on any data directory -- it is a property of
    ``annotation.rater_sheet_columns`` itself, checked unconditionally so a
    future column addition to that allowlist cannot silently reintroduce a
    leak.
    """

    leaked = set(rater_sheet_columns()) & _FORBIDDEN_RATER_FIELDS
    if leaked:
        report.error(
            "R6", f"rater sheet columns leak blinding fields: {sorted(leaked)}"
        )


def check_r1_gold_balance(report: ValidationReport, corpus: Corpus) -> None:
    """R1: no option letter should be gold on >30% of a domain x level cell."""

    cells: dict[tuple[Level, Domain], list[BenchmarkItem]] = defaultdict(list)
    for item in corpus.items.values():
        cells[(item.level, item.domain)].append(item)

    for (level, domain), cell_items in cells.items():
        total = len(cell_items)
        counts = Counter(i.gold for i in cell_items)
        for letter, n in counts.items():
            if n / total <= _GOLD_BALANCE_THRESHOLD:
                continue
            schema_ids = sorted(
                {i.schema_id or "(standalone)" for i in cell_items if i.gold == letter}
            )
            report.warn(
                "R1",
                f"{domain.value}/{level.value}: gold={letter.value} on "
                f"{n}/{total} items ({n / total:.0%}), from schema_ids {schema_ids}",
            )


_INTERPRETIVE_DOMAINS = {Domain.LITERATURE, Domain.HISTORY}


def _check_triplet(
    report: ValidationReport,
    pair_id: str,
    members: list[BenchmarkItem],
    base_step_types: dict[str, tuple[str, ...]],
) -> None:
    """One triplet: three items, one per slot, with matching level, schema and gold."""

    if len(members) != 3:
        report.error(
            "R2", f"pair_id={pair_id}: has {len(members)} items, expected 3 (dropped)"
        )
        return

    domains = {m.domain for m in members}
    if (
        len(domains) != 3
        or Domain.COMPUTATION not in domains
        or Domain.THEORETICAL not in domains
        or len(domains & _INTERPRETIVE_DOMAINS) != 1
    ):
        report.error(
            "R2",
            f"pair_id={pair_id}: must cover computation, theoretical, and "
            f"exactly one of literature/history -- got {sorted(domains)}",
        )
    if len({m.level for m in members}) != 1:
        report.error("R2", f"pair_id={pair_id}: level differs across the set")
    schema_ids = {m.schema_id for m in members}
    if len(schema_ids) != 1 or None in schema_ids:
        report.error(
            "R2", f"pair_id={pair_id}: schema_id must be identical and non-null"
        )
    if len({m.gold for m in members}) != 1:
        report.error("R2", f"pair_id={pair_id}: gold letter must match across the set")
    step_sequences = {
        base_step_types[m.item_id] for m in members if m.item_id in base_step_types
    }
    if len(step_sequences) > 1:
        report.error(
            "R2", f"pair_id={pair_id}: step-type sequence differs across the set"
        )


def _check_propagation_answers(
    report: ValidationReport,
    sets: dict[str, list[BenchmarkItem]],
    corpus: Corpus,
) -> None:
    """One propagation answer per item; matching across a triplet only warns."""

    by_item: dict[str, set[str]] = defaultdict(set)
    for chain in corpus.corrupted_chains:
        by_item[chain.item_id].add(chain.propagation_answer.value)

    for pair_id, members in sets.items():
        item_answers = {
            m.item_id: by_item[m.item_id] for m in members if m.item_id in by_item
        }
        for item_id, letters in item_answers.items():
            if len(letters) > 1:
                report.error(
                    "R2",
                    f"{item_id}: propagation_answer differs across models "
                    f"({sorted(letters)})",
                )
        if len({frozenset(letters) for letters in item_answers.values()}) > 1:
            summary = {k: sorted(v) for k, v in item_answers.items()}
            report.warn(
                "R2",
                f"pair_id={pair_id}: propagation_answer differs across the "
                f"triplet -- got {summary}",
            )


def _check_shared_step3(report: ValidationReport, corpus: Corpus) -> None:
    """Every model's chain for an item must share its corrupted and placebo step 3."""

    corrupted: dict[str, set[str]] = defaultdict(set)
    placebo: dict[str, set[str]] = defaultdict(set)
    for chain in corpus.corrupted_chains:
        corrupted[chain.item_id].add(chain.steps[2].text)
        if chain.placebo_steps is not None:
            placebo[chain.item_id].add(chain.placebo_steps[2].text)

    for name, texts_by_item in (
        ("step_3_corrupted", corrupted),
        ("step_3_placebo", placebo),
    ):
        for item_id, texts in texts_by_item.items():
            if len(texts) > 1:
                report.error(
                    "R2",
                    f"{item_id}: {name} differs across models ({len(texts)} variants)",
                )


def check_r2_matched_sets(report: ValidationReport, corpus: Corpus) -> None:
    """R2: triplets are complete and matched, and edits are shared across models.

    A triplet has one computation, one theoretical and one interpretive item
    (literature or history); matching of the propagation answer across a
    triplet is a warning only.
    """

    base_step_types: dict[str, tuple[str, ...]] = {}
    for chain in corpus.base_chains:
        base_step_types.setdefault(
            chain.item_id, tuple(s.type.value for s in chain.steps)
        )

    sets: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for item in corpus.items.values():
        if item.pair_id is not None:
            sets[item.pair_id].append(item)

    for pair_id, members in sets.items():
        _check_triplet(report, pair_id, members, base_step_types)
    _check_propagation_answers(report, sets, corpus)
    _check_shared_step3(report, corpus)


def check_r3_knowledge_probes(report: ValidationReport, corpus: Corpus) -> None:
    """R3: flag matched-set items with no knowledge_probes for review.

    Not an error -- a zero-probe item has no factual premise to fail on
    and is treated as vacuously eligible for Control 7
    (probes.knowledge_eligible), staying in the H1 analysis set.
    Surfaced as a note only so a writer can confirm each one is genuinely
    knowledge-free content, not an item where probes were simply never
    written.
    """

    for item in corpus.items.values():
        if item.pair_id is not None and len(item.knowledge_probes) == 0:
            report.warn(
                "R3",
                f"{item.item_id}: pair_id set but knowledge_probes is empty "
                "(treated as vacuously Control-7-eligible)",
            )


def check_r11_validity_gate(report: ValidationReport, corpus: Corpus) -> None:
    """R11: a base chain must still match its item's current gold answer.

    ``BaseChain`` already enforces base_answer==gold (plus language and
    no-meta-planning) at construction time (``chains.py``), so this is
    belt-and-braces defense-in-depth for any pipeline-written file --
    every check here should be vacuously true unless a JSONL line was
    hand-edited to bypass the constructor. Also catches item drift: a
    chain's own stored ``gold`` no longer matching the corpus item's
    current ``gold`` (the item was revised after the chain was built).
    """

    for chain in corpus.base_chains:
        item = corpus.items.get(chain.item_id)
        if item is None:
            continue
        if chain.gold != item.gold:
            report.warn(
                "R11",
                f"{chain.chain_id}: chain's gold ({chain.gold.value}) no longer "
                f"matches item's current gold ({item.gold.value}) -- item was "
                "revised after this chain was built",
            )


def check_variant_completeness(report: ValidationReport, corpus: Corpus) -> None:
    """Every item-model pair with any chains must have exactly one base chain
    and exactly one corrupted chain. Placebo is optional -- a curator may
    not have authored a placebo variant yet -- so its presence is reported
    as a coverage statistic, not a hard completeness requirement."""

    by_pair: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    placebo_by_pair: dict[tuple[str, str], int] = defaultdict(int)
    for base_chain in corpus.base_chains:
        by_pair[(base_chain.item_id, base_chain.model)]["base"] += 1
        if base_chain.item_id not in corpus.items:
            report.error(
                "FK",
                f"{base_chain.chain_id}: references unknown item {base_chain.item_id}",
            )
    for corrupted_chain in corpus.corrupted_chains:
        key = (corrupted_chain.item_id, corrupted_chain.model)
        by_pair[key]["corrupted"] += 1
        if corrupted_chain.placebo_steps is not None:
            placebo_by_pair[key] += 1
        if corrupted_chain.item_id not in corpus.items:
            report.error(
                "FK",
                f"{corrupted_chain.chain_id}: references unknown item "
                f"{corrupted_chain.item_id}",
            )

    expected = {"base", "corrupted"}
    for (item_id, model), counts in by_pair.items():
        if set(counts) != expected or any(n != 1 for n in counts.values()):
            report.warn(
                "structure",
                f"{item_id}/{model}: chains are {dict(counts)}, expected one "
                "each of base + corrupted (item is outside the analysed set)",
            )
        elif placebo_by_pair.get((item_id, model), 0) == 0:
            report.warn(
                "structure",
                f"{item_id}/{model}: no placebo variant authored yet "
                "(optional, not required for completeness)",
            )


def check_r12_pilot_integrity(report: ValidationReport, corpus: Corpus) -> None:
    """R12: no_cot_accuracy must match what no_cot_runs.jsonl actually shows.

    Guards against a hand-edited item (or a stale sheet applied twice): the
    stored accuracy map has to equal what ``baseline.no_cot_accuracies``
    re-derives from the raw runs, not just satisfy the band inequality
    ``items.py`` already checks.
    """

    runs = [r for r in corpus.no_cot_runs if r.condition is RunCondition.NO_COT]

    for run in runs:
        item = corpus.items.get(run.item_id)
        if item is None:
            # Not an FK error: apply_no_cot_review.py removes an item from
            # items.jsonl once it's dropped, but deliberately leaves its
            # pilot history in no_cot_runs.jsonl as an audit trail. An
            # orphaned run is the expected shape of "this item was piloted
            # and dropped," not a broken reference.
            report.warn(
                "R12",
                f"no_cot_runs.jsonl: {run.run_id} references {run.item_id}, "
                "not in items.jsonl (expected if it was dropped after "
                "piloting)",
            )
            continue
        if run.item_version != item.version:
            report.error(
                "R12",
                f"{run.run_id}: ran against item_version={run.item_version}, "
                f"item is now version={item.version}",
            )

    derived_by_item: dict[str, dict[str, float]] = defaultdict(dict)
    for (item_id, model), acc in no_cot_accuracies(runs, corpus.items).items():
        derived_by_item[item_id][model] = acc

    for item_id, item in corpus.items.items():
        if item.no_cot_accuracy is None:
            continue
        expected = derived_by_item.get(item_id, {})
        if item.no_cot_accuracy != expected:
            report.error(
                "R12",
                f"{item_id}: stored no_cot_accuracy {item.no_cot_accuracy} "
                f"does not match runs-derived {expected}",
            )


def check_r13_dropped_items_gate(report: ValidationReport, corpus: Corpus) -> None:
    """R13: report chains whose (item, model) pair failed a pilot gate.

    The probe gate and the answer-only band are covariates, not filters, so
    both only warn; the paper reports a probe-passing sensitivity subset.
    """

    all_chains: tuple[BaseChain | CorruptedChain, ...] = (
        *corpus.base_chains,
        *corpus.corrupted_chains,
    )
    if not all_chains:
        return

    no_cot = [r for r in corpus.no_cot_runs if r.condition is RunCondition.NO_COT]
    band = band_flags(no_cot, corpus.items)
    probe_ok = knowledge_ok(corpus.probe_results)
    knowledge = {
        key: knowledge_eligible(corpus.items[key[0]], key[1], probe_ok) for key in band
    }
    eligible = {key for key, ok in knowledge.items() if ok}
    band_only_ineligible = {key for key in eligible if not band.get(key, False)}

    for chain in all_chains:
        key = (chain.item_id, chain.model)
        if key not in knowledge or not knowledge[key]:
            report.warn(
                "R13",
                f"{chain.chain_id}: pair {key} did not pass the knowledge "
                "probes (kept in the analysis, covered by the probe sensitivity)",
            )
        elif key in band_only_ineligible:
            report.warn(
                "R13",
                f"{chain.chain_id}: chain built for pair {key} outside the "
                "no-CoT band (descriptive only, not a hard gate)",
            )


def compute_coverage(corpus: Corpus) -> dict[tuple[Level, Domain], int]:
    counts: dict[tuple[Level, Domain], int] = defaultdict(int)
    for item in corpus.items.values():
        counts[(item.level, item.domain)] += 1
    return dict(counts)


def validate_corpus() -> ValidationReport:
    """Run every corpus-level rule against the fixed on-disk layout."""

    corpus = load_corpus()
    report = ValidationReport(coverage=compute_coverage(corpus))

    check_r6_firewall(report)
    check_r1_gold_balance(report, corpus)
    check_r2_matched_sets(report, corpus)
    check_r3_knowledge_probes(report, corpus)
    check_variant_completeness(report, corpus)
    check_r11_validity_gate(report, corpus)
    check_r12_pilot_integrity(report, corpus)
    check_r13_dropped_items_gate(report, corpus)

    return report


def _print_findings(
    title: str, mark: str, findings: list[str], *, show: int = 3
) -> None:
    """Print a count, then the first few findings of each rule."""

    print(f"\n{title}: {len(findings)}")
    by_rule: dict[str, list[str]] = defaultdict(list)
    for finding in findings:
        by_rule[finding.split("]", 1)[0] + "]"].append(finding)
    for rule, group in by_rule.items():
        print(f"  {rule} x{len(group)}")
        for finding in group[:show]:
            print(f"    {mark} {finding}")


def _main() -> int:
    report = validate_corpus()

    print("Coverage (level x domain):")
    for level in Level:
        row = f"  {level.value:<12} "
        row += " ".join(f"{report.coverage.get((level, d), 0):<4}" for d in Domain)
        print(row)

    _print_findings("ERRORS", "x", report.errors)
    _print_findings("WARNINGS", "!", report.warnings)

    return 1 if not report.ok else 0


if __name__ == "__main__":
    sys.exit(_main())
