# Data schema

Record types live in `src/vnedu_cot/schemas/`. All records are strict and
frozen: unknown fields are rejected. Fixed paths are in
`src/vnedu_cot/storage.py`.

| File | Record | Content |
|---|---|---|
| `data/items.jsonl` | `BenchmarkItem` | One multiple-choice item: `item_id`, `level`, `domain`, `question_vi`, four `options`, `gold`, `schema_id`, `pair_id` (the triplet), `knowledge_probes`, `version` |
| `data/base_chains.jsonl` | `BaseChain` | DeepSeek-R1's temperature-0 reasoning (`raw_reasoning`, `response_text`) and `base_answer` for an item |
| `data/canonical_chains.jsonl` | `BaseChain` | The same chain condensed into four typed steps (retrieval, inference, computation, evaluation) |
| `data/corrupted_chains.jsonl` | `CorruptedChain` | The embedded base chain, the three-step corrupted prefix, `propagation_answer`, the curator attestations and the placebo prefix |
| `data/intervention_chain_ids.jsonl` | | The 162 chains used in the intervention |
| `outputs/no_cot_runs.jsonl` | `RunRecord` | Answer-only runs; `sampled_answers` are mapped back to canonical letters |
| `outputs/probe_runs.jsonl` | `KnowledgeProbeResult` | One judged answer per knowledge probe |
| `outputs/interventions/deepseek-r1/all_draws.jsonl.gz` | `InterventionDrawRecord` | The 9,720 analysed draws |
| `ratings/` | | Blinding keys and the rated workbooks, read by `scripts/paper_statistics.py` |

## Rules

Per-record rules are enforced when a record is built:

- An item has exactly four distinct options, and `gold` is one of them.
- A base chain has four contiguous steps, matches its gold answer, is observed
  Vietnamese and contains no meta-planning (a chain that fails is dropped, never
  resampled).
- A corrupted chain keeps steps 1 and 2 identical to the base chain, changes
  only step 3 and never shows step 4. `propagation_answer` differs from gold, and
  the curator attestations (`propagation_unique`, `no_world_knowledge_shortcut`,
  `option_length_balanced`) must all be true.
- Each draw has its own seed, derived from the item, model and condition, so a
  gateway cannot return one cached completion for a whole batch.

Corpus-level rules are checked by `uv run python -m vnedu_cot.stages.validation`:

| Rule | Check | Severity |
|---|---|---|
| R1 | No gold letter exceeds 30% of a domain x level cell | warning |
| R2 | A triplet has one computation, one theoretical and one interpretive item, with the same level, schema and gold letter | error; differing propagation letters across a triplet only warn |
| R3 | A matched item with no knowledge probe is listed for review | warning |
| R6 | The rater sheet never contains a blinding field | error |
| R11 | A chain's gold still matches its item | warning |
| R12 | Stored `no_cot_accuracy` matches the answer-only runs | error |
| R13 | A chain's (item, model) pair passed the probes and the difficulty band | warning |
| FK | Chains refer to existing items | error |

The probe gate and the answer-only difficulty band are covariates, not filters.
The clean arm is a descriptive manipulation check, never an eligibility gate.
