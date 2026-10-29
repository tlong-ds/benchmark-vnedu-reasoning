# VNEDU-CoT: Does Displayed Reasoning Drive the Answer?

A diagnostic benchmark for behavioural dependence on displayed reasoning in
Vietnamese educational problems. A model continues from a three-step reasoning
prefix whose step 3 is either the model's own (clean), a meaning-preserving
rewrite (placebo), or an invalid inference on which the answer depends (corrupted). If the
displayed reasoning drives the answer, the corrupted step should pull the final
answer toward the item's propagation option.

This repository contains the item corpus, the pipeline code, the analysed model
outputs and human ratings, and the paper with a script that reproduces every
reported number.

## Design

- **Corpus.** 72 matched triplets (216 multiple-choice items) across four
  education levels. Each triplet links a computational, a theoretical and a
  history/literature item that share one of six abstract inference schemas
  (S1–S6) and a rigid four-step reasoning shape, with the manipulation always
  at step 3.
- **Model.** DeepSeek-R1-0528 through OpenRouter, pinned to DeepInfra with
  fallbacks disabled. The prefix is fed as an unfinished `<think>` turn, so
  the model continues its own native reasoning.
- **Arms.** Clean, placebo and corrupted step-3 prefixes, 20 draws per arm.
  The analysed set is 162 items (54 triplets) and 9,720 generations. Of the 72
  triplets, 18 lack a valid chain for one of their three items, so only the 54
  complete triplets are analysed.
- **Primary estimand.** ΔCDE = P(propagation | corrupted) −
  P(propagation | placebo), averaged over items. The clean arm is a descriptive
  manipulation check, not a filter.
- **Covariates.** Answer-only accuracy q_i (k = 10 draws, no displayed
  reasoning) and standalone knowledge probes for factual prerequisites.

Paper: *Does Displayed Reasoning Drive the Answer? A Diagnostic Benchmark for
Vietnamese Educational Reasoning* (ICIIT 2027). `paper/main.pdf` is the
submitted paper.

Headline result: corrupted step 3 raises propagation by 10.9 percentage points
(95% whole-triplet bootstrap CI [7.3, 14.8]); placebo rewriting induces 0.0%.

## Repository layout

| Path | Contents |
|---|---|
| `src/vnedu_cot/` | The installable package: schemas, model gateway, pipeline stages, analysis |
| `tests/` | Unit and integration tests (offline, no API calls) |
| `configs/` | Model config and the intervention protocol |
| `data/items.jsonl` | All 216 items from the 72 triplets, with gold answers and knowledge probes |
| `data/base_chains.jsonl` | DeepSeek-R1's full temperature-0 reasoning and answer for 210 items |
| `data/canonical_chains.jsonl` | 203 base chains, each condensed into four typed steps (retrieval, inference, computation, evaluation) and edited into a uniform format |
| `data/corrupted_chains.jsonl` | All 198 corrupted chains (162 in the 54 analysed triplets, 36 in the 18 incomplete ones), each with its base chain, placebo step 3 and propagation answer |
| `data/intervention_chain_ids.jsonl` | The 162 corrupted chains used in the intervention |
| `outputs/no_cot_runs.jsonl` | Answer-only runs (q_i) |
| `outputs/probe_runs.jsonl` | Knowledge-probe answers and judge verdicts |
| `outputs/interventions/deepseek-r1/` | The 9,720 analysed draws (`all_draws.jsonl.gz`) and `SHA256SUMS` |
| `ratings/` | Rating keys and the rated human and LLM-judge workbooks (`rated/`) |
| `scripts/paper_statistics.py` | Reproduces every number in the paper |
| `scripts/make_canonical_template.py` | Turns base chains into a canonical-chain template with four empty steps to fill in |
| `paper/` | LaTeX sources, built PDFs and the generated `analysis/` inputs |
| `docs/` | Intervention protocol and data schema |

Raters are anonymised as Rater 1 and Rater 2.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy
```

## Reproduce the paper

```bash
uv run python scripts/paper_statistics.py
```

This reads the inputs listed above, recomputes every metric, and rewrites
`paper/analysis/statistics.json` and `paper/analysis/item_outcomes.csv`. No API
calls are made. The `.tex` files in `paper/analysis/` are written to agree with
`statistics.json`; see `paper/analysis/README.md` for the method details.
Build the PDFs with `make -C paper`.

Check the draws file against its checksum with:

```bash
cd outputs/interventions/deepseek-r1 && shasum -a 256 -c SHA256SUMS
```

## Rerun the pipeline

Model calls need an OpenRouter key in `.env` (`OPENROUTER_API_KEY=...`). Running
these commands incurs API cost. New runs write to `outputs/`, so the shipped
files in `data/` are not overwritten.

```bash
# Answer-only difficulty (q_i) and knowledge probes
uv run python -m vnedu_cot.stages.baseline run --models configs/models/ --samples 10
uv run python -m vnedu_cot.stages.probes run --models configs/models/

# Base chains (temperature 0), condensed into four steps by filling the template
# (save the filled file as outputs/canonical_chains.jsonl), then the step-3 edits
uv run python -m vnedu_cot.stages.chains generate --models configs/models/
uv run python scripts/make_canonical_template.py --base outputs/base_chains.jsonl
uv run python -m vnedu_cot.stages.corruption sheet
uv run python -m vnedu_cot.stages.corruption import

# Three-arm intervention (a dry run prints the frozen schedule)
uv run vnedu-cot intervention run --dry-run
uv run vnedu-cot intervention run
uv run vnedu-cot intervention validate --run-id <run-id>
uv run vnedu-cot intervention report --run-id <run-id>

# Corpus checks on the shipped data
uv run python -m vnedu_cot.stages.validation
```

Every stage accepts `--help`. The intervention uses
`configs/experiments/deepseek_r1_intervention.toml` unless `--protocol` is
given. See `docs/intervention_protocol.md` for the request format, calibration
gate, run records and resume rules.

## License

Code is released under the MIT License (`LICENSE`). The corpus, model outputs
and ratings are released under CC BY 4.0.
