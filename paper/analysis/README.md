# Paper analysis artifacts

Every number in the paper comes from `scripts/paper_statistics.py`, run
from the repository root:

```bash
uv run python scripts/paper_statistics.py
```

It reads the raw inputs below and rewrites `statistics.json` and
`item_outcomes.csv`. The `.tex` files here are hand-maintained and must agree
with `statistics.json`. Rerun the script and compare after any data change.

## Inputs

| Input | Content |
|---|---|
| `outputs/interventions/deepseek-r1/all_draws.jsonl.gz` | The 9,720 analysed intervention draws (162 items × 3 arms × 20). Checksum in the adjacent `SHA256SUMS`. |
| `outputs/no_cot_runs.jsonl` | Answer-only runs, k = 10 rotations per item (q_i) |
| `outputs/probe_runs.jsonl` | Knowledge-probe answers and judge verdicts. Empty responses count as failures (`judge_model = "empty_response_rule"`). |
| `data/items.jsonl` | Items, gold answers, knowledge probes |
| `ratings/rated/*.xlsx` + keys in `ratings/` (`rating_key.jsonl`, `placebo_key.jsonl`, `disclosure_rating_key.jsonl`) | Paired ratings from Rater 1 and Rater 2: Job A corruption, placebo review, disclosure. Also the `gpt-5.6-terra` disclosure judge. |

The sha256 of every input is recorded in `statistics.json` under `inputs`.

## Outputs

- `statistics.json`: every reported metric, nested by section, with method settings.
- `item_outcomes.csv`: one row per (item, arm), with the human-screen, probe and
  protocol flags used by the sensitivity analyses.
- `numbers.tex`, `results.tex`, `rating_quality.tex`, `severity_*.tex`,
  `disclosure_quality.tex`, `sensitivity.tex`: LaTeX inputs to `main.tex` and
  `supplementary.tex`.

## Method

- **Effects.** The primary estimand is ΔCDE = P(propagation | corrupted) −
  P(propagation | placebo), averaged over item-level rates. Unparseable answers
  stay in the denominator as non-propagation.
- **Intervals.** 95% percentile bootstrap over whole triplets (10,000
  replicates; seed 20261013).
- **Regressions.** Item-level OLS with triplet-clustered (CR1) standard errors.
  The p-values use t with G − 1 = 53 degrees of freedom.
- **Behaviour audit.** Uses a keyword classifier calibrated against the
  76 human-rated corrupted continuations. Both classifiers' precision and recall
  are reported. Explicit repair is ≈26–34% after calibration.
