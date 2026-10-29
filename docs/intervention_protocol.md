# Native R1 intervention protocol

The tracked protocol is
[`configs/experiments/deepseek_r1_intervention.toml`](../configs/experiments/deepseek_r1_intervention.toml).
It selects the 162 chains listed in `data/intervention_chain_ids.jsonl`. Every
selected chain resolves to a DeepSeek R1 0528 record in
`data/corrupted_chains.jsonl` with clean, placebo, and corrupted
three-step prefixes. At 20 draws per arm, the frozen schedule contains 9,720
draws: exactly the analysed set in
`outputs/interventions/deepseek-r1/all_draws.jsonl.gz`, whose recorded
prompt and prefix hashes match these stimuli. Each draw's `source_record_hash`
covers the full source record as it was run, which also carried a curator
`reviewer` field since removed from the release.

## Native continuation contract

Each request places the selected three-step prefix in an unfinished assistant
reasoning block:

```text
<think>
1. ...
2. ...
3. ...
```

The request does not close `</think>`. It sends OpenRouter reasoning effort
`high`, asks for a separate reasoning channel, pins DeepInfra, and disables
fallbacks. It does not send `prefix`, input `reasoning_content`, or
`continue_final_message` fields.

Every invocation performs one paid calibration before dataset calls. The
calibration must prove the requested model and provider were served, the
reasoning channel is non-empty, reasoning-token usage is positive, generation
continues at an apparent step 4 without echoing the complete supplied prefix,
and the visible final content has a parseable answer.

## Execution and resume

```bash
uv sync --extra dev
uv run vnedu-cot intervention run \
  --protocol configs/experiments/deepseek_r1_intervention.toml

uv run vnedu-cot intervention validate --run-id <run-id>
uv run vnedu-cot intervention report --run-id <run-id>
```

Use `--dry-run` to inspect the schedule without an API call. Pilot sizes belong
in separate tracked protocol TOML files; the CLI has no sample-size override.

Run directories live under
`outputs/interventions/<protocol-id>/<run-id>/` (not tracked). Each directory contains the frozen schedule manifest, calibration
evidence, one typed JSONL row per completed draw, a compressed append-only API
attempt log, analysis outputs, and checksums.

Transport retries reuse the same draw ID and seed. Resume skips every draw ID
already present in `draws.jsonl`. A final transport failure leaves the draw
incomplete. Unparseable answers and content-level failures are retained without
replacement.

The frozen schedule is item-complete in configurable in-flight batches. For
each draw index it queues all clean, placebo, and corrupted arms of every
in-flight item before the next index; only when that batch finishes its draws
does it move to later items. Thus `items_in_flight = 2` plus `concurrency = 6`
runs two three-arm item bundles at once while preserving complete three-arm
coverage for both before advancing.

## Prompt-cache decision

The protocol permits provider prompt/KV-cache reads. They reuse processing of
an identical prompt prefix; they do not reuse a prior completion, so a distinct
draw is still a distinct model invocation. In particular, different draw seeds
do not prevent a cache hit. Repeated draws in the same item/arm deliberately
have identical supplied steps 1--3 and may therefore receive cache reads.

Do not add per-draw cache-busting text: that would alter the prompt and confound
the matched-arm comparison. Instead, retain and audit the response usage
metadata, including `prompt_tokens_details.cached_tokens` when supplied by the
gateway, to distinguish a cache read from an independent completion.

For the pinned DeepSeek R1 route through OpenRouter/DeepInfra, OpenRouter
documents DeepSeek prompt caching as automatic and does not document a
DeepSeek-specific opt-out. The documented `prompt_cache_options.mode =
"explicit"` no-breakpoint opt-out is OpenAI GPT-5.6-and-later specific. If a
future protocol requires cache reads to be prohibited, it must use a
model/provider pair with a documented opt-out rather than modifying prompts.

## Confirmatory estimands

Reporting requires 20 protocol-valid draws in every arm for an item. The fixed
denominator includes unparseable final answers. No clean-arm observation gates
selection or reporting.

- `effect_corrupted_vs_placebo`: primary propagation effect.
- `effect_placebo_vs_clean`: wording-only manipulation check.
- `effect_corrupted_vs_clean`: total corrupted-versus-clean contrast.

The five-way corrupted-arm behavior classifier reads only generated reasoning
continuation and visible final content. Supplied steps 1–3 are excluded.

