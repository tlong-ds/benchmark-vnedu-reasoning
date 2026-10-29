"""Run the calibrated, resumable three-arm intervention against OpenRouter.

Run: ``uv run vnedu-cot intervention run --protocol <protocol.toml> [--dry-run]``
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vnedu_cot.providers.base import NormalizedResponse
from vnedu_cot.providers.openrouter import OpenRouterClient
from vnedu_cot.schemas.intervention import (
    AnswerToken,
    CalibrationRecord,
    DrawDiagnostics,
    InterventionDrawRecord,
)
from vnedu_cot.stages.intervention.prompts import (
    SYSTEM_PROMPT_VI,
    render_assistant_prefill,
    render_user_prompt,
)
from vnedu_cot.stages.intervention.protocol import InterventionProtocol
from vnedu_cot.stages.intervention.schedule import ScheduledDraw, build_schedule
from vnedu_cot.stages.intervention.sources import (
    SourceChain,
    canonical_hash,
    load_source_chains,
)
from vnedu_cot.storage import INTERVENTIONS_DIR, sha256

ClientFactory = Callable[[Callable[[dict[str, Any]], None]], OpenRouterClient]


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    run_dir: Path | None
    chain_count: int
    scheduled_draws: int
    completed_draws: int
    incomplete_draws: int
    calibration_passed: bool | None


def parse_answer(text: str) -> AnswerToken:
    normalized = text.replace("**", "")
    patterns = (
        # DeepSeek may put the label in a Markdown heading, put the option on
        # the following line, omit the colon, or wrap it in angle brackets.
        # Each form still states one explicit final option; do not infer an
        # answer from the reasoning text or an unlabeled option list.
        r"(?im)^\s*(?:→\s*)?(?:#+\s*)?đáp\s*án"
        r"(?:\s+(?:đúng|chính\s*xác|cuối\s*cùng))?\s*(?::|\blà\b)?\s*"
        r"(?:\n\s*)?<?\s*([A-D])\s*>?\b",
        r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*([A-D])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, normalized)
        if matches:
            return AnswerToken(matches[-1].upper())
    return AnswerToken.UNPARSEABLE


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _reasoning_diagnostics(
    response: NormalizedResponse, prefix: str
) -> DrawDiagnostics:
    reasoning = response.reasoning
    return DrawDiagnostics(
        reasoning_channel_present=bool(reasoning.strip()),
        positive_reasoning_tokens=response.reasoning_tokens > 0,
        prefix_echo=prefix.strip() in reasoning,
        apparent_step_4_continuation=bool(
            re.search(r"(?im)^\s*(?:bước\s*)?4[.):\-]", reasoning)
        ),
    )


def _protocol_failures(
    protocol: InterventionProtocol,
    response: NormalizedResponse,
    diagnostics: DrawDiagnostics,
) -> tuple[str, ...]:
    failures: list[str] = []
    if response.model != protocol.model:
        failures.append("model_mismatch")
    if response.provider.casefold() != protocol.provider.casefold():
        failures.append("provider_mismatch")
    if not diagnostics.reasoning_channel_present:
        failures.append("missing_reasoning")
    if not diagnostics.positive_reasoning_tokens:
        failures.append("zero_reasoning_tokens")
    return tuple(failures)


class _AttemptLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, event: dict[str, Any]) -> None:
        sanitized = json.loads(json.dumps(event, ensure_ascii=False))
        with self.lock:
            with gzip.open(self.path, "at", encoding="utf-8") as stream:
                stream.write(json.dumps(sanitized, ensure_ascii=False))
                stream.write("\n")


def _payload(
    protocol: InterventionProtocol,
    chain: SourceChain,
    draw: ScheduledDraw,
) -> tuple[dict[str, Any], str, str]:
    prefix = chain.prefixes[draw.arm]
    assistant_prefill = render_assistant_prefill(prefix)
    user_prompt = render_user_prompt(chain.question, chain.options)
    payload = OpenRouterClient.build_payload(
        model=protocol.model,
        provider=protocol.provider,
        allow_fallbacks=protocol.allow_fallbacks,
        system_prompt=SYSTEM_PROMPT_VI,
        user_prompt=user_prompt,
        assistant_prefill=assistant_prefill,
        temperature=protocol.temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        reasoning_effort=protocol.reasoning_effort,
        seed=draw.seed,
    )
    return payload, prefix, user_prompt


def _draw_record(
    *,
    protocol: InterventionProtocol,
    run_id: str,
    calibration_id: str,
    chain: SourceChain,
    draw: ScheduledDraw,
    response: NormalizedResponse,
    prefix: str,
    user_prompt: str,
    payload: dict[str, Any],
    started: dt.datetime,
    elapsed_ms: int,
) -> InterventionDrawRecord:
    diagnostics = _reasoning_diagnostics(response, prefix)
    failures = _protocol_failures(protocol, response, diagnostics)
    return InterventionDrawRecord(
        schema_version=protocol.schema_version,
        protocol_id=protocol.protocol_id,
        protocol_hash=protocol.digest,
        run_id=run_id,
        calibration_id=calibration_id,
        draw_id=draw.draw_id,
        item_id=chain.item_id,
        item_version=chain.item_version,
        chain_id=chain.chain_id,
        source_base_chain_id=chain.base_chain_id,
        gold_answer=chain.gold,
        propagation_answer=chain.propagation_answer,
        arm=draw.arm,
        draw_index=draw.draw_index,
        seed=draw.seed,
        settings=protocol.settings,
        requested_model=protocol.model,
        served_model=response.model,
        requested_provider=protocol.provider,
        served_provider=response.provider,
        reasoning_prefix=prefix,
        assistant_prefill=render_assistant_prefill(prefix),
        reasoning_continuation=response.reasoning,
        full_reasoning=prefix + response.reasoning,
        raw_final_response=response.content,
        final_response=response.content,
        parsed_answer=parse_answer(response.content),
        reasoning_details=response.reasoning_details,
        usage=response.usage,
        cost=response.cost,
        request_id=response.request_id,
        finish_reason=response.finish_reason,
        started_at=started,
        finished_at=_utc_now(),
        latency_ms=elapsed_ms,
        prompt_hash=hashlib.sha256(user_prompt.encode()).hexdigest(),
        prefix_hash=hashlib.sha256(prefix.encode()).hexdigest(),
        request_hash=canonical_hash(payload),
        source_record_hash=chain.source_record_hash,
        diagnostics=diagnostics,
        protocol_valid=not failures,
        protocol_failures=failures,
    )


def _calibrate(
    *,
    protocol: InterventionProtocol,
    run_id: str,
    chain: SourceChain,
    draw: ScheduledDraw,
    client: OpenRouterClient,
) -> CalibrationRecord:
    calibration_id = uuid.uuid4().hex
    calibration_draw = ScheduledDraw(
        draw_id=f"calibration-{calibration_id}",
        item_id=draw.item_id,
        chain_id=draw.chain_id,
        arm=draw.arm,
        draw_index=draw.draw_index,
        seed=int.from_bytes(
            hashlib.sha256(
                f"calibration\0{protocol.protocol_id}\0{run_id}".encode()
            ).digest()[:8],
            "big",
        ),
    )
    payload, prefix, _ = _payload(protocol, chain, calibration_draw)
    response = client.complete(payload)
    diagnostics = _reasoning_diagnostics(response, prefix)
    failures = list(_protocol_failures(protocol, response, diagnostics))
    if diagnostics.prefix_echo:
        failures.append("full_prefix_echo")
    if not diagnostics.apparent_step_4_continuation:
        failures.append("no_apparent_step_4_continuation")
    parsed = parse_answer(response.content)
    if parsed is AnswerToken.UNPARSEABLE:
        failures.append("unparseable_final_content")
    return CalibrationRecord(
        schema_version=protocol.schema_version,
        protocol_id=protocol.protocol_id,
        protocol_hash=protocol.digest,
        run_id=run_id,
        calibration_id=calibration_id,
        draw_id=calibration_draw.draw_id,
        requested_model=protocol.model,
        served_model=response.model,
        requested_provider=protocol.provider,
        served_provider=response.provider,
        reasoning_prefix=prefix,
        assistant_prefill=render_assistant_prefill(prefix),
        reasoning_continuation=response.reasoning,
        raw_final_response=response.content,
        final_response=response.content,
        parsed_answer=parsed,
        reasoning_tokens=response.reasoning_tokens,
        request_id=response.request_id,
        diagnostics=diagnostics,
        passed=not failures,
        failures=tuple(failures),
        timestamp=_utc_now(),
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.add(str(json.loads(line)["draw_id"]))
    return result


def _write_checksums(run_dir: Path) -> None:
    names = [
        "manifest.json",
        "calibration.json",
        "draws.jsonl",
        "api_attempts.jsonl.gz",
        "report.jsonl",
        "analysis_ready.csv",
    ]
    lines = [
        f"{sha256(run_dir / name)}  {name}"
        for name in names
        if (run_dir / name).exists()
    ]
    (run_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refuse_excluded_resume(run_dir: Path) -> None:
    manifest = run_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get(
        "posthoc_exclusion"
    ):
        raise ValueError(
            "This historical run has excluded draws and cannot be resumed; "
            "use its separate replacement protocol."
        )


def _new_manifest(
    protocol: InterventionProtocol,
    run_id: str,
    schedule: list[ScheduledDraw],
    completed: set[str],
) -> dict[str, Any]:
    return {
        "schema_version": protocol.schema_version,
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.digest,
        "run_id": run_id,
        "protocol_path": str(protocol.source_path),
        "source_checksums": {
            str(protocol.rating_key): sha256(protocol.rating_key),
            str(protocol.corrupted_chains): sha256(protocol.corrupted_chains),
            str(protocol.items): sha256(protocol.items),
        },
        # The frozen schedule is stored with arms as plain strings.
        "schedule": [{**draw.__dict__, "arm": draw.arm.value} for draw in schedule],
        "state": "calibrating",
        "scheduled_draws": len(schedule),
        "completed_draws": len(completed),
        "incomplete_draws": len(schedule) - len(completed),
        "updated_at": _utc_now().isoformat(),
    }


def _record_calibration(run_dir: Path, calibration: CalibrationRecord) -> None:
    """Append this calibration to ``calibration.json``, keeping earlier ones."""

    path = run_dir / "calibration.json"
    history: list[dict[str, Any]] = []
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        history.extend(previous.get("calibrations", [previous]))
    history.append(calibration.model_dump(mode="json"))
    _atomic_json(
        path,
        {"latest_calibration_id": calibration.calibration_id, "calibrations": history},
    )


def _execute_draws(
    protocol: InterventionProtocol,
    *,
    run_dir: Path,
    run_id: str,
    calibration_id: str,
    by_chain: dict[str, SourceChain],
    pending: list[ScheduledDraw],
    completed: set[str],
    client: OpenRouterClient,
) -> list[str]:
    """Run the pending draws, appending each record; return transport failures."""

    def execute(draw: ScheduledDraw) -> InterventionDrawRecord:
        chain = by_chain[draw.chain_id]
        payload, prefix, user_prompt = _payload(protocol, chain, draw)
        started = _utc_now()
        before = time.perf_counter()
        response = client.complete(payload)
        elapsed_ms = round((time.perf_counter() - before) * 1000)
        return _draw_record(
            protocol=protocol,
            run_id=run_id,
            calibration_id=calibration_id,
            chain=chain,
            draw=draw,
            response=response,
            prefix=prefix,
            user_prompt=user_prompt,
            payload=payload,
            started=started,
            elapsed_ms=elapsed_ms,
        )

    failures: list[str] = []
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=protocol.concurrency) as pool:
        futures = [pool.submit(execute, draw) for draw in pending]
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            with write_lock:
                with (run_dir / "draws.jsonl").open("a", encoding="utf-8") as out:
                    out.write(record.model_dump_json() + "\n")
                completed.add(record.draw_id)
    return failures


def run_protocol(
    protocol: InterventionProtocol,
    *,
    run_id: str | None = None,
    output_root: Path = INTERVENTIONS_DIR,
    dry_run: bool = False,
    client_factory: ClientFactory | None = None,
) -> RunSummary:
    chains = load_source_chains(
        rating_key=protocol.rating_key,
        corrupted_chains=protocol.corrupted_chains,
        items=protocol.items,
        model=protocol.model,
    )
    schedule = build_schedule(protocol, chains)
    if dry_run:
        return RunSummary(
            run_id=run_id or "dry-run",
            run_dir=None,
            chain_count=len(chains),
            scheduled_draws=len(schedule),
            completed_draws=0,
            incomplete_draws=len(schedule),
            calibration_passed=None,
        )

    actual_run_id = run_id or _utc_now().strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / protocol.protocol_id / actual_run_id
    _refuse_excluded_resume(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts = _AttemptLog(run_dir / "api_attempts.jsonl.gz")
    factory = client_factory or (lambda sink: OpenRouterClient(attempt_sink=sink))
    client = factory(attempts)
    by_chain = {chain.chain_id: chain for chain in chains}
    completed = _read_completed(run_dir / "draws.jsonl")

    manifest = _new_manifest(protocol, actual_run_id, schedule, completed)
    _atomic_json(run_dir / "manifest.json", manifest)

    calibration = _calibrate(
        protocol=protocol,
        run_id=actual_run_id,
        chain=by_chain[schedule[0].chain_id],
        draw=schedule[0],
        client=client,
    )
    _record_calibration(run_dir, calibration)
    if not calibration.passed:
        manifest.update(
            state="calibration_failed",
            calibration_id=calibration.calibration_id,
            updated_at=_utc_now().isoformat(),
        )
        _atomic_json(run_dir / "manifest.json", manifest)
        _write_checksums(run_dir)
        raise RuntimeError(
            "mandatory calibration failed: " + ", ".join(calibration.failures)
        )

    manifest.update(
        state="running",
        calibration_id=calibration.calibration_id,
        updated_at=_utc_now().isoformat(),
    )
    _atomic_json(run_dir / "manifest.json", manifest)

    failure_messages = _execute_draws(
        protocol,
        run_dir=run_dir,
        run_id=actual_run_id,
        calibration_id=calibration.calibration_id,
        by_chain=by_chain,
        pending=[draw for draw in schedule if draw.draw_id not in completed],
        completed=completed,
        client=client,
    )

    manifest.update(
        state="complete" if len(completed) == len(schedule) else "incomplete",
        completed_draws=len(completed),
        incomplete_draws=len(schedule) - len(completed),
        transport_failures=len(failure_messages),
        transport_failure_summaries=failure_messages,
        updated_at=_utc_now().isoformat(),
    )
    _atomic_json(run_dir / "manifest.json", manifest)
    _write_checksums(run_dir)
    return RunSummary(
        run_id=actual_run_id,
        run_dir=run_dir,
        chain_count=len(chains),
        scheduled_draws=len(schedule),
        completed_draws=len(completed),
        incomplete_draws=len(schedule) - len(completed),
        calibration_passed=True,
    )
