"""Deterministic draw schedule: one seed and one draw id per (chain, arm, draw)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from vnedu_cot.schemas.intervention import Arm
from vnedu_cot.stages.intervention.protocol import InterventionProtocol
from vnedu_cot.stages.intervention.sources import SourceChain


@dataclass(frozen=True)
class ScheduledDraw:
    draw_id: str
    item_id: str
    chain_id: str
    arm: Arm
    draw_index: int
    seed: int


def deterministic_seed(
    protocol_id: str, item_id: str, arm: Arm, draw_index: int
) -> int:
    material = f"{protocol_id}\0{item_id}\0{arm.value}\0{draw_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def build_schedule(
    protocol: InterventionProtocol, chains: list[SourceChain]
) -> list[ScheduledDraw]:
    schedule: list[ScheduledDraw] = []
    for batch_start in range(0, len(chains), protocol.items_in_flight):
        batch = chains[batch_start : batch_start + protocol.items_in_flight]
        for draw_index in range(protocol.samples_per_arm):
            for chain in batch:
                for arm in protocol.arms:
                    identity = (
                        f"{protocol.protocol_id}\0{chain.chain_id}\0"
                        f"{arm.value}\0{draw_index}"
                    )
                    draw_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
                    schedule.append(
                        ScheduledDraw(
                            draw_id=draw_id,
                            item_id=chain.item_id,
                            chain_id=chain.chain_id,
                            arm=arm,
                            draw_index=draw_index,
                            seed=deterministic_seed(
                                protocol.protocol_id,
                                item_id=chain.item_id,
                                arm=arm,
                                draw_index=draw_index,
                            ),
                        ),
                    )
    # Keep a small, deterministic batch of item bundles together. For each
    # draw index the schedule issues all three arms of every in-flight item,
    # so six workers can run two full three-arm bundles concurrently without
    # mixing in later items.
    draw_ids = {draw.draw_id for draw in schedule}
    seeds = {draw.seed for draw in schedule}
    if len(draw_ids) != len(schedule):
        raise ValueError("schedule produced duplicate draw IDs")
    if len(seeds) != len(schedule):
        raise ValueError("schedule produced duplicate seeds")
    return schedule

