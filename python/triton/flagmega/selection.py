# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Serializable default selections and optional user/agent overrides."""

from __future__ import annotations

import os
import runpy
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from triton.flagmega.errors import CheckpointError, IRVerificationError
from triton.flagmega.ir import IRModule, SelectionRecord


@dataclass(frozen=True)
class SelectionPlan:
    input_semantic_hash: str
    records: tuple[SelectionRecord, ...]

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "flagmega.selection/v1",
            "input_semantic_hash": self.input_semantic_hash,
            "records": [record.to_data() for record in self.records],
        }

    @classmethod
    def from_data(cls, data) -> SelectionPlan:
        if data.get("schema") != "flagmega.selection/v1":
            raise CheckpointError(f"Unsupported selection plan schema {data.get('schema')!r}.")
        return cls(
            str(data["input_semantic_hash"]),
            tuple(SelectionRecord.from_data(value) for value in data.get("records", ())),
        )


def apply_plan(module: IRModule, plan: SelectionPlan) -> IRModule:
    if plan.input_semantic_hash != module.semantic_hash:
        raise IRVerificationError(
            "Selection plan input semantic hash does not match the IR checkpoint.",
            stage=module.stage,
        )
    point_map = {point.id: point for point in module.selection_points}
    replacements = {record.point_id: record for record in plan.records}
    if len(replacements) != len(plan.records):
        raise IRVerificationError("A selection plan cannot override the same point twice.", stage=module.stage)
    for point_id, record in replacements.items():
        point = point_map.get(point_id)
        if point is None:
            raise IRVerificationError(f"Selection plan references unknown point {point_id!r}.", stage=module.stage)
        if record.candidate_id not in {candidate.id for candidate in point.candidates}:
            raise IRVerificationError(
                f"Candidate {record.candidate_id!r} is invalid for point {point_id!r}.", stage=module.stage)
    existing = {record.point_id: record for record in module.selections}
    existing.update(replacements)
    from dataclasses import replace

    return replace(module, selections=tuple(existing[key] for key in sorted(existing)))


def plan_source(plan: SelectionPlan) -> str:
    lines = [
        "# Editable FlagMega selection override built with real Python constructors.",
        "from triton.flagmega import ir as fm",
        "from triton.flagmega.selection import SelectionPlan",
        "",
        "PLAN = SelectionPlan(",
        f"    input_semantic_hash={plan.input_semantic_hash!r},",
        "    records=(",
    ]
    for record in plan.records:
        lines.extend((
            "        fm.SelectionRecord(",
            f"            point_id={record.point_id!r},",
            f"            candidate_id={record.candidate_id!r},",
            f"            origin={record.origin!r},",
            f"            policy={record.policy!r},",
            f"            rationale={record.rationale!r},",
            f"            evidence={record.evidence!r},",
            "        ),",
        ))
    lines.extend(("    ),", ")", ""))
    return "\n".join(lines)


def emit_plan(plan: SelectionPlan, path: str | os.PathLike[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(plan_source(plan))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_plan(path: str | os.PathLike[str]) -> SelectionPlan:
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"Selection plan does not exist: {source}.")
    try:
        namespace = runpy.run_path(str(source))
    except BaseException as error:
        raise CheckpointError(f"Failed to execute trusted selection plan {source}: {error}") from error
    plan = namespace.get("PLAN")
    if not isinstance(plan, SelectionPlan):
        raise CheckpointError(f"Selection plan {source} must define PLAN: SelectionPlan.")
    return plan


def override_plan(
    module: IRModule,
    choices: Sequence[tuple[str, str]],
    *,
    policy: str = "agent-override/v1",
    rationale: str = "",
) -> SelectionPlan:
    current = module.selection_map
    records: list[SelectionRecord] = []
    for point_id, candidate_id in choices:
        previous = current.get(point_id)
        records.append(
            SelectionRecord(
                point_id=point_id,
                candidate_id=candidate_id,
                origin="agent",
                policy=policy,
                rationale=rationale or (previous.rationale if previous else "Explicit CLI override."),
            ))
    return SelectionPlan(module.semantic_hash, tuple(records))
