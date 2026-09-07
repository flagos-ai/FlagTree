# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-compile diagnostic, pass dump, and checkpoint ownership."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from triton.flagmega.diagnostics.dump import DumpFlags, DumpManager, Dumper
from triton.flagmega.ir import IRModule

if TYPE_CHECKING:
    from triton.flagmega.passes.manager import PassExecution


@dataclass(frozen=True)
class StageReport:
    stage: str
    input_semantic_hash: str
    output_semantic_hash: str
    elapsed_ms: float
    checkpoint: str | None
    text_dump: str | None = None
    before_checkpoint: str | None = None
    before_text_dump: str | None = None
    after_checkpoint: str | None = None
    after_text_dump: str | None = None
    pass_executions: tuple[PassExecution, ...] = ()

    def to_data(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "input_semantic_hash": self.input_semantic_hash,
            "output_semantic_hash": self.output_semantic_hash,
            "elapsed_ms": self.elapsed_ms,
            "checkpoint": self.checkpoint,
            "text_dump": self.text_dump,
            "before_checkpoint": self.before_checkpoint,
            "before_text_dump": self.before_text_dump,
            "after_checkpoint": self.after_checkpoint,
            "after_text_dump": self.after_text_dump,
            "passes": [value.to_data() for value in self.pass_executions],
        }


class DiagnosticSession:
    def __init__(
        self,
        work_dir: str | Path | None = None,
        *,
        dump_flags: DumpFlags = DumpFlags.NONE,
    ) -> None:
        self.work_dir = None if work_dir is None else Path(work_dir)
        self.dump_manager = DumpManager(self.work_dir, dump_flags)
        self.dumper = self.dump_manager.root
        self.reports: list[StageReport] = []
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)

    def create_pass_manager_dumper(self, name: str) -> Dumper:
        return self.dumper.create_sub_dumper(name)

    def record(
        self,
        stage_name: str,
        source: IRModule,
        result: IRModule,
        started: float,
        *,
        dump_name: str | None = None,
        pass_executions: tuple[PassExecution, ...] = (),
    ) -> StageReport:
        checkpoint: str | None = None
        text_dump: str | None = None
        before_checkpoint: str | None = None
        before_text_dump: str | None = None
        compile_dumper = self.dumper.create_sub_dumper("Compile").create_sub_dumper(
            dump_name or f"{len(self.reports):02d}_{result.stage}")
        before = compile_dumper.dump_module(source, "Before", category=DumpFlags.COMPILE)
        after = compile_dumper.dump_module(result, "After", category=DumpFlags.COMPILE)
        if before is not None:
            before_entry = before.function(source.entry)
            # A multi-function stage is resumed from the directory.  Pointing
            # at only ``main.py`` silently drops reusable callees edited in
            # sibling files.
            before_checkpoint = str(before.directory)
            before_text_dump = str(before_entry.text_dump)
        if after is not None:
            after_entry = after.function(result.entry)
            checkpoint = str(after.directory)
            text_dump = str(after_entry.text_dump)
        report = StageReport(
            stage=stage_name,
            input_semantic_hash=source.semantic_hash,
            output_semantic_hash=result.semantic_hash,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            checkpoint=checkpoint,
            text_dump=text_dump,
            before_checkpoint=before_checkpoint,
            before_text_dump=before_text_dump,
            after_checkpoint=checkpoint,
            after_text_dump=text_dump,
            pass_executions=pass_executions,
        )
        self.reports.append(report)
        if self.work_dir is not None:
            _write_manifest(self.work_dir / "stages.json", self.reports)
        return report


def _write_manifest(destination: Path, reports: list[StageReport]) -> None:
    payload = {
        "schema": "flagmega.stage-reports/v1",
        "stages": [value.to_data() for value in reports],
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False,
        prefix=f".{destination.name}.",
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, destination)
