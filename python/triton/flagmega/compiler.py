# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Interruptible FlagMega compiler orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from triton.flagmega.diagnostics import DiagnosticSession, DumpFlags, StageReport
from triton.flagmega.errors import ReviewRequired, StageError
from triton.flagmega.ir import IRModule, emit_module, load_module, verify_module
from triton.flagmega.options import CompileOptions
from triton.flagmega.passes import (
    PIPELINE_GROUPS,
    FunctionalPass,
    PassManager,
    expand_pipeline_passes,
)
from triton.flagmega.selection import SelectionPlan, apply_plan
from triton.flagmega.stages import get_stage, stage_names
from triton.flagmega.targets import get_target


@dataclass(frozen=True)
class CompileResult:
    module: IRModule
    reports: tuple[StageReport, ...]
    checkpoint: Path | None = None


class Compiler:
    def __init__(self, options: CompileOptions | None = None) -> None:
        self.options = options or CompileOptions()
        if self.options.effective_dump_flags and self.options.work_dir is None:
            raise StageError("Dump flags require CompileOptions.work_dir (CLI: --work-dir).")
        self.target = get_target(self.options.target)
        self.diagnostics = DiagnosticSession(
            self.options.work_dir,
            dump_flags=self.options.effective_dump_flags,
        )
        self._pass_manager_index = 0

    def run_stage(
        self,
        module: IRModule,
        stage_name: str,
        *,
        plan: SelectionPlan | None = None,
        output: str | Path | None = None,
    ) -> CompileResult:
        verify_module(module)
        if plan is not None:
            module = apply_plan(module, plan)
            verify_module(module)
        stage = get_stage(stage_name)
        started = time.perf_counter()
        manager_name = f"{self._pass_manager_index:02d}_{stage.name}"
        self._pass_manager_index += 1
        pass_manager = PassManager(
            manager_name,
            dumper=self.diagnostics.create_pass_manager_dumper(manager_name),
        )
        pass_manager.add(FunctionalPass(stage.name, lambda value: stage.run(value, self.target)))
        pass_result = pass_manager.run(module)
        result = pass_result.module
        checkpoint = None
        if output is not None:
            checkpoint = emit_module(result, output)
        report = self.diagnostics.record(
            stage_name,
            module,
            result,
            started,
            dump_name=manager_name,
            pass_executions=pass_result.executions,
        )
        if stage.selection_point and self.options.require_review:
            raise ReviewRequired(
                f"Stage {stage.name!r} produced a selection point and review is required.",
                stage=result.stage,
            )
        return CompileResult(result, (report, ), checkpoint)

    def compile(self, module: IRModule, *, stop_after: str | None = None) -> CompileResult:
        verify_module(module)
        self.diagnostics.dumper.create_sub_dumper("Import").dump_module(
            module,
            "IRImport",
            category=DumpFlags.IMPORT_OPS,
        )
        reports: list[StageReport] = []
        current = module
        # A trusted checkpoint may already be at the requested boundary.
        # Checking only after scheduling another pass consumes later stages
        # (including decisions/freeze) before an agent can edit the proposal.
        stop_stages = {stop_after}
        if stop_after in stage_names():
            stop_stages.add(get_stage(stop_after).output_stage)
        stopped = current.stage in stop_stages
        for group in PIPELINE_GROUPS:
            if stopped:
                break
            if current.stage not in group.active_stages:
                continue
            manager_name = f"{self._pass_manager_index:02d}_{group.name}"
            self._pass_manager_index += 1
            pass_manager = PassManager(
                manager_name,
                dumper=self.diagnostics.create_pass_manager_dumper(manager_name),
            )
            simulated_stage = current.stage
            review_stage: str | None = None
            for member in expand_pipeline_passes(group, self.target):
                stage = get_stage(member.stage)
                if not stage.accepts(simulated_stage):
                    continue
                pass_manager.add(FunctionalPass(
                    member.name,
                    lambda value, selected_stage=stage: selected_stage.run(value, self.target),
                ))
                simulated_stage = stage.output_stage
                if stop_after is not None and stop_after in {stage.name, stage.output_stage}:
                    stopped = True
                    break
                if stage.selection_point and self.options.require_review:
                    review_stage = stage.output_stage
                    break
            source = current
            started = time.perf_counter()
            pass_result = pass_manager.run(current)
            current = pass_result.module
            if not stopped and review_stage is None and current.stage != group.output_stage:
                raise StageError(
                    f"Pass manager {group.name!r} expected output stage {group.output_stage!r}, "
                    f"got {current.stage!r}.",
                    stage=current.stage,
                )
            report = self.diagnostics.record(
                group.name,
                source,
                current,
                started,
                dump_name=manager_name,
                pass_executions=pass_result.executions,
            )
            reports.append(report)
            if review_stage is not None:
                raise ReviewRequired(
                    f"Pass manager {group.name!r} produced selection stage {review_stage!r} and review is required.",
                    stage=review_stage,
                )
            if stopped:
                break
        if not stopped and current.stage != "bufferized_tir":
            raise StageError(f"No nncase-shaped pipeline group accepts stage {current.stage!r}.")
        checkpoint = None
        if self.options.work_dir is not None:
            checkpoint = emit_module(current, self.options.work_dir / "final.py")
        return CompileResult(current, tuple(reports), checkpoint)

    def compile_checkpoint(
        self,
        input_path: str | Path,
        *,
        output: str | Path | None = None,
        stop_after: str | None = None,
    ) -> CompileResult:
        result = self.compile(load_module(input_path), stop_after=stop_after)
        checkpoint = result.checkpoint
        if output is not None:
            checkpoint = emit_module(result.module, output)
        return CompileResult(result.module, result.reports, checkpoint)
