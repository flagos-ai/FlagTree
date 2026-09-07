# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import load_artifact
from triton.flagmega.cli import main
from triton.flagmega.compiler import Compiler
from triton.flagmega.diagnostics import DumpFlags, DumpManager, parse_dump_flags
from triton.flagmega.errors import IRVerificationError, ReviewRequired
from triton.flagmega.options import CompileOptions
from triton.flagmega.passes import (
    PIPELINE_GROUPS,
    FunctionalPass,
    PassManager,
    expand_pipeline_passes,
)
from triton.flagmega.selection import apply_plan, load_plan, override_plan
from triton.flagmega.stages import next_stage, stage_names


def make_fp8_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported", metadata={"model": "fp8-unit"})
    activation = fm.tensor_type("bfloat16", [1, 128])
    weight = fm.tensor_type("float8_e4m3fn", [128, 128])
    scale = fm.tensor_type("float32", [1, 1])
    source = builder.var("source", activation, id="source")
    rhs = builder.weight("rhs", weight, source="weights.safetensors", key="rhs", id="rhs")
    rhs_scale = builder.weight("rhs_scale", scale, source="weights.safetensors", key="rhs_scale", id="rhs_scale")
    result = builder.call(
        "math.block_scaled_matmul",
        [source, rhs, rhs_scale],
        activation,
        id="result",
        attrs={"weight_block_n": 128, "weight_block_k": 128},
    )
    builder.function("main", [source], [result])
    return builder.build(entry="main")


def test_compiler_completes_without_agent_and_records_defaults(tmp_path):
    module = make_fp8_module()
    result = Compiler(CompileOptions(
        work_dir=tmp_path,
        dump_flags=DumpFlags.PASS_IR | DumpFlags.COMPILE,
    )).compile(module)

    assert result.module.stage == "bufferized_tir"
    assert result.module.dialect == "bufferized_tir"
    assert [report.stage for report in result.reports] == [
        "TargetIndependentPass",
        "AutoVectorizePass",
        "AutoPackingPass",
        "AutoDistributedPass",
        "TIRPass",
    ]
    assert [len(report.pass_executions) for report in result.reports] == [3, 2, 8, 10, 18]
    assert {record.origin for record in result.module.selections} == {"default-policy", "ortools-cp-sat"}
    assert {point.kind for point in result.module.selection_points} == {"distribution", "packing", "tir"}
    assert (tmp_path / "final.py").is_file()
    assert (tmp_path / "final.script").is_file()
    assert (tmp_path / "Compile" / "00_TargetIndependentPass" / "Before" / "main.il").is_file()
    assert (tmp_path / "Compile" / "04_TIRPass" / "After" / "main.script").is_file()
    before = tmp_path / "00_TargetIndependentPass" / "00_DecomposeComplexOps" / "Before"
    after = tmp_path / "00_TargetIndependentPass" / "00_DecomposeComplexOps" / "After"
    assert (before / "main.py").is_file()
    assert (before / "main.il").is_file()
    assert (after / "main.py").is_file()
    assert (tmp_path / "00_TargetIndependentPass" / "Before" / "main.py").is_file()
    assert (tmp_path / "00_TargetIndependentPass" / "After" / "main.py").is_file()
    freeze = tmp_path / "04_TIRPass" / "02_FreezeConstantIslands"
    tuple_boxing = tmp_path / "04_TIRPass" / "06_LowerTupleBoxing"
    lower = tmp_path / "04_TIRPass" / "08_LowerSelectedTIR"
    canonicalize_qkv = (
        tmp_path / "04_TIRPass" / "09_CanonicalizePackedQKVWeights"
    )
    select_microkernels = (
        tmp_path / "04_TIRPass" / "11_SelectTIRMicroKernels"
    )
    assert (freeze / "After" / "main.il").is_file()
    assert (tuple_boxing / "Before" / "main.py").is_file()
    assert (tuple_boxing / "After" / "main.py").is_file()
    assert (lower / "After" / "main.script").is_file()
    assert (canonicalize_qkv / "After" / "main.script").is_file()
    assert (select_microkernels / "After" / "main.script").is_file()
    dump_entries = json.loads(
        (tmp_path / "dumps.json").read_text(encoding="utf-8")
    )
    # Each small pass has Before/After pass-IR dumps; each large group also
    # has Before/After pass-IR plus Before/After compile snapshots. The one
    # PostFunctionBoundary EGraph pass additionally exposes its automatically
    # inserted Construct/Rules/Extract manager boundaries (8 module dumps).
    assert len(dump_entries) == (
        2 * sum(len(report.pass_executions) for report in result.reports)
        + 4 * len(result.reports)
        + 8
    )
    first_pass = result.reports[0].pass_executions[0]
    assert first_pass.name == "DecomposeComplexOps"
    assert first_pass.before_dump is not None
    assert first_pass.before_functions[0].function == "main"
    assert first_pass.after_functions[0].text_dump.suffix == ".il"


def test_pipeline_contains_only_implemented_pass_boundaries():
    target = Compiler().target
    groups = {
        group.name: tuple(
            item.name for item in expand_pipeline_passes(group, target)
        )
        for group in PIPELINE_GROUPS
    }

    assert tuple(groups) == (
        "TargetIndependentPass",
        "AutoVectorizePass",
        "AutoPackingPass",
        "AutoDistributedPass",
        "TIRPass",
    )
    assert groups == {
        "TargetIndependentPass": (
            "DecomposeComplexOps",
            "FormQKVRoPEWithCache",
            "HoistCallInvariantExpressions",
        ),
        "AutoVectorizePass": ("AutoVectorize", "ApplyVectorization"),
            "AutoPackingPass": (
                "AutoPacking",
            "ApplyPacking",
            "FunctionBoundaryLayoutPropagation",
            "PostFunctionBoundaryPackPropagation",
            "ThreadNormStatsAcrossFunctionBoundaries",
            "DecomposePagedAttention",
            "FormMatMulNormStatsCombine",
            "RemoveUnusedFunctions",
        ),
        "AutoDistributedPass": (
            "FormMatMulNormStatsCombine",
            "RemoveUnusedFunctions",
            "ProposeAutoDistributed",
            "AutoDistributed",
            "FoldMaterializedPackedQKVParallelLinearCombine",
            "LowerPackedQKVParallelLinearCombine",
            "SinkNormStatsBoxingAcrossFunctionBoundaries",
            "PropagatePostAutoDistributedFunctionBoundaryLayouts",
            "FinalizeNormStatsBindings",
            "SinkFinalizedNormStatsBoxingAcrossFunctionBoundaries",
            "LowerMatMulNormStatsCombine",
            "LowerVectorizationContracts",
        ),
        "TIRPass": (
            "FuseNormStatsApply",
            "ConstantCSE",
            "FreezeConstantIslands",
            "FuseGatherReduceAddNormApply",
            "FuseGatherReduceNormApply",
            "FuseGatherReduceQKVRoPEWithCache",
            "LowerTupleBoxing",
            "ProposeTIRCandidates",
            "LowerSelectedTIR",
            "CanonicalizePackedQKVWeights",
            "ProposeTIRMicroKernels",
            "SelectTIRMicroKernels",
            "FinalizeTIRPackage",
            "PlanFunctionMemory",
            "Bufferize",
            "MaterializeExecutionFunctions",
            "PlanMemorySynchronization",
            "LowerTransferPipelineRegions",
        ),
    }
    assert all(not hasattr(item, "verifier") for group in PIPELINE_GROUPS for item in group.passes)
    assert not {
        "canonicalize",
        "propose-egraph",
        "extract-egraph",
        "apply-distribution",
        "apply-tir",
    } & set(stage_names())


def test_compiler_can_stop_after_sat_bufferize_and_resume_synchronization():
    allocated = Compiler().compile(make_fp8_module(), stop_after="bufferize").module

    assert allocated.stage == "allocated_tir"
    assert allocated.metadata["buffer_plan"]["schema"] == "flagmega.buffer-plan/v6"
    assert "physical_buffers" in allocated.metadata["buffer_plan"]
    assert all(
        "mem_span" in buffer
        and "physical_id" not in buffer
        and "byte_offset" not in buffer
        and "alias_of" not in buffer
        for buffer in allocated.metadata["buffer_plan"]["buffers"]
    )
    assert "memory_synchronization" not in allocated.metadata

    resumed = Compiler().compile(allocated).module
    assert resumed.stage == "bufferized_tir"
    assert resumed.metadata["memory_synchronization"]["schema"] == (
        "flagmega.memory-synchronization/v1"
    )


def test_emit_every_stage_remains_compile_dump_alias(tmp_path):
    result = Compiler(CompileOptions(work_dir=tmp_path, emit_every_stage=True)).compile(make_fp8_module())

    assert (tmp_path / "Compile" / "00_TargetIndependentPass" / "Before" / "main.py").is_file()
    assert (tmp_path / "Compile" / "00_TargetIndependentPass" / "After" / "main.py").is_file()
    assert result.reports[0].checkpoint == str(
        tmp_path / "Compile" / "00_TargetIndependentPass" / "After")
    assert result.reports[0].before_checkpoint == str(
        tmp_path / "Compile" / "00_TargetIndependentPass" / "Before")
    assert result.reports[0].pass_executions[0].before_dump is None


def test_dump_flag_parser_scope_narrowing_and_pass_manager_freeze(tmp_path):
    assert parse_dump_flags("PassIR,Compile") == DumpFlags.PASS_IR | DumpFlags.COMPILE
    root = DumpManager(tmp_path, DumpFlags.PASS_IR).root
    child = root.create_sub_dumper("child", DumpFlags.PASS_IR | DumpFlags.COMPILE)
    assert child.is_enabled(DumpFlags.PASS_IR)
    assert not child.is_enabled(DumpFlags.COMPILE)
    manager = PassManager("unit")
    manager.seed_analysis("shape")
    manager.add(FunctionalPass("identity", lambda module: module, preserves=frozenset({"shape"})))

    result = manager.run(make_fp8_module())

    assert result.executed == ("identity", )
    assert result.invalidated_analyses == ()
    with pytest.raises(RuntimeError, match="frozen"):
        manager.add(FunctionalPass("late", lambda module: module))


def test_whole_compile_matches_explicit_stage_replay():
    module = make_fp8_module()
    whole = Compiler().compile(module).module
    current = module
    compiler = Compiler()
    while current.stage != "bufferized_tir":
        stage = next_stage(current.stage)
        assert stage is not None
        current = compiler.run_stage(current, stage.name).module

    assert current.semantic_hash == whole.semantic_hash


def test_grouped_pipeline_can_stop_after_one_small_pass(tmp_path):
    result = Compiler(CompileOptions(work_dir=tmp_path, dump_flags=DumpFlags.PASS_IR)).compile(
        make_fp8_module(), stop_after="propose-packing")

    assert result.module.stage == "packing_candidates"
    assert [report.stage for report in result.reports] == [
        "TargetIndependentPass",
        "AutoVectorizePass",
        "AutoPackingPass",
    ]
    assert [item.name for item in result.reports[-1].pass_executions] == ["AutoPacking"]
    assert (tmp_path / "02_AutoPackingPass" / "00_AutoPacking" / "After" / "main.il").is_file()


def test_grouped_pipeline_require_review_stops_inside_manager(tmp_path):
    compiler = Compiler(CompileOptions(
        work_dir=tmp_path,
        dump_flags=DumpFlags.PASS_IR,
        require_review=True,
    ))

    with pytest.raises(ReviewRequired, match="AutoVectorizePass") as raised:
        compiler.compile(make_fp8_module())

    assert raised.value.stage == "vectorization_candidates"
    assert (
        tmp_path / "01_AutoVectorizePass" / "00_AutoVectorize" / "After" / "main.il"
    ).is_file()
    assert not (tmp_path / "01_AutoVectorizePass" / "01_ApplyVectorization").exists()


def test_agent_override_is_hash_bound_and_replaces_default():
    compiler = Compiler()
    current = make_fp8_module()
    for stage_name in (
        "decompose-gdn",
        "form-qkv-rope-with-cache",
        "propose-vectorization",
        "apply-vectorization",
            "propose-packing",
            "apply-packing",
            "propagate-function-boundary-layouts",
            "post-function-boundary-pack-propagation",
            "thread-norm-stats",
        "form-matmul-norm-stats-combine",
        "auto-distributed",
    ):
        current = compiler.run_stage(current, stage_name).module
    point = next(point for point in current.selection_points if point.kind == "distribution")
    plan = override_plan(current, ((point.id, point.default_candidate), ), rationale="unit override")

    overridden = apply_plan(current, plan)

    assert overridden.selection_map[point.id].candidate_id == point.default_candidate
    assert overridden.selection_map[point.id].origin == "agent"
    with pytest.raises(IRVerificationError, match="semantic hash"):
        apply_plan(overridden, plan)


def test_agent_cli_json_and_artifact_round_trip(tmp_path, capsys):
    imported = fm.emit_module(make_fp8_module(), tmp_path / "imported.py")
    artifact_dir = tmp_path / "artifact"

    assert main([
        "compile",
        "--input",
        str(imported),
        "--output",
        str(artifact_dir),
        "--work-dir",
        str(tmp_path / "work"),
        "--json",
    ]) == 0
    compile_result = json.loads(capsys.readouterr().out)
    assert compile_result["schema"] == "flagmega.cli/v1"
    assert compile_result["ok"] is True
    assert compile_result["module"]["stage"] == "bufferized_tir"

    manifest, loaded = load_artifact(artifact_dir)
    assert manifest["semantic_hash"] == loaded.semantic_hash
    assert main(["artifact", "verify", str(artifact_dir), "--json"]) == 0
    verify_result = json.loads(capsys.readouterr().out)
    assert verify_result["ok"] is True


def test_cli_dump_flags_create_hierarchical_pass_and_compile_dumps(tmp_path, capsys):
    imported = fm.emit_module(make_fp8_module(), tmp_path / "imported.py")
    work = tmp_path / "work"

    assert main([
        "compile",
        "--input",
        str(imported),
        "--output",
        str(tmp_path / "artifact"),
        "--work-dir",
        str(work),
        "--dump-flags",
        "PassIR,Compile",
        "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["dump_flags"] == "pass-ir,compile"
    assert (work / "00_TargetIndependentPass" / "Before" / "main.il").is_file()
    assert (work / "00_TargetIndependentPass" / "After" / "main.il").is_file()
    assert (work / "Compile" / "04_TIRPass" / "Before" / "main.il").is_file()
    assert (work / "Compile" / "04_TIRPass" / "After" / "main.script").is_file()


def test_cli_can_write_selection_override(tmp_path, capsys):
    compiler = Compiler()
    current = make_fp8_module()
    for stage_name in (
        "decompose-gdn",
        "form-qkv-rope-with-cache",
        "propose-vectorization",
        "apply-vectorization",
            "propose-packing",
            "apply-packing",
            "propagate-function-boundary-layouts",
            "post-function-boundary-pack-propagation",
            "thread-norm-stats",
        "form-matmul-norm-stats-combine",
        "auto-distributed",
    ):
        current = compiler.run_stage(current, stage_name).module
    point = next(point for point in current.selection_points if point.kind == "distribution")
    checkpoint = fm.emit_module(current, tmp_path / "candidates.py")
    plan = tmp_path / "distribution.py"

    assert main([
        "select",
        str(checkpoint),
        "--point",
        point.id,
        "--candidate",
        point.default_candidate,
        "--output",
        str(plan),
        "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert plan.is_file()
    source = plan.read_text(encoding="utf-8")
    assert "fm.SelectionRecord(" in source
    assert "SPEC =" not in source
    assert "from_data" not in source
    loaded = load_plan(plan)
    assert loaded.input_semantic_hash == current.semantic_hash
    assert loaded.records[0].candidate_id == point.default_candidate
