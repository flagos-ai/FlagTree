"""Import, specialize and resume through the normal FlagMega compiler.

Each profile is an immutable trial directory. No generated-source edits or
hidden build/flagmega dependencies. Put this directory on PYTHONPATH for resume.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.diagnostics import DumpFlags
from triton.flagmega.importer import DirectoryCheckpoint, import_model, apply_numerical_profile, VLLM_INDUCTOR_LEVEL3
from triton.flagmega.ir import emit_module, load_module
from triton.flagmega.options import CompileOptions
from triton.flagmega.passes import PassManager, PIPELINE_GROUPS, expand_pipeline_passes
from triton.flagmega.stages import get_stage
from triton.flagmega.passes.auto_distributed import DistributedCandidateProviderRegistry, build_search_graph, solve_search_graph
from triton.flagmega.passes.auto_distributed.materializer import distribution_selection_state
from triton.flagmega.selection import emit_plan, override_plan

from local_optimizations import create_target, serving_passes
from local_optimizations.distribution_recipe import CHOICES, sharded_residual_norm_choices, replicated_norm_choices
from local_optimizations.kernel_recipe import select_tir


def select_distribution(module, target, *, residual_layout="auto", norm_layout="auto"):
    points = {point.id: point for point in module.selection_points if point.kind == "distribution"}
    choices = []
    for point_id, candidate_id in CHOICES:
        if point_id not in points:
            raise ValueError(f"The workload changed: missing distribution point {point_id}")
        if not any(candidate.id == candidate_id for candidate in points[point_id].candidates):
            raise ValueError(f"Recipe candidate is no longer legal: {candidate_id}")
        choices.append((point_id, candidate_id))
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    placements = target.distributed_placements(module)
    if len(placements) != 1:
        raise ValueError("Choose a single mesh before applying this workload recipe")
    graph = build_search_graph(module, placements[0], registry,
                               target.distributed_reshard_realization_policy(),
                               target.distributed_reshard_cost_model(), target.distributed_operation_cost_model())
    fixed = {point.removeprefix("distribution."): candidate for point, candidate in choices}
    if residual_layout in {"sharded", "sharded-casts"}:
        fixed.update(sharded_residual_norm_choices(graph, module, shard_casts=residual_layout == "sharded-casts"))
    if norm_layout in {"replicated", "replicated-residual"}:
        fixed.update(replicated_norm_choices(graph, module, residual_only=norm_layout == "replicated-residual"))
    result = solve_search_graph(graph, fixed_selections=fixed)
    _, records = distribution_selection_state(result, policy=target.distribution_policy.identity)
    return override_plan(module, [(record.point_id, record.candidate_id) for record in records],
                         rationale=f"H800 batch-1 QKV split-K, residual layout={residual_layout}, norm layout={norm_layout}; re-solved all coupled layout constraints")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rdata-cache-dir", type=Path, help="Share immutable packed weights between trial directories")
    parser.add_argument("--profile", choices=("baseline", "scheduled", "serving"), required=True)
    parser.add_argument("--residual-layout", choices=("auto", "sharded", "sharded-casts"), default="auto")
    parser.add_argument("--norm-layout", choices=("auto", "replicated", "replicated-residual"), default="auto")
    parser.add_argument("--residual-kernel", choices=("direct", "staged", "async"), default="direct")
    parser.add_argument("--glu-reduction-group", type=int, choices=(32, 64, 128), default=32)
    reselect = parser.add_mutually_exclusive_group()
    reselect.add_argument("--reselect-distribution", action="store_true",
                        help="Explicitly apply new local choices to a saved distribution proposal")
    reselect.add_argument("--reselect-tir", action="store_true",
                         help="Explicitly apply new kernel choices to a saved TIR proposal")
    parser.add_argument("--input", type=Path, help="Trusted Python IR file or Before/After directory for edit-and-resume")
    parser.add_argument("--stop-after", help="Normal compiler stage, e.g. propose-tir")
    parser.add_argument("--publish-kernels", type=Path, help="Export generated Python source, never weights")
    args = parser.parse_args()
    trial = args.work_dir / args.profile
    trial.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoint = DirectoryCheckpoint(args.checkpoint)
    module = (load_module(args.input) if args.input
              else import_model(checkpoint, numerical_profile=VLLM_INDUCTOR_LEVEL3))
    if module.stage == "imported":
        module = apply_numerical_profile(module, VLLM_INDUCTOR_LEVEL3)
    elif module.metadata.get("numerical_contract") != VLLM_INDUCTOR_LEVEL3:
        raise ValueError("Resume requires the tutorial's pinned numerical contract")
    emit_module(module, trial / "imported.py")
    compiler = Compiler(CompileOptions(
        work_dir=trial / "dumps", dump_flags=DumpFlags.COMPILE | DumpFlags.PASS_IR | DumpFlags.EGRAPH_COST))
    if args.profile != "baseline":
        compiler.target = create_target(glu_reduction_group=args.glu_reduction_group)
    if args.reselect_distribution and (module.stage != "distribution_candidates" or args.profile == "baseline"):
        raise ValueError("Reselect requires a saved distribution proposal and a scheduled/serving profile")
    if args.reselect_tir and (module.stage not in {"tuple_boxing_lowered", "selected_tir_variants"} or args.profile == "baseline"):
        raise ValueError("Reselect TIR requires a saved TIR proposal and a scheduled/serving profile")
    if module.stage != "imported" and not (args.reselect_distribution or args.reselect_tir):
        # Resume with the same workload implementation model that selected
        # the saved kernel parameters; no numerical or selection reapplication.
        module = compiler.compile(module, stop_after=args.stop_after).module
        finish(module, compiler, checkpoint, args, trial, started, resumed=True)
        return
    manager = PassManager("WorkloadSpecialization", dumper=
                          compiler.diagnostics.create_pass_manager_dumper("LocalWorkloadPasses"))
    if args.profile == "serving" and module.stage == "imported":
        for module_pass in serving_passes():
            manager.add(module_pass)
    module = manager.run(module).module
    emit_module(module, trial / "specialized.py")

    stages = [get_stage(member.stage) for group in PIPELINE_GROUPS
              for member in expand_pipeline_passes(group, compiler.target)]
    rank = {"imported": -1}
    for index, stage in enumerate(stages):
        rank[stage.name] = rank[stage.output_stage] = index
    if args.stop_after and args.stop_after not in rank:
        raise ValueError(f"Unknown stop stage {args.stop_after!r}")

    def pause():
        emit_module(module, trial / "final.py")
        print(f"Paused at {module.stage}: {trial / 'final.py'}", flush=True)

    def compile_to(boundary):
        nonlocal module
        requested = args.stop_after
        terminal = requested if requested and rank[requested] <= rank[boundary] else boundary
        if terminal != module.stage:
            module = compiler.compile(module, stop_after=terminal).module
        if requested and rank[requested] <= rank[boundary]:
            pause()
            return True
        return False

    if args.stop_after == "imported":
        pause()
        return
    if args.profile != "baseline":
        if not args.reselect_tir:
            if compile_to("propose-distribution"):
                return
            emit_module(module, trial / "distribution.proposal.py")
            plan = select_distribution(module, compiler.target, residual_layout=args.residual_layout, norm_layout=args.norm_layout)
            emit_plan(plan, trial / "distribution.plan.py")
            module = compiler.run_stage(module, "auto-distributed", plan=plan).module
            emit_module(module, trial / "distributed.py")
            if args.stop_after and rank[args.stop_after] == rank["auto-distributed"]:
                pause()
                return
        if compile_to("propose-tir"):
            return
        emit_module(module, trial / "tir.proposal.py")
        plan = select_tir(module, residual_kernel=args.residual_kernel)
        emit_plan(plan, trial / "tir.plan.py")
        module = compiler.run_stage(module, "lower-tir", plan=plan).module
        if args.stop_after and rank[args.stop_after] == rank["lower-tir"]:
            pause()
            return
    module = compiler.compile(module, stop_after=args.stop_after).module
    finish(module, compiler, checkpoint, args, trial, started)


def finish(module, compiler, checkpoint, args, trial, started, *, resumed=False):
    emit_module(module, trial / "final.py")
    if args.stop_after:
        print(f"Paused at {module.stage}: {trial / 'final.py'}", flush=True)
        return
    artifact = write_artifact(module, trial / "artifact", target=compiler.target.name,
                              checkpoint=checkpoint, emit_executable=True,
                              rdata_cache_dir=args.rdata_cache_dir or args.work_dir / ".rdata-cache")
    source = artifact / "generated_kernels.py"
    if args.publish_kernels:
        args.publish_kernels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, args.publish_kernels)
    report = {"profile": args.profile, "residual_layout": None if resumed or args.reselect_tir else args.residual_layout,
              "norm_layout": None if resumed or args.reselect_tir else args.norm_layout,
              "resumed_from": str(args.input) if args.input else None,
              "reselected_distribution": args.reselect_distribution,
              "reselected_tir": args.reselect_tir,
              "residual_kernel": None if resumed else args.residual_kernel,
              "glu_reduction_group": args.glu_reduction_group,
              "artifact": str(artifact), "semantic_hash": module.semantic_hash,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "compile_wall_seconds": time.perf_counter() - started}
    (trial / "compile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
