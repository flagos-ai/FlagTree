"""Compile the complete GLM-5.3 model with the explicit local TP4 override."""

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

from agent_optimizations import activate
from tp4_distribution import Tp4DistributionTarget
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import DirectoryCheckpoint
from triton.flagmega.ir import emit_module, load_module, verify_buffer_plan, verify_module
from triton.flagmega.options import CompileOptions, parse_placement
from triton.flagmega.passes import FunctionalPass
from triton.flagmega.passes.constants import ConstnessAnalysis
from triton.flagmega.passes.functions import function_nodes, static_function_invocation_counts
from triton.flagmega.targets import NvidiaSm90Target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--integrity-report", type=Path,
                        help="Reuse a completed local source audit; otherwise digest source files during publication.")
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--gemv-compute-warps", type=int, choices=(8, 16), default=8)
    parser.add_argument("--arithmetic-profile", choices=("nncase", "vllm-0.31-tp4"), default="nncase",
                        help="Opt into the explicit native arithmetic experiment; official prefix acceptance is required.")
    parser.add_argument("--compile-only", action="store_true", help="Emit verified IR and source without materializing weights.")
    args = parser.parse_args()
    activate()
    args.trial.mkdir(parents=True, exist_ok=False)
    for name in ("tp4_distribution.py", "compile_tp4.py", "native_profile.py", "native_tp4.py", "native_mla.py"):
        shutil.copy2(Path(__file__).parent / name, args.trial / name)
    shutil.copytree(Path(__file__).parent / "agent_optimizations", args.trial / "agent_optimizations",
                    ignore=shutil.ignore_patterns("__pycache__"))
    started = time.perf_counter()
    module = load_module(args.input)
    native = args.arithmetic_profile == "vllm-0.31-tp4"
    if native:
        if args.gemv_compute_warps != 8:
            raise ValueError("The measured native profile requires 8 compute warps.")
        from native_profile import apply_native_profile
        module = apply_native_profile(module)
    load_seconds = time.perf_counter() - started
    if module.metadata.get("num_hidden_layers") != 45 or module.metadata.get("scope") != "text-decoder":
        raise ValueError("This override requires the complete 45-layer GLM-5.3 text decoder.")
    placement = parse_placement("cyx=4x8x16/cbb")
    target_type, catalog_options = Tp4DistributionTarget, {}
    if native:
        from native_tp4 import NativeTp4DistributionTarget, native_implementation_model
        target_type = NativeTp4DistributionTarget
        catalog_options = {"triton_implementation_model": native_implementation_model()}
    target = target_type(
        options=replace(NvidiaSm90Target().options, placements=(placement,)),
        evidence_dir=args.trial, gemv_compute_warps=args.gemv_compute_warps,
        **catalog_options,
    )
    compiler = Compiler(CompileOptions(target="nvidia-sm90", placement=placement))
    compiler.target = target
    original_run = FunctionalPass.run
    timings, acceptance = [], []

    def measured_run(module_pass, source):
        phase = time.perf_counter()
        result = original_run(module_pass, source)
        row = {"pass": module_pass.name, "seconds": time.perf_counter() - phase,
               "stage": result.stage, "nodes": len(result.nodes)}
        timings.append(row)
        print(json.dumps(row), flush=True)
        if result.stage in {"unused_functions_removed", "vector_contracts_lowered", "tuple_boxing_lowered"}:
            constants = ConstnessAnalysis.analyze(result).constants
            reachable = static_function_invocation_counts(result)
            counts = [{"function": f.name,
                       "compute_pack": sum(n.op == "tensors.pack" and n.id not in constants
                                           for n in function_nodes(result, f))}
                      for f in result.functions if f.name in reachable]
            if any(c["compute_pack"] for c in counts):
                emit_module(result, args.trial / f"rejected-{result.stage}.py")
                rejected = [{"function": function.name, "id": node.id, "inputs": node.inputs,
                             "type": str(node.type), "attrs": dict(node.attrs)}
                            for function in result.functions if function.name in reachable
                            for node in function_nodes(result, function)
                            if node.op == "tensors.pack" and node.id not in constants]
                (args.trial / "rejected-compute-pack.json").write_text(json.dumps(rejected, indent=2) + "\n")
                raise AssertionError(counts)
            acceptance.append({"stage": result.stage, "functions": counts})
        return result

    FunctionalPass.run = measured_run
    compile_started = time.perf_counter()
    try:
        result = compiler.compile(module)
        if native:
            from native_tp4 import rewrite_mla_schedules, rewrite_qkv_norm_schedules
            result = replace(result, module=verify_module(replace(result.module,
                kernel_definitions=rewrite_mla_schedules(result.module))))
            result = replace(result, module=verify_module(replace(result.module,
                kernel_definitions=rewrite_qkv_norm_schedules(result.module))))
    finally:
        FunctionalPass.run = original_run
    compile_seconds = time.perf_counter() - compile_started
    plan = verify_buffer_plan(result.module)
    binding = describe_function_runtime_binding(result.module)
    external = {plan.buffer_map[a["buffer"]].mem_span.buffer.id:
                plan.buffer_map[a["buffer"]].mem_span.buffer.nbytes for a in binding["arguments"]}
    budget = {"rdata_per_chip_bytes": plan.rdata_bytes,
              "external_bytes": sum(external.values()), "pools": binding["pools"],
              "minimum_bytes_per_chip": sum(external.values()) + sum(p["nbytes"] for p in binding["pools"])}
    (args.trial / "memory-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    print(json.dumps({"memory_budget": budget}), flush=True)
    publish_started = time.perf_counter()
    if args.compile_only:
        from triton.flagmega.codegen.triton import render_triton_package
        emit_module(result.module, args.trial / "packaged.py")
        render_triton_package(result.module, args.trial / "codegen")
        artifact = None
    else:
        artifact = write_artifact(
            result.module, args.trial / "artifact", target="nvidia-sm90", emit_executable=True,
            checkpoint=DirectoryCheckpoint(args.checkpoint, integrity_report=args.integrity_report),
            rdata_cache_dir=None,
        )
    publish_seconds = time.perf_counter() - publish_started
    report = {"input": str(args.input.resolve()), "input_semantic_hash": module.semantic_hash,
              "output_semantic_hash": result.module.semantic_hash,
              "artifact": None if artifact is None else str(artifact.resolve()),
              "arithmetic_profile": args.arithmetic_profile, "compile_only": args.compile_only,
              "gemv_compute_warps": args.gemv_compute_warps,
              "override_sha256": hashlib.sha256((Path(__file__).parent / "tp4_distribution.py").read_bytes()).hexdigest(),
              "load_seconds": load_seconds, "compile_seconds": compile_seconds,
              "publish_seconds": publish_seconds, "rdata_cache": None,
              "integrity_report": None if args.integrity_report is None else str(args.integrity_report.resolve()),
              "total_seconds": time.perf_counter() - started, "timings": timings,
              "memory_budget": budget, "ir_acceptance": acceptance,
              "numerical_correctness_verified": False, "performance_verified": False}
    (args.trial / "compile-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"timings","ir_acceptance"}}), flush=True)


if __name__ == "__main__":
    main()
