"""Rebuild the accepted native-BF16 decode policy through normal compiler stages."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

from agent_optimizations.decode import QKV_PIPELINE, create_target, install
from agent_optimizations.shared_route import fuse_shared_route
from agent_optimizations.residual_stats import fuse_residual_stats
from optimize import input_module
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.diagnostics import DumpFlags
from triton.flagmega.importer import DirectoryCheckpoint
from triton.flagmega.options import CompileOptions
from triton.flagmega.passes.functions import lift_constant_parameter_expressions
from triton.flagmega.passes.manager import FunctionalPass, PassManager
from triton.flagmega.selection import emit_plan, load_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Imported, distribution_candidates, or frozen_constants Python IR")
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--rdata-cache-dir", type=Path)
    parser.add_argument("--bufferize-opt-level", choices=("fast", "optimized"), default="optimized")
    parser.add_argument("--qkv-kernel", choices=("mma", "gemv"), default="mma",
                        help="Validated K64 two-stage P/C, or the matched merged-GEMV baseline")
    parser.add_argument("--distribution-plan", type=Path, help="Trusted SelectionPlan bound to this exact proposal")
    args = parser.parse_args()
    args.trial.mkdir(parents=True, exist_ok=False)
    install()
    checkpoint = DirectoryCheckpoint(args.checkpoint)
    module = input_module(checkpoint, args.input, "nncase")
    fm.verify_module(module)
    if module.stage not in {"imported", "distribution_candidates", "frozen_constants"}:
        raise ValueError("Resume before implementation selection, not from an allocated artifact")
    if args.distribution_plan and module.stage == "frozen_constants":
        raise ValueError("A distribution plan cannot change an already distributed input")
    root = Path(__file__).parent
    paths = (Path(__file__), root / "optimize.py", *sorted((root / "agent_optimizations").rglob("*.py")),
             *sorted((root / "agent_optimizations").rglob("*.jinja")),
             *sorted((root / "local_optimizations").rglob("*.py")))
    sources = {}
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root)
        destination = args.trial / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    fm.emit_module(module, args.trial / "input.py")
    compiler = Compiler(CompileOptions(work_dir=args.trial / "dumps",
        dump_flags=DumpFlags.COMPILE | DumpFlags.PASS_IR, bufferize_opt_level=args.bufferize_opt_level))
    compiler.target = create_target(qkv_kernel=args.qkv_kernel, bufferize_opt_level=args.bufferize_opt_level)
    started = perf_counter()
    original_hash = module.semantic_hash
    if module.stage == "imported":
        module = lift_constant_parameter_expressions(fuse_shared_route(module, fuse_router=True))
        fm.emit_module(module, args.trial / "prepared_import.py")
        module = compiler.compile(module, stop_after="propose-distribution").module
    if module.stage == "distribution_candidates":
        plan = load_plan(args.distribution_plan) if args.distribution_plan else None
        if plan is not None:
            emit_plan(plan, args.trial / "distribution.plan.py")
        module = compiler.run_stage(module, "auto-distributed", plan=plan).module
        module = compiler.compile(module, stop_after="freeze-constants").module
    prepared = PassManager("DecodeResidualStats").add(
        FunctionalPass("FuseResidualStats", fuse_residual_stats)).run(module).module
    fm.emit_module(prepared, args.trial / "distributed.py")
    result = compiler.compile(prepared).module
    fm.emit_module(result, args.trial / "final.py")
    implementations = Counter(value.dispatch.microkernel.implementation for value in result.kernel_definitions)
    if args.qkv_kernel == "mma" and not implementations[QKV_PIPELINE]:
        raise ValueError("The requested QKV P/C implementation was not selected")
    report = {"input_hash": original_hash, "distributed_hash": prepared.semantic_hash,
              "semantic_hash": result.semantic_hash, "numerical_contract": "nncase", "stage": result.stage,
              "qkv_kernel": args.qkv_kernel, "bufferize_opt_level": args.bufferize_opt_level,
              "implementations": dict(implementations), "sources": sources,
              "note": "Compilation is not numerical or performance acceptance."}
    if not args.compile_only:
        artifact = write_artifact(result, args.trial / "artifact", target=compiler.target.name,
                                  emit_executable=True, checkpoint=checkpoint, rdata_cache_dir=args.rdata_cache_dir)
        report["source_sha256"] = hashlib.sha256((artifact / "generated_kernels.py").read_bytes()).hexdigest()
    for relative, digest in sources.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during build: {relative}")
    report["compile_seconds"] = perf_counter() - started
    (args.trial / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
