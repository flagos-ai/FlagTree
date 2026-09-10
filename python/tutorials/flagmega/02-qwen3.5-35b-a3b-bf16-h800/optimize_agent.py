"""Replay accepted local passes from a fresh decode import, without frozen plans."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from agent_optimizations import IMPLEMENTATION, create_target, install
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.artifacts import write_artifact
from agent_optimizations.distribution_plan import bind_plan
from triton.flagmega.diagnostics import DumpFlags, DumpManager
from triton.flagmega.importer import DirectoryCheckpoint, import_model
from triton.flagmega.options import CompileOptions
from triton.flagmega.passes import FunctionalPass, PassManager
from agent_optimizations.routing.fusion import fuse_staged_routing
from agent_optimizations.scalar_scale.fusion import fuse_scalar_scale
from agent_optimizations.gated_epilogue.fusion import fuse_gated_epilogue
from agent_optimizations.sigmoid_product.fusion import fuse_sigmoid_product


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--bufferize-opt-level", choices=("fast", "optimized"), default="optimized")
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--rdata-cache-dir", type=Path)
    parser.add_argument("--distribution-plan", type=Path, help="Trusted edited SelectionPlan Python file")
    args = parser.parse_args()
    args.trial.mkdir(parents=True, exist_ok=False)
    install()
    started = perf_counter()
    module = fm.load_module(args.input) if args.input else import_model(
        DirectoryCheckpoint(args.checkpoint), mode="decode-1", num_tokens=1,
        revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        numerical_profile="vllm-ae10e855a-inductor-level3",
    )
    fm.verify_module(module)
    if module.stage not in {"imported", "distribution_candidates"}:
        raise ValueError("Resume from imported IR or a complete distribution proposal")
    assert module.metadata["numerical_contract"] == "vllm-ae10e855a-inductor-level3"
    fm.emit_module(module, args.trial / "imported.py")
    flags = DumpFlags.COMPILE | DumpFlags.PASS_IR
    compiler = Compiler(CompileOptions(
        work_dir=args.trial / "dumps", dump_flags=flags, bufferize_opt_level=args.bufferize_opt_level))
    compiler.target = create_target(bufferize_opt_level=args.bufferize_opt_level)
    proposal = compiler.compile(module, stop_after="propose-distribution").module
    from triton.flagmega.selection import emit_plan, load_plan
    plan = load_plan(args.distribution_plan) if args.distribution_plan else bind_plan(proposal)
    emit_plan(plan, args.trial / "distribution.plan.py")
    distributed = compiler.run_stage(proposal, "auto-distributed", plan=plan).module
    prepared = compiler.compile(distributed, stop_after="freeze-constants").module
    assert prepared.stage == "frozen_constants"
    manager = PassManager("AcceptedAgentFusions", dumper=DumpManager(args.trial / "fusion", flags).root)
    for name, transform in (
        ("StagedRouting", fuse_staged_routing),
        ("ScalarScale", fuse_scalar_scale),
        ("GatedResidualStats", fuse_gated_epilogue),
        ("SigmoidProduct", fuse_sigmoid_product),
    ):
        manager.add(FunctionalPass(name, transform))
    fused = manager.run(prepared).module
    counts = {op: sum(n.op == op for n in fused.nodes)
              for op in ("local.staged_routing", "local.gated_residual_norm_stats", "local.sigmoid_product")}
    if counts != {"local.staged_routing": 2, "local.gated_residual_norm_stats": 2, "local.sigmoid_product": 1}:
        raise ValueError(f"Accepted fusion coverage changed: {counts}")
    result = compiler.compile(fused).module
    if result.metadata["buffer_plan"]["optimization_level"] != args.bufferize_opt_level:
        raise ValueError("The compiled allocator does not match the requested level")
    selected = tuple(k.dispatch.microkernel.implementation for k in result.kernel_definitions
                     if k.dispatch.semantic_op == "ntt.paged_attention_partial")
    if selected != (IMPLEMENTATION,):
        raise ValueError(f"Attention choice changed: {selected}")
    fm.emit_module(result, args.trial / "final.py")
    if not args.compile_only:
        write_artifact(result, args.trial / "artifact", target=compiler.target.name, emit_executable=True,
                       checkpoint=DirectoryCheckpoint(args.checkpoint), rdata_cache_dir=args.rdata_cache_dir)
    source = Path(__file__).read_bytes()
    report = {
        "input": str(args.input) if args.input else "fresh import", "input_hash": module.semantic_hash,
        "pre_fusion_hash": prepared.semantic_hash, "fused_hash": fused.semantic_hash,
        "output_hash": result.semantic_hash, "stage": result.stage,
        "bufferize_opt_level": result.metadata["buffer_plan"]["optimization_level"],
        "fusions": counts, "implementation": IMPLEMENTATION,
        "script_sha256": hashlib.sha256(source).hexdigest(),
        "wall_seconds": perf_counter() - started,
        "note": "Numerical and performance acceptance are separate from compilation and packaging.",
    }
    import agent_optimizations
    package = Path(agent_optimizations.__file__).parent
    sources = [Path(__file__), *sorted(package.rglob("*.py")), *sorted(package.rglob("*.jinja"))]
    report["local_sources"] = {str(path.relative_to(package.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources}
    (args.trial / "optimize_agent.py").write_bytes(source)
    (args.trial / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
