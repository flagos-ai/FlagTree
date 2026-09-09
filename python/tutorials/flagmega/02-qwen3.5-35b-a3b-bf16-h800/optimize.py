"""Import or resume editable Python IR with explicit workload-local expert/GDN tiles.

Each invocation owns a fresh trial. Candidate/resource contracts are regenerated
through normal compiler stages; saved catalog selections are never patched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.diagnostics import DumpFlags
from triton.flagmega.importer import DirectoryCheckpoint, import_model
from triton.flagmega.ir import emit_module, load_module
from triton.flagmega.options import CompileOptions

from local_optimizations import create_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Trusted Python module or complete Before/After directory")
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--rdata-cache-dir", type=Path)
    parser.add_argument("--compile-only", action="store_true", help="Emit final IR without loading or packing weights")
    parser.add_argument("--stop-after")
    parser.add_argument("--bufferize-opt-level", choices=("fast", "optimized"), default="optimized")
    parser.add_argument("--gate-n", type=int, default=8)
    parser.add_argument("--gate-k", type=int, default=128)
    parser.add_argument("--down-n", type=int, default=8)
    parser.add_argument("--down-k", type=int, default=128)
    parser.add_argument("--gdn-value-tile", type=int, help="Override the recurrent value tile; default keeps the catalog")
    parser.add_argument("--gdn-projection-tile", type=int, help="Override the recurrent A/B projection K tile")
    args = parser.parse_args()
    if args.trial.exists():
        raise ValueError("Use a fresh trial directory; previous trials are immutable")
    target = create_target(gate_n=args.gate_n, gate_k=args.gate_k, down_n=args.down_n, down_k=args.down_k,
                           gdn_value_tile=args.gdn_value_tile, gdn_projection_tile=args.gdn_projection_tile,
                           bufferize_opt_level=args.bufferize_opt_level)
    checkpoint = DirectoryCheckpoint(args.checkpoint)
    module = load_module(args.input) if args.input else import_model(
        checkpoint, mode="decode-1", num_tokens=1,
        revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        numerical_profile="vllm-ae10e855a-inductor-level3")
    if module.metadata.get("numerical_contract") != "vllm-ae10e855a-inductor-level3":
        raise ValueError("The tutorial requires its pinned vLLM numerical contract")
    args.trial.mkdir(parents=True, exist_ok=False)
    source_paths = (Path(__file__), Path(__file__).parent / "local_optimizations" / "target.py")
    identity = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    for path in source_paths:
        (args.trial / path.name).write_bytes(path.read_bytes())
    emit_module(module, args.trial / "input.py")
    compiler = Compiler(CompileOptions(work_dir=args.trial / "dumps", bufferize_opt_level=args.bufferize_opt_level,
                                      dump_flags=DumpFlags.COMPILE | DumpFlags.PASS_IR))
    compiler.target = target
    started = time.perf_counter()
    result = compiler.compile(module, stop_after=args.stop_after)
    emit_module(result.module, args.trial / "final.py")
    report = {"input": str(args.input.resolve()) if args.input else "fresh import", "input_hash": module.semantic_hash,
              "semantic_hash": result.module.semantic_hash, "stage": result.module.stage,
              "local_sources": identity, "expert_tiles": {"gate": [args.gate_n, args.gate_k],
                                                           "down": [args.down_n, args.down_k]},
              "gdn_tiles": {"value": args.gdn_value_tile, "projection": args.gdn_projection_tile},
              "bufferize_opt_level": args.bufferize_opt_level, "compile_seconds": time.perf_counter() - started}
    if not args.compile_only and not args.stop_after:
        artifact = write_artifact(result.module, args.trial / "artifact", target=target.name, checkpoint=checkpoint,
                                  emit_executable=True, rdata_cache_dir=args.rdata_cache_dir)
        report["artifact"] = str(artifact)
        report["source_sha256"] = hashlib.sha256((artifact / "generated_kernels.py").read_bytes()).hexdigest()
    (args.trial / "compile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
