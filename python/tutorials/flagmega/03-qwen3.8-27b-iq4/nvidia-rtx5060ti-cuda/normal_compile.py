"""Original-GGUF reconstruction through accepted typed stages and normal writer.

This worker receives only the import produced by compile.py in this fresh trial.
It never loads an old physical checkpoint or edits generated source/native code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RECIPE = HERE / "local_optimizations/recipes/u4_l_fullhead48.json"
GRAPH_STAGE = "form-u4-full12-input-norm-common-c0-c1-c2"


def validate_options(args):
    if args.input or args.layer is not None or args.cpp_operator_profile or args.cpp_max_registers is not None:
        raise ValueError("Ordinary reconstruction requires original full GGUF and the accepted compiler resource policy")
    if args.placement != "yx=4x24/bb":
        raise ValueError("Ordinary reconstruction requires --placement yx=4x24/bb")
    if args.recipe is None or json.loads(args.recipe.read_text()) != json.loads(RECIPE.read_text()):
        raise ValueError("Ordinary reconstruction requires the accepted u4_l_fullhead48 recipe")


def register_recipe(recipe_path):
    from local_optimizations import install, reset49
    from local_optimizations.u4_common_q8_kernels import register
    from local_optimizations.u4_norm_q8_c1_c2_regions import register_input_norm_common_q8_stage
    recipe = json.loads(Path(recipe_path).read_text())
    if recipe != json.loads(RECIPE.read_text()):
        raise ValueError("Unexpected ordinary reconstruction recipe")
    install(**recipe)
    register()
    reset49.register()
    return register_input_norm_common_q8_stage()


def prepare_graph(compiler, stats):
    from triton.flagmega.ir import verify_module
    from local_optimizations.u4_execution_plan import prepare_u4_distribution
    from local_optimizations.u4_norm_q8_c1_c2_regions import verify_input_norm_common_q8
    graph = compiler.run_stage(stats, GRAPH_STAGE).module
    verify_input_norm_common_q8(graph)
    prepared = prepare_u4_distribution(graph, gdn_fullhead48=True)
    verify_module(prepared)
    return prepared


def reconstruct(imported, *, checkpoint, output, work_dir, rdata_cache_dir, recipe, schedule):
    from triton.flagmega import ir as fm
    from triton.flagmega.artifacts import write_artifact, load_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions
    from triton.flagmega.diagnostics import DumpFlags
    from triton.flagmega.importer.checkpoint import checkpoint_from_path
    from triton.flagmega.runtime.cpp_programs import verified_cpp_programs
    from local_optimizations import reset49
    from local_optimizations.u4_common_q8_distribution import validate_common_q8_physical_contract
    from download_checkpoint import REVISION, SHA256

    register_recipe(recipe)
    source = fm.load_module(imported)
    expected = dict(revision=REVISION, numerical_contract="nncase", num_hidden_layers=64,
                    scope="text-decoder", mode="decode-1", tokens_per_call=1)
    if any(source.metadata.get(k) != v for k, v in expected.items()) or source.stage != "imported":
        raise ValueError("Worker requires the fresh original full64 nncase decode-1 import")
    options = CompileOptions(target="nvidia-sm120", codegen_backend="cpp",
        placement=fm.Placement((4, 24), "yx", "bb"), work_dir=work_dir,
        dump_flags=DumpFlags.COMPILE | DumpFlags.PASS_IR | DumpFlags.EGRAPH_COST)
    compiler = Compiler(options)
    # Default Bufferize is the accepted D9 policy; do not substitute the old
    # public wrapper's explicit 'fast' allocator for that policy.
    phases = {}
    def phase(name, operation):
        start = time.perf_counter()
        value = operation()
        phases[name] = time.perf_counter() - start
        return value
    stats = phase("normal_to_stats_combined", lambda: compiler.compile(source, stop_after="stats_combined").module)
    fm.emit_module(stats, work_dir / "original_stats_combined.py")
    prepared = phase("accepted_graph_and_distribution_requests", lambda: prepare_graph(compiler, stats))
    fm.emit_module(prepared, work_dir / "prepared_stats_combined.py")
    physical = phase("normal_physical_lowering", lambda: compiler.compile(prepared).module)
    fm.verify_module(physical)
    plan = fm.verify_buffer_plan(physical)
    audit = validate_common_q8_physical_contract(physical, plan,
        records=prepared.metadata["u4_input_norm_q8_records"])
    fm.emit_module(physical, work_dir / "ordinary_bufferized_tir.py")
    current = reset49.apply(physical, schedule)
    fm.verify_module(current)
    fm.emit_module(current, work_dir / "scheduled_bufferized_tir.py")
    original = checkpoint_from_path(checkpoint)
    # This is the normal original readonly source. No CacheOnlyCheckpoint,
    # previous RDATA or old artifact is an input to the writer.
    phase("normal_writer_and_native_build", lambda: write_artifact(current, output,
        target="nvidia-sm120", checkpoint=original, emit_executable=True,
        codegen_backend="cpp", rdata_cache_dir=rdata_cache_dir, require_reflink=True,
        dumper=compiler.diagnostics.dumper))
    manifest, loaded = load_artifact(output, verify_rdata_payload=False)
    programs = verified_cpp_programs(manifest["codegen"], loaded)
    epoch = manifest["codegen"]["cpp_counter_epoch_plan"]
    resets = 49 if schedule == "candidate49" else 275
    if (len(programs["entries"]), len(programs["invocations"]), len(epoch["resets"])) != (33, 869, resets):
        raise ValueError("Ordinary complete invocation/reset census changed")
    imports = {}
    for name, module in tuple(sys.modules.items()):
        if (name.startswith("triton.flagmega") or name.startswith("local_optimizations")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(REPO):
                raise ValueError("Non-public compiler import: " + str(path))
            imports[name] = str(path.relative_to(REPO))
    report = dict(schema="flagmega.original-gguf-normal-reconstruction/v1", schedule=schedule,
        original_checkpoint=str(checkpoint), expected_checkpoint_sha256=SHA256,
        normal_import=True, normal_Compiler=True, normal_writer=True,
        payload_clone_policy="require-reflink",
        old_physical_input=False, old_rdata_dependency=False,
        source_module_hash=source.semantic_hash, stats_module_hash=stats.semantic_hash,
        prepared_module_hash=prepared.semantic_hash, physical_module_hash=physical.semantic_hash,
        module_semantic_hash=loaded.semantic_hash, ordinary_program_plan_sha256=programs["plan_sha256"],
        counter_epoch_plan_sha256=epoch["plan_sha256"], entries=33, invocations=869,
        resets=resets, feedback_nodes_additional=True, distribution_audit=audit,
        phases_seconds=phases, imports=imports, reproduction_accepted=False)
    (work_dir / "reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("imported", "checkpoint", "output", "work-dir", "rdata-cache-dir", "recipe"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--schedule", choices=("baseline275", "candidate49"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh artifact output")
    print(json.dumps(reconstruct(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
