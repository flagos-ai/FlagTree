"""Emit the accepted full model from explicitly supplied complete final IR.

This entry uses normal artifact generation, immutable readonly-cache lookup,
and runtime validation. It does not claim to rebuild the original-input pipeline.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-ir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--integrity-report', type=Path, required=True)
    parser.add_argument('--rdata-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tutorial = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(tutorial / 'local_optimizations', output / 'local_optimizations',
                    ignore=shutil.ignore_patterns('__pycache__'))
    sys.path.insert(0, str(output))
    from local_optimizations.kernel_recipe import activate
    activate()
    from local_optimizations.accepted_schedules import select
    from triton.flagmega import ir as fm
    from triton.flagmega.artifacts import rdata, write_artifact
    from triton.flagmega.importer import DirectoryCheckpoint
    from triton.flagmega.runtime import load
    begin = time.monotonic()
    original = fm.load_module(args.final_ir)
    module, schedules = select(original)
    recipe_path = output / 'local_optimizations/plan.json'
    recipe = json.loads(recipe_path.read_text())
    if recipe['semantic_hash'] != original.semantic_hash:
        raise ValueError('Recipe and supplied final IR disagree')
    recipe['semantic_hash'] = module.semantic_hash
    recipe_path.write_text(json.dumps(recipe, indent=2) + '\n')
    checkpoint = DirectoryCheckpoint(args.checkpoint, integrity_report=args.integrity_report)
    key_data = rdata._rdata_cache_key_data(module, checkpoint)
    if key_data is None or key_data != rdata._rdata_cache_key_data(original, checkpoint):
        raise ValueError('Readonly model data changed')
    key = rdata._rdata_cache_key(module, checkpoint)
    index = rdata._load_rdata_cache_entry(args.rdata_cache / key, key)
    if index is None or index['segment_nbytes'] != 80876114688:
        raise ValueError('The verified full-model readonly cache is required')
    artifact = write_artifact(module, output / 'artifact', target='nvidia-sm90',
                              emit_executable=True, checkpoint=checkpoint,
                              rdata_cache_dir=args.rdata_cache)
    manifest = json.loads((artifact / 'artifact.json').read_text())
    codegen = manifest['codegen']
    if len(codegen['kernels']) != 228 or len({s['entry_symbol'] for s in codegen['kernels']}) != 26:
        raise ValueError('Full-model program count changed')
    runtime = load(artifact)
    try:
        if len(runtime._kernel_launch_specs()) != 228:
            raise ValueError('Full-model runtime invocation count changed')
    finally:
        runtime.close()
    import torch
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, normal_CPU_runtime_passed=True,
                  fresh_original_IR_rebuilt=False, numerical_acceptance=False,
                  performance_acceptance=False, source_sha256=codegen['source_sha256'],
                  semantic_hash=module.semantic_hash, cache_key=key,
                  normal_cache_reuse=True, logical_invocations=228, unique_entries=26,
                  schedules=schedules, seconds=time.monotonic() - begin)
    (output / 'exit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
