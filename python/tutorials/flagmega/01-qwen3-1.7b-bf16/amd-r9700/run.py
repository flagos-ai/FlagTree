"""Compile and run Qwen3-1.7B BF16 on an AMD RDNA GPU (R9700 / gfx1201).

Example:

    PYTHONPATH="$TUTORIAL/amd-r9700" python "$TUTORIAL/amd-r9700/run.py" \
        --checkpoint /models/Qwen3-1.7B --benchmark

Compiles through the amd-gfx1201 FlagMega target with the RDNA distribution
recipe (see recipe.py), renders the megakernel, loads it on cuda:0 (ROCm maps
HIP devices through the cuda namespace), and runs a single-token decode.
"""

import argparse
import time
from pathlib import Path

import torch

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import (
    DirectoryCheckpoint,
    apply_numerical_profile,
    import_model,
    VLLM_INDUCTOR_LEVEL3,
)
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load as load_runtime

from recipe import create_target, recipe_compile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Qwen3-1.7B HF directory")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/qwen3-1.7b-amd"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--glu-reduction-group", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--benchmark", action="store_true", help="Time steady-state decode latency")
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    target = create_target(glu_reduction_group=args.glu_reduction_group)
    compiler = Compiler(CompileOptions(target="amd-gfx1201"))
    compiler.target = target

    checkpoint = DirectoryCheckpoint(args.checkpoint)
    print("Importing Qwen3-1.7B weights...", flush=True)
    module = import_model(checkpoint, numerical_profile=VLLM_INDUCTOR_LEVEL3)
    if module.stage == "imported":
        module = apply_numerical_profile(module, VLLM_INDUCTOR_LEVEL3)

    print("Compiling with the RDNA distribution recipe...", flush=True)
    module = recipe_compile(module, compiler, target)
    assert module.stage == "bufferized_tir", module.stage

    args.work_dir.mkdir(parents=True, exist_ok=True)
    artifact = write_artifact(module, args.work_dir / "artifact", target=target.name,
                              checkpoint=checkpoint, emit_executable=True)
    print(f"Artifact: {artifact}", flush=True)

    runtime = load_runtime(artifact, device=args.device)
    runtime._pool_values["workspace"].fill_(0xA5)
    state = runtime.create_state()
    input_ids = torch.tensor([9707], dtype=torch.int32, device=args.device)  # "Hello"
    logits, next_token = runtime.create_outputs()
    runtime.prepare(input_ids, state, logits=logits, next_token=next_token)
    runtime.run_into(logits, next_token, input_ids, state)
    torch.cuda.synchronize()
    print(f"next_token={int(next_token.item())}  logits_finite={bool(torch.isfinite(logits).all())}", flush=True)

    if args.benchmark:
        for _ in range(10):
            runtime.run_into(logits, next_token, input_ids, state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            runtime.run_into(logits, next_token, input_ids, state)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / args.iters * 1e6
        print(f"decode latency: {us:.1f} us/token ({1e6 / us:.1f} tok/s), "
              f"spill={runtime.resource_report.get('spill_bytes')}", flush=True)


if __name__ == "__main__":
    main()
