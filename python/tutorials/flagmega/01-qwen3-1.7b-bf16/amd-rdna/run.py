"""Compile and run Qwen3-1.7B BF16 on an AMD RDNA GPU.

One tutorial for both RDNA parts -- pick the gfx target with ``--arch``:
``gfx1201`` (R9700 / RDNA4, default) or ``gfx1100`` (W7900 / RDNA3).

Example:

    TUT=python/tutorials/flagmega/01-qwen3-1.7b-bf16
    PYTHONPATH="$TUT/amd-rdna" python "$TUT/amd-rdna/run.py" \
        --checkpoint /models/Qwen3-1.7B --arch gfx1100 --benchmark

Compiles through the amd-<arch> FlagMega target with the RDNA distribution
recipe (see recipe.py), renders the megakernel, loads it on cuda:0 (ROCm maps
HIP devices through the cuda namespace), and runs a single-token decode.

``--benchmark`` times decode at each ``--context-lengths`` entry (tokens
already in the KV cache, default 1024,4096,8192). The cache is filled with
real decode steps, and every timed step appends one more token, so each
result covers ``L .. L+warmup+iters``.
"""

import argparse
import math
import time
from pathlib import Path

import torch

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import (
    DirectoryCheckpoint,
    apply_numerical_profile,
    import_qwen3_model,
    VLLM_INDUCTOR_LEVEL3,
)
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load as load_runtime

from recipe import create_target, recipe_compile

_KV_BLOCK_SIZE = 256
_DEFAULT_KV_BLOCKS = 16
_WARMUP_STEPS = 10


def _context_lengths(text):
    try:
        lengths = sorted({int(value) for value in text.split(",") if value.strip()})
    except ValueError:
        raise SystemExit(f"--context-lengths must be comma-separated integers, got {text!r}")
    if not lengths or lengths[0] < 1:
        raise SystemExit("--context-lengths needs at least one length >= 1")
    return lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Qwen3-1.7B HF directory")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/qwen3-1.7b-amd"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arch", choices=("gfx1201", "gfx1100"), default="gfx1201",
                        help="RDNA gfx target: gfx1201 (R9700) or gfx1100 (W7900)")
    parser.add_argument("--glu-reduction-group", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--benchmark", action="store_true", help="Time steady-state decode latency")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--context-lengths", default="1024,4096,8192",
                        help="Comma-separated KV context lengths to time decode at (with --benchmark)")
    args = parser.parse_args()

    target = create_target(arch=args.arch, glu_reduction_group=args.glu_reduction_group)
    compiler = Compiler(CompileOptions(target=f"amd-{args.arch}"))
    compiler.target = target

    checkpoint = DirectoryCheckpoint(args.checkpoint)
    lengths = _context_lengths(args.context_lengths) if args.benchmark else []
    # The kernel does not bound-check the block table, so size the cache for
    # the longest context plus the warmup and timed steps that follow it.
    tokens_needed = (lengths[-1] + _WARMUP_STEPS + args.iters) if lengths else 1
    max_positions = int(checkpoint.config.get("max_position_embeddings", tokens_needed))
    if tokens_needed > max_positions:
        raise SystemExit(f"context {tokens_needed} exceeds max_position_embeddings={max_positions}")
    num_blocks = max(_DEFAULT_KV_BLOCKS, math.ceil(tokens_needed / _KV_BLOCK_SIZE))

    print(f"Importing Qwen3-1.7B weights (KV cache {num_blocks * _KV_BLOCK_SIZE} tokens)...", flush=True)
    module = import_qwen3_model(checkpoint, block_size=_KV_BLOCK_SIZE, num_blocks=num_blocks)
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

    position = 1  # tokens in the KV cache after the correctness step above
    for length in lengths:
        if length > position:
            print(f"Filling KV cache to {length} tokens...", flush=True)
            for _ in range(length - position):
                runtime.run_into(logits, next_token, input_ids, state)
            torch.cuda.synchronize()
            position = length
        start = position
        for _ in range(_WARMUP_STEPS):
            runtime.run_into(logits, next_token, input_ids, state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            runtime.run_into(logits, next_token, input_ids, state)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / args.iters * 1e6
        position += _WARMUP_STEPS + args.iters
        print(f"context {start}..{position}: decode latency {us:.1f} us/token "
              f"({1e6 / us:.1f} tok/s), spill={runtime.resource_report.get('spill_bytes')}",
              flush=True)


if __name__ == "__main__":
    main()
