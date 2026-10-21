# Qwen3-1.7B BF16 on AMD RDNA (Radeon AI PRO R9700 / gfx1201)

Runs the same Qwen3-1.7B decode megakernel as the H800 tutorial, retargeted to
RDNA4 through the `amd-gfx1201` FlagMega target.

## Run

```bash
TUTORIAL=python/tutorials/flagmega/01-qwen3-1.7b-bf16
PYTHONPATH="$TUTORIAL/amd-r9700" python "$TUTORIAL/amd-r9700/run.py" \
    --checkpoint /models/Qwen3-1.7B --benchmark
```

The generated megakernel compiles and runs against a ROCm/HIP device (torch maps
HIP GPUs through the `cuda` device namespace).

## What is AMD-specific

The FlagMega distribution recipe is **mesh-topology based** (an 8×16 `bb` block
grid), so it is vendor-agnostic — `recipe.py` reuses the measured distribution
`CHOICES` and the residual/norm layout helpers from `../nvidia-h800`. Only three
things differ (see `recipe.py`):

1. `create_target` wraps `AmdGfx1201Target` (the batch-1 kernel-parameter tweaks
   — reduction/tile sizes — are software, not NV hardware features).
2. `residual_layout="sharded-casts"` + `norm_layout="replicated-residual"` keep
   the residual/norm shards full-mesh contiguous, avoiding the gather-reduce
   owner-group codegen path the default auto-distribution hits at hidden=2048.
3. The H800 **TMA** kernel recipe (`select_tir`) is skipped — RDNA has no
   tensor-descriptor, so the default TIR selection uses the portable
   `packed_k_major_gemv` kernels.

## Measured baseline (R9700 / gfx1201, batch-1 decode)

| Model | Latency | Throughput |
| --- | --- | --- |
| Qwen3-1.7B, 28 layers | ~17 ms/token | ~59 tok/s |

This is a correctness-first bring-up baseline; the decode path is GEMV
(memory-bandwidth) bound. Remaining performance work: RDNA-coalesced/vectorized
weight loads (`buffer_load`), occupancy / waves-per-eu tuning, re-enabling the
no-spill residency gate, and an AMD-specific cost model. Matrix-core (WMMA)
kernels help prefill (M>1), not single-token decode.
