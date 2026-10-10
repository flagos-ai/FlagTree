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

## Decode optimization (split-K)

Batch-1 decode is GEMV (memory-bandwidth) bound. The lm_head (152K vocab) and the
wide per-layer projections were under-parallelized: their output was sharded on a
single mesh axis, so the GEMV ran on only 16 of the 128 CTAs. `recipe.py` pins
**split-K** candidates (split the reduction over y + shard the output over x = all
128 CTAs), flooding the CUs as the amd_radeon_kernels / FreeToken RDNA GEMV
references do. Measured gain on real Qwen3-1.7B / R9700:

| config | latency | throughput |
| --- | --- | --- |
| baseline | 16999 us/token | 58.8 tok/s |
| + lm_head split-K | 14913 us/token | 67.1 tok/s |
| + per-layer split-K | 10452 us/token | **95.7 tok/s** |

**1.63x throughput**, identical next token.

## Prefill (WMMA, M>1)

`prefill_wmma.py` is the tuned RDNA bf16 WMMA GEMM microkernel (`tl.dot`,
`num_stages=1`). FlagMega renders M>1 matmuls as a loop of GEMVs (~3% of peak at
M=128); the WMMA GEMM is **~10x faster at M=128 and ~30x at M=512 (84.6 TFLOP/s,
~89% of the bf16 WMMA peak)**. It is the kernel body intended for a
`tir.dense_matmul.mma` candidate; the remaining work is wiring it into the
catalog (`implementations.py`) + the distributed/packed-ABI renderer
(`kernel_call_renderers.py`), since FlagMega's Qwen3 path is currently decode-only.

Remaining perf work: integrate the WMMA candidate for a prefill graph,
occupancy / waves-per-eu tuning, re-enabling the no-spill residency gate, and an
AMD-specific cost model.
