# Qwen3-1.7B BF16 on AMD RDNA (R9700 / gfx1201 and W7900 / gfx1100)

Runs the same Qwen3-1.7B decode megakernel as the H800 tutorial, retargeted to
RDNA through the `amd-gfx1201` / `amd-gfx1100` FlagMega targets. One recipe
serves both RDNA4 (R9700) and RDNA3 (W7900) — pick the gfx target with `--arch`.

## Run

```bash
TUTORIAL=python/tutorials/flagmega/01-qwen3-1.7b-bf16

# R9700 / RDNA4 (default)
PYTHONPATH="$TUTORIAL/amd-rdna" python "$TUTORIAL/amd-rdna/run.py" \
    --checkpoint /models/Qwen3-1.7B --arch gfx1201 --benchmark

# W7900 / RDNA3
PYTHONPATH="$TUTORIAL/amd-rdna" python "$TUTORIAL/amd-rdna/run.py" \
    --checkpoint /models/Qwen3-1.7B --arch gfx1100 --benchmark \
    --context-lengths 1,1024,4096,8192
```

The generated megakernel compiles and runs against a ROCm/HIP device (torch maps
HIP GPUs through the `cuda` device namespace). `--benchmark` decodes
autoregressively (each step feeds back the predicted token) up to every
`--context-lengths` entry (default `1024,4096,8192`), then times decode there.

## What is AMD-specific

The FlagMega distribution recipe is **mesh-topology based** (an 8×16 `bb` block
grid), so it is vendor- and gfx-agnostic — `recipe.py` reuses the measured
distribution `CHOICES` and the residual/norm layout helpers from `../nvidia-h800`,
and the same recipe serves gfx1100 and gfx1201. What differs (see `recipe.py`):

1. `create_target(arch=...)` wraps `AmdGfx1100Target` / `AmdGfx1201Target` (the
   batch-1 kernel-parameter tweaks — reduction/tile sizes — are software, not NV
   hardware features); the gfx target machine is the only per-arch difference.
2. `residual_layout="sharded-casts"` + `norm_layout="replicated-residual"` keep
   the residual/norm shards full-mesh contiguous. The norm pin is the
   load-bearing one: under the default distribution the fused
   `gather_reduce_add_norm_apply` gets a block-cyclic partial input its renderer
   rejects.
3. The H800 **TMA** kernel recipe (`select_tir`) is skipped — RDNA has no
   tensor-descriptor, so the default TIR selection uses the portable
   `packed_k_major_gemv` kernels, with three RDNA pins: **split-K** for the
   lm_head and the wide per-layer projections (all 128 CTAs instead of 16), a
   64-row / 32-wide-K gate+up GEMV tile, and the per-row `decode_t16`
   paged-attention partial.

Kernel templates under `platforms/amd/` (and `platforms/amd/gfx1100|gfx1201/`)
override shared templates only where RDNA was measured to benefit.

## Decode performance (batch 1, bf16)

Median ms/token for `run.py --benchmark` (3 interleaved rounds). "Upstream" is
the `feature/flagmega` baseline (`b33b91f85`) built the same way.

| Context | R9700 | R9700 upstream | W7900 | W7900 upstream |
| --- | --- | --- | --- | --- |
| 1 | **6.49** (154 tok/s) | 7.50 | **5.99** (167 tok/s) | 7.92 |
| 1K | **6.66** (150 tok/s) | 7.97 | **6.17** (162 tok/s) | 8.30 |
| 4K | **7.22** (139 tok/s) | 9.59 | **6.69** (149 tok/s) | 10.06 |
| 8K | **7.97** (125 tok/s) | 11.71 | **7.34** (136 tok/s) | 12.31 |

### Bandwidth utilization

Decode reads every weight once per token plus the KV cache, so effective
bandwidth = bytes per token / latency, with bytes per token = 3.44 GB of bf16
weights (28 layers + lm_head) + ctx × 28 layers × 8 KV heads × 128 × 2 (K, V) ×
2 B. This is a lower bound: it ignores activations and duplicated reads.

| Context | R9700 GB/s | % of measured (631) | % of nominal (645) | W7900 GB/s | % of measured (700) | % of nominal (864) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 530 | 84% | 82% | 574 | 82% | 66% |
| 1K | 534 | 85% | 83% | 576 | 82% | 67% |
| 4K | 541 | 86% | 84% | 584 | 83% | 68% |
| 8K | 549 | 87% | 85% | 597 | 85% | 69% |

"Measured" is the streaming-read ceiling of the same card (a 4 GB
`torch.sum()`); it is the meaningful roofline. The large GEMV phases (gate+up,
down, lm_head) stream at roughly 90–100% of it; most of the remaining gap is
the small per-layer phases (norms, RoPE, attention combine) and their grid
barriers.

### Why the W7900 reaches only ~700 of 864 GB/s: inline ECC

The W7900's nominal 864 GB/s (384-bit GDDR6 at 18 Gbps) is not reachable on the
measured board because its **VRAM ECC is enabled**:

- GDDR6 ECC on RDNA is *inline*: 1/16 of VRAM holds the check bits, so the
  board exposes 46064 MiB instead of 48 GiB (15/16), and every access also moves
  ECC data.
- On two otherwise identical R9700 hosts (same clocks and power), the card with
  ECC active (30576 MiB visible, 15/16 of 32 GiB) streams **552 GB/s** and the one
  without (32624 MiB) **631 GB/s**: ECC costs about 12.5%. Decode is 11–12%
  slower on the ECC card.
- Clocks are not the limit on the W7900: MCLK stays at its 1124 MHz (18 Gbps)
  maximum under load, and forcing a higher fabric clock (`high` performance
  level, FCLK 1600 → 2000–2200 MHz) leaves streaming bandwidth unchanged
  (~703 vs ~706 GB/s).

So ECC explains most of the W7900's 864 → ~700 GB/s gap; the R9700 numbers above
are for a card without ECC. Check yours with `cat
/sys/class/drm/card*/device/mem_info_vram_total` — a total of 15/16 of the
nominal VRAM means inline ECC is active, and ~12% lower bandwidth should be
expected. Use the measured ceiling of your own card when comparing.

### Decode optimizations

Batch-1 decode is GEMV (memory-bandwidth) bound; each change below was measured
with per-phase in-kernel timestamps and kept only if it did not raise the
megakernel's peak VGPRs (a register increase slows every phase):

- **split-K** for the lm_head and wide per-layer projections (output-only
  sharding left the GEMV on 16 of 128 CTAs);
- an RDNA QKV GEMV template without per-K-step barriers and full load drains;
- K-contiguity hints so split-K GEMV weight loads become 128-bit;
- a 64-row / 32-wide-K gate+up tile (the default tile split contiguous weight
  runs; W7900 went from 71% to ~100% of its bandwidth on that phase);
- gfx1201 `s_prefetch_data` of the next QKV / down / o-proj weight tiles and a
  per-owner QKV tile rotation;
- paged-attention partial: V loaded next to K, per-token-row online softmax (no
  cross-warp reduction or barrier in the loop), and on gfx1100 straight-line
  tile loads with the next block ids fetched alongside the current K/V;
- on gfx1201, small-phase loads stay global loads instead of serial buffer-load
  waterfalls;
- the grid barrier reads the workgroup count with aligned scalar loads.

## Prefill (WMMA, M>1)

`prefill_wmma.py` is the tuned RDNA bf16 WMMA GEMM microkernel (`tl.dot`,
`num_stages=1`). A loop of GEMVs reaches ~3% of peak at M=128; the WMMA GEMM is
**~10x faster at M=128 and ~30x at M=512 (84.6 TFLOP/s, ~89% of the bf16 WMMA
peak)**. M>1 matmuls select the WMMA `tir.dense_matmul.mma` candidate.
