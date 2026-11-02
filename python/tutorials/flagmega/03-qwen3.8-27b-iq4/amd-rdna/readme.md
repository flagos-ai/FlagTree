# Qwen3.8-27B IQ4_XS decode on AMD RDNA (R9700 / gfx1201, W7900 / gfx1100)

FlagMega compiles the original IQ4_XS GGUF used by
[the RTX 5060 Ti tutorial](../nvidia-rtx5060ti/readme.md) into a single cooperative
decode megakernel for the `amd-gfx1201` / `amd-gfx1100` targets. The megakernel covers all 64 layers
(48 GDN, 16 gated full attention), the IQ4_XS output head and GPU greedy sampling. Weights
stay packed IQ4_XS from IR to GPU memory.

## Performance (batch 1 decode, ROCm 7.2, one GPU)

| GPU | FlagMega | llama.cpp (HIP) | Speedup | Weight bandwidth | % of measured peak | % of nominal peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Radeon AI PRO R9700 (gfx1201) | **26.3 ms** (38.1 tok/s) | 29.8 ms (33.6 tok/s) | **1.13x** | 518 GB/s | 82% of 631 | 80% of 644.6 |
| Radeon PRO W7900 (gfx1100) | **25.8 ms** (38.8 tok/s) | 29.7 ms (33.7 tok/s) | **1.15x** | 528 GB/s | 75% of ~700 | 61% of 864 |

- Measured at `bb0edd2a2`, `benchmark.py --measurement host-token` (graph replay, GPU sampling,
  one token returned to the host per step, the same boundary as `llama_performance.py`).
  Synthetic32 and Chinese64 workloads, three interleaved FlagMega / llama.cpp rounds of three repeats.
  FlagMega rounds vary by less than 0.1 ms; llama.cpp rounds by less than 0.4 ms.
- Weight bandwidth = 13,611,666,080 bytes read per decode call / latency.
  - The measured peak is a large streaming read on the same card.
  - The W7900 has inline VRAM ECC, which caps it at about 700 of 864 GB/s; see the
    [Qwen3-1.7B readme](../../01-qwen3-1.7b-bf16/amd-rdna/readme.md).
- Starting point: the generic native GGUF GEMV took 257 ms (R9700) and 192 ms (W7900) per token.

## What makes it fast

| Optimization | Where | Effect |
| --- | --- | --- |
| RDNA IQ4_XS / Q4_K GEMV templates: wide loads, `v_perm_b32` codebook lookup, one scale per 32-value group, one cross-thread reduction per tile | both | 257 → 32 ms (R9700) |
| q8_1 activations dotted with `v_dot4_i32_iu8` (llama.cpp MMVQ contract), quantized once per block into LDS | both | R9700 30.1 → 28.2 ms, W7900 34.3 → 28.4 ms |
| Attention heads sharded over the mesh; GDN q/k norms and gates computed once per head | both | R9700 31.8 → 29.9 ms |
| gate+up row tiles taken from an atomic tile queue, so slower blocks take fewer tiles | both | W7900 −1.1 ms, R9700 −0.3 ms |
| 8-row projection tiles with each row reduced within one wave | both | −0.3 to −0.4 ms |
| Compiler: same-chip parameter reshards read as views; residual add fused with the next RMSNorm statistics | both | W7900 26.7 → 25.7 ms, R9700 26.6 → 26.3 ms |
| 64 KB `s_prefetch_data` of each weight during quantization; write-through GDN state stores | gfx1201 | 28.2 → 27.3 ms |
| Pointers kept in SGPRs and 64-bit activation offsets (no `readfirstlane` waterfalls) | gfx1100 | 42.7 → 38.5 ms |
| 8x24 = 192-block mesh (128 blocks cannot saturate ECC GDDR6) | gfx1100 | 37.2 → 34.3 ms |

The tutorial templates live in `agent_optimizations/templates/kernels/**/platforms/amd/`
(arch-specific ones under `gfx1100/` and `gfx1201/`). The core templates only gain
default-off hooks, so other targets render unchanged.

## Correctness

Checked against llama.cpp HIP built from the same pinned commit, on the same GGUF and GPU:

- **Chinese64** (17-token prompt, 64 steps): all 64 tokens equal llama.cpp on both cards.
- **Synthetic32** (token 1, 32 steps):
  - R9700: all 32 tokens equal llama.cpp.
  - W7900: the first 18 tokens equal llama.cpp. Step 18 is a near-tie that llama.cpp
    ranks by 0.022 logits.
- **Near-tie rule:** `benchmark.py` also accepts a first difference where llama.cpp's
  top-1/top-2 margin is below `--near-tie-margin` (default 0.1) and FlagMega's token is
  llama.cpp's second choice.
- **Reproducibility:** three independent processes reproduce the 64-step tokens and logits
  bitwise. This is required on gfx1100, where FlagTree's LLVM 22 has a schedule-sensitive
  codegen defect.
- **Accuracy:** against an fp64 one-step reference on the R9700, measured before the fused
  norm statistics, the logits max error is 0.19, against 0.22 for llama.cpp HIP; both have
  the same top-5.
- **Fused residual/norm statistics:** they sum in a different order, so logits are not
  bitwise equal to the unfused build. With teacher forcing on five prompts (up to 128
  steps), the error against llama.cpp is unchanged.

## Reproduce

Run from the repository root in the FlagTree environment (validated with PyTorch 2.10 +
ROCm 7.2.4). The checkpoint and requirements are shared with the NVIDIA tutorial.

```sh
export TUTORIAL="$PWD/python/tutorials/flagmega/03-qwen3.8-27b-iq4/amd-rdna"
export NVIDIA_TUTORIAL="$TUTORIAL/../nvidia-rtx5060ti"
export LOCAL="$TUTORIAL/.local"
export CHECKPOINT="$LOCAL/checkpoint/qwen3.8-27b-IQ4_XS-pure.gguf"
export TRITON_CACHE_DIR="$LOCAL/triton-cache"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export ARCH=gfx1201   # gfx1100 on a W7900
python -m pip install -r "$NVIDIA_TUTORIAL/requirements.txt"
python "$NVIDIA_TUTORIAL/download_checkpoint.py" --output "$LOCAL/checkpoint"

# 1. Compile (CPU only, ~1-2 min) and smoke-run four steps.
python "$TUTORIAL/compile.py" --checkpoint "$CHECKPOINT" --arch $ARCH --trial "$LOCAL/trials/baseline-001"
python "$TUTORIAL/run.py" --artifact "$LOCAL/trials/baseline-001/artifact" --steps 4

# 2. Build the pinned llama.cpp (commit 4413a192410fad1f054e9216114087a3e01527a0) with HIP.
export LLAMA_SOURCE="$LOCAL/llama.cpp" LLAMA_BUILD="$LOCAL/llama-build"
cmake -S "$LLAMA_SOURCE" -B "$LLAMA_BUILD" -DGGML_HIP=ON -DAMDGPU_TARGETS=$ARCH \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_COMMON=OFF
cmake --build "$LLAMA_BUILD" --target llama -j16

# 3. llama.cpp golden + performance, then FlagMega performance and correctness (Chinese64).
PROMPT='请用中文解释为什么量化模型的单 token decode 通常受显存带宽限制。'
LLAMA="--arch $ARCH --llama-source $LLAMA_SOURCE --llama-build $LLAMA_BUILD --checkpoint $CHECKPOINT"
python "$TUTORIAL/llama_golden.py" $LLAMA --output "$LOCAL/golden-chinese64" --steps 64 --prompt "$PROMPT"
python "$TUTORIAL/llama_performance.py" $LLAMA --output "$LOCAL/llama-perf-chinese64" \
  --steps 64 --repeats 3 --prompt "$PROMPT"
python "$TUTORIAL/benchmark.py" --arch $ARCH --artifact "$LOCAL/trials/baseline-001/artifact" \
  --llama-reference "$LOCAL/golden-chinese64" --independent-generation --golden-token-prefix 8 \
  --steps 64 --repeats 3 --measurement host-token --output "$LOCAL/flagmega-chinese64.json"
```

For Synthetic32, use `--steps 32 --token-id 1` instead of `--steps 64 --prompt ...`.

`benchmark.py` reports ms/token, GB/s and whether the tokens are accepted against llama.cpp.
Run GPU jobs one at a time and alternate the FlagMega / llama.cpp order between rounds.

Compile options:
- `--q8-activations off`: floating-point activation GEMVs.
- `--mesh 8x16`: the gfx1100 target's default mesh instead of 8x24.
