# Qwen3.8-27B IQ4_XS decode on AMD RDNA (R9700 / gfx1201, W7900 / gfx1100)

This tutorial compiles the same original GGUF as
[the RTX 5060 Ti tutorial](../nvidia-rtx5060ti/readme.md) for the `amd-gfx1201`
and `amd-gfx1100` FlagMega targets: all 64 text decoder layers (48 GDN, 16 gated full attention),
the IQ4_XS output head and GPU greedy sampling, in one cooperative megakernel.
IQ4_XS weights stay packed in IR, readonly data and GPU storage. It uses the
current default compiler (the core baseline path); the NVIDIA `pc_tma` agent
optimizations rely on TMA, DP4A and PTX and are not used here.

The native IQ4_XS GEMVs never quantize activations, so they also run on gfx1100,
which has no fp8: they select the fp8-free `tir.block_fp8.native_simt` /
`tir.matmul_glu.native_simt` implementations there.

## Accepted results: R9700 (gfx1201), ROCm 7.2, one GPU

| Backend | ms/decode call | tokens/s |
| --- | ---: | ---: |
| FlagMega compiler baseline | 257 | 3.9 |
| llama.cpp (HIP, gfx1201) | 29.9 | 33.5 |

Three rotated rounds per workload on one idle card; host time around graph
replay, GPU greedy sampling and one CPU token return, the same boundary as the
NVIDIA tutorial and `llama_performance.py`. Chinese64: 257.8/258.2/258.2 ms vs
llama.cpp 29.8/29.7/29.8 ms; Synthetic32: 255.7/257.3/258.0 ms vs 29.9/30.1/30.0 ms.
Model weight bandwidth is 13,611,666,080 bytes per call / latency: about
53 GB/s, 8% of the R9700's 644.6 GB/s. The megakernel uses 248 VGPRs per thread.

Correctness is checked against llama.cpp built for HIP from the same pinned
commit on the same GGUF and GPU:

- **Synthetic32** (token 1, 32 calls): all 32 greedy tokens equal llama.cpp,
  including the first 8 generated independently.
- **Chinese64** (17 prompt tokens, 64 calls): the generated prefix equals
  llama.cpp up to step 22, a genuine near-tie: llama.cpp HIP ranks 98832 over
  4960 by 0.009 logits and FlagMega ranks them the other way by 0.071 (the CUDA
  build of llama.cpp also picks 4960). With the same input tokens forced, 63 of
  64 steps agree.
- Logits are finite; three independent processes reproduce the 64-step
  trajectory and logits bitwise.

## Results: W7900 (gfx1100), ROCm 7.2, one GPU

| Backend | ms/decode call | tokens/s |
| --- | ---: | ---: |
| FlagMega compiler baseline | 192 | 5.2 |
| llama.cpp (HIP, gfx1100) | 29.6 | 33.8 |

Two rounds per workload, three repeats each (Chinese64: 191.8 / 192.3 ms vs
llama.cpp 29.5 ms; Synthetic32: 192.5 / 192.4 ms vs 29.7 ms). That is about 71 GB/s of weight
bandwidth, 10% of the ~700 GB/s this W7900 streams. Its VRAM ECC is enabled,
which limits streaming bandwidth to about 700 of the 864 GB/s nominal (see the
Qwen3-1.7B [amd-rdna readme](../../01-qwen3-1.7b-bf16/amd-rdna/readme.md)).
The megakernel uses 248 VGPRs, no spills, and builds in about 67 s.

Correctness against llama.cpp HIP built for gfx1100 from the same pinned commit:

- **Chinese64**: all 64 tokens equal llama.cpp. The gfx1100 llama.cpp build
  itself picks 4960 at the step-22 near-tie, like FlagMega and the CUDA build.
- **Synthetic32**: equal through step 17; at step 18 llama.cpp ranks 14791 over
  56239 by 0.022 logits and FlagMega picks 56239, its second choice (a near-tie).
- Logits are finite.
- Four independent processes (one before and three after a host reboot)
  reproduce the 64-step Chinese64 tokens and logits bitwise.

On gfx1100 every FlagMega change must reproduce bitwise across independent
processes like this, because FlagTree's LLVM 22 has a schedule-sensitive
gfx1100 codegen defect.

`benchmark.py` therefore accepts a generated prefix that equals llama.cpp's, or
whose first differing step is a golden-logit near-tie (top-2 margin below
`--near-tie-margin`, default 0.1, with our token as llama.cpp's second choice).
After such a step the two contexts differ, so later prefix tokens are not
compared.

## Reproduce

Run from the repository root in the FlagTree environment (validated with
PyTorch 2.10 + ROCm 7.2.4). The checkpoint, `prepare_checkpoint.py` and
requirements are shared with the NVIDIA tutorial.

```sh
export TUTORIAL="$PWD/python/tutorials/flagmega/03-qwen3.8-27b-iq4/amd-rdna"
export NVIDIA_TUTORIAL="$TUTORIAL/../nvidia-rtx5060ti"
export LOCAL="$TUTORIAL/.local"
export CHECKPOINT="$LOCAL/checkpoint/qwen3.8-27b-IQ4_XS-pure.gguf"
export TRITON_CACHE_DIR="$LOCAL/triton-cache"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -r "$NVIDIA_TUTORIAL/requirements.txt"
python "$NVIDIA_TUTORIAL/download_checkpoint.py" --output "$LOCAL/checkpoint"

python "$TUTORIAL/compile.py" --checkpoint "$CHECKPOINT" --arch gfx1201 \
  --trial "$LOCAL/trials/baseline-001"
python "$TUTORIAL/run.py" --artifact "$LOCAL/trials/baseline-001/artifact" --steps 4
```

Use `--arch gfx1100` (and `-DAMDGPU_TARGETS=gfx1100` for llama.cpp) on a W7900.
Build the pinned llama.cpp with HIP for the same GPU, record the goldens and
compare (Synthetic32: `--steps 32 --token-id 1` and no `--prompt`):

```sh
export LLAMA_SOURCE="$LOCAL/llama.cpp"
export LLAMA_BUILD="$LOCAL/llama-build"
# Fetch commit 4413a192410fad1f054e9216114087a3e01527a0 as in the NVIDIA tutorial, then:
cmake -S "$LLAMA_SOURCE" -B "$LLAMA_BUILD" -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_COMMON=OFF
cmake --build "$LLAMA_BUILD" --target llama -j16

PROMPT='请用中文解释为什么量化模型的单 token decode 通常受显存带宽限制。'
python "$TUTORIAL/llama_golden.py" --arch gfx1201 --llama-source "$LLAMA_SOURCE" \
  --llama-build "$LLAMA_BUILD" --checkpoint "$CHECKPOINT" \
  --output "$LOCAL/llama-golden-chinese" --steps 64 --prompt "$PROMPT"
python "$TUTORIAL/llama_performance.py" --arch gfx1201 --llama-source "$LLAMA_SOURCE" \
  --llama-build "$LLAMA_BUILD" --checkpoint "$CHECKPOINT" \
  --output "$LOCAL/llama-performance-chinese" --steps 64 --repeats 3 --prompt "$PROMPT"
python "$TUTORIAL/benchmark.py" --arch gfx1201 \
  --artifact "$LOCAL/trials/baseline-001/artifact" \
  --llama-reference "$LOCAL/llama-golden-chinese" --independent-generation \
  --golden-token-prefix 8 --steps 64 --repeats 3 --measurement host-token \
  --output "$LOCAL/baseline-chinese.json"
```

`run.py`, `llama_golden.py` and `llama_performance.py` run the vendor-neutral
NVIDIA scripts and only relabel the llama.cpp backend. `benchmark.py` runs the
NVIDIA graph-replay protocol (HIP graphs capture the cooperative launch) and
reports bandwidth against the AMD card. Keep GPU runs sequential and rotate the
backend order when comparing repeated performance runs.
