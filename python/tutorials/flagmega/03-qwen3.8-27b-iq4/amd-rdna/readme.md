# Qwen3.8-27B IQ4_XS decode on AMD RDNA (R9700 / gfx1201, W7900 / gfx1100)

This tutorial compiles the same original GGUF as
[the RTX 5060 Ti tutorial](../nvidia-rtx5060ti/readme.md) for the `amd-gfx1201`
and `amd-gfx1100` FlagMega targets: all 64 text decoder layers (48 GDN, 16 gated full attention),
the IQ4_XS output head and GPU greedy sampling, in one cooperative megakernel.
IQ4_XS weights stay packed in IR, readonly data and GPU storage. The NVIDIA
`pc_tma` agent optimizations rely on TMA, DP4A and PTX and are not used here.
On AMD the native IQ4_XS/Q4_K GEMVs render
from RDNA templates (`kernels/{block_fp8,matmul_glu}/platforms/amd/`): one
thread owns a 32-value group of two rows, loads its 16 packed bytes with wide
loads, maps the IQ4_NL codebook with `v_perm_b32` table lookups (as llama.cpp's
`get_int_from_table_16`), applies each group's `d*(scale-32)` once to the group
sum and reduces across threads once per tile; a 16-row tile runs as two
8-row halves and skips a half with no active row. By default `compile.py`
also installs this tutorial's `agent_optimizations` (below), which give the
IQ4_XS GEMVs int8 activations; with `--q8-activations off` activations stay
floating point. The GDN gate projections (`in_proj_b` / `in_proj_a`, one row per tile)
and the IQ4_XS embedding use the same group step and lookup
(`kernels/{gdn_recurrent,embedding}/platforms/amd/`). The AMD GDN body computes
each head's normalized q/k and both gate rows once per head and steps the fp32
state in (128, 16) tiles (`gdn_recurrent/platforms/amd/`). For decode attention
the RDNA targets split the context on the mesh axis whose remaining extent
divides every query-head count: Qwen3.8's 24 query heads do not divide the
default 8x16 mesh's x axis, so the context is split 16 ways over x and each
block computes 3 heads instead of all 24.

The native IQ4_XS GEMVs never use fp8, so they also run on gfx1100,
which has no fp8: they select the fp8-free `tir.block_fp8.native_simt` /
`tir.matmul_glu.native_simt` implementations there, which render the same RDNA
templates.

### q8_1 activations (`--q8-activations`, default `on`)

`agent_optimizations.install()` puts the tutorial's templates in front of
FlagMega's for the compile process, the same mechanism as the NVIDIA tutorial.
Each IQ4_XS GEMV call (projection, GLU gate/up, down and the output head) then
quantizes its activation once per block into LDS with the NVIDIA tutorial's
`q8_1_rne` contract, which is also llama.cpp's MMVQ activation: per 32 values,
`int8 = rne(fl32(x * fl32(127 / amax)))` packed four per int32 word and an fp16
scale `amax / 127`. The GEMV maps each weight word's eight nibbles to int8 codes
with `v_perm_b32` and dots them with `v_dot4_i32_iu8` (both operands signed), so
a group is one int32 sum scaled once by `(ls - 32) * d_w * d_x`. The core
templates only gain a default-off hook (`native_gguf_activation_prologue`) and
render byte-identically without the package; Q4_K, the GDN gate rows and the
NVIDIA targets keep their kernels.

LDS extents must be powers of two, so a call's groups are split into the
largest power-of-two main buffer and a power-of-two tail (5120: 128 + 32,
6144: 128 + 64, 17408: 512 + 32), each with its own K loop. The quantization
loads all of a call's values before packing any (one memory latency) and gets
round-to-nearest-even with `v_mul_f32` + `v_add_f32 0x4b400000` (1.5 * 2^23)
instead of `v_rndne_f32` + `v_cvt_i32_f32`. The GEMV's inner loop runs 156
VALU instructions with 16 `v_dot4` for 64 weight values per lane, about 2.4 per
value against about 8 (a convert and an FMA per value) for floating-point
activations; the megakernel has 960 `v_dot4`, still 248 VGPRs and no spills,
and 26,688 B of LDS per block instead of 4,096.

On the R9700 the int8 K loops are 3.4 ms faster than the floating-point ones
(removing every GEMV K loop leaves 2.1 ms). The first per-block quantization,
a TG=128 loop with derived indices and four masked stores, gave all of that back
(30.0 ms either way); loading every piece first and the cheaper rounding brought
its cost to 1.6 ms, of which 0.6 ms is the activation load latency.

On gfx1201 each IQ4_XS call first issues `s_prefetch_data` for the first 64 KB
of every weight it streams (gate and up for the GLU), from this block's first
row, so DRAM is already busy while the activation is quantized. The prefetch
changes no arithmetic: logits are bitwise equal to the build without it. Sizes
measured on one card (ms/token, Synthetic32 / Chinese64): none 28.26 / 28.27,
16 KB 28.19 / 28.18, 32 KB 27.86 / 27.87, **64 KB 27.48 / 27.45**, 128 KB
27.61 / 27.57, 256 KB 27.76 / 27.77, the whole block slice 28.70 / 28.70. The 64 KB build keeps 248 VGPRs and
72 B of scratch (13 SGPR spills go to VGPR lanes). gfx1100 has no
`s_prefetch_data` and renders unchanged.

On gfx1201 the GDN layers also write the fp32 recurrent state with a
nontemporal store (`.cs`; only its owner reads it again, on the next token), and
the convolution tile loads all history slots before it shifts the state, so the
loads no longer wait behind the shifted stores
(`gdn_{recurrent,convolution}/platforms/amd/gfx1201/`). Both keep the arithmetic
order: logits are bitwise equal to the build without them, with 248 VGPRs and
72 B of scratch. Together they take 0.14 ms off a token. gfx1100 renders
unchanged: on the W7900 one of three processes with these changes diverged at
step 3. Removing the GDN convolution and recurrent phases entirely saves about
1.3 ms per token, against about 0.74 ms for llama.cpp's whole GDN. The
recurrent core (about 1.1 ms of that) is latency-bound on layout conversions
between the `[16, 128]` state tiles and the per-value vectors. Preloading the
next state tile or deferring the state stores kept logits bitwise but was
slower (28.6 vs 27.5 ms). Keeping the norm in the head-tile layout spilled to
360 B of scratch and was 1.3 ms slower. All of these run inside the
248-VGPR megakernel.

Sharing one quantization across CTAs or across the calls that consume the same
input (in_proj qkv/z, attention q/k/v) was evaluated and not done: the norm
that produces a GEMV input writes 40-value shards per CTA (5120 / 128), so no
producer owns whole 32-value q8 groups; an invocation-scoped workspace cannot
outlive the call; and quantizing once per call into global memory needs a grid
barrier before the dot, about the cost of the per-block quantization it
replaces. The NVIDIA optimized path also quantizes per CTA.

## Accepted results: R9700 (gfx1201), ROCm 7.2, one GPU

| Backend | ms/decode call | tokens/s |
| --- | ---: | ---: |
| FlagMega, q8_1 activations + `v_dot4` + 64 KB weight prefetch + GDN stores (default) | **27.3** | **36.6** |
| FlagMega, q8_1 activations + `v_dot4` + 64 KB weight prefetch | 27.5 | 36.4 |
| FlagMega, q8_1 activations + `v_dot4`, no prefetch | 28.2 | 35.4 |
| FlagMega, floating-point activations (`--q8-activations off`) | 30.1 | 33.2 |
| FlagMega, RDNA native GGUF templates (before the attention/GDN changes) | 32.1 | 31.2 |
| FlagMega, generic native GGUF GEMV (before) | 257 | 3.9 |
| llama.cpp (HIP, gfx1201) | 29.8 | 33.6 |

Rotated FlagMega / llama.cpp runs of three repeats on one idle card (.101, no
ECC); host time around graph replay, GPU greedy sampling and one CPU token
return, the same boundary as the NVIDIA tutorial and `llama_performance.py`.
q8_1 activations, three interleaved rounds of q8 / floating point / llama.cpp
(the third on a second idle card): Synthetic32 28.27/28.21/28.33 ms vs
29.93/30.12/29.96 ms and llama.cpp 29.99/29.98/29.88 ms; Chinese64
28.24/28.22/28.30 ms vs 30.10/30.18/30.13 ms and llama.cpp 29.77/29.70/29.78 ms,
5–6% faster than llama.cpp at about 481 GB/s (75% of 644.6 GB/s).
With the gfx1201 weight prefetch (same card, three interleaved rounds against
the build without it): Synthetic32 27.54/27.47/27.45 ms vs 28.29/28.24/28.24 ms,
Chinese64 27.46/27.43/27.44 ms vs 28.30/28.25/28.25 ms, about 8% faster than
llama.cpp; tokens and logits are bitwise equal to the build without prefetch.
With the gfx1201 GDN stores (three interleaved rounds against the build without
them, graph replay): Synthetic32 27.33/27.31/27.40 ms vs 27.48/27.53/27.45 ms;
Chinese64 27.34 ms vs 27.45 ms, about 9% faster than llama.cpp; 64 tokens and
logits bitwise equal.
Floating-point activations, before:
Synthetic32: 29.90/30.09/30.04/30.11 ms (llama.cpp 29.93/29.87 ms);
Chinese64: 30.08/30.20 ms (llama.cpp 29.62/29.60 ms). Model weight bandwidth
is 13,611,666,080 bytes per call / latency: about 453 GB/s, 70% of the R9700's
644.6 GB/s (72% of the ~631 GB/s it streams). Sharding the attention heads took
decode from 31.8 to 29.9 ms together with the GDN body (CUDA-graph replay of the
same artifacts, three interleaved rounds: 31.72–31.77 vs 29.84–29.92 ms).
The IQ4 GEMVs were 97% of the
generic kernel's time: its 15-deep `tl.where` codebook, per-element byte loads
and a cross-thread reduction every K step left it latency-bound at ~55 GB/s.
In isolation the RDNA templates stream 450–560 GB/s on the model's projection
shapes; skipping empty half tiles (owners of 40 and 8 rows end in a half-full
tile) took decode from 37.0 to 34.4 ms. The GDN gate projections then still
walked K in 40 serial 128-value steps of byte gathers and a `tl.where`
codebook in each of the 48 GDN layers; dotting their row with the GEMV's group
step took decode from 34.4 to 32.1 ms. The megakernel still uses 248 VGPRs per
thread (the GEMV helpers use 170) and 88 SGPRs, with no register spills and
72 B of scratch.

Correctness is checked against llama.cpp built for HIP from the same pinned
commit on the same GGUF and GPU:

- **Synthetic32** (token 1, 32 calls): all 32 greedy tokens equal llama.cpp,
  including the first 8 generated independently, with either activation path.
- **Chinese64** (17 prompt tokens, 64 calls): with q8_1 activations all 64
  tokens equal llama.cpp. With floating-point activations the prefix equals
  llama.cpp up to step 22, a genuine near-tie: llama.cpp HIP ranks 98832 over
  4960 by 0.009 logits and FlagMega picks 4960 (as does the CUDA build of
  llama.cpp).
- Against a one-step fp64 reference (GGUF dequantization, token 1), the logits
  max error is 0.187 with q8_1 activations, 0.096 with floating-point ones and
  0.223 for llama.cpp HIP itself (0.156 for the generic kernel); all have the
  same top-5. q8_1 also doubles the worst GDN convolution / KV cache state
  errors (0.625 / 0.5 vs 0.25 / 0.25): within llama.cpp's own error, but not
  the floating-point path's.
- Logits are finite; three independent processes reproduce the 64-step
  trajectory and logits bitwise (q8_1 activations on an R9700 with ECC
  enabled; floating-point activations likewise after the attention and GDN
  changes).

## Results: W7900 (gfx1100), ROCm 7.2, one GPU

| Backend | ms/decode call | tokens/s |
| --- | ---: | ---: |
| FlagMega, 8x24, q8_1 activations + `v_dot4` (default) | **28.4** | **35.2** |
| FlagMega, 8x24 block mesh, floating-point activations | 34.3 | 29.2 |
| FlagMega, 8x16 block mesh (`--mesh 8x16`) | 37.2 | 26.9 |
| FlagMega, 8x16, before the attention/GDN changes | 38.7 | 25.8 |
| FlagMega, before the gfx1100 address changes | 42.7 | 23.4 |
| FlagMega, generic native GGUF GEMV (before) | 192 | 5.2 |
| llama.cpp (HIP, gfx1100) | 29.6 | 33.8 |

Three repeats per round. q8_1 activations, two interleaved rounds against the
floating-point build: Synthetic32 28.34 / 28.40 ms vs 34.33 / 34.47 ms,
Chinese64 28.39 / 28.44 ms vs 34.65 / 34.70 ms, about 480 GB/s; llama.cpp (not
rerun) 29.5 / 29.7 ms. The int8 dot pays off more than on the R9700 because the
floating-point GEMV was further from the bandwidth limit here. The 26,688 B of LDS
per block leave 4 blocks per WGP (5 with 4 KB), 192 on 48 WGPs: still the whole
8x24 grid. The megakernel keeps 248 VGPRs and 44 B of scratch (the quantization
holds its loaded activations as bf16 until it packs them; converting them first
spilled the down projection to 128 B).

128 blocks of 4 waves cannot keep the W7900's ECC
GDDR6 busy (the GEMV streams 371–389 GB/s at 128 blocks but 456–495 GB/s with
more waves; the R9700 already saturates at 128), so the tutorial compiles for
gfx1100 on an 8x24 = 192-block mesh through the compiler's `--placement`
option. At 248 VGPRs (5 wave32 waves per SIMD) a WGP holds 5 blocks of 4
waves, about 240 on the W7900's 48 WGPs, and each block uses 4 KB of LDS, so
all 192 blocks of the grid barrier stay co-resident. `--mesh 8x16` restores
the gfx1100 target default; other tutorials and the target default are unchanged. 8x24:
Synthetic32 34.13 / 34.32 ms, Chinese64 34.33 / 34.50 ms; 8x16: Chinese64
37.15 ms. With the RDNA templates before the attention/GDN changes (8x16):
Synthetic32 38.35 / 38.69 ms, Chinese64 38.90 ms. Before
the two gfx1100 address changes below: Chinese64 42.74 / 43.12 ms, Synthetic32
42.56 / 42.64 ms (45.7 ms with the RDNA GEMVs but the generic GDN gate
projections). With the generic kernel: Chinese64 191.8 / 192.3 ms, Synthetic32
192.5 / 192.4 ms. llama.cpp: 29.5 / 29.7 ms. On 8x24 that is about 397 GB/s
of weight bandwidth (71 GB/s with the generic kernel), 57% of the ~700 GB/s
this W7900 streams. Its VRAM ECC is enabled, which limits streaming
bandwidth to about 700 of the 864 GB/s nominal (see the Qwen3-1.7B
[amd-rdna readme](../../01-qwen3-1.7b-bf16/amd-rdna/readme.md)). The megakernel
uses 248 VGPRs, 105 SGPRs and 44 B of scratch (160 B before), and builds in
about 60–75 s.

gfx1100 still converts global loads to buffer loads (gfx1201 does not), and a
buffer load needs its base in SGPRs. Activation pointers built from 32-bit
offsets in the GEMVs, and pointer parameters of the noinline device functions
(which arrive in VGPRs), turned into `readfirstlane` waterfall loops. 64-bit
activation offsets keep the GEMV loads as global loads (42.6 → 39.6 ms), and
on gfx1100 every non-GEMV device function reads its pointer parameters back
into SGPRs (39.6 → 38.5 ms): `s_cbranch_execnz` loops drop from 258 to 28.
Both keep the logits bitwise.

Correctness against llama.cpp HIP built for gfx1100 from the same pinned commit:

- **Chinese64**: all 64 tokens equal llama.cpp, with either activation path.
  The gfx1100 llama.cpp build itself picks 4960 at the step-22 near-tie, like
  FlagMega and the CUDA build.
- **Synthetic32**: the first 18 tokens equal llama.cpp, with either activation
  path. Step 18 is a near-tie: llama.cpp ranks 14791 over 56239 by 0.022 logits
  and FlagMega picks 56239, its second choice.
- Logits are finite.
- Three independent processes reproduce the 64-step Chinese64 tokens and logits
  bitwise with q8_1 activations on the 8x24 mesh, and with floating-point
  activations on both the 8x24 and the 8x16 mesh, as with each earlier kernel
  (four processes for the generic kernel, one before and three after a host
  reboot).

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

Use `--arch gfx1100` (and `-DAMDGPU_TARGETS=gfx1100` for llama.cpp) on a W7900;
it compiles on the 8x24 mesh unless `--mesh 8x16` is given. `--q8-activations
off` compiles the floating-point activation GEMVs instead.
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
