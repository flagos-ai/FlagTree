# Qwen3.8 27B IQ4_XS — NVIDIA RTX 5060 Ti 16 GB

Decode-only inference for a 64-layer Qwen3.8 27B text decoder stored as IQ4_XS, with optional
multi-token prediction (MTP) using the model's own draft layer.

The target is a single 16 GB consumer GPU at 180 W, so the whole design is bandwidth-shaped: every
weight byte is read once per decode step and the arithmetic is chosen to keep the integer dot products
fed rather than to minimise FLOPs.

## What this example covers

- IQ4_XS weight streaming with `dp4a` dot products on CUDA cores, including the per-32-element
  quantisation records and the gate/up fused layout.
- A persistent, multi-column projection kernel shape shared by all weight families, with the
  per-family column counts and occupancy that measured best on this card.
- Gated delta-net (GDN) recurrent state and paged attention state, both kept on device.
- Multi-token verification: one forward call evaluates `T` token rows against a single pass over the
  weights, so weight traffic is amortised over the rows.
- MTP speculative decoding driven by the checkpoint's own draft layer (blk.64), byte-exact to the
  GGUF's IQ4_XS payloads.

## Prerequisites

- An NVIDIA GPU with compute capability 12.0 (this example is tuned for RTX 5060 Ti: 36 SMs,
  448 GB/s nominal HBM bandwidth, 180 W cap).
- CUDA 13.x and a Python environment with a matching PyTorch distribution.
- The Qwen3.8 27B IQ4_XS GGUF checkpoint (14,534,384,640 bytes,
  SHA256 `ea5a3c45d407f9b9e5d2c0d647f0ea600f486f6b86b92b56d0823ba073dae675`).
- A llama.cpp build from the pinned upstream commit `5ce06fc71f5e938301f7dc51697a59ae308bd6b7` if you
  want to reproduce the reference comparison.

Build FlagTree from the repository root first:

```sh
FLAGTREE_BACKEND=nvidia FLAGTREE_TLE=1 python -m pip install '.[flagmega]'
```

## Quick start

Compile the full 64-layer decoder into an immutable trial directory:

```sh
python compile.py \
  --checkpoint /path/to/qwen3.8-27b-IQ4_XS.gguf \
  --trial .local/trials/full64
```

`compile.py` verifies the checkpoint hash and size, packs the weights into read-only images, builds the
graph, and emits the compiled artifact together with a receipt of the toolchain and source identities.
`--rdata-cache-dir` (default `.local/cache/weights`) holds the packed weight images so repeated builds
and benchmark runs reuse them. `--layer` builds a single-layer probe and is intended for backend
bring-up only.

Measure decode throughput against a fresh llama.cpp run:

```sh
python benchmark/measure_candidate.py \
  --output .local/results/baseline \
  --candidate .local/trials/full64/artifact \
  --baseline .local/trials/r000/artifact
```

`measure_candidate.py` runs a numerical gate before timing anything: the first generated tokens must
equal a freshly generated llama.cpp golden, all logits must be finite, and the eager, graph and
reset trajectories must agree. Timing uses host milliseconds per generated token, includes the GPU
greedy selection and one CPU token return, and excludes load, JIT, warmup, state reset and prompt
injection.

For multi-token verification, the harness in `mtp/` builds a `T`-row artifact and times the full
verify step on a CUDA graph:

```sh
python mtp/build_t5_plain.py .local/trials/t4 1 4 1
python mtp/time_forward_graph.py .local/trials/t4/artifact 4 25
python mtp/compare_rows.py .local/trials/t4/artifact .local/trials/t1/artifact 1 2 3 4
```

`build_t5_plain.py OUT MIN MAX SEGMENT` builds rows `MIN..MAX`; `time_forward_graph.py ARTIFACT
TOKENS ITERS` reports the median CUDA-graph replay; `compare_rows.py` verifies each row against a
sequential decode-1 reference; `profile_programs.py` aggregates per-program GPU time by kernel family.

## Layout

| Path | Contents |
| --- | --- |
| `compile.py` | Checkpoint verification, weight packing, graph build, artifact emission |
| `agent_optimizations/iq4_streaming/graph.py` | Graph construction and Q8/Q4 node distribution for the IQ4 path |
| `agent_optimizations/iq4_streaming/native.py` | Native IQ4_XS renderer, including the `dp4a` weight decode |
| `agent_optimizations/iq4_streaming/industry/` | Pinned industry reference kernel used as the fidelity comparator |
| `benchmark/` | Numerical gates, decode timing, llama.cpp reference drivers, report rendering |
| `mtp/` | Multi-token verification build, timing, row verification and per-program profiling |
| `tests/` | Graph and native-lowering tests for the IQ4/Q8 path |

## Measured results

RTX 5060 Ti 16 GB, 180 W, default clocks. Reference is the pinned llama.cpp built from the same
commit, using the same GGUF and the same input token IDs, greedy sampling, k = 4 drafts
(`--spec-type draft-mtp --spec-draft-n-max 4`), 4096-token context.

Whole-model decode, one token per step, no speculation:

| | this example | llama.cpp (no speculation) |
| --- | ---: | ---: |
| ms per token | 35.37 | 36.00 |
| tokens per second | 28.3 | 27.8 |

With MTP speculation, verification width `T = 4` (three drafts per step) is the best fixed operating
point on both workloads:

| Workload | this example | llama.cpp (k = 4) | ratio |
| --- | ---: | ---: | ---: |
| Chat reference, 69 input tokens | 13.173 ms/token (75.9 tok/s) | 14.023 ms/token (71.3 tok/s) | 0.939 |
| Eight raw prompts, aggregate | 15.356 ms/token (65.1 tok/s) | 16.337 ms/token (61.2 tok/s) | 0.940 |

These two rows compare each implementation at its own best fixed width: this example at `T = 4`
(three drafts) and llama.cpp at `n_max = 4` (four drafts), so llama.cpp performs more verification
work per step. At equal width `T = 5` / `k = 4` the chat reference is 13.878 ms/token against
14.023, a 1.0 % advantage. On the raw reference prompt llama.cpp's acceptance falls to 0.44 where the
chat template gives 0.77, so the 32 % gap on that prompt is dominated by draft acceptance rather than
by kernel throughput.

## Notes and limitations

- Everything above is measured at default clocks, not frequency-normalised. The memory clock was
  13801 MHz on both sides; automatic SM boost differed slightly between the two implementations.
- The draft capacity differs: this example allocates 64 positions for raw prompts and 128 for chat
  prompts, while llama.cpp allocates against the full 4096-token target context. All measured
  sequences fit.
- Generation windows are short (32–35 timed tokens). Longer contexts and longer generations were not
  measured, and no cross-backend floating-point equivalence is claimed — only token identity against
  a sequential decode-1 reference on this implementation.
- The multi-token path is numerically exact against sequential decode-1 on every row, but it is not
  bitwise identical to llama.cpp.
