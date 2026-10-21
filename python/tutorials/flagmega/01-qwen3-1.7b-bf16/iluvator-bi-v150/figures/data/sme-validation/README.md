# SME pipeline validation

These compressed logs preserve the SME correctness and performance evidence
collected before attention staging and the final CLI measurement audit.
`manifest.json` hashes the uncompressed contents. The current compiler tests are
in `python/test/flagmega/codegen/cpp/test_sme_batches.py` and
`test_readonly_layout.py`, relative to the repository root.

## Optimizations and synchronization contracts

`codegen/cpp/ftt/include/ftt/arch/iluvatar/primitive_ops.h` uses flat global-to-shared
copies, preserving canonical byte order. The 2-D rowxfb instruction swizzles its
destination and requires a source row pitch of at least 64 bytes. The flat
pipeline waits on the issuing lane's FIFO counter; a CTA barrier alone does not
complete other lanes' transfers. Consumers drain their own shared-read queues
before a consumed slot is reused. GPU probes validate every staged element for
multiple batches, projections and K-group counts.

`kernels/_sme_batches.cpp.jinja` issues a two-projection batch as
`gate0, gate1, up0, up1`. It retains a complete newer batch while waiting for the
older batch. This changes descriptor issue order while preserving each
projection's arithmetic order. Fused QKV uses eight slots per warp, requiring
128 KiB for 16 warps. `_sme_arena_elems` sizes QKV storage independently of
whether another dense/GLU call exists; the renderer checks the completed shared
arena against the target capability.

`readonly_layout.py::select_readonly_layouts` proves the actual readonly roots
and all consumers' address contracts before selecting GLU/down device layouts.
`apply_readonly_layout` validates the entire record set before changing bytes,
then permutes `(K groups, owners, group elements, bytes)` to
`(owners, K groups, group elements, bytes)` once before graph capture. Canonical
IR, source-layout programs and artifact weights retain their original bytes.
CPU/GPU tests cover the byte permutation, untouched neighboring bytes, a
canonical second reader, malformed later records and canonical source reloads.

Large-N GEMV overlaps the scalar tail with the final SME tile only when complete
idle warps can cover it. For N=9496, six idle warps process the 24-element tail.
The scalar payload retains its four K accumulation chains and 16-lane fold,
accesses independent output elements and never accesses the shared arena.
Other valid geometries retain their own legal schedules.

## Missing SME group-address alignment

Root cause: `dense_matmul/_function.cpp.jinja`, SME implementation selection.
The newly expanded large-N SME schedule originally checked tile capacity but
omitted the instruction's alignment requirement for each K-group address.
For unpadded N=549, the BF16 group stride is `549 * 16 * 2 = 17568` bytes,
which is not a multiple of 64; odd groups produced incorrect front elements.
N=1023 exposed the same defect, while scalar tail elements remained correct.

`test-tail-gpu.log.gz` records the two failing GPU cases. Selection now requires
`weight_group_stride % 32 == 0` in BF16 elements, proving every group offset is
64-byte aligned. Unaligned layouts use their canonical scalar implementation;
aligned layouts continue to use SME without changing the model or arithmetic.
`test-tail-gpu-after.log.gz` records 12 passing checks, including padded/unpadded
neighboring capacities. The Qwen3 layout already meets the instruction contract,
so this legality correction preserves its generated source and performance.

## Shared QKV output-reader lifetime

Root cause: `qkv_parallel_linear/_fused_sme.cpp.jinja`, partial reduction and
function exit. A later shared-memory consumer could overwrite the arena while
delayed output reducers were still reading it. The minimal GPU test delays
reducers with 1000 `sl_nop` instructions and immediately reuses the arena from
other warps. Removing the exit CTA synchronization reproduces the failure in
`test-qkv-fused-fail-before.log.gz`; retaining it passes both QKV variants in
`test-qkv-fused-exit.log.gz`.

The exit synchronization establishes the required read-completion-to-overwrite
ordering. `renderer.py::_mark_shared_exit_barriers` merges it with a following
schedule barrier only when every enclosing caller provably reaches that barrier.
Unknown callers and recursive paths cannot establish that proof. Regression
tests cover multiple callers and standalone QKV-only arena allocation.

Other arena users retain disjoint live regions. The down-projection RMS
statistics occupy a region separate from its warp partials; greedy argmax uses
separate float-value and int64-index regions. `test-argmax-arena.log.gz` covers
maximum-at-end, tied maxima and NaN behavior against the reference.

## Performance and validation boundaries

Two independent formal measurements on GPU 8 used three alternating rounds,
40 samples per family per round and the median of round medians:

| Measurement | Sum of five families minus five shells | Net bandwidth utilization |
|---|---:|---:|
| First | 5.245924578 ms | 80.0878026% |
| Second | 5.249496375 ms | 80.0333102% |

The fixed denominator is 819 GB/s; modeled family weight traffic is
3.440902144 GB. The formula is
`3.440902144 / net_ms * 1000 / 819 * 100`. GLU, down, QKV, attention output
and logits are isolated families. This is distinct from complete decode latency
and request throughput. Device weight reordering and startup are excluded.
`sites-formal-tail-gig-{all,repeat}.log.gz` and `final-metrics.json.gz` preserve
the individual family and shell measurements. Later complete decode charts use
the separately documented final attention implementation and measurement protocol.

`validate-final-aligned-long.log.gz` records 600 steps with all 151936 logits
bit equal to the earlier safe implementation, including KV tile/page boundaries.
`test-cpp-final.log.gz` records the then-current 71-test C++ suite.

The adjacent CPU source-rdata checks passed 29 tests. Separate small-file GPU
source loading reproduced `cuMemHostRegister_v2` status 1 in the existing generic
`runtime/source_rdata.py::_RegisteredFile`, before C++ device permutation.
`test-source-cuda-alone.log.gz` preserves that unresolved historical failure;
this checkpoint does not claim to fix that loader or pass those GPU cases.
