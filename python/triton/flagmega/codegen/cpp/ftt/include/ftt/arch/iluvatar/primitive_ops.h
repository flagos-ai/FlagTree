/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "arch_types.h"

// Iluvatar lane/warp/CTA primitives.
//
// Same role as NTT's `arch/<name>/primitive_ops.h`: wrap the vendor builtins
// so generic code never names them directly.  Everything here was checked
// against the COREX builtin table (BuiltinsIluvatar.def) rather than assumed
// from the CUDA equivalents, because several differ:
//
//   * `warp_size` is 64, so a lane mask is 6 bits.
//   * `__syncthreads_alu` is a CTA-WIDE rendezvous, not a warp one -- it was
//     measured that a single warp cannot clear it alone.  Use `warp_sync()`
//     when only one warp must converge.
//   * `vote.all` IS warp-scope and does order the lanes' prior shared-memory
//     writes against lane 0's read, which is what the warp-aggregated
//     reductions below rely on.

namespace ftt {
namespace arch {

FTT_DEVICE_INLINE int lane_id() {
#if defined(FTT_DEVICE_PASS)
    return __ivcorex_lane_id();
#else
    return 0;
#endif
}

FTT_DEVICE_INLINE int thread_id() {
#if defined(FTT_DEVICE_PASS)
    return static_cast<int>(threadIdx.x);
#else
    return 0;
#endif
}

FTT_DEVICE_INLINE int warp_id() {
    return thread_id() / iluvatar_bi_v150::warp_size;
}

FTT_DEVICE_INLINE int block_id() {
#if defined(FTT_DEVICE_PASS)
    return static_cast<int>(blockIdx.x);
#else
    return 0;
#endif
}

// Whole-CTA rendezvous with a full memory fence.
FTT_DEVICE_INLINE void cta_sync() {
#if defined(FTT_DEVICE_PASS)
    __syncthreads();
#endif
}

// Whole-CTA rendezvous without the memory-system fence.  Cheaper than
// `cta_sync()` but NOT warp-scope: every warp of the CTA must reach it.
FTT_DEVICE_INLINE void cta_sync_alu() {
#if defined(FTT_DEVICE_PASS)
    __syncthreads_alu();
#endif
}

// Warp-scope convergence.  A single warp can clear this on its own.
FTT_DEVICE_INLINE void warp_sync() {
#if defined(FTT_DEVICE_PASS)
    (void)__ivcorex_vote_all(1);
#endif
}

// Release-side writeback-invalidate.  Required before publishing a counter
// that another CTA will observe, because this machine has no inter-SM L1
// coherence.  Must sit in the same basic block as the publishing atomic: the
// vendor backend does not emit the invalidate across a conditional branch.
FTT_DEVICE_INLINE void release_fence() {
#if defined(FTT_DEVICE_PASS)
    __threadfence();
#endif
}

FTT_DEVICE_INLINE float shuffle(float value, int src_lane) {
#if defined(FTT_DEVICE_PASS)
    return __shfl_sync(0xffffffffffffffffull, value, src_lane,
                       iluvatar_bi_v150::warp_size);
#else
    (void)src_lane;
    return value;
#endif
}

// Sum across the 64 lanes of one warp; every lane receives the total.
FTT_DEVICE_INLINE float warp_reduce_sum(float value) {
#if defined(FTT_DEVICE_PASS)
    FTT_UNROLL
    for (int offset = iluvatar_bi_v150::warp_size / 2; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffffffffffull, value, offset,
                                 iluvatar_bi_v150::warp_size);
#endif
    return value;
}

// Sum across a contiguous group of `Width` lanes within one warp; every lane
// in the group receives the group's total.  `Width` must be a power of two no
// larger than the warp.
//
// This is the reduction a packed-layout GEMV needs.  The weight layout stores
// `KLane` consecutive K values contiguously for one N, so the loads only
// coalesce if `KLane` lanes cooperate on one N and each takes a different K --
// which leaves a partial sum per lane that has to be folded across exactly
// those lanes, not across the whole warp.
template <int Width>
FTT_DEVICE_INLINE float lane_group_reduce_sum(float value) {
    static_assert(Width > 0 && (Width & (Width - 1)) == 0,
                  "Lane-group width must be a power of two.");
    static_assert(Width <= iluvatar_bi_v150::warp_size,
                  "A lane group cannot span more than one warp.");
#if defined(FTT_DEVICE_PASS)
    FTT_UNROLL
    for (int offset = Width / 2; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffffffffffull, value, offset,
                                 iluvatar_bi_v150::warp_size);
#endif
    return value;
}

FTT_DEVICE_INLINE float warp_reduce_max(float value) {
#if defined(FTT_DEVICE_PASS)
    FTT_UNROLL
    for (int offset = iluvatar_bi_v150::warp_size / 2; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffffffffffull, value, offset,
                                            iluvatar_bi_v150::warp_size);
        value = value > other ? value : other;
    }
#endif
    return value;
}

// Position of this CTA along one axis of the launch's logical block mesh.
//
// A FlagMega distributed entry addresses its shards through a row-major mesh
// over the x block dimension, so the shard coordinate is a decomposition of
// `blockIdx.x` -- the same mapping the Triton backend gets from
// `tle.shard_id`.  `Axis` is the mesh axis and `Extents` its full shape, both
// known at codegen time.
template <int Axis, class... Extents>
FTT_DEVICE_INLINE dim_t mesh_shard_id(Extents... extents) {
    constexpr int rank = sizeof...(Extents);
    static_assert(Axis >= 0 && Axis < rank, "Mesh axis is outside the mesh rank.");
    const dim_t sizes[rank] = {static_cast<dim_t>(extents)...};
    dim_t stride = 1;
    for (int axis = Axis + 1; axis < rank; ++axis)
        stride *= sizes[axis];
    return (static_cast<dim_t>(block_id()) / stride) % sizes[Axis];
}

// Grid-wide rendezvous over a caller-provided counter in global memory.
//
// This machine has no cooperative-launch admission API and no hardware grid
// barrier, so this is a software counter barrier, and it is only safe when the
// launch is no larger than `sm_count` CTAs -- otherwise a non-resident CTA can
// never arrive and the resident ones spin forever.  The runtime's resource
// contract enforces that bound (`ResourceContract.available_sm_count`); this
// function assumes it.
//
// The counter only ever increases: the nth barrier releases when the count
// crosses `n * blocks`, so no phase reset is needed and a CTA cannot race
// ahead into the next barrier's count.
//
// Three details are load-bearing on this machine, and getting any of them
// wrong is silent -- the barrier still returns, just too early, and the
// caller reads another CTA's stale data (measured: 2 of 16 CTAs saw 14 stale
// values each):
//
//   * The spin-wait must re-read the counter from memory every iteration.
//     `__ldg` is a read-only-cache load, so it can spin forever on a stale
//     line or -- worse, since the loop is bounded in practice -- observe the
//     release late. `atomicAdd(counter, 0)` is the read that provably
//     bypasses the cache hierarchy on COREX.
//   * `release_fence()` must sit in the same basic block as the arriving
//     atomic: this backend does not emit the writeback-invalidate across a
//     conditional branch, so another SM's L1 would hold a stale count.
//   * There must be an acquire-side fence after the wait. This machine has no
//     inter-SM L1 coherence, so passing the barrier does not by itself make
//     the *data* other CTAs published visible -- only the counter. Without
//     the second fence the arriving CTA reads its own cached copies.
FTT_DEVICE_INLINE void grid_barrier(unsigned int *counter, int blocks) {
#if defined(FTT_DEVICE_PASS)
    cta_sync();
    if (thread_id() == 0) {
        release_fence();
        const unsigned int generation =
            atomicAdd(counter, 1u) / static_cast<unsigned int>(blocks);
        while (atomicAdd(counter, 0u) / static_cast<unsigned int>(blocks)
               == generation) {
        }
        release_fence();
    }
    cta_sync();
#else
    (void)counter;
    (void)blocks;
#endif
}


// Asynchronous global->shared tile copy on the SME engine.
//
// `rowxfb16` moves 16 source rows of 64-bit granules into shared memory and
// interleaves two rows per 32-bit word: shared slot `i` holds
// `src(row = i % 2, col = i / 2)`. That is exactly the `k_major_n8_k16` packed
// weight order, which is why the GEMV kernels can consume the tile directly.
//
// Three things about this engine are load-bearing, all measured (see the
// tutorial's .local/perf-iteration/cpp-backend-iteration.zh.md):
//
//   * `sme_wait()` drains *every* outstanding G2S transaction; there is no
//     "wait for the oldest N" form. So a classic double-buffer that issues the
//     next tile and then waits on the current one gains nothing -- measured
//     61 GB/s against 88 for plain scalar loads.
//   * MANY transactions queue per thread. Issuing 12 tiles back to back from
//     one thread and draining once with `__ivcorex_sl_waitcnt` lands all
//     6144 elements, under both wait masks 8 and 12.
//
//     An earlier note here claimed "only ONE transaction per CTA is in
//     flight", from a probe whose filled-slot count scaled as 1/N. That was a
//     defect in the probe, not the engine: it dispatched one tile per *thread*
//     (`threadIdx.x < n`), and `sl_waitcnt` is per-thread, so the inspecting
//     thread awaited only its own transaction and read the rest before they
//     landed. Same tile count issued from a single thread lands completely;
//     dispatched one-per-thread it reports 512/6144. Issue a batch from one
//     thread, or have every issuing thread wait for its own tiles.
//
//     An even earlier claim of 816 GB/s was also wrong, for a different
//     reason: that probe only accumulated a checksum and never validated the
//     staged bytes. Validate staged data before believing any number here.
//   * Cross-warp staging works: a producer warp can stage tiles that other
//     warps consume, with the producer's `waitcnt` making the data visible and
//     a CTA barrier making the producer's issue visible (measured 0 unfilled
//     slots of 6144 from the consumer side). A CTA barrier alone does NOT
//     drain the queue -- the wait is required.
//
//     Even so, SME producer staging does not pay for the GEMV path here, and
//     the reason is not the engine: those kernels are bound by load-issue
//     slots, and staging does not reduce the consumer's load count, it only
//     moves the reads from global to shared. A 16-byte shared read measures
//     0.95x the cost of a 16-byte global one -- the same issue port -- so the
//     consumer saves ~5%, worth ~0.15 ms/token on the measured 3.58 ms GLU
//     cost, while dedicating 2 of 16 warps to production costs ~0.45 ms of
//     consumer issue throughput. Net negative before counting the per-k-group
//     producer/consumer handshake. This is the same reason round 10.3's
//     shared-memory activation staging lost 30.69 vs 19.91 ms/token.
//
//     SME is the right tool when the consumer is an MMA that reads its
//     operands from shared anyway, which is how the Triton backend uses it
//     (`emitIluvatarSmeTileLoads`, predicated on `smeWarpsPerCTA`): there the
//     shared read is not an added cost because it was already required. A
//     batch-1 GEMV reads its weights exactly once, so there is no reuse to
//     amortize the extra hop.
//   * The descriptor's third lane must be -1 and its fourth the source row
//     stride in *bytes*.
//   * The row stride has a 64-byte minimum. Below it the instruction silently
//     reads only 16 source elements and repeats them across the destination --
//     no fault, no partial write, just duplicated data. A 32-byte stride
//     (16 bf16, which is what a `k_major_n8_k16` lane looks like) is therefore
//     unusable, and that single fact produced four rounds of apparently
//     unrelated wrong-token bugs.
//   * One instruction moves 16 rows x 32 columns = 512 elements, independent of
//     the stride; the stride only sets the row pitch. So it cannot densely copy
//     a contiguous run whose rows are narrower than 32 elements -- it is built
//     for taking a 16x32 tile out of a wider matrix, which is how the Triton
//     backend uses it for dot operands.
//   * The destination permutation is regular only over the first 256 slots
//     (`smem[i] = src(row = 2*(i/32) + i%2, col = (i/2)%16)`); slots 256..511
//     follow a different pattern (slot 256 holds row 4, not row 0). Measure it
//     with an integer-only probe before relying on it. `__bf16` and FTT's own
//     `bfloat16_t` both read back as zero in a standalone translation unit on
//     this toolchain, which makes float-based probes report "nothing was
//     copied" when the copy in fact succeeded.
//
// The destination must be 64-bit aligned shared memory with room for
// 16 * `granules_per_row` 16-bit elements.
FTT_DEVICE_INLINE void sme_copy_tile(void *destination, const void *source,
                                     int source_row_stride_bytes,
                                     int source_byte_offset = 0) {
#if defined(FTT_DEVICE_PASS)
    using descriptor_t = int __attribute__((__vector_size__(16)));
    const unsigned long long address =
        reinterpret_cast<unsigned long long>(source);
    const descriptor_t base = {
        static_cast<int>(address & 0xffffffffu),
        static_cast<int>(address >> 32),
        -1,
        source_row_stride_bytes,
    };
    __ivcorex_sme_load_16x1b64_rowxfb16(
        static_cast<int>(reinterpret_cast<unsigned long long>(destination)),
        base, source_byte_offset, 0);
#else
    (void)destination;
    (void)source;
    (void)source_row_stride_bytes;
    (void)source_byte_offset;
#endif
}

// Drain the G2S queue and rendezvous the CTA, so every thread may read any
// tile the batch just published. Both halves are required: the wait makes the
// data visible, the barrier makes the *producer's* issue visible to consumers.
//
// The wait is per-thread and only covers transactions this thread issued, so
// a caller that spreads tile issue across threads must have each of them call
// this, not just one. Verified both ways: 12 tiles issued from one thread land
// completely, while one-tile-per-thread with a single waiter reports 512 of
// 6144. The Triton backend uses mask 8 (G2S only) where this uses 12 (G2S plus
// shared stores); both land the data, and 12 is the safer default when the
// caller may also have written the same buffer with ordinary shared stores.
FTT_DEVICE_INLINE void sme_wait() {
#if defined(FTT_DEVICE_PASS)
    __ivcorex_sl_waitcnt(12);  // bit2 = shared stores, bit3 = G2S
    cta_sync();
#endif
}

} // namespace arch
} // namespace ftt
