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

} // namespace arch
} // namespace ftt
