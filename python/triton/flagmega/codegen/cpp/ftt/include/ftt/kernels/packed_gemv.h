/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "../ftt.h"

// Packed-weight GEMV: out[n] = sum_k act[k] * w[k, n].
//
// This is the port of the Triton `packed_fused_gemv` template, and it keeps
// that template's central property: the weight is walked in PACKED PHYSICAL
// ORDER, not logical (n, k) order.
//
// Layout `k_major_n8_k16`: each (n_lane x k_lane) block is contiguous
// row-major, blocks sit back to back along n, and k groups are strided.  So
// the payload axis `tile_n * k_lane` is contiguous in memory and a lane-major
// walk over it is coalesced.  Addressing logical (n, k) instead turns the same
// memory into a k_group_stride-strided scatter -- measured at ~16-25 GB/s
// against full bandwidth for the packed walk.
//
// The K axes are reduced only at loop exit, via a register-only reshape:
// accumulate into [k_groups][tile_n * k_lane], then fold k_lane and k_groups.
// No shared-memory exchange is involved.

namespace ftt {
namespace kernels {

// `TileN`      logical n values produced per program
// `KLane`      contiguous k values inside one packed block
// `NLane`      n values inside one packed block
// `BlockK`     k values consumed per iteration (multiple of KLane)
template <dim_t TileN, dim_t KLane, dim_t NLane, dim_t BlockK, class TAct,
          class TWeight, class TOut>
FTT_DEVICE_INLINE void packed_gemv(const TAct *activation,
                                   const TWeight *weight, TOut *output,
                                   dim_t k_extent, dim_t n_start,
                                   dim_t n_capacity,
                                   dim_t weight_k_group_stride,
                                   dim_t weight_n_group_stride,
                                   dim_t weight_group_start, dim_t lane,
                                   dim_t lanes) {
    constexpr dim_t payload = TileN * KLane;
    constexpr dim_t k_groups = BlockK / KLane;

    // Accumulator lives in registers: [k_groups][payload] folded at exit.
    float acc[k_groups][payload];
    FTT_UNROLL
    for (dim_t g = 0; g < k_groups; ++g)
        FTT_UNROLL
        for (dim_t p = 0; p < payload; ++p)
            acc[g][p] = 0.0f;

    for (dim_t k_base = 0; k_base < k_extent; k_base += BlockK) {
        FTT_UNROLL
        for (dim_t g = 0; g < k_groups; ++g) {
            FTT_UNROLL
            for (dim_t p = 0; p < payload; ++p) {
                const dim_t n_local = n_start + p / KLane;
                const dim_t k_in_lane = p % KLane;
                const dim_t k = k_base + g * KLane + k_in_lane;
                if (k >= k_extent || n_local >= n_capacity)
                    continue;

                // Packed physical address: k group, then n group, then the
                // (n, k) position inside the block.
                const dim_t w_offset =
                    (k_base / KLane + g) * weight_k_group_stride +
                    (weight_group_start + n_local / NLane) *
                        weight_n_group_stride +
                    (n_local % NLane) * KLane + k_in_lane;

                acc[g][p] += to_float(weight[w_offset]) *
                             to_float(activation[k]);
            }
        }
    }

    // Fold the two K axes.  `payload` is (tile_n, k_lane) interleaved, so
    // summing k_lane collapses the inner axis and summing k_groups the outer.
    FTT_UNROLL
    for (dim_t n = 0; n < TileN; ++n) {
        if (n_start + n >= n_capacity)
            continue;
        float total = 0.0f;
        FTT_UNROLL
        for (dim_t g = 0; g < k_groups; ++g)
            FTT_UNROLL
            for (dim_t kl = 0; kl < KLane; ++kl)
                total += acc[g][n * KLane + kl];
        output[n_start + n] = static_cast<TOut>(total);
    }

    (void)lane;
    (void)lanes;
}

} // namespace kernels
} // namespace ftt
