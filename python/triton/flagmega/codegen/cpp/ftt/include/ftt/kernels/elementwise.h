/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "../ftt.h"

// Elementwise and GLU kernels.

namespace ftt {
namespace kernels {

template <class TIn, class TOut>
FTT_DEVICE_INLINE void cast(const TIn *input, TOut *output, dim_t count,
                            dim_t lane, dim_t lanes) {
    for (dim_t i = lane; i < count; i += lanes)
        output[i] = static_cast<TOut>(to_float(input[i]));
}

template <class TA, class TB, class TOut>
FTT_DEVICE_INLINE void add(const TA *lhs, const TB *rhs, TOut *output,
                           dim_t count, dim_t lane, dim_t lanes) {
    for (dim_t i = lane; i < count; i += lanes)
        output[i] = static_cast<TOut>(to_float(lhs[i]) + to_float(rhs[i]));
}

// SwiGLU: out = silu(gate) * up.
//
// Qwen3's MLP produces gate and up as one fused projection, so they arrive as
// two halves of the same buffer rather than separate tensors.
template <class TIn, class TOut>
FTT_DEVICE_INLINE void swiglu(const TIn *gate_up, TOut *output,
                              dim_t intermediate, dim_t lane, dim_t lanes) {
    for (dim_t i = lane; i < intermediate; i += lanes) {
        const float gate = to_float(gate_up[i]);
        const float up = to_float(gate_up[intermediate + i]);
        output[i] = static_cast<TOut>(silu(gate) * up);
    }
}

// Residual add then RMSNorm statistic, fused.  The decode layer needs the
// statistic of the post-residual tensor, so computing it in the same pass
// avoids re-reading the hidden state.
template <class TIn, class TRes, class TOut>
FTT_DEVICE_INLINE float add_norm_stats(const TIn *input, const TRes *residual,
                                       TOut *output, dim_t count, dim_t lane,
                                       dim_t lanes) {
    float sum_squares = 0.0f;
    for (dim_t i = lane; i < count; i += lanes) {
        const float value = to_float(input[i]) + to_float(residual[i]);
        output[i] = static_cast<TOut>(value);
        sum_squares += value * value;
    }
    return arch::warp_reduce_sum(sum_squares) / static_cast<float>(count);
}

} // namespace kernels
} // namespace ftt
