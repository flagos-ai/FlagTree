/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "../ftt.h"

// RMS-norm statistics and application.
//
// Qwen3 decode uses RMSNorm, so the statistic is the mean of squares with no
// mean subtraction (NTT's `layer_norm.h` carries a `UseMean` switch for the
// general case; FlagMega's IR has already specialized that away by the time
// the kernel is generated).
//
// The accumulation is fp32 even when the payload is bf16: the IR's cast
// contract requires it, and accumulating in bf16 loses the tail of the sum
// across 2048 channels.

namespace ftt {
namespace kernels {

// Mean of squares over `count` contiguous elements, reduced across one warp.
// Every lane receives the result.
template <class T>
FTT_DEVICE_INLINE float rms_stats(const T *input, dim_t count, dim_t lane,
                                  dim_t lanes) {
    float sum_squares = 0.0f;
    for (dim_t i = lane; i < count; i += lanes) {
        const float value = to_float(input[i]);
        sum_squares += value * value;
    }
    return arch::warp_reduce_sum(sum_squares) / static_cast<float>(count);
}

// Apply RMSNorm with a per-channel weight: out = in * rsqrt(ms + eps) * w.
template <class TIn, class TWeight, class TOut>
FTT_DEVICE_INLINE void rms_apply(const TIn *input, const TWeight *weight,
                                 TOut *output, dim_t count, float mean_square,
                                 float epsilon, dim_t lane, dim_t lanes) {
    const float scale = rsqrt(mean_square + epsilon);
    for (dim_t i = lane; i < count; i += lanes) {
        const float normalized = to_float(input[i]) * scale;
        output[i] = static_cast<TOut>(normalized * to_float(weight[i]));
    }
}

// Fused form: one pass for the statistic, one for the application.  Kept
// separate from the two helpers above because the generated code sometimes
// needs the statistic published to another kernel (the `norm_stats_combine`
// family) rather than consumed locally.
template <class TIn, class TWeight, class TOut>
FTT_DEVICE_INLINE void rms_norm(const TIn *input, const TWeight *weight,
                                TOut *output, dim_t count, float epsilon,
                                dim_t lane, dim_t lanes) {
    const float mean_square = rms_stats(input, count, lane, lanes);
    rms_apply(input, weight, output, count, mean_square, epsilon, lane, lanes);
}

} // namespace kernels
} // namespace ftt
