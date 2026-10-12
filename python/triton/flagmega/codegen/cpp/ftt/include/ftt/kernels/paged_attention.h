/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "../ftt.h"

// Paged attention, decode form (one query token).
//
// Split into `partial` and `combine` to match the FlagMega IR: the partial
// pass walks a slice of the KV blocks and emits a running (max, sum, acc)
// triple, and the combine pass merges those triples.  That split is what lets
// the KV sequence be distributed across programs.
//
// The merge uses the standard online-softmax rescaling: when combining two
// partial results with different maxima, each contribution is rescaled by
// exp(local_max - merged_max) before summing.  Doing it this way keeps the
// numerics stable without a second pass over the KV cache.

namespace ftt {
namespace kernels {

struct softmax_partial {
    float max_score;
    float sum_exp;
};

// Attend over KV blocks [block_begin, block_end) for one head.
//
// `kv_cache` is indexed via `block_table`: logical block i lives at physical
// block `block_table[i]`.  Layout per block is [2][block_size][head_dim],
// key first then value.
template <class TQ, class TKV, class TAcc>
FTT_DEVICE_INLINE softmax_partial paged_attention_partial(
    const TQ *query, const TKV *kv_cache, const int *block_table,
    TAcc *accumulator, dim_t head_dim, dim_t block_size, dim_t block_begin,
    dim_t block_end, dim_t seq_len, float scale, dim_t kv_block_stride,
    dim_t lane, dim_t lanes) {
    float running_max = -__builtin_huge_valf();
    float running_sum = 0.0f;

    for (dim_t i = lane; i < head_dim; i += lanes)
        accumulator[i] = static_cast<TAcc>(0.0f);

    for (dim_t block = block_begin; block < block_end; ++block) {
        const dim_t physical = static_cast<dim_t>(block_table[block]);
        const TKV *keys = kv_cache + physical * kv_block_stride;
        const TKV *values = keys + block_size * head_dim;

        for (dim_t slot = 0; slot < block_size; ++slot) {
            const dim_t position = block * block_size + slot;
            if (position >= seq_len)
                break;

            // score = dot(q, k) * scale, reduced across the warp.
            float dot = 0.0f;
            for (dim_t d = lane; d < head_dim; d += lanes)
                dot += to_float(query[d]) * to_float(keys[slot * head_dim + d]);
            const float score = arch::warp_reduce_sum(dot) * scale;

            // Online rescale: fold the new score into (max, sum, acc).
            const float new_max = score > running_max ? score : running_max;
            const float rescale = running_max == -__builtin_huge_valf()
                                      ? 0.0f
                                      : __builtin_expf(running_max - new_max);
            const float weight = __builtin_expf(score - new_max);

            for (dim_t d = lane; d < head_dim; d += lanes) {
                const float previous = to_float(accumulator[d]) * rescale;
                accumulator[d] = static_cast<TAcc>(
                    previous + weight * to_float(values[slot * head_dim + d]));
            }
            running_sum = running_sum * rescale + weight;
            running_max = new_max;
        }
    }

    return softmax_partial{running_max, running_sum};
}

// Merge `count` partial results into `output`.
//
// `partial_acc` is [count][head_dim] and `partials` the matching triples.
template <class TAcc, class TOut>
FTT_DEVICE_INLINE void paged_attention_combine(
    const TAcc *partial_acc, const softmax_partial *partials, TOut *output,
    dim_t head_dim, dim_t count, dim_t lane, dim_t lanes) {
    float merged_max = -__builtin_huge_valf();
    FTT_UNROLL
    for (dim_t i = 0; i < count; ++i)
        merged_max =
            partials[i].max_score > merged_max ? partials[i].max_score : merged_max;

    float merged_sum = 0.0f;
    for (dim_t i = 0; i < count; ++i)
        merged_sum +=
            partials[i].sum_exp * __builtin_expf(partials[i].max_score - merged_max);

    const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;

    for (dim_t d = lane; d < head_dim; d += lanes) {
        float total = 0.0f;
        for (dim_t i = 0; i < count; ++i) {
            const float rescale =
                __builtin_expf(partials[i].max_score - merged_max);
            total += to_float(partial_acc[i * head_dim + d]) * rescale;
        }
        output[d] = static_cast<TOut>(total * inv_sum);
    }
}

} // namespace kernels
} // namespace ftt
