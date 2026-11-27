/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "../ftt.h"

// Rotary position embedding (RoPE), decode form.
//
// Qwen3 uses the half-split convention: the head dimension is halved and the
// pair (x[i], x[i + half]) is rotated together.  That differs from the
// interleaved convention some models use, and getting it wrong is silent --
// the output stays finite and only the attention scores degrade.  So the
// pairing is written explicitly rather than inferred from a stride.

namespace ftt {
namespace kernels {

// Rotate one head in place.  `cos`/`sin` are indexed by the position, and
// hold `half` entries each.
template <class T, class TTrig>
FTT_DEVICE_INLINE void rope_rotate_head(T *head, const TTrig *cos,
                                        const TTrig *sin, dim_t head_dim,
                                        dim_t lane, dim_t lanes) {
    const dim_t half = head_dim / 2;
    for (dim_t i = lane; i < half; i += lanes) {
        const float lo = to_float(head[i]);
        const float hi = to_float(head[i + half]);
        const float c = to_float(cos[i]);
        const float s = to_float(sin[i]);
        head[i] = static_cast<T>(lo * c - hi * s);
        head[i + half] = static_cast<T>(hi * c + lo * s);
    }
}

// Apply RoPE across all heads of one token.
template <class T, class TTrig>
FTT_DEVICE_INLINE void rope_decode(T *tensor, const TTrig *cos,
                                   const TTrig *sin, dim_t heads,
                                   dim_t head_dim, dim_t lane, dim_t lanes) {
    for (dim_t h = 0; h < heads; ++h)
        rope_rotate_head(tensor + h * head_dim, cos, sin, head_dim, lane,
                         lanes);
}

} // namespace kernels
} // namespace ftt
