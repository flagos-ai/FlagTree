/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "../ftt.h"

// Token embedding lookup for the decode path.
//
// One token per step, so this is a strided row copy rather than a gather:
// the row index comes from the input id and every lane copies a slice of the
// hidden dimension.

namespace ftt {
namespace kernels {

template <class TTable, class TOut>
FTT_DEVICE_INLINE void embedding_decode(const TTable *table, const int *ids,
                                        TOut *output, dim_t hidden,
                                        dim_t token_index, dim_t lane,
                                        dim_t lanes) {
    const dim_t row = static_cast<dim_t>(ids[token_index]) * hidden;
    for (dim_t i = lane; i < hidden; i += lanes)
        output[i] = static_cast<TOut>(to_float(table[row + i]));
}

} // namespace kernels
} // namespace ftt
