/* Copyright 2019-2021 Canaan Inc.
 * Modified for FTT: C++17 host and device support.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "dimension.h"

namespace ftt {
namespace detail {
template <class Callable, size_t... I>
FTT_HOST_DEVICE constexpr void loop_impl(Callable &body, index_sequence<I...>) {
    (body(fixed_dim_v<I>), ...);
}
} // namespace detail

template <dim_t N, class Callable>
FTT_HOST_DEVICE constexpr void loop(Callable &&body) {
    static_assert(N >= 0, "Loop extent must be nonnegative");
    detail::loop_impl(body, detail::make_index_sequence<N>{});
}
} // namespace ftt
