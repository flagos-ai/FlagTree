/* Copyright 2019-2021 Canaan Inc.
 * Modified for FTT: C++17 freestanding span utilities.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "shape.h"

namespace ftt {
template <class U, class T> FTT_HOST_DEVICE U *typed_pointer_reinterpret(T *source) noexcept { return reinterpret_cast<U *>(source); }
template <class U, class T, size_t Extent>
FTT_HOST_DEVICE auto typed_span_reinterpret(span<T, Extent> source) noexcept {
    static_assert(Extent == dynamic_extent || (Extent * sizeof(T)) % sizeof(U) == 0, "Span size must divide the destination element size");
    using result_type = span<U, Extent == dynamic_extent ? dynamic_extent : Extent * sizeof(T) / sizeof(U)>;
    return result_type{typed_pointer_reinterpret<U>(source.data()), source.size_bytes() / sizeof(U)};
}
template <class U, class T, size_t Extent>
FTT_HOST_DEVICE auto span_cast(span<T, Extent> source) noexcept { return typed_span_reinterpret<U>(source); }
template <class T, size_t SourceExtent, class Offset, class Extent>
FTT_HOST_DEVICE constexpr auto make_subspan(span<T, SourceExtent> source, const Offset &offset, const Extent &extent) noexcept {
    constexpr size_t result_extent = []() constexpr {
        if constexpr (is_fixed_dim_v<Extent>) { return size_t(Extent::value); } else { return dynamic_extent; }
    }();
    return span<T, result_extent>{source.data() + dim_value(offset), size_t(dim_value(extent))};
}
} // namespace ftt
