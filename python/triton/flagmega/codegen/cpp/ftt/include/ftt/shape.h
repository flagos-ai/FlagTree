/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "compiler_defs.h"
#include "detail/dimensions.h"

namespace ftt {

template <dim_t... Dims> struct fixed_shape : fixed_shape_t<Dims...> {
    using fixed_shape_t<Dims...>::rank;
    FTT_HOST_DEVICE static constexpr bool is_fixed() noexcept { return true; }
    static constexpr dim_t dims[rank > 0 ? rank : 1] = {Dims...};

    FTT_HOST_DEVICE static constexpr dim_t at(dim_t axis) {
        const dim_t values[rank > 0 ? rank : 1] = {Dims...};
        return values[axis < 0 ? axis + dim_t(rank) : axis];
    }
    template <dim_t Axis> FTT_HOST_DEVICE static constexpr auto at() { return fixed_shape_t<Dims...>{}.template at<Axis>(); }
    template <dim_t Axis> FTT_HOST_DEVICE static constexpr auto at(fixed_dim<Axis>) { return at<Axis>(); }

    FTT_HOST_DEVICE static constexpr dim_t length() {
        dim_t total = 1;
        for (size_t i = 0; i < rank; ++i)
            total *= at(i);
        return total;
    }
};

template <dim_t... Strides> struct fixed_strides : fixed_strides_t<Strides...> {
    using fixed_strides_t<Strides...>::rank;
    FTT_HOST_DEVICE static constexpr bool is_fixed() noexcept { return true; }
    static constexpr dim_t strides[rank > 0 ? rank : 1] = {Strides...};

    FTT_HOST_DEVICE static constexpr dim_t at(dim_t axis) {
        const dim_t values[rank > 0 ? rank : 1] = {Strides...};
        return values[axis < 0 ? axis + dim_t(rank) : axis];
    }
    template <dim_t Axis> FTT_HOST_DEVICE static constexpr auto at() { return fixed_strides_t<Strides...>{}.template at<Axis>(); }
    template <dim_t Axis> FTT_HOST_DEVICE static constexpr auto at(fixed_dim<Axis>) { return at<Axis>(); }
};

namespace detail {

// Row-major stride for `axis` is the product of all extents to its right.
// Written as a constexpr function rather than a recursive class template: the
// terminating specialization would need to be
// `row_major_stride<TShape, TShape::rank - 1>`, whose non-type argument
// depends on a template parameter of the partial specialization, which is
// ill-formed.
template <class TShape> FTT_HOST_DEVICE constexpr dim_t row_major_stride_of(size_t axis) {
    dim_t stride = 1;
    for (size_t i = axis + 1; i < TShape::rank; ++i)
        stride *= TShape{}.at(i);
    return stride;
}

template <class TShape, size_t Axis> struct row_major_stride {
    static constexpr dim_t value = row_major_stride_of<TShape>(Axis);
};

template <class TShape, class I> struct make_row_major;

template <class TShape, size_t... I>
struct make_row_major<TShape, detail::index_sequence<I...>> {
    using type = fixed_strides<row_major_stride<TShape, I>::value...>;
};

} // namespace detail

// Row-major (last axis contiguous) strides for a fixed shape.
template <class TShape>
using row_major_strides_t =
    typename detail::make_row_major<TShape,
                                    detail::make_index_sequence<TShape::rank>>::type;

// Linearize an index tuple against a stride set.
template <class TStrides, class... TIndices>
FTT_HOST_DEVICE constexpr dim_t linear_offset(TIndices... indices) {
    static_assert(sizeof...(TIndices) == TStrides::rank,
                  "index rank must match stride rank");
    const dim_t idx[sizeof...(TIndices) ? sizeof...(TIndices) : 1] = {static_cast<dim_t>(indices)...};
    dim_t offset = 0;
    FTT_UNROLL
    for (size_t axis = 0; axis < TStrides::rank; ++axis)
        offset += idx[axis] * TStrides{}.at(axis);
    return offset;
}

} // namespace ftt

#include "detail/shape_ops.h"
