/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "compiler_defs.h"
#include <cstddef>

// Compile-time shapes and strides.
//
// NTT parameterizes these over `Dimension` concepts so a single tensor type
// can mix fixed and dynamic extents.  FlagMega's generated kernels always
// know their extents at codegen time (the IR has resolved every shape before
// the package plan is produced), so FTT keeps only the fixed form.  That
// removes the dynamic-dimension machinery while preserving the part the
// kernels actually use: static extents, static strides, and row-major
// linearization.

namespace ftt {

template <dim_t... Dims> struct fixed_shape {
    static constexpr size_t rank = sizeof...(Dims);
    static constexpr dim_t dims[rank > 0 ? rank : 1] = {Dims...};

    static constexpr dim_t at(size_t axis) { return dims[axis]; }

    static constexpr dim_t length() {
        dim_t total = 1;
        for (size_t i = 0; i < rank; ++i)
            total *= dims[i];
        return total;
    }
};

template <dim_t... Strides> struct fixed_strides {
    static constexpr size_t rank = sizeof...(Strides);
    static constexpr dim_t strides[rank > 0 ? rank : 1] = {Strides...};

    static constexpr dim_t at(size_t axis) { return strides[axis]; }
};

namespace detail {

// Row-major stride for `axis` is the product of all extents to its right.
// Written as a constexpr function rather than a recursive class template: the
// terminating specialization would need to be
// `row_major_stride<TShape, TShape::rank - 1>`, whose non-type argument
// depends on a template parameter of the partial specialization, which is
// ill-formed.
template <class TShape> constexpr dim_t row_major_stride_of(size_t axis) {
    dim_t stride = 1;
    for (size_t i = axis + 1; i < TShape::rank; ++i)
        stride *= TShape::at(i);
    return stride;
}

template <class TShape, size_t Axis> struct row_major_stride {
    static constexpr dim_t value = row_major_stride_of<TShape>(Axis);
};

template <class TShape, class I> struct make_row_major;

template <class TShape, size_t... I>
struct make_row_major<TShape, std::index_sequence<I...>> {
    using type = fixed_strides<row_major_stride<TShape, I>::value...>;
};

} // namespace detail

// Row-major (last axis contiguous) strides for a fixed shape.
template <class TShape>
using row_major_strides_t =
    typename detail::make_row_major<TShape,
                                    std::make_index_sequence<TShape::rank>>::type;

// Linearize an index tuple against a stride set.
template <class TStrides, class... TIndices>
FTT_HOST_DEVICE constexpr dim_t linear_offset(TIndices... indices) {
    static_assert(sizeof...(TIndices) == TStrides::rank,
                  "index rank must match stride rank");
    const dim_t idx[] = {static_cast<dim_t>(indices)...};
    dim_t offset = 0;
    FTT_UNROLL
    for (size_t axis = 0; axis < TStrides::rank; ++axis)
        offset += idx[axis] * TStrides::at(axis);
    return offset;
}

} // namespace ftt
