/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "shape.h"

// Loop skeletons.
//
// NTT's `apply.h` walks an arbitrary-rank shape with a callable, handling
// tiling and dynamic extents.  FlagMega's kernels are generated per-rank with
// static extents, so FTT keeps the two forms the generated code needs:
// a rank-generic static walk, and a lane-strided walk for the data-parallel
// kernels where each lane owns a slice of the last axis.

namespace ftt {

namespace detail {

template <class TShape, size_t Axis, class Callable, class... TIndices>
FTT_DEVICE_INLINE void apply_axis(Callable &&body, TIndices... outer) {
    if constexpr (Axis == TShape::rank) {
        body(outer...);
    } else {
        FTT_UNROLL
        for (dim_t i = 0; i < TShape::at(Axis); ++i)
            apply_axis<TShape, Axis + 1>(body, outer...,
                                         i);
    }
}

} // namespace detail

// Visit every index of `TShape` in row-major order.
template <class TShape, class Callable>
FTT_DEVICE_INLINE void apply(Callable &&body) {
    detail::apply_axis<TShape, 0>(body);
}

namespace detail {
template <size_t Axis, class Shape, class Callable, class Index>
FTT_HOST_DEVICE constexpr void apply_shape_axis(const Shape &shape, Callable &body, const Index &index) {
    if constexpr (Axis == Shape::rank()) { body(index); }
    else {
        FTT_UNROLL
        for (dim_t i = 0; i < shape.template at<Axis>(); ++i) { apply_shape_axis<Axis + 1>(shape, body, index.append(i)); }
    }
}
} // namespace detail

template <class Shape, class Callable>
FTT_HOST_DEVICE constexpr void apply(const Shape &shape, Callable &&body) {
    detail::apply_shape_axis<0>(shape, body, make_shape());
}

// Visit `count` elements, striding by the CTA's thread count so each thread
// takes a disjoint subset.  This is the shape most of the elementwise and
// normalization kernels want.
template <class Callable>
FTT_DEVICE_INLINE void apply_strided(dim_t count, dim_t thread, dim_t threads,
                                     Callable &&body) {
    for (dim_t i = thread; i < count; i += threads)
        body(i);
}

} // namespace ftt
