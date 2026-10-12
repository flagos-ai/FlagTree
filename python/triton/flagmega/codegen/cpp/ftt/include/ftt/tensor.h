/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "shape.h"

// Tensor views over caller-owned memory.
//
// NTT's `basic_tensor` owns or views storage and carries an `IsView` flag.
// FlagMega kernels never allocate: every buffer comes from the artifact's
// workspace / block_local_data / global arguments, with offsets already fixed
// by the buffer plan.  So FTT provides views only -- construction from a raw
// pointer plus a compile-time shape -- and leaves ownership to the runtime.

namespace ftt {

template <class T, class TShape, class TStrides = row_major_strides_t<TShape>>
class tensor_view {
  public:
    using element_type = T;
    using shape_type = TShape;
    using strides_type = TStrides;

    static constexpr size_t rank = TShape::rank;

    FTT_HOST_DEVICE explicit tensor_view(T *data) noexcept : data_(data) {}

    FTT_HOST_DEVICE T *data() const noexcept { return data_; }

    static constexpr dim_t extent(size_t axis) { return TShape::at(axis); }
    static constexpr dim_t stride(size_t axis) { return TStrides::at(axis); }
    static constexpr dim_t size() { return TShape::length(); }

    template <class... TIndices>
    FTT_HOST_DEVICE T &operator()(TIndices... indices) const noexcept {
        return data_[linear_offset<TStrides>(indices...)];
    }

    // Reinterpret this view's storage under a different shape.  Used where the
    // IR reshapes without moving data (for example the register-only reshape
    // the packed GEMV templates do before the K-axis reduction).
    template <class TNewShape,
              class TNewStrides = row_major_strides_t<TNewShape>>
    FTT_HOST_DEVICE tensor_view<T, TNewShape, TNewStrides> reshape() const noexcept {
        static_assert(TNewShape::length() == TShape::length(),
                      "reshape must preserve element count");
        return tensor_view<T, TNewShape, TNewStrides>(data_);
    }

    // Narrow to the sub-view starting at `offset` along axis 0.
    template <dim_t Extent>
    FTT_HOST_DEVICE auto slice(dim_t offset) const noexcept {
        return tensor_view<T, TShape, TStrides>(data_ + offset * TStrides::at(0));
    }

  private:
    T *data_;
};

template <class T, class TShape>
FTT_HOST_DEVICE auto make_tensor_view(T *data) noexcept {
    return tensor_view<T, TShape>(data);
}

} // namespace ftt
