/* Copyright 2019-2021 Canaan Inc.
 * Modified for FTT: C++17 host and device support.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../tensor_traits.h"

namespace ftt {
template <class A, class B, detail::enable_if_t<is_dimensions_v<A> && is_dimensions_v<B>, int> = 0>
FTT_HOST_DEVICE constexpr bool operator==(const A &a, const B &b) noexcept {
    if constexpr (A::rank() != B::rank()) { return false; }
    else {
        for (dim_t i = 0; i < A::rank(); ++i) { if (a.at(i) != b.at(i)) { return false; } }
        return true;
    }
}
template <class A, class B, detail::enable_if_t<is_dimensions_v<A> && is_dimensions_v<B>, int> = 0>
FTT_HOST_DEVICE constexpr bool operator!=(const A &a, const B &b) noexcept { return !(a == b); }

namespace detail {
template <class A, dim_t Rank> FTT_HOST_DEVICE constexpr bool valid_axes() noexcept {
    const A axes{};
    for (dim_t i = 0; i < A::rank(); ++i) {
        if (axes.at(i) < 0 || axes.at(i) >= Rank) { return false; }
        for (dim_t j = 0; j < i; ++j) { if (axes.at(i) == axes.at(j)) { return false; } }
    }
    return true;
}
template <class S, class T, size_t... I>
FTT_HOST_DEVICE constexpr auto canonicalize_strides_impl(const S &shape, const T &strides, index_sequence<I...>) noexcept {
    return make_strides(where(shape.template at<I>() == dim_one, dim_zero, strides.template at<I>())...);
}
template <class I, class S, size_t... Axis>
FTT_HOST_DEVICE constexpr auto linear_offset_impl(const I &index, const S &strides, index_sequence<Axis...>) noexcept {
    return (dim_zero + ... + (index.template at<Axis>() * strides.template at<Axis>()));
}
template <template <class...> class D, size_t Axis, class O, class S, class R>
FTT_HOST_DEVICE constexpr auto unravel_impl(const O &offset, const S &shape, const R &result) noexcept {
    if constexpr (Axis == 0) { return result; }
    else { return unravel_impl<D, Axis - 1>(offset / shape.template at<Axis - 1>(), shape, result.prepend(offset % shape.template at<Axis - 1>())); }
}
template <size_t Axis, class S, class T, class M, class N>
FTT_HOST_DEVICE constexpr auto linear_size_impl(const S &shape, const T &strides, M max_stride, N max_shape) noexcept {
    if constexpr (Axis == S::rank()) { return max_stride * max_shape; }
    else {
        const auto extent = shape.template at<Axis>();
        const auto stride = strides.template at<Axis>();
        const auto next_stride = where(extent == dim_one, max_stride, where(stride >= max_stride, stride, max_stride));
        const auto next_shape = where(extent == dim_one, max_shape, where(stride >= max_stride, extent, max_shape));
        return linear_size_impl<Axis + 1>(shape, strides, next_stride, next_shape);
    }
}
template <class S, class T>
FTT_HOST_DEVICE constexpr dim_t contiguous_dims_impl(const S &shape, const T &strides) noexcept {
    const auto canonical = default_strides(shape);
    const auto regular = default_strides<S, false>(shape);
    for (dim_t i = S::rank() - 1; i >= 0; --i) {
        if (strides.at(i) != canonical.at(i) && !(shape.at(i) == 1 && strides.at(i) == regular.at(i))) { return S::rank() - i - 1; }
    }
    return S::rank();
}
template <size_t Axis, class D, class A, class R>
FTT_HOST_DEVICE constexpr auto squeeze_impl(const D &dims, const A &axes, const R &result) noexcept {
    if constexpr (Axis == D::rank()) { return result; }
    else if constexpr (A{}.contains(fixed_dim_v<Axis>)) { return squeeze_impl<Axis + 1>(dims, axes, result); }
    else { return squeeze_impl<Axis + 1>(dims, axes, result.append(dims.template at<Axis>())); }
}
template <size_t Axis, size_t Source, class D, class A, class V, class R>
FTT_HOST_DEVICE constexpr auto unsqueeze_impl(const D &dims, const A &axes, const V &value, const R &result) noexcept {
    if constexpr (Axis == D::rank() + A::rank()) { return result; }
    else if constexpr (A{}.contains(fixed_dim_v<Axis>)) { return unsqueeze_impl<Axis + 1, Source>(dims, axes, value, result.append(value)); }
    else { return unsqueeze_impl<Axis + 1, Source + 1>(dims, axes, value, result.append(dims.template at<Source>())); }
}
template <class A, class D, size_t... I>
FTT_HOST_DEVICE constexpr auto positive_axes_impl(const A &axes, const D &dim, index_sequence<I...>) noexcept { return make_shape(positive_index(axes.template at<I>(), dim)...); }
template <class I, class S, size_t... Axis>
FTT_HOST_DEVICE constexpr auto positive_index_impl(const I &index, const S &shape, index_sequence<Axis...>) noexcept { return make_shape(positive_index(index.template at<Axis>(), shape.template at<Axis>())...); }
} // namespace detail

template <class S, class T>
FTT_HOST_DEVICE constexpr auto canonicalize_strides(const S &shape, const T &strides) noexcept {
    static_assert(S::rank() == T::rank(), "Shape and strides ranks must match");
    return detail::canonicalize_strides_impl(shape, strides, detail::make_index_sequence<S::rank()>{});
}
template <dim_t ExtendRank, class S> FTT_HOST_DEVICE constexpr auto broadcast_strides(const S &strides) noexcept {
    static_assert(ExtendRank >= 0, "Extend rank must be nonnegative");
    return make_zeros_strides<ExtendRank>().concat(strides);
}
template <class I, class S, detail::enable_if_t<is_dimensions_v<I> && is_dimensions_v<S>, int> = 0>
FTT_HOST_DEVICE constexpr auto linear_offset(const I &index, const S &shape_or_strides) noexcept {
    static_assert(I::rank() == S::rank(), "Index and strides ranks must match");
    if constexpr (S::usage() == dims_usage::shape) { return linear_offset(index, default_strides(shape_or_strides)); }
    else { return detail::linear_offset_impl(index, shape_or_strides, detail::make_index_sequence<I::rank()>{}); }
}
template <template <class...> class D = shape_t, class O, class S>
FTT_HOST_DEVICE constexpr auto unravel_index(const O &offset, const S &shape) noexcept { return detail::unravel_impl<D, S::rank()>(offset, shape, D<>{}); }
template <class S, class T> FTT_HOST_DEVICE constexpr auto linear_size(const S &shape, const T &strides) noexcept {
    static_assert(S::rank() == T::rank(), "Shape and strides ranks must match");
    return detail::linear_size_impl<0>(shape, strides, dim_one, dim_one);
}
template <class S, class T> FTT_HOST_DEVICE constexpr auto contiguous_dims(const S &shape, const T &strides) noexcept {
    static_assert(S::rank() == T::rank(), "Shape and strides ranks must match");
    if constexpr (S::is_fixed() && T::is_fixed()) { return fixed_dim_v<detail::contiguous_dims_impl(S{}, T{})>; }
    else { return detail::contiguous_dims_impl(shape, strides); }
}
template <class S, class T> inline constexpr size_t max_size_v = []() constexpr {
    if constexpr (S::is_fixed() && T::is_fixed()) { return size_t(linear_size(S{}, T{})); }
    else { return dynamic_extent; }
}();
template <class I, class S> FTT_HOST_DEVICE constexpr bool in_bound(const I &index, const S &shape) noexcept {
    if constexpr (I::rank() != S::rank()) { return false; }
    else {
        for (dim_t i = 0; i < I::rank(); ++i) { if (index.at(i) < 0 || index.at(i) >= shape.at(i)) { return false; } }
        return true;
    }
}
template <class A, class D> FTT_HOST_DEVICE constexpr auto positive_axes(const A &axes, const D &dim) noexcept {
    return detail::positive_axes_impl(axes, dim, detail::make_index_sequence<A::rank()>{});
}
template <class I, class S, detail::enable_if_t<is_dimensions_v<I> && is_dimensions_v<S>, int> = 0>
FTT_HOST_DEVICE constexpr auto positive_index(const I &index, const S &shape) noexcept {
    static_assert(I::rank() == S::rank(), "Index and shape ranks must match");
    return detail::positive_index_impl(index, shape, detail::make_index_sequence<I::rank()>{});
}
template <class D, class A> FTT_HOST_DEVICE constexpr auto squeeze_dims(const D &dims, const A &axes) noexcept {
    static_assert(A::is_fixed(), "Squeeze axes must be fixed");
    const auto positive = positive_axes(axes, fixed_dim_v<D::rank()>);
    static_assert(detail::valid_axes<decltype(positive), D::rank()>(), "Squeeze axes must be unique and in bounds");
    return detail::squeeze_impl<0>(dims, positive, empty_dims_alike_t<D>{});
}
template <class D, class A, class V> FTT_HOST_DEVICE constexpr auto unsqueeze_dims(const D &dims, const A &axes, const V &value) noexcept {
    static_assert(A::is_fixed(), "Unsqueeze axes must be fixed");
    const auto positive = positive_axes(axes, fixed_dim_v<D::rank() + A::rank()>);
    static_assert(detail::valid_axes<decltype(positive), D::rank() + A::rank()>(), "Unsqueeze axes must be unique and in bounds");
    return detail::unsqueeze_impl<0, 0>(dims, positive, value, empty_dims_alike_t<D>{});
}
} // namespace ftt
