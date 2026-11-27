/* Copyright 2019-2021 Canaan Inc.
 * Modified for FTT: C++17 freestanding vector storage.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../shape.h"

namespace ftt {
template <class T, class Lanes> class basic_vector;
template <class T, class Lanes> struct vector_storage_traits;

namespace detail {
template <class Traits, class Index, class = void> struct has_element_at : integral_constant<bool, false> {};
template <class Traits, class Index>
struct has_element_at<Traits, Index, void_t<decltype(Traits::element_at(declval<typename Traits::buffer_type &>(), declval<const Index &>()))>> : integral_constant<bool, true> {};

template <class Traits, class Index> class vector_storage_element_proxy {
  public:
    using element_type = typename Traits::element_type;
    using buffer_type = typename Traits::buffer_type;
    FTT_HOST_DEVICE constexpr vector_storage_element_proxy(buffer_type &buffer, Index index) noexcept : buffer_(buffer), index_(index) {}
    FTT_HOST_DEVICE constexpr operator element_type() const noexcept { return Traits::get_element(buffer_, index_); }
    FTT_HOST_DEVICE constexpr vector_storage_element_proxy &operator=(element_type value) noexcept {
        Traits::set_element(buffer_, index_, value);
        return *this;
    }
    FTT_HOST_DEVICE constexpr vector_storage_element_proxy &operator=(const vector_storage_element_proxy &value) noexcept { return operator=(static_cast<element_type>(value)); }
  private:
    buffer_type &buffer_;
    Index index_;
};
} // namespace detail

template <class Traits, class Index>
FTT_HOST_DEVICE constexpr auto unwrap_proxy(const detail::vector_storage_element_proxy<Traits, Index> &value) noexcept { return static_cast<typename Traits::element_type>(value); }

template <class T, class Lane> struct vector_storage_traits<T, shape_t<Lane>> {
    static_assert(is_fixed_dim_v<Lane> && Lane::value > 0, "Vector lanes must be positive and fixed");
    using buffer_type = array<T, Lane::value>;
    using element_type = T;
    template <class I> FTT_HOST_DEVICE static constexpr T &element_at(buffer_type &buffer, const I &index) noexcept {
        static_assert(I::rank() == 1, "Index rank must be one");
        return buffer[index.at(0)];
    }
    template <class I> FTT_HOST_DEVICE static constexpr const T &element_at(const buffer_type &buffer, const I &index) noexcept {
        static_assert(I::rank() == 1, "Index rank must be one");
        return buffer[index.at(0)];
    }
};
template <class T, class Outer, class Head, class... Tail>
struct vector_storage_traits<T, shape_t<Outer, Head, Tail...>> {
    static_assert(is_fixed_dim_v<Outer> && Outer::value > 0, "Vector lanes must be positive and fixed");
    using buffer_type = array<basic_vector<T, shape_t<Head, Tail...>>, Outer::value>;
    using element_type = T;
    template <class I> FTT_HOST_DEVICE static constexpr decltype(auto) element_at(buffer_type &buffer, const I &index) noexcept {
        return buffer[index.at(0)](index.template slice<1>());
    }
    template <class I> FTT_HOST_DEVICE static constexpr decltype(auto) element_at(const buffer_type &buffer, const I &index) noexcept {
        return buffer[index.at(0)](index.template slice<1>());
    }
};
} // namespace ftt
