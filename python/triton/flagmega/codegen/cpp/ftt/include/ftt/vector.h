/* Copyright 2019-2021 Canaan Inc.
 * Modified for FTT: C++17 host and device vectors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "detail/vector_storage.h"

namespace ftt {
template <class T, class Lanes> class basic_vector {
    static_assert(Lanes::is_fixed() && Lanes::rank() > 0, "Vector shape must be nonempty and fixed");
  public:
    static constexpr bool IsVector = true;
    using element_type = T;
    using shape_type = Lanes;
    using strides_type = decltype(default_strides(Lanes{}));
    using traits_type = vector_storage_traits<T, Lanes>;
    using buffer_type = typename traits_type::buffer_type;
    FTT_HOST_DEVICE static constexpr auto rank() noexcept { return Lanes::rank(); }
    FTT_HOST_DEVICE static constexpr Lanes shape() noexcept { return {}; }
    FTT_HOST_DEVICE static constexpr strides_type strides() noexcept { return {}; }
    FTT_HOST_DEVICE static constexpr size_t size() noexcept { return Lanes{}.length(); }
    template <size_t I> FTT_HOST_DEVICE static constexpr auto lane() noexcept { return Lanes{}.template at<I>(); }
    FTT_HOST_DEVICE constexpr basic_vector() noexcept = default;
    FTT_HOST_DEVICE constexpr basic_vector(const buffer_type &buffer) noexcept : buffer_(buffer) {}
    FTT_HOST_DEVICE constexpr explicit basic_vector(T value) noexcept : basic_vector(from_scalar(value)) {}
    FTT_HOST_DEVICE constexpr const buffer_type &buffer() const noexcept { return buffer_; }
    FTT_HOST_DEVICE constexpr buffer_type &buffer() noexcept { return buffer_; }
    FTT_HOST_DEVICE constexpr operator const buffer_type &() const noexcept { return buffer_; }
    FTT_HOST_DEVICE constexpr operator buffer_type &() noexcept { return buffer_; }

    template <class I, detail::enable_if_t<is_dimensions_v<I>, int> = 0>
    FTT_HOST_DEVICE constexpr decltype(auto) operator()(const I &index) noexcept {
        static_assert(I::rank() <= rank(), "Index rank exceeds vector rank");
        if constexpr (I::rank() == 0) { return (*this); }
        else if constexpr (detail::has_element_at<traits_type, I>::value) { return traits_type::element_at(buffer_, index); }
        else { return detail::vector_storage_element_proxy<traits_type, I>(buffer_, index); }
    }
    template <class I, detail::enable_if_t<is_dimensions_v<I>, int> = 0>
    FTT_HOST_DEVICE constexpr decltype(auto) operator()(const I &index) const noexcept {
        static_assert(I::rank() <= rank(), "Index rank exceeds vector rank");
        if constexpr (I::rank() == 0) { return (*this); }
        else if constexpr (detail::has_element_at<traits_type, I>::value) { return traits_type::element_at(buffer_, index); }
        else { return traits_type::get_element(buffer_, index); }
    }
    template <class... I, detail::enable_if_t<(is_dimension_v<I> && ...), int> = 0>
    FTT_HOST_DEVICE constexpr decltype(auto) operator()(const I &...index) noexcept { return (*this)(make_shape(index...)); }
    template <class... I, detail::enable_if_t<(is_dimension_v<I> && ...), int> = 0>
    FTT_HOST_DEVICE constexpr decltype(auto) operator()(const I &...index) const noexcept { return (*this)(make_shape(index...)); }
    FTT_HOST_DEVICE constexpr decltype(auto) flat_at(dim_t i) noexcept { return (*this)(unravel_index(i, Lanes{})); }
    FTT_HOST_DEVICE constexpr T flat_at(dim_t i) const noexcept { return (*this)(unravel_index(i, Lanes{})); }

    template <class U> FTT_HOST_DEVICE static constexpr basic_vector from_scalar(const U &value) noexcept {
        basic_vector result{};
        if constexpr (is_vector_v<U>) {
            static_assert(U::rank() <= rank(), "Broadcast rank exceeds vector rank");
            static_assert(Lanes{}.template slice<rank() - U::rank()>() == U::shape(), "Broadcast lanes must match");
            for (size_t i = 0; i < size(); ++i) { result.flat_at(i) = static_cast<T>(value.flat_at(i % U::size())); }
        } else {
            for (size_t i = 0; i < size(); ++i) { result.flat_at(i) = static_cast<T>(value); }
        }
        return result;
    }
    template <class U> FTT_HOST_DEVICE static constexpr basic_vector unaligned_load_from(const U *source) noexcept {
        basic_vector result{};
        if constexpr (is_vector_v<U>) {
            static_assert(U::rank() <= rank(), "Load rank exceeds vector rank");
            static_assert(Lanes{}.template slice<rank() - U::rank()>() == U::shape(), "Load lanes must match");
            for (size_t i = 0; i < size(); ++i) { result.flat_at(i) = static_cast<T>(source[i / U::size()].flat_at(i % U::size())); }
        } else {
            for (size_t i = 0; i < size(); ++i) { result.flat_at(i) = static_cast<T>(source[i]); }
        }
        return result;
    }
    template <class U> FTT_HOST_DEVICE constexpr void unaligned_store_to(U *destination) const noexcept {
        for (size_t i = 0; i < size(); ++i) { destination[i] = static_cast<U>(flat_at(i)); }
    }
  private:
    buffer_type buffer_{};
};

template <class T, size_t... Lanes> using vector = basic_vector<T, fixed_shape_t<Lanes...>>;
template <class T, class U> struct replace_element_type { using type = U; };
template <class T, class Lanes, class U> struct replace_element_type<basic_vector<T, Lanes>, U> { using type = basic_vector<U, Lanes>; };
template <class T, class U> using replace_element_t = typename replace_element_type<detail::remove_cvref_t<T>, U>::type;
template <class T, size_t... Lanes> using replace_lanes_t = vector<typename T::element_type, Lanes...>;
template <class T, size_t... Lanes> using prepend_lanes_t = basic_vector<typename T::element_type, decltype(T::shape().prepend(fixed_dim_v<Lanes>...))>;
template <class T, bool = is_vector_v<T>> struct vector_rank { static constexpr dim_t value = 0; };
template <class T> struct vector_rank<T, true> { static constexpr dim_t value = detail::remove_cvref_t<T>::rank(); };
template <class T> FTT_CONSTANT fixed_dim<vector_rank<T>::value> vector_rank_v{};
template <class T> using get_last_lane_vector_t = vector<typename T::element_type, T::shape().back()>;
} // namespace ftt
