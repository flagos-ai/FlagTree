/* Copyright 2019-2021 Canaan Inc.
 * Modified for FTT: C++17 type traits.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "detail/dimensions.h"

namespace ftt {
namespace detail {
template <class T, class = void> struct dimensions_traits : integral_constant<bool, false> {};
template <class T> struct dimensions_traits<T, void_t<decltype(T::usage()), decltype(T::is_fixed())>> : integral_constant<bool, true> {};
template <class T, class = void> struct vector_traits : integral_constant<bool, false> {};
template <class T> struct vector_traits<T, void_t<decltype(T::IsVector)>> : integral_constant<bool, T::IsVector> {};
template <class T, class = void> struct element_or_scalar { using type = T; };
template <class T> struct element_or_scalar<T, void_t<typename T::element_type>> { using type = typename T::element_type; };
} // namespace detail
template <class T> inline constexpr bool is_dimensions_v = detail::dimensions_traits<detail::remove_cvref_t<T>>::value;
template <class T> inline constexpr bool is_vector_v = detail::vector_traits<detail::remove_cvref_t<T>>::value;
template <class T, bool = is_dimensions_v<T>> struct is_fixed_dimensions : detail::integral_constant<bool, false> {};
template <class T> struct is_fixed_dimensions<T, true> : detail::integral_constant<bool, detail::remove_cvref_t<T>::is_fixed()> {};
template <class T> inline constexpr bool is_fixed_dimensions_v = is_fixed_dimensions<T>::value;
template <class T> using element_or_scalar_t = typename detail::element_or_scalar<detail::remove_cvref_t<T>>::type;
template <class T, bool = is_vector_v<T>> struct element_scalar_count : detail::integral_constant<size_t, 1> {};
template <class T> struct element_scalar_count<T, true> : detail::integral_constant<size_t, detail::remove_cvref_t<T>::size()> {};
template <class T> inline constexpr size_t element_scalar_count_v = element_scalar_count<T>::value;
template <class T> FTT_HOST_DEVICE constexpr auto unwrap_proxy(const T &value) noexcept { return value; }
} // namespace ftt
