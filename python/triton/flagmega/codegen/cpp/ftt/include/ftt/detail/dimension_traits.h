/* Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "../compiler_defs.h"
#include <stddef.h>
#if __has_include(<type_traits>)
#include <type_traits>
#endif

namespace ftt {
namespace detail {

template <class T, T Value> struct integral_constant {
    using value_type = T;
    using type = integral_constant;
    static constexpr T value = Value;
    FTT_HOST_DEVICE constexpr operator T() const noexcept { return Value; }
    FTT_HOST_DEVICE constexpr T operator()() const noexcept { return Value; }
};

template <class T> struct remove_cv { using type = T; };
template <class T> struct remove_cv<const T> { using type = T; };
template <class T> struct remove_cv<volatile T> { using type = T; };
template <class T> struct remove_cv<const volatile T> { using type = T; };
template <class T> struct remove_reference { using type = T; };
template <class T> struct remove_reference<T &> { using type = T; };
template <class T> struct remove_reference<T &&> { using type = T; };
template <class T> using remove_cvref_t = typename remove_cv<typename remove_reference<T>::type>::type;
template <class A, class B> struct is_same : integral_constant<bool, false> {};
template <class T> struct is_same<T, T> : integral_constant<bool, true> {};
template <bool B, class T = void> struct enable_if {};
template <class T> struct enable_if<true, T> { using type = T; };
template <bool B, class T = void> using enable_if_t = typename enable_if<B, T>::type;
template <bool B, class T, class F> struct conditional { using type = T; };
template <class T, class F> struct conditional<false, T, F> { using type = F; };
template <bool B, class T, class F> using conditional_t = typename conditional<B, T, F>::type;
template <class...> using void_t = void;
template <class T> typename remove_reference<T>::type &&declval() noexcept;
template <class T, class = void> struct is_boolean_condition : integral_constant<bool, false> {};
template <class T> struct is_boolean_condition<T, void_t<decltype(bool(declval<const T &>()))>> : integral_constant<bool, true> {};

template <class T> struct is_integral : integral_constant<bool, false> {};
#define FTT_INTEGRAL_TYPE(T) template <> struct is_integral<T> : integral_constant<bool, true> {};
FTT_INTEGRAL_TYPE(bool)
FTT_INTEGRAL_TYPE(char)
FTT_INTEGRAL_TYPE(signed char)
FTT_INTEGRAL_TYPE(unsigned char)
FTT_INTEGRAL_TYPE(short)
FTT_INTEGRAL_TYPE(unsigned short)
FTT_INTEGRAL_TYPE(int)
FTT_INTEGRAL_TYPE(unsigned int)
FTT_INTEGRAL_TYPE(long)
FTT_INTEGRAL_TYPE(unsigned long)
FTT_INTEGRAL_TYPE(long long)
FTT_INTEGRAL_TYPE(unsigned long long)
FTT_INTEGRAL_TYPE(wchar_t)
FTT_INTEGRAL_TYPE(char16_t)
FTT_INTEGRAL_TYPE(char32_t)
#if defined(__cpp_char8_t)
FTT_INTEGRAL_TYPE(char8_t)
#endif
#undef FTT_INTEGRAL_TYPE

template <class T, class = void> struct is_static_bool : integral_constant<bool, false> {};
template <bool Value>
struct is_static_bool<integral_constant<bool, Value>, void> : integral_constant<bool, true> {};
#if __has_include(<type_traits>)
template <bool Value>
struct is_static_bool<std::integral_constant<bool, Value>, void> : integral_constant<bool, true> {};
#endif

template <size_t... I> struct index_sequence {};
template <size_t N, size_t... I> struct make_index_sequence_impl : make_index_sequence_impl<N - 1, N - 1, I...> {};
template <size_t... I> struct make_index_sequence_impl<0, I...> { using type = index_sequence<I...>; };
template <size_t N> using make_index_sequence = typename make_index_sequence_impl<N>::type;
template <size_t I, class Head, class... Tail> struct type_at : type_at<I - 1, Tail...> {};
template <class Head, class... Tail> struct type_at<0, Head, Tail...> { using type = Head; };

} // namespace detail
} // namespace ftt
