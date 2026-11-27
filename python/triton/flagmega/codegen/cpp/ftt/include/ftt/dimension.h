/* Copyright 2019-2021 Canaan Inc.
 * Modified for FTT: C++17 freestanding dimension and shape support.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "detail/dimension_traits.h"

namespace ftt {

template <dim_t Value> struct fixed_dim : detail::integral_constant<dim_t, Value> {};
template <dim_t Value> FTT_CONSTANT fixed_dim<Value> fixed_dim_v{};
FTT_CONSTANT fixed_dim<0> dim_zero{};
FTT_CONSTANT fixed_dim<1> dim_one{};
template <class T> struct is_fixed_dim_t : detail::integral_constant<bool, false> {};
template <dim_t Value> struct is_fixed_dim_t<fixed_dim<Value>> : detail::integral_constant<bool, true> {};
template <class T> inline constexpr bool is_fixed_dim_v = is_fixed_dim_t<detail::remove_cvref_t<T>>::value;
template <class T> inline constexpr bool is_dimension_v = is_fixed_dim_v<T> || detail::is_integral<detail::remove_cvref_t<T>>::value;

template <class T, detail::enable_if_t<is_dimension_v<T>, int> = 0>
FTT_HOST_DEVICE constexpr dim_t dim_value(const T &dim) noexcept {
    if constexpr (is_fixed_dim_v<T>) { return detail::remove_cvref_t<T>::value; }
    else { return static_cast<dim_t>(dim); }
}

namespace detail {
template <char... Digits> FTT_HOST_DEVICE constexpr dim_t dimension_literal() {
    static_assert(((Digits >= '0' && Digits <= '9') && ...), "Invalid dimension literal");
    dim_t value = 0;
    ((value = value * 10 + (Digits - '0')), ...);
    return value;
}
} // namespace detail

template <char... Digits> FTT_HOST_DEVICE constexpr auto operator""_dim() {
    return fixed_dim_v<detail::dimension_literal<Digits...>()>;
}
template <dim_t Value> FTT_HOST_DEVICE constexpr auto operator-(fixed_dim<Value>) noexcept {
    return fixed_dim_v<-Value>;
}

#define FTT_DIM_BINARY_OP(op) \
    template <class L, class R, detail::enable_if_t<is_dimension_v<L> && is_dimension_v<R> && (is_fixed_dim_v<L> || is_fixed_dim_v<R>), int> = 0> \
    FTT_HOST_DEVICE constexpr auto operator op(const L &lhs, const R &rhs) noexcept { \
        if constexpr (is_fixed_dim_v<L> && is_fixed_dim_v<R>) { return fixed_dim_v<L::value op R::value>; } \
        else { return dim_value(lhs) op dim_value(rhs); } \
    }
FTT_DIM_BINARY_OP(+)
FTT_DIM_BINARY_OP(-)
FTT_DIM_BINARY_OP(*)
FTT_DIM_BINARY_OP(/)
FTT_DIM_BINARY_OP(%)
#undef FTT_DIM_BINARY_OP

#define FTT_DIM_COMPARE_OP(op) \
    template <class L, class R, detail::enable_if_t<is_dimension_v<L> && is_dimension_v<R> && (is_fixed_dim_v<L> || is_fixed_dim_v<R>), int> = 0> \
    FTT_HOST_DEVICE constexpr auto operator op(const L &lhs, const R &rhs) noexcept { \
        if constexpr (is_fixed_dim_v<L> && is_fixed_dim_v<R>) { return detail::integral_constant<bool, (L::value op R::value)>{}; } \
        else { return dim_value(lhs) op dim_value(rhs); } \
    }
FTT_DIM_COMPARE_OP(==)
FTT_DIM_COMPARE_OP(!=)
FTT_DIM_COMPARE_OP(<)
FTT_DIM_COMPARE_OP(<=)
FTT_DIM_COMPARE_OP(>)
FTT_DIM_COMPARE_OP(>=)
#undef FTT_DIM_COMPARE_OP

template <class Cond, class T, class F, detail::enable_if_t<detail::is_static_bool<Cond>::value || (detail::is_boolean_condition<Cond>::value && is_dimension_v<T> && is_dimension_v<F>), int> = 0>
FTT_HOST_DEVICE constexpr auto where(const Cond &cond, const T &true_dim, const F &false_dim) noexcept {
    if constexpr (detail::is_static_bool<Cond>::value) {
        if constexpr (Cond::value) { return true_dim; }
        else { return false_dim; }
    } else {
        static_assert(is_dimension_v<T> && is_dimension_v<F>, "Dynamic where requires dimensions");
        return cond ? dim_value(true_dim) : dim_value(false_dim);
    }
}

template <class N, class D, detail::enable_if_t<is_dimension_v<N> && is_dimension_v<D>, int> = 0>
FTT_HOST_DEVICE constexpr auto ceil_div(const N &num, const D &denom) noexcept {
    if constexpr (is_fixed_dim_v<N> && is_fixed_dim_v<D>) { return (num + denom - dim_one) / denom; }
    else { return (dim_value(num) + dim_value(denom) - 1) / dim_value(denom); }
}
template <class A, class B, detail::enable_if_t<is_dimension_v<A> && is_dimension_v<B>, int> = 0>
FTT_HOST_DEVICE constexpr auto min(const A &a, const B &b) noexcept {
    if constexpr (is_fixed_dim_v<A> && is_fixed_dim_v<B>) { return where(a < b, a, b); }
    else { return where(dim_value(a) < dim_value(b), a, b); }
}
template <class A, class B, detail::enable_if_t<is_dimension_v<A> && is_dimension_v<B>, int> = 0>
FTT_HOST_DEVICE constexpr auto max(const A &a, const B &b) noexcept {
    if constexpr (is_fixed_dim_v<A> && is_fixed_dim_v<B>) { return where(a > b, a, b); }
    else { return where(dim_value(a) > dim_value(b), a, b); }
}
template <class D, detail::enable_if_t<is_dimension_v<D>, int> = 0> FTT_HOST_DEVICE constexpr auto min(const D &dim) noexcept {
    if constexpr (is_fixed_dim_v<D>) { return dim; } else { return dim_value(dim); }
}
template <class D, detail::enable_if_t<is_dimension_v<D>, int> = 0> FTT_HOST_DEVICE constexpr auto max(const D &dim) noexcept {
    if constexpr (is_fixed_dim_v<D>) { return dim; } else { return dim_value(dim); }
}
template <class A, class B, class C, class... Rest>
FTT_HOST_DEVICE constexpr auto min(const A &a, const B &b, const C &c, const Rest &...rest) noexcept { return min(min(a, b), c, rest...); }
template <class A, class B, class C, class... Rest>
FTT_HOST_DEVICE constexpr auto max(const A &a, const B &b, const C &c, const Rest &...rest) noexcept { return max(max(a, b), c, rest...); }
namespace detail {
FTT_HOST_DEVICE constexpr dim_t dimension_lcm(dim_t a, dim_t b) noexcept {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    if (!a || !b) { return 0; }
    dim_t x = a, y = b;
    while (y) { const auto next = x % y; x = y; y = next; }
    return a / x * b;
}
} // namespace detail
template <class A, class B, detail::enable_if_t<is_dimension_v<A> && is_dimension_v<B>, int> = 0>
FTT_HOST_DEVICE constexpr auto lcm(const A &a, const B &b) noexcept {
    if constexpr (is_fixed_dim_v<A> && is_fixed_dim_v<B>) { return fixed_dim_v<detail::dimension_lcm(A::value, B::value)>; }
    else { return detail::dimension_lcm(dim_value(a), dim_value(b)); }
}
template <class D, class L, class U, detail::enable_if_t<is_dimension_v<D> && is_dimension_v<L> && is_dimension_v<U>, int> = 0>
FTT_HOST_DEVICE constexpr auto clamp(const D &dim, const L &lower, const U &upper) noexcept {
    if constexpr (is_fixed_dim_v<L> && is_fixed_dim_v<U>) { static_assert(L::value <= U::value, "Invalid clamp bounds"); }
    return min(max(dim, lower), upper);
}
template <class I, class D, detail::enable_if_t<is_dimension_v<I> && is_dimension_v<D>, int> = 0>
FTT_HOST_DEVICE constexpr auto positive_index(const I &index, const D &dim) noexcept {
    if constexpr (is_fixed_dim_v<I>) {
        if constexpr (I::value < 0) { return index + dim; }
        else { return index; }
    } else {
        return index < 0 ? index + dim : index;
    }
}

} // namespace ftt
