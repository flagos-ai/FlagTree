/* Copyright 2019-2021 Canaan Inc.
 * Modified for FTT: C++17 host and device vector operations.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "vector.h"
#include "loop.h"

namespace ftt {
namespace detail {
template <class T, size_t Rank, size_t Axis>
FTT_HOST_DEVICE constexpr dim_t broadcast_extent() noexcept {
    if constexpr (Axis + vector_rank_v<T> < Rank) { return 1; }
    else { return T::shape().template at<Axis + vector_rank_v<T> - Rank>(); }
}
template <class A, class B, size_t Rank, size_t Axis>
FTT_HOST_DEVICE constexpr dim_t broadcast_lane() noexcept {
    constexpr auto a = broadcast_extent<A, Rank, Axis>();
    constexpr auto b = broadcast_extent<B, Rank, Axis>();
    static_assert(a == b || a == 1 || b == 1, "Vector broadcast lanes must match");
    return a > b ? a : b;
}
template <class E, class A, class B, class Seq> struct broadcast_vector;
template <class E, class A, class B, size_t... I>
struct broadcast_vector<E, A, B, index_sequence<I...>> {
    static constexpr size_t rank = sizeof...(I);
    using type = vector<E, broadcast_lane<A, B, rank, I>()...>;
};
template <class E, class A, class B>
using broadcast_vector_t = typename broadcast_vector<E, A, B, make_index_sequence<(vector_rank_v<A> > vector_rank_v<B> ? vector_rank_v<A> : vector_rank_v<B>)>>::type;
template <class A, class B> using binary_element_t = conditional_t<is_vector_v<A>, element_or_scalar_t<A>, element_or_scalar_t<B>>;
template <class T, class Index, size_t... I>
FTT_HOST_DEVICE constexpr auto broadcast_index(const Index &index, index_sequence<I...>) noexcept {
    return make_shape((T::shape().template at<I>() == 1 ? dim_t(0) : dim_value(index.template at<Index::rank() - T::rank() + I>()))...);
}
template <class T, class Index>
FTT_HOST_DEVICE constexpr auto broadcast_element(const T &value, const Index &index) noexcept {
    if constexpr (is_vector_v<T>) { return value(broadcast_index<T>(index, make_index_sequence<T::rank()>{})); }
    else { return value; }
}
template <template <class> class Op, class V>
FTT_HOST_DEVICE constexpr V map_unary(const V &input) noexcept {
    V result{};
    for (size_t i = 0; i < V::size(); ++i) { result.flat_at(i) = Op<typename V::element_type>{}(input.flat_at(i)); }
    return result;
}
template <template <class, class> class Op, class E, class A, class B>
FTT_HOST_DEVICE constexpr auto map_binary(const A &a, const B &b) noexcept {
    using result_type = broadcast_vector_t<E, A, B>;
    result_type result{};
    for (size_t i = 0; i < result_type::size(); ++i) {
        const auto index = unravel_index(dim_t(i), result_type::shape());
        result(index) = Op<element_or_scalar_t<A>, element_or_scalar_t<B>>{}(broadcast_element(a, index), broadcast_element(b, index));
    }
    return result;
}
template <template <class, class, class> class Op, class E, class A, class B, class C>
FTT_HOST_DEVICE constexpr auto map_ternary(const A &a, const B &b, const C &c) noexcept {
    using intermediate_type = conditional_t<is_vector_v<A> || is_vector_v<B>, broadcast_vector_t<E, A, B>, E>;
    using result_type = broadcast_vector_t<E, intermediate_type, C>;
    result_type result{};
    for (size_t i = 0; i < result_type::size(); ++i) {
        const auto index = unravel_index(dim_t(i), result_type::shape());
        result(index) = Op<element_or_scalar_t<A>, element_or_scalar_t<B>, element_or_scalar_t<C>>{}(broadcast_element(a, index), broadcast_element(b, index), broadcast_element(c, index));
    }
    return result;
}
} // namespace detail

namespace ops {
#define FTT_VECTOR_UNARY(name, expression) \
    template <class T> struct name { \
        FTT_HOST_DEVICE constexpr auto operator()(const T &value) const noexcept { \
            if constexpr (is_vector_v<T>) { return detail::map_unary<name>(value); } \
            else { return static_cast<T>(expression); } \
        } \
    };
FTT_VECTOR_UNARY(abs, value < T(0) ? -value : value)
FTT_VECTOR_UNARY(copy, value)
FTT_VECTOR_UNARY(neg, -value)
FTT_VECTOR_UNARY(sign, (T(0) < value) - (value < T(0)))
FTT_VECTOR_UNARY(square, value * value)
FTT_VECTOR_UNARY(round, __builtin_nearbyint(double(value)))
#undef FTT_VECTOR_UNARY
#define FTT_VECTOR_MATH(name) \
    template <class T> struct name { \
        FTT_HOST_DEVICE constexpr auto operator()(const T &value) const noexcept { \
            if constexpr (is_vector_v<T>) { return detail::map_unary<name>(value); } \
            else if constexpr (!detail::is_integral<T>::value && sizeof(T) <= sizeof(float)) { return static_cast<T>(__builtin_##name##f(float(value))); } \
            else { return static_cast<T>(__builtin_##name(double(value))); } \
        } \
    };
FTT_VECTOR_MATH(acos)
FTT_VECTOR_MATH(acosh)
FTT_VECTOR_MATH(asin)
FTT_VECTOR_MATH(asinh)
FTT_VECTOR_MATH(ceil)
FTT_VECTOR_MATH(cos)
FTT_VECTOR_MATH(cosh)
FTT_VECTOR_MATH(erf)
FTT_VECTOR_MATH(exp)
FTT_VECTOR_MATH(floor)
FTT_VECTOR_MATH(log)
FTT_VECTOR_MATH(sin)
FTT_VECTOR_MATH(sinh)
FTT_VECTOR_MATH(sqrt)
FTT_VECTOR_MATH(tanh)
#undef FTT_VECTOR_MATH
template <class T> struct rsqrt {
    FTT_HOST_DEVICE constexpr auto operator()(const T &value) const noexcept {
        if constexpr (is_vector_v<T>) { return detail::map_unary<rsqrt>(value); }
        else { return static_cast<T>(T(1) / sqrt<T>{}(value)); }
    }
};
template <class T> struct swish {
    FTT_HOST_DEVICE constexpr auto operator()(const T &value) const noexcept {
        if constexpr (is_vector_v<T>) { return detail::map_unary<swish>(value); }
        else { return static_cast<T>(value / (T(1) + exp<T>{}(static_cast<T>(-value)))); }
    }
};

#define FTT_VECTOR_BINARY(name, expression) \
    template <class A, class B> struct name { \
        FTT_HOST_DEVICE constexpr auto operator()(const A &a, const B &b) const noexcept { \
            if constexpr (is_vector_v<A> || is_vector_v<B>) { return detail::map_binary<name, detail::binary_element_t<A, B>>(a, b); } \
            else { return expression; } \
        } \
    };
FTT_VECTOR_BINARY(add, a + b)
FTT_VECTOR_BINARY(sub, a - b)
FTT_VECTOR_BINARY(mul, a * b)
FTT_VECTOR_BINARY(div, a / b)
FTT_VECTOR_BINARY(ceil_div, a / b + (a % b != 0))
FTT_VECTOR_BINARY(min, a < b ? a : b)
FTT_VECTOR_BINARY(max, a > b ? a : b)
FTT_VECTOR_BINARY(pow, static_cast<A>(__builtin_pow(double(a), double(b))))
FTT_VECTOR_BINARY(swishb, static_cast<A>(double(a) / (1.0 + __builtin_exp(-double(a) * double(b)))))
FTT_VECTOR_BINARY(logical_and, bool(a) && bool(b))
FTT_VECTOR_BINARY(logical_or, bool(a) || bool(b))
FTT_VECTOR_BINARY(logical_xor, bool(a) != bool(b))
#undef FTT_VECTOR_BINARY

template <class A, class B> struct mod {
    FTT_HOST_DEVICE constexpr auto operator()(const A &a, const B &b) const noexcept {
        if constexpr (is_vector_v<A> || is_vector_v<B>) { return detail::map_binary<mod, detail::binary_element_t<A, B>>(a, b); }
        else if constexpr (detail::is_integral<A>::value && detail::is_integral<B>::value) { return a % b; }
        else { return static_cast<A>(__builtin_fmod(double(a), double(b))); }
    }
};
template <class A, class B> struct floor_mod {
    FTT_HOST_DEVICE constexpr auto operator()(const A &a, const B &b) const noexcept {
        if constexpr (is_vector_v<A> || is_vector_v<B>) { return detail::map_binary<floor_mod, detail::binary_element_t<A, B>>(a, b); }
        else if constexpr (detail::is_integral<A>::value && detail::is_integral<B>::value) {
            const auto remainder = a % b;
            return remainder != 0 && ((remainder < 0) != (b < 0)) ? remainder + b : remainder;
        } else { return static_cast<A>(double(a) - __builtin_floor(double(a) / double(b)) * double(b)); }
    }
};
#define FTT_VECTOR_COMPARE(name, expression) \
    template <class A, class B> struct name { \
        FTT_HOST_DEVICE constexpr auto operator()(const A &a, const B &b) const noexcept { \
            if constexpr (is_vector_v<A> || is_vector_v<B>) { return detail::map_binary<name, bool>(a, b); } \
            else { return bool(expression); } \
        } \
    };
FTT_VECTOR_COMPARE(equal, a == b)
FTT_VECTOR_COMPARE(not_equal, a != b)
FTT_VECTOR_COMPARE(less, a < b)
FTT_VECTOR_COMPARE(less_or_equal, a <= b)
FTT_VECTOR_COMPARE(greater, a > b)
FTT_VECTOR_COMPARE(greater_or_equal, a >= b)
#undef FTT_VECTOR_COMPARE

template <class A, class B, class C> struct mul_add {
    FTT_HOST_DEVICE constexpr auto operator()(const A &a, const B &b, const C &c) const noexcept {
        if constexpr (is_vector_v<A> || is_vector_v<B> || is_vector_v<C>) { return detail::map_ternary<mul_add, element_or_scalar_t<C>>(a, b, c); }
        else { return static_cast<C>(a * b) + c; }
    }
};
template <class A, class B, class C> struct mul_sub {
    FTT_HOST_DEVICE constexpr auto operator()(const A &a, const B &b, const C &c) const noexcept {
        if constexpr (is_vector_v<A> || is_vector_v<B> || is_vector_v<C>) { return detail::map_ternary<mul_sub, element_or_scalar_t<C>>(a, b, c); }
        else { return static_cast<C>(a * b) - c; }
    }
};
template <class C, class A, class B> struct where {
    FTT_HOST_DEVICE constexpr auto operator()(const C &condition, const A &a, const B &b) const noexcept {
        if constexpr (is_vector_v<C> || is_vector_v<A> || is_vector_v<B>) { return detail::map_ternary<where, detail::binary_element_t<A, B>>(condition, a, b); }
        else { return condition ? a : b; }
    }
};
template <class V, class T> struct clamp {
    FTT_HOST_DEVICE constexpr auto operator()(const V &value, const T &lower, const T &upper) const noexcept {
        return min<V, T>{}(max<V, T>{}(value, lower), upper);
    }
};
template <template <class, class> class Op, class R, class V> struct reduce {
    FTT_HOST_DEVICE constexpr R operator()(const V &value, R seed) const noexcept {
        if constexpr (is_vector_v<V>) {
            for (size_t i = 0; i < V::size(); ++i) { seed = Op<R, typename V::element_type>{}(seed, value.flat_at(i)); }
            return seed;
        } else { return Op<R, V>{}(seed, value); }
    }
    FTT_HOST_DEVICE constexpr R operator()(const V &value) const noexcept {
        if constexpr (is_vector_v<V>) {
            R seed = static_cast<R>(value.flat_at(0));
            for (size_t i = 1; i < V::size(); ++i) { seed = Op<R, typename V::element_type>{}(seed, value.flat_at(i)); }
            return seed;
        } else { return static_cast<R>(value); }
    }
};
template <class From, class To> struct cast_elem {
    FTT_HOST_DEVICE constexpr auto operator()(const From &value) const noexcept {
        if constexpr (!is_vector_v<From>) { return static_cast<To>(value); }
        else {
            using E = typename From::element_type;
            if constexpr (detail::is_same<E, bool>::value || detail::is_same<To, bool>::value || sizeof(E) == sizeof(To)) {
                replace_element_t<From, To> result{};
                for (size_t i = 0; i < From::size(); ++i) { result.flat_at(i) = static_cast<To>(value.flat_at(i)); }
                return result;
            } else if constexpr ((sizeof(E) < sizeof(To) && From::rank() > 1) || (sizeof(E) > sizeof(To) && From::rank() > 2)) {
                using Inner = detail::remove_cvref_t<decltype(value(0_dim))>;
                using ResultInner = decltype(cast_elem<Inner, To>{}(value(0_dim)));
                prepend_lanes_t<ResultInner, From::shape().front()> result{};
                for (dim_t i = 0; i < From::shape().front(); ++i) { result(i) = cast_elem<Inner, To>{}(value(i)); }
                return result;
            } else if constexpr (sizeof(E) > sizeof(To)) {
                static_assert(From::rank() == 2 && From::shape().front() == sizeof(E) / sizeof(To), "Narrow cast requires a leading size-ratio axis");
                vector<To, From::size()> result{};
                for (size_t i = 0; i < From::size(); ++i) { result.flat_at(i) = static_cast<To>(value.flat_at(i)); }
                return result;
            } else {
                static_assert(From::size() % (sizeof(To) / sizeof(E)) == 0, "Wide cast lanes must divide the size ratio");
                vector<To, sizeof(To) / sizeof(E), From::size() / (sizeof(To) / sizeof(E))> result{};
                for (size_t i = 0; i < From::size(); ++i) { result.flat_at(i) = static_cast<To>(value.flat_at(i)); }
                return result;
            }
        }
    }
};
} // namespace ops

#define FTT_VECTOR_UNARY_FUNCTION(name) \
    template <class T, detail::enable_if_t<is_vector_v<T>, int> = 0> \
    FTT_HOST_DEVICE constexpr auto name(const T &value) noexcept { return ops::name<T>{}(value); }
FTT_VECTOR_UNARY_FUNCTION(abs)
FTT_VECTOR_UNARY_FUNCTION(copy)
FTT_VECTOR_UNARY_FUNCTION(neg)
FTT_VECTOR_UNARY_FUNCTION(sign)
FTT_VECTOR_UNARY_FUNCTION(square)
FTT_VECTOR_UNARY_FUNCTION(acos)
FTT_VECTOR_UNARY_FUNCTION(acosh)
FTT_VECTOR_UNARY_FUNCTION(asin)
FTT_VECTOR_UNARY_FUNCTION(asinh)
FTT_VECTOR_UNARY_FUNCTION(ceil)
FTT_VECTOR_UNARY_FUNCTION(cos)
FTT_VECTOR_UNARY_FUNCTION(cosh)
FTT_VECTOR_UNARY_FUNCTION(erf)
FTT_VECTOR_UNARY_FUNCTION(exp)
FTT_VECTOR_UNARY_FUNCTION(floor)
FTT_VECTOR_UNARY_FUNCTION(log)
FTT_VECTOR_UNARY_FUNCTION(round)
FTT_VECTOR_UNARY_FUNCTION(rsqrt)
FTT_VECTOR_UNARY_FUNCTION(sin)
FTT_VECTOR_UNARY_FUNCTION(sinh)
FTT_VECTOR_UNARY_FUNCTION(sqrt)
FTT_VECTOR_UNARY_FUNCTION(tanh)
FTT_VECTOR_UNARY_FUNCTION(swish)
#undef FTT_VECTOR_UNARY_FUNCTION
#define FTT_SCALAR_UNARY_FUNCTION(name) \
    template <class T, detail::enable_if_t<!is_vector_v<T> && !is_fixed_dim_v<T> && !is_dimensions_v<T>, int> = 0> \
    FTT_HOST_DEVICE constexpr auto name(const T &value) noexcept { return ops::name<T>{}(value); }
FTT_SCALAR_UNARY_FUNCTION(copy)
FTT_SCALAR_UNARY_FUNCTION(neg)
FTT_SCALAR_UNARY_FUNCTION(sign)
FTT_SCALAR_UNARY_FUNCTION(square)
FTT_SCALAR_UNARY_FUNCTION(acos)
FTT_SCALAR_UNARY_FUNCTION(acosh)
FTT_SCALAR_UNARY_FUNCTION(asin)
FTT_SCALAR_UNARY_FUNCTION(asinh)
FTT_SCALAR_UNARY_FUNCTION(ceil)
FTT_SCALAR_UNARY_FUNCTION(cos)
FTT_SCALAR_UNARY_FUNCTION(cosh)
FTT_SCALAR_UNARY_FUNCTION(erf)
FTT_SCALAR_UNARY_FUNCTION(floor)
FTT_SCALAR_UNARY_FUNCTION(round)
FTT_SCALAR_UNARY_FUNCTION(sin)
FTT_SCALAR_UNARY_FUNCTION(sinh)
FTT_SCALAR_UNARY_FUNCTION(tanh)
FTT_SCALAR_UNARY_FUNCTION(swish)
#if !defined(FTT_ARCH_NVIDIA) && !defined(FTT_ARCH_ILUVATAR)
FTT_SCALAR_UNARY_FUNCTION(abs)
FTT_SCALAR_UNARY_FUNCTION(exp)
FTT_SCALAR_UNARY_FUNCTION(log)
FTT_SCALAR_UNARY_FUNCTION(rsqrt)
FTT_SCALAR_UNARY_FUNCTION(sqrt)
#endif
#undef FTT_SCALAR_UNARY_FUNCTION
#define FTT_VECTOR_BINARY_FUNCTION(name) \
    template <class A, class B, detail::enable_if_t<is_vector_v<A> || is_vector_v<B>, int> = 0> \
    FTT_HOST_DEVICE constexpr auto name(const A &a, const B &b) noexcept { return ops::name<A, B>{}(a, b); }
FTT_VECTOR_BINARY_FUNCTION(add)
FTT_VECTOR_BINARY_FUNCTION(sub)
FTT_VECTOR_BINARY_FUNCTION(mul)
FTT_VECTOR_BINARY_FUNCTION(div)
FTT_VECTOR_BINARY_FUNCTION(mod)
FTT_VECTOR_BINARY_FUNCTION(floor_mod)
FTT_VECTOR_BINARY_FUNCTION(ceil_div)
FTT_VECTOR_BINARY_FUNCTION(min)
FTT_VECTOR_BINARY_FUNCTION(max)
FTT_VECTOR_BINARY_FUNCTION(pow)
FTT_VECTOR_BINARY_FUNCTION(swishb)
FTT_VECTOR_BINARY_FUNCTION(logical_and)
FTT_VECTOR_BINARY_FUNCTION(logical_or)
FTT_VECTOR_BINARY_FUNCTION(logical_xor)
FTT_VECTOR_BINARY_FUNCTION(equal)
FTT_VECTOR_BINARY_FUNCTION(not_equal)
FTT_VECTOR_BINARY_FUNCTION(less)
FTT_VECTOR_BINARY_FUNCTION(less_or_equal)
FTT_VECTOR_BINARY_FUNCTION(greater)
FTT_VECTOR_BINARY_FUNCTION(greater_or_equal)
#undef FTT_VECTOR_BINARY_FUNCTION
#define FTT_SCALAR_BINARY_FUNCTION(name) \
    template <class A, class B, detail::enable_if_t<!is_vector_v<A> && !is_vector_v<B> && !is_dimensions_v<A> && !is_dimensions_v<B>, int> = 0> \
    FTT_HOST_DEVICE constexpr auto name(const A &a, const B &b) noexcept { return ops::name<A, B>{}(a, b); }
FTT_SCALAR_BINARY_FUNCTION(add)
FTT_SCALAR_BINARY_FUNCTION(sub)
FTT_SCALAR_BINARY_FUNCTION(mul)
FTT_SCALAR_BINARY_FUNCTION(div)
FTT_SCALAR_BINARY_FUNCTION(mod)
FTT_SCALAR_BINARY_FUNCTION(floor_mod)
FTT_SCALAR_BINARY_FUNCTION(pow)
FTT_SCALAR_BINARY_FUNCTION(swishb)
FTT_SCALAR_BINARY_FUNCTION(logical_and)
FTT_SCALAR_BINARY_FUNCTION(logical_or)
FTT_SCALAR_BINARY_FUNCTION(logical_xor)
FTT_SCALAR_BINARY_FUNCTION(equal)
FTT_SCALAR_BINARY_FUNCTION(not_equal)
FTT_SCALAR_BINARY_FUNCTION(less)
FTT_SCALAR_BINARY_FUNCTION(less_or_equal)
FTT_SCALAR_BINARY_FUNCTION(greater)
FTT_SCALAR_BINARY_FUNCTION(greater_or_equal)
#undef FTT_SCALAR_BINARY_FUNCTION
#if !defined(FTT_ARCH_NVIDIA) && !defined(FTT_ARCH_ILUVATAR)
template <class A, class B, detail::enable_if_t<!is_vector_v<A> && !is_vector_v<B> && !(is_dimension_v<A> && is_dimension_v<B>), int> = 0>
FTT_HOST_DEVICE constexpr auto min(const A &a, const B &b) noexcept { return ops::min<A, B>{}(a, b); }
template <class A, class B, detail::enable_if_t<!is_vector_v<A> && !is_vector_v<B> && !(is_dimension_v<A> && is_dimension_v<B>), int> = 0>
FTT_HOST_DEVICE constexpr auto max(const A &a, const B &b) noexcept { return ops::max<A, B>{}(a, b); }
#endif
template <class C, class A, class B, detail::enable_if_t<!detail::is_static_bool<C>::value && !(detail::is_boolean_condition<C>::value && is_dimension_v<A> && is_dimension_v<B>), int> = 0>
FTT_HOST_DEVICE constexpr auto where(const C &condition, const A &a, const B &b) noexcept { return ops::where<C, A, B>{}(condition, a, b); }
template <class V, class T, detail::enable_if_t<!(is_dimension_v<V> && is_dimension_v<T>), int> = 0>
FTT_HOST_DEVICE constexpr auto clamp(const V &value, const T &lower, const T &upper) noexcept { return ops::clamp<V, T>{}(value, lower, upper); }
template <class A, class B, class C> FTT_HOST_DEVICE constexpr auto mul_add(const A &a, const B &b, const C &c) noexcept { return ops::mul_add<A, B, C>{}(a, b, c); }
template <class A, class B, class C> FTT_HOST_DEVICE constexpr auto mul_sub(const A &a, const B &b, const C &c) noexcept { return ops::mul_sub<A, B, C>{}(a, b, c); }
template <class To, class From> FTT_HOST_DEVICE constexpr auto cast_elem(const From &value) noexcept { return ops::cast_elem<From, To>{}(value); }
template <class To, class From> FTT_HOST_DEVICE constexpr auto cast(const From &value) noexcept { return cast_elem<element_or_scalar_t<To>>(value); }
template <template <class, class> class Op, class R, class V> FTT_HOST_DEVICE constexpr R reduce(const V &value, R seed) noexcept { return ops::reduce<Op, R, V>{}(value, seed); }
template <template <class, class> class Op, class R, class V> FTT_HOST_DEVICE constexpr R reduce(const V &value) noexcept { return ops::reduce<Op, R, V>{}(value); }
#define FTT_VECTOR_REDUCE_FUNCTION(name, op) \
    template <class R, class V> FTT_HOST_DEVICE constexpr R name(const V &value, R seed) noexcept { return reduce<ops::op, R>(value, seed); } \
    template <class R, class V> FTT_HOST_DEVICE constexpr R name(const V &value) noexcept { return reduce<ops::op, R>(value); }
FTT_VECTOR_REDUCE_FUNCTION(reduce_sum, add)
FTT_VECTOR_REDUCE_FUNCTION(reduce_min, min)
FTT_VECTOR_REDUCE_FUNCTION(reduce_max, max)
#undef FTT_VECTOR_REDUCE_FUNCTION
template <class V, detail::enable_if_t<is_vector_v<V>, int> = 0> FTT_HOST_DEVICE constexpr auto operator-(const V &value) noexcept { return neg(value); }
#define FTT_VECTOR_OPERATOR(op, name) \
    template <class A, class B, detail::enable_if_t<is_vector_v<A> || is_vector_v<B>, int> = 0> \
    FTT_HOST_DEVICE constexpr auto operator op(const A &a, const B &b) noexcept { return name(a, b); } \
    template <class A, class B, detail::enable_if_t<is_vector_v<A>, int> = 0> \
    FTT_HOST_DEVICE constexpr A &operator op##=(A &a, const B &b) noexcept { a = name(a, b); return a; }
FTT_VECTOR_OPERATOR(+, add)
FTT_VECTOR_OPERATOR(-, sub)
FTT_VECTOR_OPERATOR(*, mul)
FTT_VECTOR_OPERATOR(/, div)
FTT_VECTOR_OPERATOR(%, mod)
#undef FTT_VECTOR_OPERATOR

template <class A, class B> FTT_HOST_DEVICE constexpr auto inner_product(const A &a, const B &b) noexcept {
    static_assert(A::shape() == B::shape(), "Inner product shapes must match");
    using E = typename A::element_type;
    using Acc = detail::conditional_t<!detail::is_integral<E>::value && sizeof(E) < sizeof(float), float, E>;
    Acc result{};
    for (size_t i = 0; i < A::size(); ++i) { result += static_cast<Acc>(a.flat_at(i)) * static_cast<Acc>(b.flat_at(i)); }
    return static_cast<E>(result);
}
template <class A, class B> FTT_HOST_DEVICE constexpr auto outer_product(const A &a, const B &b) noexcept {
    static_assert(A::rank() == 1 && B::rank() == 1, "Outer product requires one-dimensional vectors");
    vector<typename A::element_type, A::size(), B::size()> result{};
    for (size_t i = 0; i < A::size(); ++i) {
        for (size_t j = 0; j < B::size(); ++j) { result(i, j) = a(i) * b(j); }
    }
    return result;
}
template <bool AccC, bool TransA = false, class A, class B, class C>
FTT_HOST_DEVICE constexpr C vmma(const A &a, const B &b, const C &c) noexcept {
    static_assert(A::rank() == 2 && B::rank() == 2 && C::rank() == 2, "MMA requires two-dimensional vectors");
    static_assert(A::shape().at(TransA ? 0 : 1) == B::shape().at(0), "MMA inner dimensions must match");
    static_assert(A::shape().at(TransA ? 1 : 0) == C::shape().at(0) && B::shape().at(1) == C::shape().at(1), "MMA output dimensions must match");
    C result = AccC ? c : C{};
    for (dim_t m = 0; m < C::shape().at(0); ++m) {
        for (dim_t k = 0; k < B::shape().at(0); ++k) {
            for (dim_t n = 0; n < C::shape().at(1); ++n) {
                const auto product = static_cast<typename C::element_type>((TransA ? a(k, m) : a(m, k)) * b(k, n));
                result(m, n) = static_cast<typename C::element_type>(result(m, n)) + product;
            }
        }
    }
    return result;
}
namespace vector_ops {
template <class V> struct vload_scalar { template <class U> FTT_HOST_DEVICE constexpr V operator()(const U &value) const noexcept { return V::from_scalar(value); } };
template <class V, class U> struct vunaligned_load { FTT_HOST_DEVICE constexpr V operator()(const U *source) const noexcept { return V::unaligned_load_from(source); } };
template <bool AccC, bool TransA, class A, class B, class C> struct vmma { FTT_HOST_DEVICE constexpr C operator()(const A &a, const B &b, const C &c) const noexcept { return ftt::vmma<AccC, TransA>(a, b, c); } };
} // namespace vector_ops
} // namespace ftt
