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
#include "../dimension.h"
#include "../std_containers.h"

namespace ftt {

enum class dims_usage { normal, shape, strides };
template <class... Dims> struct dims_t;
template <class... Dims> struct shape_t;
template <class... Dims> struct strides_t;

namespace detail {

struct default_aggregate_selector {
    template <class T> FTT_HOST_DEVICE constexpr auto operator()(T value) const noexcept { return value; }
};

template <size_t Axis, class Dims, class Seed, class Func, class Selector>
FTT_HOST_DEVICE constexpr auto aggregate_impl(const Dims &dims, Seed seed, Func &func, Selector &selector) noexcept {
    if constexpr (Axis == Dims::rank()) { return selector(seed); }
    else { return aggregate_impl<Axis + 1>(dims, func(seed, dims.template at<Axis>(), fixed_dim_v<Axis>), func, selector); }
}

template <class D, class V, bool Fixed = is_fixed_dim_v<D> && is_fixed_dim_v<V>>
struct fixed_match : integral_constant<bool, false> {};
template <class D, class V> struct fixed_match<D, V, true> : integral_constant<bool, D::value == V::value> {};

template <class T> using canonical_dim_t = conditional_t<is_fixed_dim_v<T>, remove_cvref_t<T>, dim_t>;
template <size_t I, class D, bool Fixed = is_fixed_dim_v<D>> struct dimension_slot {
    D value{};
    FTT_HOST_DEVICE constexpr dimension_slot() noexcept = default;
    FTT_HOST_DEVICE constexpr dimension_slot(D dim) noexcept : value(dim) {}
    FTT_HOST_DEVICE constexpr D get() const noexcept { return value; }
    FTT_HOST_DEVICE constexpr D &get() noexcept { return value; }
};
template <size_t I, class D> struct dimension_slot<I, D, true> {
    FTT_HOST_DEVICE constexpr dimension_slot() noexcept = default;
    FTT_HOST_DEVICE constexpr dimension_slot(D) noexcept {}
    FTT_HOST_DEVICE constexpr D get() const noexcept { return {}; }
};
template <class Sequence, class... Dims> struct dimensions_storage;
template <size_t... I, class... Dims>
struct dimensions_storage<index_sequence<I...>, Dims...> : dimension_slot<I, Dims>... {
    FTT_HOST_DEVICE constexpr dimensions_storage(Dims... dims) noexcept : dimension_slot<I, Dims>(dims)... {}
};

template <template <class...> class Derived, class... Dims>
FTT_HOST_DEVICE constexpr auto make_dims_impl(const Dims &...dims) noexcept {
    static_assert((is_dimension_v<Dims> && ...), "Dimensions must be integral or fixed");
    return Derived<canonical_dim_t<Dims>...>{static_cast<canonical_dim_t<Dims>>(dims)...};
}

template <dims_usage Usage, template <class...> class Derived, class... Dims>
struct dimensions_base : dimensions_storage<make_index_sequence<sizeof...(Dims)>, Dims...> {
    static_assert((is_dimension_v<Dims> && ...), "Dimensions must be integral or fixed");
    using storage = dimensions_storage<make_index_sequence<sizeof...(Dims)>, Dims...>;
    inline static constexpr fixed_dim<sizeof...(Dims)> rank{};
    FTT_HOST_DEVICE constexpr dimensions_base(Dims... dims) noexcept : storage(dims...) {}
    template <size_t N = sizeof...(Dims), enable_if_t<N != 0, int> = 0>
    FTT_HOST_DEVICE constexpr dimensions_base() noexcept : storage(Dims{}...) {}
    FTT_HOST_DEVICE static constexpr dims_usage usage() noexcept { return Usage; }
    FTT_HOST_DEVICE static constexpr auto fixed_rank() noexcept { return fixed_dim_v<(0 + ... + (is_fixed_dim_v<Dims> ? 1 : 0))>; }
    FTT_HOST_DEVICE static constexpr auto dynamic_rank() noexcept { return fixed_dim_v<sizeof...(Dims)> - fixed_rank(); }
    FTT_HOST_DEVICE static constexpr bool is_fixed() noexcept { return (is_fixed_dim_v<Dims> && ...); }

    template <dim_t Index> FTT_HOST_DEVICE constexpr auto at() const noexcept {
        constexpr dim_t axis = Index < 0 ? Index + dim_t(sizeof...(Dims)) : Index;
        static_assert(axis >= 0 && axis < sizeof...(Dims), "Dimension index out of bounds");
        using dim = typename type_at<axis, Dims...>::type;
        return static_cast<const dimension_slot<axis, dim> &>(*this).get();
    }
    template <dim_t Index> FTT_HOST_DEVICE constexpr decltype(auto) at() noexcept {
        constexpr dim_t axis = Index < 0 ? Index + dim_t(sizeof...(Dims)) : Index;
        static_assert(axis >= 0 && axis < sizeof...(Dims), "Dimension index out of bounds");
        using dim = typename type_at<axis, Dims...>::type;
        return static_cast<dimension_slot<axis, dim> &>(*this).get();
    }
    template <dim_t Index> FTT_HOST_DEVICE constexpr auto at(fixed_dim<Index>) const noexcept { return at<Index>(); }
    template <dim_t Index> FTT_HOST_DEVICE constexpr decltype(auto) at(fixed_dim<Index>) noexcept { return at<Index>(); }
    template <size_t I> FTT_HOST_DEVICE constexpr decltype(auto) get() & noexcept { return at<I>(); }
    template <size_t I> FTT_HOST_DEVICE constexpr auto get() const & noexcept { return at<I>(); }
    template <size_t I> FTT_HOST_DEVICE constexpr auto get() && noexcept { return at<I>(); }
    template <size_t... I> FTT_HOST_DEVICE constexpr dim_t at_impl(dim_t index, index_sequence<I...>) const noexcept {
        const dim_t values[sizeof...(Dims) ? sizeof...(Dims) : 1] = {dim_value(at<I>())...};
        return values[index < 0 ? index + sizeof...(Dims) : index];
    }
    FTT_HOST_DEVICE constexpr dim_t at(dim_t index) const noexcept { return at_impl(index, make_index_sequence<sizeof...(Dims)>{}); }
    template <class I> FTT_HOST_DEVICE constexpr auto operator[](const I &index) const noexcept { return at(index); }
    template <dim_t Index> FTT_HOST_DEVICE constexpr decltype(auto) operator[](fixed_dim<Index>) noexcept { return at<Index>(); }
    FTT_HOST_DEVICE constexpr auto front() const noexcept { return at<0>(); }
    FTT_HOST_DEVICE constexpr decltype(auto) front() noexcept { return at<0>(); }
    FTT_HOST_DEVICE constexpr auto back() const noexcept { return at<sizeof...(Dims) - 1>(); }
    FTT_HOST_DEVICE constexpr decltype(auto) back() noexcept { return at<sizeof...(Dims) - 1>(); }

    template <class Seed, class Func, class Selector = default_aggregate_selector>
    FTT_HOST_DEVICE constexpr auto aggregate(Seed seed, Func func, Selector selector = {}) const noexcept {
        return aggregate_impl<0>(*this, seed, func, selector);
    }
    template <class V, size_t... I>
    FTT_HOST_DEVICE constexpr bool contains_impl(const V &value, index_sequence<I...>) const noexcept {
        return (false || ... || (at<I>() == value));
    }
    template <class V> FTT_HOST_DEVICE constexpr auto contains(const V &value) const noexcept {
        static_assert(is_dimension_v<V>, "Value must be a dimension");
        if constexpr (sizeof...(Dims) == 0) { return integral_constant<bool, false>{}; }
        else if constexpr ((fixed_match<Dims, V>::value || ...)) { return integral_constant<bool, true>{}; }
        else if constexpr (is_fixed() && is_fixed_dim_v<V>) { return integral_constant<bool, false>{}; }
        else { return contains_impl(value, make_index_sequence<sizeof...(Dims)>{}); }
    }
    template <size_t Axis, class V, class Acc>
    FTT_HOST_DEVICE constexpr auto index_of_impl(const V &value, Acc acc) const noexcept {
        if constexpr (Axis == sizeof...(Dims)) { return acc; }
        else { return index_of_impl<Axis + 1>(value, where(acc == fixed_dim_v<-1>, where(at<Axis>() == value, fixed_dim_v<Axis>, fixed_dim_v<-1>), acc)); }
    }
    template <class V> FTT_HOST_DEVICE constexpr auto index_of(const V &value) const noexcept {
        return index_of_impl<0>(value, fixed_dim_v<-1>);
    }

    template <size_t... I, class... U>
    FTT_HOST_DEVICE constexpr auto append_impl(index_sequence<I...>, const U &...values) const noexcept {
        return make_dims_impl<Derived>(at<I>()..., values...);
    }
    template <class... U> FTT_HOST_DEVICE constexpr auto append(const U &...values) const noexcept {
        return append_impl(make_index_sequence<sizeof...(Dims)>{}, values...);
    }
    template <size_t... I, class... U>
    FTT_HOST_DEVICE constexpr auto prepend_impl(index_sequence<I...>, const U &...values) const noexcept {
        return make_dims_impl<Derived>(values..., at<I>()...);
    }
    template <class... U> FTT_HOST_DEVICE constexpr auto prepend(const U &...values) const noexcept {
        return prepend_impl(make_index_sequence<sizeof...(Dims)>{}, values...);
    }
    template <class Other, size_t... I, size_t... J>
    FTT_HOST_DEVICE constexpr auto concat_impl(const Other &other, index_sequence<I...>, index_sequence<J...>) const noexcept {
        return make_dims_impl<Derived>(at<I>()..., other.template at<J>()...);
    }
    template <class... U> FTT_HOST_DEVICE constexpr auto concat(const Derived<U...> &other) const noexcept {
        return concat_impl(other, make_index_sequence<sizeof...(Dims)>{}, make_index_sequence<sizeof...(U)>{});
    }
    template <size_t Start, size_t... I>
    FTT_HOST_DEVICE constexpr auto slice_impl(index_sequence<I...>) const noexcept { return make_dims_impl<Derived>(at<Start + I>()...); }
    template <size_t Start, size_t Count = sizeof...(Dims) - Start>
    FTT_HOST_DEVICE constexpr auto slice() const noexcept {
        static_assert(Start <= sizeof...(Dims) && Count <= sizeof...(Dims) - Start, "Slice out of bounds");
        return slice_impl<Start>(make_index_sequence<Count>{});
    }
    template <dim_t Axis, size_t... I>
    FTT_HOST_DEVICE constexpr auto remove_impl(index_sequence<I...>) const noexcept { return make_dims_impl<Derived>(at<(I < Axis ? I : I + 1)>()...); }
    template <dim_t Index> FTT_HOST_DEVICE constexpr auto remove_at() const noexcept {
        constexpr dim_t axis = Index < 0 ? Index + dim_t(sizeof...(Dims)) : Index;
        static_assert(axis >= 0 && axis < dim_t(sizeof...(Dims)), "Dimension index out of bounds");
        return remove_impl<axis>(make_index_sequence<sizeof...(Dims) - 1>{});
    }
    template <dim_t Axis, size_t I, class V>
    FTT_HOST_DEVICE constexpr auto replace_dim(const V &value) const noexcept {
        if constexpr (Axis == I) { return value; } else { return at<I>(); }
    }
    template <dim_t Axis, class V, size_t... I>
    FTT_HOST_DEVICE constexpr auto replace_impl(const V &value, index_sequence<I...>) const noexcept { return make_dims_impl<Derived>(replace_dim<Axis, I>(value)...); }
    template <dim_t Index, class V> FTT_HOST_DEVICE constexpr auto replace_at(const V &value) const noexcept {
        constexpr dim_t axis = Index < 0 ? Index + dim_t(sizeof...(Dims)) : Index;
        static_assert(axis >= 0 && axis < dim_t(sizeof...(Dims)), "Dimension index out of bounds");
        return replace_impl<axis>(value, make_index_sequence<sizeof...(Dims)>{});
    }
    template <size_t... I> FTT_HOST_DEVICE constexpr auto reverse_impl(index_sequence<I...>) const noexcept { return make_dims_impl<Derived>(at<sizeof...(Dims) - 1 - I>()...); }
    FTT_HOST_DEVICE constexpr auto reverse() const noexcept { return reverse_impl(make_index_sequence<sizeof...(Dims)>{}); }
    template <class Indices, size_t... I>
    FTT_HOST_DEVICE constexpr auto select_impl(const Indices &indices, index_sequence<I...>) const noexcept { return make_dims_impl<Derived>(at(indices.template at<I>())...); }
    template <class Indices> FTT_HOST_DEVICE constexpr auto select(const Indices &indices) const noexcept {
        static_assert(Indices::is_fixed(), "Selection indices must be fixed");
        return select_impl(indices, make_index_sequence<Indices::rank()>{});
    }
    template <size_t... I> FTT_HOST_DEVICE constexpr auto to_array_impl(index_sequence<I...>) const noexcept { return array<dim_t, sizeof...(Dims)>{{dim_value(at<I>())...}}; }
    FTT_HOST_DEVICE constexpr auto to_array() const noexcept { return to_array_impl(make_index_sequence<sizeof...(Dims)>{}); }
    template <class Other, size_t... I>
    FTT_HOST_DEVICE constexpr void copy_to_impl(Other &other, index_sequence<I...>) const noexcept { ((other[fixed_dim_v<I>] = dim_value(at<I>())), ...); }
    template <class Other> FTT_HOST_DEVICE constexpr void copy_to(Other &other) const noexcept { copy_to_impl(other, make_index_sequence<sizeof...(Dims)>{}); }
    template <size_t... I> FTT_HOST_DEVICE constexpr auto length_impl(index_sequence<I...>) const noexcept {
        return (dim_one * ... * at<I>());
    }
    FTT_HOST_DEVICE constexpr auto length() const noexcept { return length_impl(make_index_sequence<sizeof...(Dims)>{}); }
};

template <template <class...> class Derived, class Sequence> struct dynamic_dims;
template <template <class...> class Derived, size_t... I> struct dynamic_dims<Derived, index_sequence<I...>> {
    template <size_t> using element = dim_t;
    using type = Derived<element<I>...>;
};

} // namespace detail

template <class T> struct empty_dims_alike_type;
template <class T> using empty_dims_alike_t = typename empty_dims_alike_type<detail::remove_cvref_t<T>>::type;
template <class T, size_t Rank> struct zeros_dims_alike_type;
template <class T, size_t Rank> FTT_CONSTANT auto zeros_dims_alike_v = zeros_dims_alike_type<detail::remove_cvref_t<T>, Rank>::value;
namespace detail {
template <template <class...> class Derived, class V, size_t... I>
FTT_HOST_DEVICE constexpr auto repeat_dims(const V &value, index_sequence<I...>) noexcept { return make_dims_impl<Derived>(((void)I, value)...); }
template <template <class...> class Derived, dim_t Start, size_t... I>
FTT_HOST_DEVICE constexpr auto index_dims(index_sequence<I...>) noexcept { return make_dims_impl<Derived>(fixed_dim_v<Start + dim_t(I)>...); }
template <template <class...> class Derived, class Generator, size_t... I>
FTT_HOST_DEVICE constexpr auto generate_dims(Generator &generator, index_sequence<I...>) noexcept { return make_dims_impl<Derived>(generator(fixed_dim_v<I>)...); }
} // namespace detail

#define FTT_DIMENSIONS(name, kind) \
    template <class... Dims> struct name##_t : detail::dimensions_base<dims_usage::kind, name##_t, Dims...> { \
        using base = detail::dimensions_base<dims_usage::kind, name##_t, Dims...>; \
        using base::base; \
    }; \
    template <class... Dims> FTT_HOST_DEVICE constexpr auto make_##name(const Dims &...dims) noexcept { return detail::make_dims_impl<name##_t>(dims...); } \
    template <size_t Rank> using dynamic_##name##_t = typename detail::dynamic_dims<name##_t, detail::make_index_sequence<Rank>>::type; \
    template <dim_t... Dims> using fixed_##name##_t = name##_t<fixed_dim<Dims>...>; \
    template <dim_t... Dims> FTT_CONSTANT fixed_##name##_t<Dims...> fixed_##name##_v{}; \
    template <class... Dims> struct empty_dims_alike_type<name##_t<Dims...>> { using type = name##_t<>; }; \
    template <size_t Rank, class V> FTT_HOST_DEVICE constexpr auto make_repeat_##name(const V &value) noexcept { return detail::repeat_dims<name##_t>(value, detail::make_index_sequence<Rank>{}); } \
    template <size_t Rank> FTT_HOST_DEVICE constexpr auto make_zeros_##name() noexcept { return make_repeat_##name<Rank>(dim_zero); } \
    template <size_t Rank> FTT_HOST_DEVICE constexpr auto make_ones_##name() noexcept { return make_repeat_##name<Rank>(dim_one); } \
    template <size_t Rank, dim_t Start = 0> FTT_HOST_DEVICE constexpr auto make_index_##name() noexcept { return detail::index_dims<name##_t, Start>(detail::make_index_sequence<Rank>{}); } \
    template <size_t Rank, class G> FTT_HOST_DEVICE constexpr auto generate_##name(G &&generator) noexcept { return detail::generate_dims<name##_t>(generator, detail::make_index_sequence<Rank>{}); } \
    template <class... Dims, size_t Rank> struct zeros_dims_alike_type<name##_t<Dims...>, Rank> { inline static constexpr auto value = make_zeros_##name<Rank>(); };
FTT_DIMENSIONS(dims, normal)
FTT_DIMENSIONS(shape, shape)
FTT_DIMENSIONS(strides, strides)
#undef FTT_DIMENSIONS

namespace detail {
template <class Shape, size_t Axis>
FTT_HOST_DEVICE constexpr auto row_stride(const Shape &shape) noexcept {
    if constexpr (Axis + 1 == Shape::rank()) { return dim_one; }
    else { return shape.template at<Axis + 1>() * row_stride<Shape, Axis + 1>(shape); }
}
template <bool Canonical, class Shape, size_t I>
FTT_HOST_DEVICE constexpr auto default_stride(const Shape &shape) noexcept {
    if constexpr (Canonical) { return where(shape.template at<I>() == dim_one, dim_zero, row_stride<Shape, I>(shape)); }
    else { return row_stride<Shape, I>(shape); }
}
template <bool Canonical, class Shape, size_t... I>
FTT_HOST_DEVICE constexpr auto default_strides_select(const Shape &shape, index_sequence<I...>) noexcept {
    return make_strides(default_stride<Canonical, Shape, I>(shape)...);
}
} // namespace detail

template <class Shape, bool Canonical = true>
FTT_HOST_DEVICE constexpr auto default_strides(const Shape &shape) noexcept {
    return detail::default_strides_select<Canonical>(shape, detail::make_index_sequence<Shape::rank()>{});
}

} // namespace ftt

#if __has_include(<tuple>)
#include <tuple>
namespace std {
#define FTT_DIMENSIONS_TUPLE(name) \
    template <class... Dims> struct tuple_size<ftt::name##_t<Dims...>> : integral_constant<size_t, sizeof...(Dims)> {}; \
    template <size_t I, class... Dims> struct tuple_element<I, ftt::name##_t<Dims...>> { using type = typename ftt::detail::type_at<I, Dims...>::type; };
FTT_DIMENSIONS_TUPLE(dims)
FTT_DIMENSIONS_TUPLE(shape)
FTT_DIMENSIONS_TUPLE(strides)
#undef FTT_DIMENSIONS_TUPLE
} // namespace std
#endif
