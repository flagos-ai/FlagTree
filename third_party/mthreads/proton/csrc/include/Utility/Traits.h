// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PROTON_UTILITY_TRAITS_H_
#define PROTON_UTILITY_TRAITS_H_

#include <type_traits>
#include <variant>

namespace proton {

namespace details {

template <class T, class Variant> struct variant_index;

template <class T, class... Ts> struct variant_index<T, std::variant<Ts...>> {
  static constexpr std::size_t value = []() constexpr {
    std::size_t i = 0;
    (void)((std::is_same_v<T, Ts> ? true : (++i, false)) || ...);
    return i;
  }();
};

} // namespace details
template <class T, class... Ts>
struct is_one_of : std::disjunction<std::is_same<T, Ts>...> {};

template <class T> struct always_false : std::false_type {};

template <class T, class Variant>
inline constexpr std::size_t variant_index_v =
    details::variant_index<T, Variant>::value;

} // namespace proton

#endif // PROTON_UTILITY_TRAITS_H_
