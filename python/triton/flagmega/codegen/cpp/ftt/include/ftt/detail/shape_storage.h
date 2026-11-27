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
#include "../shape.h"

namespace ftt::detail {
template <class Shape, bool Fixed = Shape::is_fixed()> class shape_storage {
  public:
    FTT_HOST_DEVICE constexpr shape_storage(Shape shape) noexcept : shape_(shape) {}
    FTT_HOST_DEVICE static constexpr auto rank() noexcept { return Shape::rank; }
    FTT_HOST_DEVICE constexpr const Shape &shape() const noexcept { return shape_; }
    FTT_HOST_DEVICE constexpr size_t size() const noexcept { return shape_.length(); }
  private:
    Shape shape_;
};
template <class Shape> class shape_storage<Shape, true> {
  public:
    FTT_HOST_DEVICE constexpr shape_storage() noexcept = default;
    FTT_HOST_DEVICE constexpr shape_storage(Shape) noexcept {}
    FTT_HOST_DEVICE static constexpr auto rank() noexcept { return Shape::rank; }
    FTT_HOST_DEVICE static constexpr Shape shape() noexcept { return {}; }
    FTT_HOST_DEVICE static constexpr size_t size() noexcept { return Shape{}.length(); }
};
template <class Strides, bool Fixed = Strides::is_fixed()> class strides_storage {
  public:
    FTT_HOST_DEVICE constexpr strides_storage(Strides strides) noexcept : strides_(strides) {}
    FTT_HOST_DEVICE constexpr const Strides &strides() const noexcept { return strides_; }
  private:
    Strides strides_;
};
template <class Strides> class strides_storage<Strides, true> {
  public:
    FTT_HOST_DEVICE constexpr strides_storage() noexcept = default;
    FTT_HOST_DEVICE constexpr strides_storage(Strides) noexcept {}
    FTT_HOST_DEVICE static constexpr Strides strides() noexcept { return {}; }
};
template <class Shape, class Strides>
struct tensor_size_impl : shape_storage<Shape>, strides_storage<Strides> {
    FTT_HOST_DEVICE constexpr tensor_size_impl() noexcept = default;
    FTT_HOST_DEVICE constexpr tensor_size_impl(Shape shape, Strides strides) noexcept : shape_storage<Shape>(shape), strides_storage<Strides>(strides) {}
};
} // namespace ftt::detail
