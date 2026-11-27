/* Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "compiler_defs.h"
#include <stddef.h>

namespace ftt {

template <class T, size_t N> struct array {
    T elements[N ? N : 1];
    FTT_HOST_DEVICE constexpr T &operator[](size_t i) noexcept { return elements[i]; }
    FTT_HOST_DEVICE constexpr const T &operator[](size_t i) const noexcept { return elements[i]; }
    FTT_HOST_DEVICE constexpr T *data() noexcept { return elements; }
    FTT_HOST_DEVICE constexpr const T *data() const noexcept { return elements; }
    FTT_HOST_DEVICE static constexpr size_t size() noexcept { return N; }
    FTT_HOST_DEVICE constexpr T *begin() noexcept { return data(); }
    FTT_HOST_DEVICE constexpr const T *begin() const noexcept { return data(); }
    FTT_HOST_DEVICE constexpr T *end() noexcept { return data() + N; }
    FTT_HOST_DEVICE constexpr const T *end() const noexcept { return data() + N; }
};

inline constexpr size_t dynamic_extent = size_t(-1);
template <class T, size_t Extent = dynamic_extent> class span {
  public:
    FTT_HOST_DEVICE constexpr span(T *data, size_t size) noexcept : data_(data), size_(size) {}
    FTT_HOST_DEVICE constexpr T *data() const noexcept { return data_; }
    FTT_HOST_DEVICE constexpr size_t size() const noexcept { return Extent == dynamic_extent ? size_ : Extent; }
    FTT_HOST_DEVICE constexpr size_t size_bytes() const noexcept { return size() * sizeof(T); }
    FTT_HOST_DEVICE constexpr T &operator[](size_t i) const noexcept { return data_[i]; }
    FTT_HOST_DEVICE constexpr T *begin() const noexcept { return data_; }
    FTT_HOST_DEVICE constexpr T *end() const noexcept { return data_ + size(); }
  private:
    T *data_;
    size_t size_;
};

} // namespace ftt
