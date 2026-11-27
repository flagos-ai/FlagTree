/* Copyright 2019-2021 Canaan Inc.
 * Modified for FTT: C++17 host and device storage adapters.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "detail/vector_storage.h"

#define FTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, ...) \
    namespace ftt { template <> struct vector_storage_traits<element_type_, fixed_shape_t<__VA_ARGS__>> { \
        using buffer_type = native_type; \
        using element_type = element_type_;
#define FTT_END_DEFINE_NATIVE_VECTOR() }; }
#define FTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(element_type_, native_type, ...) \
    FTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, __VA_ARGS__) \
    template <class I> FTT_HOST_DEVICE static element_type get_element(const buffer_type &buffer, const I &index) noexcept { \
        static_assert(I::rank() == 1, "Native vector index must be one-dimensional"); \
        return buffer[index.at(0)]; \
    } \
    template <class I> FTT_HOST_DEVICE static void set_element(buffer_type &buffer, const I &index, element_type value) noexcept { \
        static_assert(I::rank() == 1, "Native vector index must be one-dimensional"); \
        buffer[index.at(0)] = value; \
    }
#define FTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT_CAST(element_type_, native_type, cast_type, ...) \
    FTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, __VA_ARGS__) \
    template <class I> FTT_HOST_DEVICE static element_type get_element(const buffer_type &buffer, const I &index) noexcept { \
        static_assert(I::rank() == 1, "Native vector index must be one-dimensional"); \
        return cast_type(buffer)[index.at(0)]; \
    } \
    template <class I> FTT_HOST_DEVICE static void set_element(buffer_type &buffer, const I &index, element_type value) noexcept { \
        static_assert(I::rank() == 1, "Native vector index must be one-dimensional"); \
        auto converted = cast_type(buffer); \
        converted[index.at(0)] = value; \
        buffer = (native_type)converted; \
    }
#define FTT_DEFINE_NATIVE_VECTOR_DEFAULT_BITCAST(element_type_, native_type, bitcast_type, lanes) \
    FTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, lanes) \
    template <class I> FTT_HOST_DEVICE static element_type get_element(const buffer_type &buffer, const I &index) noexcept { \
        static_assert(I::rank() == 1 && sizeof(element_type) == sizeof(bitcast_type), "Invalid native vector bitcast"); \
        const bitcast_type bits = buffer[index.at(0)]; \
        element_type value; __builtin_memcpy(&value, &bits, sizeof(value)); return value; \
    } \
    template <class I> FTT_HOST_DEVICE static void set_element(buffer_type &buffer, const I &index, element_type value) noexcept { \
        static_assert(I::rank() == 1 && sizeof(element_type) == sizeof(bitcast_type), "Invalid native vector bitcast"); \
        bitcast_type bits; __builtin_memcpy(&bits, &value, sizeof(bits)); buffer[index.at(0)] = bits; \
    }
