/* Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "arch_types.h"
#include <cuda/atomic>
#include <cstdint>

namespace ftt {
namespace arch {

// Only a valid, positive complete packed-row prefix may reach this check.
// Generic pointers can name local/shared memory. Alignment alone is insufficient.
FTT_DEVICE_INLINE bool try_aligned_global_u32_address(
    const unsigned char *row, unsigned long long &global_address) {
#if defined(FTT_DEVICE_PASS)
    static_assert(sizeof(std::uintptr_t) == sizeof(unsigned long long));
    if ((reinterpret_cast<std::uintptr_t>(row) & 3u) != 0u)
        return false;
    unsigned int is_global;
    asm volatile("{ .reg .pred p; isspacep.global p, %1; selp.u32 %0, 1, 0, p; }"
        : "=r"(is_global) : "l"(row) : "memory");
    if (!is_global)
        return false;
    // Convert only after the generic address is proven to belong to global space.
    asm volatile("cvta.to.global.u64 %0, %1;"
        : "=l"(global_address) : "l"(row) : "memory");
    return true;
#else
    (void)row;
    (void)global_address;
    return false;
#endif
}

// Precondition: naturally aligned converted global address, valid four-byte span.
// The memory clobber and volatile assembly expose the read to the compiler without
// a retyped uint32 dereference or speculation beyond the caller's validity guard.
FTT_DEVICE_INLINE unsigned int load_global_u32(unsigned long long address) {
#if defined(FTT_DEVICE_PASS)
    unsigned int result;
    asm volatile("ld.global.u32 %0, [%1];"
        : "=r"(result) : "l"(address) : "memory");
    return result;
#else
    // The checked reader never selects this target-only operation on the host.
    (void)address;
    return 0u;
#endif
}

// Precondition: the caller has a valid positive span containing an eight-byte
// packet. Generic shared/local pointers and weaker ABI alignment retain fallback.
FTT_DEVICE_INLINE bool try_aligned_global_u64_address(
    const void *source, unsigned long long &global_address) {
#if defined(FTT_DEVICE_PASS)
    static_assert(sizeof(std::uintptr_t) == sizeof(unsigned long long));
    if ((reinterpret_cast<std::uintptr_t>(source) & 7u) != 0u)
        return false;
    unsigned int is_global;
    asm volatile("{ .reg .pred p; isspacep.global p, %1; selp.u32 %0, 1, 0, p; }"
        : "=r"(is_global) : "l"(source) : "memory");
    if (!is_global)
        return false;
    asm volatile("cvta.to.global.u64 %0, %1;"
        : "=l"(global_address) : "l"(source) : "memory");
    return true;
#else
    (void)source;
    (void)global_address;
    return false;
#endif
}

// Precondition: converted naturally aligned global address, valid eight-byte span.
// Raw assembly avoids a typed alias and exposes its memory effect to the compiler.
FTT_DEVICE_INLINE unsigned long long load_global_u64(unsigned long long address) {
#if defined(FTT_DEVICE_PASS)
    unsigned long long result;
    asm volatile("ld.global.u64 %0, [%1];"
        : "=l"(result) : "l"(address) : "memory");
    return result;
#else
    (void)address;
    return 0ull;
#endif
}

// Diagnostic only: calibration/scale must be verified on the loaded target.
FTT_DEVICE_INLINE unsigned long long global_timer64() {
#if defined(FTT_DEVICE_PASS)
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) : : "memory");
    return value;
#else
    return 0ull; // Host instrumentation is rejected before rendering.
#endif
}

// Default PRMT uses selector bit3 to replicate the selected byte's sign bit.
// CUDA __byte_perm clears that bit, so it cannot implement signed extraction.
FTT_DEVICE_INLINE unsigned int byte_permute_default(unsigned int a,
                                                   unsigned int b,
                                                   unsigned int selector) {
#if defined(FTT_DEVICE_PASS)
    unsigned int result;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(selector));
    return result;
#else
    const unsigned long long bytes = static_cast<unsigned long long>(a)
        | (static_cast<unsigned long long>(b) << 32);
    unsigned int result = 0;
    for (int component = 0; component < 4; ++component) {
        const unsigned int selected = (selector >> (4 * component)) & 15;
        unsigned int value = static_cast<unsigned int>((bytes >> (8 * (selected & 7))) & 255);
        if (selected & 8) value = (value & 128) ? 255 : 0;
        result |= value << (8 * component);
    }
    return result;
#endif
}

template <int Component>
FTT_DEVICE_INLINE int signed_byte_extract(unsigned int word) {
    static_assert(Component >= 0 && Component < 4, "Byte component is outside the word.");
#if defined(FTT_DEVICE_PASS)
    int result;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result)
        : "r"(word), "r"(0u), "n"(0x8880u + Component * 0x1111u));
    return result;
#else
    const unsigned int value = (word >> (8 * Component)) & 255;
    return static_cast<int>(value) - static_cast<int>((value & 128) << 1);
#endif
}

FTT_DEVICE_INLINE int thread_id() {
#if defined(FTT_DEVICE_PASS)
    return static_cast<int>(threadIdx.x);
#else
    return 0;
#endif
}
FTT_DEVICE_INLINE int lane_id() { return thread_id() % nvidia::warp_size; }
FTT_DEVICE_INLINE int warp_id() { return thread_id() / nvidia::warp_size; }
FTT_DEVICE_INLINE int block_id() {
#if defined(FTT_DEVICE_PASS)
    return static_cast<int>(blockIdx.x);
#else
    return 0;
#endif
}
FTT_DEVICE_INLINE void cta_sync() {
#if defined(FTT_DEVICE_PASS)
    __syncthreads();
#endif
}
FTT_DEVICE_INLINE void cta_sync_alu() { cta_sync(); }
FTT_DEVICE_INLINE void warp_sync() {
#if defined(FTT_DEVICE_PASS)
    __syncwarp(__activemask());
#endif
}
FTT_DEVICE_INLINE void release_fence() {
#if defined(FTT_DEVICE_PASS)
    __threadfence();
#endif
}
FTT_DEVICE_INLINE float shuffle(float value, int src_lane) {
#if defined(FTT_DEVICE_PASS)
    return __shfl_sync(__activemask(), value, src_lane, nvidia::warp_size);
#else
    (void)src_lane;
    return value;
#endif
}
template <int Width>
FTT_DEVICE_INLINE float lane_group_reduce_sum(float value) {
    static_assert(Width > 0 && (Width & (Width - 1)) == 0 && Width <= nvidia::warp_size,
                  "Lane-group width must be a power of two within a warp.");
#if defined(FTT_DEVICE_PASS)
    // Some consumers invoke a complete lane group from a divergent branch.
    // Include exactly that group, rather than naming other inactive lanes.
    const unsigned int mask = (0xffffffffu >> (32 - Width)) << ((lane_id() / Width) * Width);
    FTT_UNROLL
    for (int offset = Width / 2; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(mask, value, offset, Width);
#endif
    return value;
}
FTT_DEVICE_INLINE float warp_reduce_sum(float value) {
    return lane_group_reduce_sum<nvidia::warp_size>(value);
}
FTT_DEVICE_INLINE float warp_reduce_max(float value) {
#if defined(FTT_DEVICE_PASS)
    FTT_UNROLL
    for (int offset = nvidia::warp_size / 2; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, value, offset, nvidia::warp_size);
        value = value > other ? value : other;
    }
#endif
    return value;
}
template <int Axis, class... Extents>
FTT_DEVICE_INLINE dim_t mesh_shard_id(Extents... extents) {
    constexpr int rank = sizeof...(Extents);
    static_assert(Axis >= 0 && Axis < rank, "Mesh axis is outside the mesh rank.");
    const dim_t sizes[rank] = {static_cast<dim_t>(extents)...};
    dim_t stride = 1;
    for (int axis = Axis + 1; axis < rank; ++axis)
        stride *= sizes[axis];
    return (static_cast<dim_t>(block_id()) / stride) % sizes[Axis];
}

// The host shim admits the complete resident grid with a cooperative launch.
// Device-scope release/acquire orders publication through each axis group's
// counter. Every writer fences before the CTA leader publishes its arrival.
// Each runtime owns its counters; the launch stream clears them per replay.
FTT_DEVICE_INLINE void grid_barrier(unsigned int *counter, int blocks) {
#if defined(FTT_DEVICE_PASS)
    release_fence();
    cta_sync();
    if (thread_id() == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> arrival(*counter);
        const unsigned int generation =
            arrival.fetch_add(1u, cuda::memory_order_acq_rel) / static_cast<unsigned int>(blocks);
        while (arrival.load(cuda::memory_order_acquire) / static_cast<unsigned int>(blocks) == generation) {
        }
    }
    cta_sync();
#else
    (void)counter;
    (void)blocks;
#endif
}

} // namespace arch
} // namespace ftt
