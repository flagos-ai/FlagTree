// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
struct FlagmegaProgramTraceRecord {
    uint64_t tick, epoch;
    uint32_t entry_id, invocation, warp, event;
};
struct FlagmegaProgramTrace {
    uint64_t buffer, capacity, epoch;
    uint32_t entry_id, invocation;
};
static_assert(sizeof(FlagmegaProgramTraceRecord) == 32, "trace ABI drift");
static_assert(sizeof(FlagmegaProgramTrace) == 32, "trace binding ABI drift");
#ifdef __CUDACC__
static __device__ __forceinline__ void flagmega_program_event(FlagmegaProgramTrace trace, uint32_t site) {
    if (!trace.buffer || (threadIdx.x & 31)) return;
    uint64_t tick;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick) :: "memory");
    auto *header = reinterpret_cast<unsigned long long *>(trace.buffer);
    const uint64_t index = atomicAdd(header, 1ULL);
    if (index >= trace.capacity) { atomicAdd(header + 1, 1ULL); return; }
    auto *records = reinterpret_cast<FlagmegaProgramTraceRecord *>(header + 2);
    const uint32_t block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    records[index] = {tick, trace.epoch, trace.entry_id, trace.invocation,
                     block * (blockDim.x / 32) + threadIdx.x / 32, site};
}
#endif
