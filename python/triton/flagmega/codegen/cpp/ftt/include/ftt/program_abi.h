// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
#include <stddef.h>
#define FLAGMEGA_CPP_PROGRAM_ABI_VERSION 3
struct FlagmegaArgSlot {
    uint32_t tag, reserved;
    uint64_t payload;
};
struct FlagmegaResourceRecord {
    uint32_t version, record_bytes, entry_id, device, cc_major, cc_minor;
    uint32_t sm_count, warp_size, registers_per_thread, max_threads_per_block;
    uint32_t block_threads, resident_blocks_per_sm, cooperative_supported, reserved;
    uint64_t static_shared_bytes, dynamic_shared_bytes, local_bytes;
};
static_assert(sizeof(FlagmegaArgSlot) == 16, "ArgSlot ABI drift");
static_assert(sizeof(FlagmegaResourceRecord) == 80, "ResourceRecord ABI drift");
extern "C" {
uint32_t flagmega_program_count_v3();
const char *flagmega_program_manifest_v3();
int flagmega_program_resource_info_v3(uint32_t entry_id, FlagmegaResourceRecord *, uint32_t record_bytes);
int flagmega_launch_program_v3(uint32_t entry_id, const FlagmegaArgSlot *, uint32_t arg_count,
                               void *stream, void *scratch, uint64_t scratch_bytes);
}

// Additive normal whole-plan ABI. Rows cannot supply entry IDs or reset policy.
#define FLAGMEGA_COUNTER_EPOCH_ABI_VERSION 1
struct FlagmegaProgramRow {
    const FlagmegaArgSlot *args;
    uint32_t arg_count, reserved;
};
struct FlagmegaPlanResult {
    uint32_t version, record_bytes, ordinal, stage;
    int32_t status;
    uint32_t submitted, reserved0, reserved1;
};
static_assert(sizeof(FlagmegaProgramRow) == 16, "ProgramRow ABI drift");
static_assert(sizeof(FlagmegaPlanResult) == 32, "PlanResult ABI drift");
extern "C" {
const char *flagmega_counter_epoch_manifest_v1();
int flagmega_launch_counter_epoch_plan_v1(const char *plan_sha256,
    const FlagmegaProgramRow *rows, uint32_t row_count, void *stream,
    void *scratch, uint64_t scratch_bytes, FlagmegaPlanResult *out);
}
