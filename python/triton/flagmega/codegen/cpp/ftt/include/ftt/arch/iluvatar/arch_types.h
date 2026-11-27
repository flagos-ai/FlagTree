/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "../../compiler_defs.h"

// Iluvatar BI-V150 (COREX, ivcore11) machine constants and element types.
//
// Mirrors NTT's `arch/<name>/arch_types.h` role: this is the only place that
// states machine shape, so kernels and the generic layers stay arch-neutral.
//
// The numbers here are the ones the FlagMega target descriptor already
// asserts for this machine (see targets/iluvatar_bi_v150.py); keeping them in
// one header means a second arch only adds a sibling directory.

namespace ftt {
namespace arch {

struct iluvatar_bi_v150 {
    static constexpr const char *name = "iluvatar-bi-v150";

    // A warp is 64 lanes on ivcore11, not 32.  Every lane-level reduction in
    // the kernels below depends on this.
    static constexpr int warp_size = 64;

    static constexpr int sm_count = 16;

    // Shared memory per CTA, in bytes.
    static constexpr int shared_memory_bytes = 128 * 1024;
    static constexpr bool has_flat_sme_count_wait = true;

    // BI-V150 has no inter-SM L1 coherence, so cross-CTA publication needs an
    // explicit writeback-invalidate on the release side.  Recorded here
    // because kernels must not assume plain release ordering suffices.
    static constexpr bool has_inter_sm_l1_coherence = false;

    // No hardware mbarrier and no hardware named barriers: pipelined kernels
    // must emulate them in shared memory.
    static constexpr bool has_hardware_mbarrier = false;
    static constexpr bool has_hardware_named_barrier = false;
};

using current = iluvatar_bi_v150;

} // namespace arch

// Element types used by the Qwen3 decode path.
#if defined(FTT_DEVICE_PASS)
using bfloat16_t = __bf16;
#else
// Host-side placeholder so the headers stay parseable off-device; the storage
// size matters for buffer-plan arithmetic, the arithmetic does not.
struct bfloat16_t {
    unsigned short bits;
};
#endif

using float32_t = float;
using int32_t_ = int;

} // namespace ftt
