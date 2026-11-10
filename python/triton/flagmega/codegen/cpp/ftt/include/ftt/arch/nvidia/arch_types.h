/* Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "../../compiler_defs.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace ftt {
namespace arch {

// Device counts and resource capacities belong to the target contract and
// launch admission, rather than a header shared by different NVIDIA devices.
struct nvidia {
    static constexpr int warp_size = 32;
    static constexpr bool has_flat_sme_count_wait = false;
};
using current = nvidia;

} // namespace arch
using bfloat16_t = __nv_bfloat16;
using float32_t = float;
using int32_t_ = int;
} // namespace ftt
