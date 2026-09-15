// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

// Based on CANN 9.1 kernel_reg.h::PipeBarrierImpl for dav_c220.
// Local CUBE pipeline barriers; these do not synchronize cores or allocate
// state.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_cube_begin(int32_t token) {
  pipe_barrier(PIPE_ALL);
}

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_cube_end(int32_t token) {
  pipe_barrier(PIPE_ALL);
}
