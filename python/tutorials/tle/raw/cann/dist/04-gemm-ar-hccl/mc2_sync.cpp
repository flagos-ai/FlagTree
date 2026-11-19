// Copyright 2026- FlagOS Contributors
// Barriers for Triton mix-CV MatmulAllReduce:
//   cube:  PipeBarrier + CrossCore (ops Mc2SyncAll ON_CUBE) — flush MM stores
//   vector: SyncAll() among all AIVs — wait until every program finished MM
// Without the AIV SyncAll, vector block0 Commit races other cubes still writing
// WINDOW (tile=1 / large strips show wrong rows; multi-tile often corrupts
// strip 0 only).

#include "Utils.h"
#include "kernel_operator.h"

using namespace AscendC;

extern "C" {

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_mc2_sync_all_cube(memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  PipeBarrier<PIPE_ALL>();
  CrossCoreSetFlag<0x0, PIPE_FIX>(3);
  CrossCoreWaitFlag(3);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_mc2_sync_all_vector(memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  // FFTS AIV-only barrier; all vector programs must enter.
  SyncAll();
}

} // extern "C"
