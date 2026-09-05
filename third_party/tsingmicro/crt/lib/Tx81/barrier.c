//===------------------------ Barrier.c -----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Barrier see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81_run.h"
#include <stdint.h>

// Full-cluster barrier: synchronizes all tiles on the chip.
void __Barrier() {
  INTRNISIC_RUN_SWITCH;
  RcsWaitfinish();
}

// Subgroup barrier: synchronizes a subset of tiles identified by
// `physical_ids`.  For the MVP this conservatively falls back to a
// full-cluster barrier.
void __BarrierSubgroup(const uint32_t *physical_ids, uint32_t count) {
  (void)physical_ids;
  (void)count;
  INTRNISIC_RUN_SWITCH;
  RcsWaitfinish();
}
