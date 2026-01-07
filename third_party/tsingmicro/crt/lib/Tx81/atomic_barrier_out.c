//===------------------------ AtomicBarrierOut.c --------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::AtomicBarrierOut see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __AtomicBarrierOut() {
#ifdef USE_SIM_MODE
#else
  atomic_barrier_out();
#endif
}
