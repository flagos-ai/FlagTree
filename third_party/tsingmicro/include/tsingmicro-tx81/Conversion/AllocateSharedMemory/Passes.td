//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef ALLOCATE_SHARED_MEMORY_PASSES
#define ALLOCATE_SHARED_MEMORY_PASSES

include "mlir/Pass/PassBase.td"

def AllocateSharedMemory : Pass<"spmd-allocate-shared-memory", "mlir::ModuleOp"> {
  let summary = "SPMD(single program multi-data) mode: add metadata for shared memory allocation";

  let description = [{
    This pass uses the `ModuleAllocation` analysis to:
      - Annotate modules with an attribute with the amount of shared/local
        memory used.
      - Annotate operations with an offset into the total shared/local memory.
  }];
}

#endif
