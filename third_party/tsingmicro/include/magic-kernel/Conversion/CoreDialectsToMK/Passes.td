//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef CORE_DIALECTS_TO_MK_CONVERSION_PASSES
#define CORE_DIALECTS_TO_MK_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def CoreDialectsToMK : Pass<"core-dialects-to-mk", "mlir::ModuleOp"> {
  let summary = "Convert core dialects including Linalg, Memref etc to MK";
  let constructor = "triton::createCoreDialectsToMKPass()";
}

#endif
