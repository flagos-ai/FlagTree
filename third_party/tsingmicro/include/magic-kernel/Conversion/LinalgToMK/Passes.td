//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_TO_MK_CONVERSION_PASSES
#define LINALG_TO_MK_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToMK : Pass<"linalg-to-mk", "mlir::ModuleOp"> {
  let summary = "Convert linalg operations into magic kernel operations";

  let options = [];
}

#endif
