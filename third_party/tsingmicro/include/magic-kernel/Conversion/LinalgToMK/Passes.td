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

  let options = [
     Option<"precisionMode", "precision-mode", "int", /*default*/"0",
       "Precision mode: 0 disables it, 1 preserves i64 precision, 2 preserves full integer precision (i32/i64 handled by CPU)">
  ];
}

#endif
