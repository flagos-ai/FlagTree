//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef TLE_TO_MK_CONVERSION_PASSES
#define TLE_TO_MK_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TLEToMK : Pass<"tle-to-mk", "mlir::ModuleOp"> {
  let summary = "Convert TLE communication operations into magic kernel operations";

  let options = [];
}

#endif
