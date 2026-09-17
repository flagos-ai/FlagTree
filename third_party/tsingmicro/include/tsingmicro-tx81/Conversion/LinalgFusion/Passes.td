//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_LINALG_FUSION_PASSES
#define TRITON_CONVERSION_LINALG_FUSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgFusion : Pass<"linalg-fusion", "mlir::ModuleOp"> {
  let summary = "Apply fusion transformation to linalg operations";
  let description = [{
    This pass applies scalar fusion transformation to linalg operations.
    It fuses scalar input linalg ops to reduce redundant memory read and write operations.
    Elementwise fusion needs to be implemented later.
  }];
  let constructor = "triton::createLinalgFusionPass()";
}

#endif // TRITON_CONVERSION_LINALG_FUSION_PASSES
