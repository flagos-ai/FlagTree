/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_TO_HFUSION_CONVERSION_PASSES
#define TRITON_TO_HFUSION_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToHFusion : Pass<"triton-to-hfusion", "mlir::ModuleOp"> {
  let summary = "Convert Triton to HFusion dialect";
  let constructor = "triton::createTritonToHFusionPass()";
  let dependentDialects = ["hfusion::HFusionDialect", "arith::ArithDialect",
                           "linalg::LinalgDialect", "scf::SCFDialect",
                           "tensor::TensorDialect"];
}

#endif // TRITON_TO_HFUSION_CONVERSION_PASSES
