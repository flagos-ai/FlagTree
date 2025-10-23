/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef DISCRETE_MASK_ACCESS_CONVERSION_PASSES
#define DISCRETE_MASK_ACCESS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def DiscreteMaskAccessConversion : Pass<"discrete-mask-access-conversion", "mlir::ModuleOp"> {
  let summary = "Recognize and convert discrete mask memory access";
  let constructor = "triton::createDiscreteMaskAccessConversionPass()";
}

#endif // DISCRETE_MASK_ACCESS_CONVERSION_PASSES
