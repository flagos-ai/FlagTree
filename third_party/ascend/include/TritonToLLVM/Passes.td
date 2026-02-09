/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_TO_LLVM_CONVERSION_PASSES
#define TRITON_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLLVM : Pass<"triton-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Triton to LLVM dialect";
  let constructor = "triton::createTritonToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "tensor::TensorDialect", "arith::ArithDialect"];
}

#endif // TRITON_TO_LLVM_CONVERSION_PASSES
