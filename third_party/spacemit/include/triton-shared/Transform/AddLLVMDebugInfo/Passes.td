//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef ADD_LLVM_DEBUG_INFO_TRANSFORM_PASSES
#define ADD_LLVM_DEBUG_INFO_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def AddLLVMDebugInfo : Pass<"add-llvm-debug-info", "mlir::ModuleOp"> {
  let summary = "Add LLVM debug info";
  let constructor = "triton::createAddLLVMDebugInfoPass()";
}

#endif
