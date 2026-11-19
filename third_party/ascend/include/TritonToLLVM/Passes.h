/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_ADAPTER_TRITON_TO_LLVM_CONVERSION_PASSES_H
#define TRITON_ADAPTER_TRITON_TO_LLVM_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

/// Creates a pass to convert Triton dialect to LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createTritonToLLVMPass();

/// Creates a pass to normalize locations for debug line table generation.
std::unique_ptr<OperationPass<ModuleOp>>
createNormalizeDebugLineLocationsPass();

#define GEN_PASS_REGISTRATION
#include "ascend/include/TritonToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_LLVM_CONVERSION_PASSES_H
