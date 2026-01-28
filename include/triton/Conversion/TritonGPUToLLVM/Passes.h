#ifndef TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PASSES_H
#define TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

namespace gpu {
std::unique_ptr<OperationPass<ModuleOp>> createAllocateSharedMemoryPass();

} // namespace gpu

// Forward declaration for pipeline intrinsics pass
std::unique_ptr<OperationPass<ModuleOp>> createPipelineIntrinsicsToLLVMPass();

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
