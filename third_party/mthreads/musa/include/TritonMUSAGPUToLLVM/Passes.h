#ifndef TRITONMUSAGPU_CONVERSION_TRITONMUSAGPUTOLLVM_PASSES_H
#define TRITONMUSAGPU_CONVERSION_TRITONMUSAGPUTOLLVM_PASSES_H

#include "Dialect/MUSA/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "musa/include/TritonMUSAGPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonMUSAGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonMUSAGPUToLLVMPass(int32_t computeCapability);
std::unique_ptr<OperationPass<ModuleOp>>
createAllocateMUSASharedMemoryPass(int32_t computeCapability);

#define GEN_PASS_REGISTRATION
#include "musa/include/TritonMUSAGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
