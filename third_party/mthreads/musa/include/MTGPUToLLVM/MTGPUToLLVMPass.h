#ifndef TRITON_CONVERSION_MTGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_MTGPU_TO_LLVM_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

namespace mtgpu {

void populateMTGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

} // namespace mtgpu

std::unique_ptr<OperationPass<ModuleOp>> createConvertMTGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertMTGPUToLLVMPass(int32_t computeCapability);

#define GEN_PASS_DECL
#include "musa/include/MTGPUToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "musa/include/MTGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif // TRITON_CONVERSION_MTGPU_TO_LLVM_PASS_H
