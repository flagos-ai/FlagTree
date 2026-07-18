//===----------------------------------------------------------------------===//
//
// This provides registration calls for LLVMXPU dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_XPU_XPUTOLLVMIRTRANSLATION_SDNN_H
#define MLIR_TARGET_LLVMIR_DIALECT_XPU_XPUTOLLVMIRTRANSLATION_SDNN_H

#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "llvm/IR/IRBuilder.h"

namespace mlir {

LogicalResult SDNNConvertOperation(Operation &op, llvm::IRBuilderBase &builder,
                                   LLVM::ModuleTranslation &moduleTranslation);
void registerLLVMXPUSDNNDialectTranslation(DialectRegistry &registry);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_XPU_XPUTOLLVMIRTRANSLATION_SDNN_H
