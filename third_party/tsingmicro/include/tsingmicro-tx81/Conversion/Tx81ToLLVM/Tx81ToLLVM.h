//===------------------- Tx81ToLLVM.h -------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TX81_TO_LLVM_H
#define TRITON_CONVERSION_TX81_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/Passes.h.inc"

void populateTx81ToLLVMConversionPatterns(RewritePatternSet &patterns,
                                          ConversionTarget &target,
                                          LLVMTypeConverter &converter);

std::unique_ptr<OperationPass<ModuleOp>> createTx81ToLLVMPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TX81_TO_LLVM_H
