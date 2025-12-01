//===------------------- MKToTx81.h ---------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//
//
// Lowering magic kernel ops to TsingMicro Tx81 target.
//
//===----------------------------------------------------------------------===//

#ifndef ZTC_CONVERSION_MK_TO_TX81_H
#define ZTC_CONVERSION_MK_TO_TX81_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "tsingmicro-tx81/Conversion/MKToTx81/Passes.h.inc"

void populateMKToTx81CanonicalizationPatterns(RewritePatternSet &patterns);

void populateMKToTx81ConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createMKToTx81Pass();

} // namespace triton
} // namespace mlir

#endif // ZTC_CONVERSION_MK_TO_TX81_H
