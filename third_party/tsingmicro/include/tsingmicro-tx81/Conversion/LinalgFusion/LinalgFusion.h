#ifndef TRITON_CONVERSION_LINALG_FUSION_H
#define TRITON_CONVERSION_LINALG_FUSION_H


#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "tsingmicro-tx81/Conversion/LinalgFusion/Passes.h.inc"

void populateLinalgBinaryOpFusionPatterns(RewritePatternSet &patterns);

// TODO: Support linalg elementwise op fusion. 
#if 0
void populateLinalgFusionPatterns(RewritePatternSet &patterns);
#endif

std::unique_ptr<OperationPass<ModuleOp>> createLinalgFusionPass();

}
}

#endif