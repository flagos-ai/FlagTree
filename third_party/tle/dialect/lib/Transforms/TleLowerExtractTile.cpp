// flagtree tle
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLELOWEREXTRACTTILE
#include "tle/dialect/include/Transforms/Passes.h.inc"

// ============================================================================
// Pass 实现
// ============================================================================
class TleLowerExtractTile 
    : public impl::TritonTleLowerExtractTileBase<TleLowerExtractTile> {
  
  void runOnOperation() override {
    return;
  }
};

// ============================================================================
// 工厂函数
// ============================================================================
//std::unique_ptr<Pass> createTritonTleLowerExtractTile() {
//  return std::make_unique<TleLowerExtractTile>();
//}

} // namespace mlir::triton::tle