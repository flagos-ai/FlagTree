// flagtree tle
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLELOWERINSERTTILE
#include "tle/dialect/include/Transforms/Passes.h.inc"

// ============================================================================
// Pass 实现
// ============================================================================
class TleLowerInsertTile 
    : public impl::TritonTleLowerInsertTileBase<TleLowerInsertTile> {
  
  void runOnOperation() override {
    return;
  }
};


} // namespace mlir::triton::tle