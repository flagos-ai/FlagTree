// flagtree tle
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLELOWERINSERTTILE
#include "tle/dialect/include/Transforms/Passes.h.inc"

class TleLowerInsertTile
    : public impl::TritonTleLowerInsertTileBase<TleLowerInsertTile> {

  void runOnOperation() override {}
};

} // namespace mlir::triton::tle
