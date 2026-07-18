// flagtree tle
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLELOWEREXTRACTTILE
#include "tle/dialect/include/Transforms/Passes.h.inc"

class TleLowerExtractTile
    : public impl::TritonTleLowerExtractTileBase<TleLowerExtractTile> {

  void runOnOperation() override {}
};

} // namespace mlir::triton::tle
