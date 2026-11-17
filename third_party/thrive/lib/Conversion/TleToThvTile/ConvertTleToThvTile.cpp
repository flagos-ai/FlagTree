#include "Conversion/TleToThvTile/Passes.h"
#include "Conversion/TleToThvTile/TleToThvTilePatterns.h"
#include "Dialect/ThvTile/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::thvtile {
#define GEN_PASS_DEF_CONVERTTLETOTHVTILE
#include "Conversion/TleToThvTile/Passes.h.inc"

namespace {

struct ConvertTleToThvTilePass
    : public impl::ConvertTleToThvTileBase<ConvertTleToThvTilePass> {
  using ConvertTleToThvTileBase::ConvertTleToThvTileBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    TleToThvTileTypeConverter typeConverter;

    ConversionTarget target(*ctx);
    // Mark TLE tile ops as illegal (to be converted).
    target.addLegalDialect<thvtile::ThvTileDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<tle::ExtractTileOp, tle::InsertTileOp,
                        tle::ExclusiveCumsumOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(ctx);
    populateSliceOpsToThvTilePatterns(typeConverter, patterns);
    populateCumsumOpToThvTilePatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertTleToThvTilePass() {
  return std::make_unique<ConvertTleToThvTilePass>();
}

} // namespace mlir::thvtile
