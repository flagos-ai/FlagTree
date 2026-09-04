#include "Dialect/ThvTile/IR/Dialect.h"
#include "tle/dialect/include/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace tle = mlir::triton::tle;

namespace {

struct GetNumPesOpToLibdevice : public OpRewritePattern<tle::GetNumPesOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tle::GetNumPesOp op,
                                PatternRewriter &rewriter) const override {
    auto call = rewriter.create<thvtile::LibdeviceCallOp>(
        op.getLoc(), TypeRange{op.getResult().getType()},
        StringRef("__shmem_n_pes"), ValueRange{}, /*pure=*/true);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct GetDeviceIdOpToLibdevice : public OpRewritePattern<tle::GetDeviceIdOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tle::GetDeviceIdOp op,
                                PatternRewriter &rewriter) const override {
    auto call = rewriter.create<thvtile::LibdeviceCallOp>(
        op.getLoc(), TypeRange{op.getResult().getType()},
        StringRef("__shmem_my_pe"), ValueRange{}, /*pure=*/true);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct DistributedBarrierOpToLibdevice
    : public OpRewritePattern<tle::DistributedBarrierOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tle::DistributedBarrierOp op,
                                PatternRewriter &rewriter) const override {
    auto space = op.getSpace();
    if (!space || *space != "chiplet")
      return failure();

    auto zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
    rewriter.create<thvtile::LibdeviceCallOp>(
        op.getLoc(), TypeRange{}, StringRef("__shmem_barrier_cluster"),
        ValueRange{zero}, /*pure=*/false);
    rewriter.eraseOp(op);
    return success();
  }
};

struct RemotePointersOpToLibdevice
    : public OpRewritePattern<tle::RemotePointersOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tle::RemotePointersOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSpace() != "chiplet")
      return failure();

    auto call = rewriter.create<thvtile::LibdeviceCallOp>(
        op.getLoc(), TypeRange{op.getResult().getType()},
        StringRef("__shmem_ptr"), ValueRange{op.getSrc(), op.getShardId()},
        /*pure=*/false);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct ChipletToThvTilePass
    : public PassWrapper<ChipletToThvTilePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ChipletToThvTilePass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<GetNumPesOpToLibdevice, GetDeviceIdOpToLibdevice,
                 DistributedBarrierOpToLibdevice, RemotePointersOpToLibdevice>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::thrive {

std::unique_ptr<Pass> createChipletToThvTilePass() {
  return std::make_unique<ChipletToThvTilePass>();
}

} // namespace mlir::thrive
