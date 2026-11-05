/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "Dialect/TritonPPUGPU/IR/Dialect.h"
#include "TritonPPUGPUToLLVM/AIUUtility.h"
#include "TritonPPUGPUTransforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/CacheControls.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {
namespace {

class AIULoadLowering : public OpRewritePattern<AIULoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIULoadOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    auto tensorType = op.getResult().getType();
    assert(tensorType.getRank() > 1);
    SmallVector<unsigned> order(op.getOrder().begin(), op.getOrder().end());
    auto ctaLayout = getCTALayout(tensorType.getEncoding());
    auto elemBytes = tensorType.getElementTypeBitWidth() / 8;
    int numWarps = lookupNumWarps(op);

    auto tileShape = tensorType.getShape();
    size_t rank = tileShape.size();
    auto tileC = tileShape[rank - 1];
    auto tileW = tileShape[rank - 2];
    if (order[rank - 1] != 0) {
      tileC = tileShape[rank - 2];
      tileW = tileShape[rank - 1];
    }

    auto mod = op->getParentOfType<ModuleOp>();
    auto computeCapability = getPPUComputeCapability(mod);
    unsigned version = (computeCapability == 80) ? 1 : 2;
    auto loadStrategy =
        LLVM::PPU::AIULoadStrategy(numWarps, tileW, tileC, elemBytes, version);
    Attribute encoding = PPUAIUSharedEncodingAttr::get(
        tensorType.getContext(), version, loadStrategy, order, ctaLayout);

    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = rewriter.create<LocalAllocOp>(loc, memDescType, Value());

    Operation *copy =
        rewriter.create<triton::ppu_gpu::AsyncAIUCopyGlobalToLocalOp>(
            loc, op.getSrcPtr(), op.getIndices(), op.getShape(), alloc,
            op.getCache(), op.getEvict());

    Value Zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 0);

    Operation *commmit = rewriter.create<triton::gpu::AsyncCommitGroupOp>(
        loc, copy->getResult(0));
    Operation *wait = rewriter.create<triton::gpu::AsyncWaitOp>(
        loc, commmit->getResult(0), 0);
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op.getType(), alloc);
    return success();
  }
};

} // namespace
} // namespace gpu
} // namespace triton

#define GEN_PASS_DEF_TRITONPPUAIULOWERINGPASS
#include "TritonPPUGPUTransforms/Passes.h.inc"

class TritonPPUAIULoweringPass
    : public impl::TritonPPUAIULoweringPassBase<TritonPPUAIULoweringPass> {
public:
  using impl::TritonPPUAIULoweringPassBase<
      TritonPPUAIULoweringPass>::TritonPPUAIULoweringPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // Native AIU coordinates must not wrap the 24-bit encoding. In
    // particular, negative X/Z offsets fault on the tested SDK 2.1 path.
    // Check the actual indices after block-pointer advance has been lowered.
    bool invalid = false;
    OpBuilder builder(context);
    m.walk([&](triton::AIULoadOp op) {
      if (!triton::isPPUAIUCache(op.getCache()))
        return;
      builder.setInsertionPoint(op);
      Value valid;
      auto tile = op.getType().getShape();
      for (auto [axis, coordinate] : llvm::enumerate(op.getIndices())) {
        int64_t maximum = (1 << 24) - tile[axis];
        APInt constant;
        if (matchPattern(coordinate, m_ConstantInt(&constant))) {
          if (constant.getSExtValue() < 0 ||
              constant.getSExtValue() > maximum) {
            op.emitError("PPU AIU copy requires nonnegative X/Z offsets within "
                         "the 24-bit coordinate range");
            invalid = true;
          }
          continue;
        }
        Value limit = arith::ConstantOp::create(
            builder, op.getLoc(),
            builder.getIntegerAttr(coordinate.getType(), maximum));
        Value fits = arith::CmpIOp::create(
            builder, op.getLoc(), arith::CmpIPredicate::ule, coordinate, limit);
        valid = valid ? arith::AndIOp::create(builder, op.getLoc(), valid, fits)
                      : fits;
      }
      if (valid)
        triton::AssertOp::create(builder, op.getLoc(), valid,
                                 "PPU AIU copy requires nonnegative X/Z "
                                 "offsets within the 24-bit coordinate range");
    });
    if (invalid) {
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet patterns(context);
    patterns.add<mlir::triton::gpu::AIULoadLowering>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
