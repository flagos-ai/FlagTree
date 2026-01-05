// MIT License

// Copyright (c) 2025 The FlagOS Contributors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// flagtree tle

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace nvidia_gpu;

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLELOWERTMACOPY
#include "tle/dialect/include/Transforms/Passes.h.inc"

class TMACopyLowering : public OpRewritePattern<TMACopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  enum class TransferDirection {
    GM_TO_SHARMEMORY = 0,
    SHARMEMORY_TO_GM = 1,
    INVALID = -1
  };

  TMACopyLowering(mlir::MLIRContext *context)
      : OpRewritePattern<TMACopyOp>(context) {}

  LogicalResult matchAndRewrite(TMACopyOp op,
                                PatternRewriter &rewriter) const override {
    TransferDirection direction = TransferDirection::INVALID;
    auto loc = op.getLoc();

    // Determine direction based on operand types
    if (isa<TensorDescType>(op.getSrc().getType()) &&
        isa<MemDescType>(op.getDst().getType())) {
      direction = TransferDirection::GM_TO_SHARMEMORY;
    } else if (isa<MemDescType>(op.getSrc().getType()) &&
               isa<TensorDescType>(op.getDst().getType())) {
      direction = TransferDirection::SHARMEMORY_TO_GM;
    } else {
      return failure();
    }

    if (direction == TransferDirection::GM_TO_SHARMEMORY) {
      // Load from global memory to shared memory
      auto srcType = cast<TensorDescType>(op.getSrc().getType());
      auto tensorType = srcType.getBlockType();

      // Use the existing shared memory allocation (should use #shared encoding
      // like Gluon)
      Value dstMemDesc = op.getDst();

      // Create minimal mbarrier allocation with #shared2 encoding (similar to
      // our current implementation)
      auto mbarrierCTALayout =
          gpu::CTALayoutAttr::get(tensorType.getContext(), {1}, {1}, {0});
      auto mbarrierEncoding = gpu::SwizzledSharedEncodingAttr::get(
          tensorType.getContext(), 1, 1, 1, {0}, mbarrierCTALayout);
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(op.getContext());

      gpu::MemDescType mbarrierMemDescType =
          gpu::MemDescType::get({1}, rewriter.getI64Type(), mbarrierEncoding,
                                sharedMemorySpace, /*mutableMemory=*/true);

      Value mbarrierAlloc =
          rewriter.create<gpu::LocalAllocOp>(loc, mbarrierMemDescType);
      rewriter.create<InitBarrierOp>(loc, mbarrierAlloc, 1);

      // Calculate size in bytes
      auto encoding = getEncodingFromDescriptor(op, tensorType, op.getSrc());
      auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
      int sizeInBytes = product(shapePerCTA) *
                        tensorType.getElementType().getIntOrFloatBitWidth() / 8;

      Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
      rewriter.create<triton::nvidia_gpu::BarrierExpectOp>(loc, mbarrierAlloc,
                                                           sizeInBytes, pred);

      // Create TMA indices
      auto indices = translateTMAIndices(rewriter, op.getLoc(),
                                         srcType.getBlockType().getEncoding(),
                                         op.getIndices());

      // Perform async TMA copy from global to existing shared memory
      rewriter.create<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(
          op.getLoc(), op.getSrc(), indices, mbarrierAlloc, dstMemDesc, pred);

      // Wait for completion and invalidate barrier
      Value phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
      rewriter.create<WaitBarrierOp>(loc, mbarrierAlloc, phase);
      rewriter.create<InvalBarrierOp>(loc, mbarrierAlloc);

    } else {
      // Store from shared memory to global memory
      auto dstType = cast<TensorDescType>(op.getDst().getType());
      auto tensorType = dstType.getBlockType();

      // Fence shared memory before store
      rewriter.create<triton::nvidia_gpu::FenceAsyncSharedOp>(loc, false);

      // Create TMA indices
      auto indices = translateTMAIndices(rewriter, op.getLoc(),
                                         dstType.getBlockType().getEncoding(),
                                         op.getIndices());

      // Perform async TMA copy from shared to global memory
      rewriter.create<triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(
          op.getLoc(), op.getDst(), indices, op.getSrc());

      // Wait for store completion
      rewriter.create<triton::nvidia_gpu::TMAStoreWaitOp>(loc, 0);
    }

    // Remove the TMACopyOp after processing
    rewriter.eraseOp(op);
    return success();
  }
};

class TritonTleLowerTmaCopy
    : public impl::TritonTleLowerTmaCopyBase<TritonTleLowerTmaCopy> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TMACopyLowering>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir::triton::tle
