/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utility>

#include "Conversion/TritonToGCU/TritonToGCUPass.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tle-to-triton-gcu"

namespace mlir {
#define GEN_PASS_DEF_TLETOTRITONGCUPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

#ifdef ENABLE_TLE

/// Compute row-major strides from a static shape.
/// For shape [S0, S1, ..., S_{n-1}]:
///   stride[i] = S_{i+1} * S_{i+2} * ... * S_{n-1}
///   stride[n-1] = 1
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

/// Try to look through a tt.broadcast to find the pre-broadcast source.
static Value lookThroughBroadcast(Value v) {
  if (auto broadcastOp = v.getDefiningOp<triton::BroadcastOp>())
    return broadcastOp.getSrc();
  return v;
}

/// Rewrite `tle.local_pointers(%memdesc, %idx0, %idx1, ...)`
/// using the standard Triton pointer-arithmetic pattern:
///
/// For a 2D memdesc<S0 x S1 x elemTy>:
///
///   %base = triton_gcu.memdesc_to_ptr %memdesc -> !tt.ptr<elemTy, 3>
///
///   // dim 0 (highest): splat base to idx0's shape, addptr with idx0*stride0
///   %base_splat = tt.splat %base -> tensor<S0 x 1 x !tt.ptr<elemTy, 3>>
///   %stride0_cst = arith.constant dense<stride0> : tensor<S0 x 1 x i32>
///   %off0 = arith.muli %idx0_pre, %stride0_cst
///   %ptr0 = tt.addptr %base_splat, %off0   (shape: S0 x 1)
///
///   // broadcast ptr tensor to include dim 1
///   %ptr0_bc = tt.broadcast %ptr0 -> tensor<S0 x S1 x !tt.ptr<elemTy, 3>>
///
///   // dim 1 (lowest, stride=1): addptr with idx1
///   %idx1_bc = tt.broadcast %idx1_pre -> tensor<S0 x S1 x i32>
///   %result = tt.addptr %ptr0_bc, %idx1_bc  (shape: S0 x S1)
///
/// This mirrors the standard Triton IR pattern for global pointer arithmetic.
struct ConvertLocalPointersOp : public RewritePattern {
  explicit ConvertLocalPointersOp(MLIRContext *ctx)
      : RewritePattern("tle.local_pointers", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();

    auto loc = op->getLoc();

    Value memDescVal = op->getOperand(0);
    auto memDescTy = dyn_cast<triton::gpu::MemDescType>(memDescVal.getType());
    if (!memDescTy)
      return rewriter.notifyMatchFailure(op, "operand 0 is not MemDescType");

    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "result is not RankedTensorType");

    auto resultShape = resultTy.getShape();
    auto encoding = resultTy.getEncoding();
    auto elemTy = memDescTy.getElementType();
    auto bufferShape = memDescTy.getShape();
    unsigned bufferRank = bufferShape.size();

    unsigned numIndices = op->getNumOperands() - 1;
    if (numIndices != bufferRank)
      return rewriter.notifyMatchFailure(
          op, "indices count does not match buffer rank");

    auto ptrElemTy = dyn_cast<triton::PointerType>(resultTy.getElementType());
    if (!ptrElemTy)
      return rewriter.notifyMatchFailure(op,
                                         "result element type is not tt.ptr");

    auto strides = computeRowMajorStrides(bufferShape);

    constexpr int kSharedMemAddrSpace = 3;
    auto basePtrTy = triton::PointerType::get(elemTy, kSharedMemAddrSpace);
    auto i32Ty = rewriter.getI32Type();

    // %base_ptr = triton_gcu.memdesc_to_ptr %memdesc
    auto basePtr = rewriter.create<triton::gcu::MemDescToPtrOp>(loc, basePtrTy,
                                                                memDescVal);

    // Process dimensions from high (dim 0) to low (dim N-1).
    // At each step we maintain a running pointer tensor `ptrAccum`.
    Value ptrAccum;

    for (unsigned dim = 0; dim < bufferRank; ++dim) {
      Value idx = op->getOperand(dim + 1);

      // Try to look through tt.broadcast to get the pre-broadcast source,
      // which typically has a "1" in non-owning dimensions.
      Value idxSrc = lookThroughBroadcast(idx);
      auto idxSrcTy = cast<RankedTensorType>(idxSrc.getType());
      auto idxSrcShape = idxSrcTy.getShape();
      auto idxSrcEncoding = idxSrcTy.getEncoding();

      // Ensure index element type is i32.
      if (idxSrcTy.getElementType() != i32Ty) {
        auto castTy = RankedTensorType::get(idxSrcShape, i32Ty, idxSrcEncoding);
        idxSrc = rewriter.create<arith::TruncIOp>(loc, castTy, idxSrc);
        idxSrcTy = cast<RankedTensorType>(idxSrc.getType());
        idxSrcShape = idxSrcTy.getShape();
        idxSrcEncoding = idxSrcTy.getEncoding();
      }

      // Compute offset contribution: idx * stride (skip multiply if stride=1).
      Value offset;
      int64_t strideVal = strides[dim];
      if (strideVal == 1) {
        offset = idxSrc;
      } else {
        auto strideCstTy =
            RankedTensorType::get(idxSrcShape, i32Ty, idxSrcEncoding);
        auto strideCst = rewriter.create<arith::ConstantOp>(
            loc, DenseElementsAttr::get(strideCstTy,
                                        rewriter.getI32IntegerAttr(
                                            static_cast<int32_t>(strideVal))));
        offset = rewriter.create<arith::MulIOp>(loc, idxSrc, strideCst);
      }

      if (!ptrAccum) {
        auto splatPtrTy =
            RankedTensorType::get(idxSrcShape, ptrElemTy, idxSrcEncoding);
        ptrAccum = rewriter.create<triton::SplatOp>(loc, splatPtrTy, basePtr);
        ptrAccum = rewriter.create<triton::AddPtrOp>(loc, splatPtrTy, ptrAccum,
                                                     offset);
      } else {
        auto ptrAccumTy = cast<RankedTensorType>(ptrAccum.getType());
        auto ptrAccumShape = ptrAccumTy.getShape();

        SmallVector<int64_t> targetShape(bufferRank);
        for (unsigned d = 0; d < bufferRank; ++d)
          targetShape[d] = std::max(ptrAccumShape[d], idxSrcShape[d]);

        auto targetPtrTy =
            RankedTensorType::get(targetShape, ptrElemTy, encoding);
        auto targetI32Ty = RankedTensorType::get(targetShape, i32Ty, encoding);

        if (ptrAccumShape != ArrayRef<int64_t>(targetShape))
          ptrAccum =
              rewriter.create<triton::BroadcastOp>(loc, targetPtrTy, ptrAccum);

        if (idxSrcShape != ArrayRef<int64_t>(targetShape))
          offset =
              rewriter.create<triton::BroadcastOp>(loc, targetI32Ty, offset);

        auto addPtrResultTy = cast<RankedTensorType>(ptrAccum.getType());
        ptrAccum = rewriter.create<triton::AddPtrOp>(loc, addPtrResultTy,
                                                     ptrAccum, offset);
      }
    }

    // Final broadcast to result shape if needed (should already match).
    auto ptrAccumShape = cast<RankedTensorType>(ptrAccum.getType()).getShape();
    if (ptrAccumShape != resultShape) {
      auto finalPtrTy = RankedTensorType::get(resultShape, ptrElemTy, encoding);
      ptrAccum =
          rewriter.create<triton::BroadcastOp>(loc, finalPtrTy, ptrAccum);
    }

    rewriter.replaceOp(op, ptrAccum);
    return success();
  }
};

#endif // ENABLE_TLE

struct TLEToTritonGCUPass
    : public impl::TLEToTritonGCUPassBase<TLEToTritonGCUPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::TritonDialect, triton::gcu::TritonGCUDialect>();
  }

  void runOnOperation() override {
#ifdef ENABLE_TLE
    auto module = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertLocalPointersOp>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return signalPassFailure();
#endif // ENABLE_TLE
  }
};

} // namespace
