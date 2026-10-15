/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "TritonILUVATARGPUTransforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONILUVATARGPUOPTIMIZEEPILOGUE
#include "TritonILUVATARGPUTransforms/Passes.h.inc"

namespace {

bool isOneOperandElementwiseOp(Operation *op) {
  if (llvm::isa<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::FPToSIOp,
                arith::FPToUIOp, arith::NegFOp, arith::SIToFPOp,
                arith::TruncFOp, arith::TruncIOp, arith::UIToFPOp>(op))
    return true;
  if (llvm::isa<math::AbsFOp, math::AbsIOp, math::AtanOp, math::Atan2Op,
                math::CeilOp, math::CosOp, math::SinOp,
                math::CountLeadingZerosOp, math::CountTrailingZerosOp,
                math::CtPopOp, math::ErfOp, math::ExpOp, math::Exp2Op,
                math::ExpM1Op, math::FloorOp, math::LogOp, math::Log10Op,
                math::Log1pOp, math::Log2Op, math::SqrtOp, math::RsqrtOp,
                math::TanhOp>(op))
    return true;
  if (llvm::isa<triton::IntToPtrOp, triton::PtrToIntOp, triton::BitcastOp,
                triton::FpToFpOp>(op))
    return true;
  if (auto externElementwiseOp = dyn_cast<triton::ExternElementwiseOp>(op))
    return op->getNumOperands() == 1 && op->getNumResults() == 1 &&
           externElementwiseOp.getPure();
  return false;
}

// Bypassing shared memory only pays off if the mma tile can reach memory
// coalesced, and the TCU tile puts its fast lane bits on the LAST tensor dim
// (see iluvatarMmaTile: lane covers N first, then the low M offsets). So does
// every tile chooseIluvatarStoreLayout derives from it, down to the register
// bit it vectorizes on. A transposed epilogue -- e.g. col-major-S flash
// attention, whose accumulator is O[d, m] but whose pointers are the row-major
// O[m, d], making dim0 the contiguous one -- would therefore have adjacent
// lanes writing addresses `rowStride` apart: one 2-byte transaction per
// element.
//
// Coalescing dim0 instead is unreachable without shared memory. The layout that
// does it disagrees with #mma about which warp owns which column, so
// minimalCvtLayout keeps a `warp` out dimension and cvtNeedsWarpShuffle returns
// false at its out-dim check, before it ever counts transpositions -- shuffles
// cannot move data across warps, so no threshold would help. Nothing to win by
// bypassing, then: leave the store on the generic #mma -> #blocked
// shared-memory path, which does produce wide coalesced stores. Without this
// guard the fp16 store degrades to per-element 2-byte writes.
//
// Ask getOrderForMemory, not getOrder: the latter reports register contiguity,
// which says nothing about addresses once a thread holds a single element per
// tile. A fully coalesced [128, 512] f32 store whose lanes and warps already
// span all of dim1 has to spend its register bases repeating along dim0, so
// getOrder answers [0, 1] and the guard would reject a store that is perfectly
// contiguous. getOrderForMemory falls back to the thread order exactly in that
// degenerate case.
//
// Only ask this of stores that really are the epilogue: see the caller.
static bool storeMatchesMmaContiguity(RankedTensorType ptrType,
                                      Attribute mmaEncoding) {
  if (!isa<triton::gpu::IluvatarMmaEncodingAttr>(mmaEncoding))
    return true;
  auto order = triton::gpu::getOrderForMemory(ptrType);
  if (order.empty())
    return true;
  return order.front() == ptrType.getRank() - 1;
}

// Tries to optimize oldStoreOp with v_permlane*_swap instruction when possible.
// Returns null store op if not suitable.
static triton::StoreOp
usePermlaneSwapToOptimizeStore(PatternRewriter &rewriter, Value ptr, Value val,
                               Value mask, triton::StoreOp oldStoreOp) {
  auto ptrType = cast<RankedTensorType>(ptr.getType());
  auto valType = cast<RankedTensorType>(val.getType());

  // Build a store-friendly layout: each thread holds 2 consecutive columns
  // (-> 32-bit 2xfp16/bf16 global stores) AND adjacent lanes map to adjacent
  // columns (coalesced). The 16x32 tile differs from the TCU mma tile by
  // exactly one mixed register<->lane transposition, so its convert clears the
  // default cvtNeedsWarpShuffle gate (<2) and lowers to a register-only
  // multi-shuffle (prmt + slb.shfl, no shared-memory round-trip). The
  // single-16x16 fallback tile needs two and therefore does not clear that
  // gate: it still gets the wide coalesced store, but pays a shared-memory
  // round-trip for the convert. See the comments on iluvatarStoreTile2TCU.
  // Coalescing is load-bearing: an uncoalesced 2-element layout regresses vs
  // the blocked+SMEM baseline, and only reaching 32B instead of 64B per warp
  // costs ~1.5% on fp16 matmul.
  std::optional<triton::LinearLayout> storeLL =
      triton::gpu::chooseIluvatarStoreLayout(valType);
  if (!storeLL)
    return nullptr;

  Attribute newEncoding = triton::gpu::LinearEncodingAttr::get(
      oldStoreOp.getContext(), storeLL.value());
  auto newPtrType = ptrType.cloneWithEncoding(newEncoding);
  Value newPtr = triton::gpu::ConvertLayoutOp::create(rewriter, ptr.getLoc(),
                                                      newPtrType, ptr);

  auto newValType = valType.cloneWithEncoding(newEncoding);
  Value newVal = triton::gpu::ConvertLayoutOp::create(rewriter, val.getLoc(),
                                                      newValType, val);

  Value newMask = mask;
  if (mask) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    auto newMaskType = maskType.cloneWithEncoding(newEncoding);
    newMask = triton::gpu::ConvertLayoutOp::create(rewriter, mask.getLoc(),
                                                   newMaskType, mask);
  }

  return triton::StoreOp::create(rewriter, oldStoreOp.getLoc(), newPtr, newVal,
                                 newMask, oldStoreOp.getCache(),
                                 oldStoreOp.getEvict());
}

// convert(val) : xmma -> blocked
// elementWiseOp(val) : blocked
// ...
// elementWiseOp(val) : blocked
// tt.store(ptr, val, mask, ...) : blocked
// ==>
// convert(ptr) : blocked -> xmma
// convert(mask) : blocked -> xmma
// elementWiseOp(val) : xmma
// ...
// elementWiseOp(val) : xmma
// tt.store(ptr, val, mask, ...) : xmma
//
// Store with xmma layout directly
//
// xmma layout is either MFMA or WMMA
class BypassEpilogueSMEM : public mlir::OpRewritePattern<triton::StoreOp> {

public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp stOp,
                  mlir::PatternRewriter &rewriter) const override {

    Value ptr = stOp.getPtr();
    Value val = stOp.getValue();
    Value mask = stOp.getMask();
    auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
    auto valType = dyn_cast<RankedTensorType>(val.getType());
    if (!ptrType || !valType ||
        !isa<triton::gpu::BlockedEncodingAttr>(ptrType.getEncoding()) ||
        !isa<triton::gpu::BlockedEncodingAttr>(valType.getEncoding()))
      return mlir::failure();

    llvm::SmallVector<mlir::Operation *> chainedOps;
    while (true) {
      auto chainedOp = val.getDefiningOp();
      if (!chainedOp)
        return mlir::failure();
      if (llvm::isa<triton::gpu::ConvertLayoutOp>(chainedOp))
        break;
      if (!chainedOp->hasOneUse())
        return mlir::failure();
      if (!isOneOperandElementwiseOp(chainedOp))
        return mlir::failure();
      val = chainedOp->getOperand(0);
      chainedOps.push_back(chainedOp);
    }

    auto cvtOp = val.getDefiningOp<triton::gpu::ConvertLayoutOp>();
    if (!cvtOp)
      return mlir::failure();

    auto encoding = cvtOp.getSrc().getType().getEncoding();
    if (!isa<triton::gpu::MmaEncodingTrait>(encoding))
      return mlir::failure();

    if (!cvtOp.getResult().hasOneUse())
      return mlir::failure();

    // Trading the round trip for coalescing is only a good deal on the way out
    // of the kernel. A store inside a loop pays that trip every iteration --
    // including the barrier pair, which serializes against the pipelined loads
    // it shares the loop with -- and its scratch stays allocated for the whole
    // kernel. Col-major-S flash attention hits this with its debug softmax
    // mask: a [BLOCK_N, BLOCK_M] f32 store in the KV loop, whose 128KB of
    // scratch on top of the epilogue's own put the kernel over the
    // shared-memory limit.
    if (!stOp->getParentOfType<LoopLikeOpInterface>() &&
        !storeMatchesMmaContiguity(ptrType, encoding))
      return mlir::failure();

    auto newEncoding =
        cast<RankedTensorType>(cvtOp.getSrc().getType()).getEncoding();

    auto newPtrType = ptrType.cloneWithEncoding(newEncoding);
    Value newPtr = triton::gpu::ConvertLayoutOp::create(rewriter, ptr.getLoc(),
                                                        newPtrType, ptr);

    auto newVal = cvtOp.getSrc();

    for (auto chainedOp : llvm::reverse(chainedOps)) {
      auto oldType =
          cast<mlir::RankedTensorType>(chainedOp->getResult(0).getType());
      chainedOp->setOperand(0, newVal);
      newVal = llvm::cast<mlir::TypedValue<RankedTensorType>>(
          chainedOp->getResult(0));

      auto newType = oldType.cloneWithEncoding(newEncoding);
      newVal.setType(newType);
    }

    Value newMask = mask;
    if (mask) {
      auto maskType = dyn_cast<RankedTensorType>(mask.getType());
      auto newMaskType = maskType.cloneWithEncoding(newEncoding);
      newMask = triton::gpu::ConvertLayoutOp::create(rewriter, mask.getLoc(),
                                                     newMaskType, mask);
    }
    triton::StoreOp newStoreOp =
        usePermlaneSwapToOptimizeStore(rewriter, newPtr, newVal, newMask, stOp);
    if (!newStoreOp) {
      newStoreOp =
          triton::StoreOp::create(rewriter, stOp.getLoc(), newPtr, newVal,
                                  newMask, stOp.getCache(), stOp.getEvict());
    }

    rewriter.replaceOp(stOp, newStoreOp);
    return mlir::success();
  }
};

} // anonymous namespace

class TritonILUVATARGPUOptimizeEpiloguePass
    : public impl::TritonILUVATARGPUOptimizeEpilogueBase<
          TritonILUVATARGPUOptimizeEpiloguePass> {

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<BypassEpilogueSMEM>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createTritonILUVATARGPUOptimizeEpiloguePass() {
  return std::make_unique<TritonILUVATARGPUOptimizeEpiloguePass>();
}

} // namespace mlir
