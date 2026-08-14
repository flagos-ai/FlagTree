/*
 * Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
 * All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License"); you may
 *    not use this file except in compliance with the License. You may obtain
 *    a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *    License for the specific language governing permissions and limitations
 *    under the License.
 */

#include "TritonILUVATARGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>

namespace mlir {
namespace triton {
namespace gpu {

namespace {

int computeCapabilityToSMEVersion(int computeCapability) {
  if (computeCapability <= 70) {
    return 0;
  } else if (computeCapability <= 80) {
    return 1;
  }
  assert(false && "computeCapability >80 not supported");
  return 2;
}

Value getSmeStride(LoadOp &loadOp, mlir::PatternRewriter &rewriter) {
  Value res = NULL;
  Value initArg = NULL;
  if (auto forOp =
          llvm::dyn_cast<scf::ForOp>(loadOp->getBlock()->getParentOp())) {
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(loadOp.getPtr())) {
      initArg = forOp.getTiedLoopInit(blockArg)->get();
    } else {
      initArg = loadOp.getPtr();
    }
  } else if (auto funOp =
                 llvm::dyn_cast<FuncOp>(loadOp->getBlock()->getParentOp())) {
    initArg = loadOp.getPtr();
  }
  if (!initArg)
    return res;

  SetVector<Operation *> bwdSlices;
  (void)mlir::getBackwardSlice(initArg, &bwdSlices);
  for (auto op : bwdSlices) {
    if (auto muliOp = dyn_cast<arith::MulIOp>(op)) {
      Type valueTy = muliOp.getResult().getType();
      auto muli_res = mlir::dyn_cast<RankedTensorType>(valueTy);
      if (muli_res) {
        Value in = NULL;
        if (mlir::isa_and_nonnull<ExpandDimsOp>(
                muliOp.getOperand(0).getDefiningOp()))
          in = muliOp.getOperand(1);
        else if (mlir::isa_and_nonnull<ExpandDimsOp>(
                     muliOp.getOperand(1).getDefiningOp()))
          in = muliOp.getOperand(0);
        else
          break;
        auto inPreOp = in.getDefiningOp();
        auto muliEncoding =
            mlir::dyn_cast<BlockedEncodingAttr>(muli_res.getEncoding());
        if (inPreOp && muliEncoding && muli_res.getShape().size() == 2) {
          if (auto constantOp = dyn_cast<arith::ConstantOp>(inPreOp)) {
            if (auto int_attr = dyn_cast<mlir::DenseIntElementsAttr>(
                    constantOp.getValue())) {
              int stride = (*(*(int_attr.begin())).getRawData());
              res = mlir::Value(mlir::arith::ConstantIntOp::create(
                  rewriter, rewriter.getUnknownLoc(), stride, 32));
              break;
            }
          } else if (auto splatOp = dyn_cast<SplatOp>(inPreOp)) {
            Type dataType = splatOp->getOperand(0).getType();
            if (dataType.isInteger()) {
              res = splatOp.getSrc();
              break;
            }
          }
        }
      }
    }
  }
  return res;
}

Value cloneMaskComputation(Value root, PatternRewriter &rewriter) {
  auto withSmeMaskEncoding = [](Type type) -> Type {
    auto tensorTy = dyn_cast<RankedTensorType>(type);
    if (!tensorTy)
      return type;
    Attribute encoding = tensorTy.getEncoding();
    MLIRContext *ctx = encoding.getContext();
    if (auto slice = dyn_cast<SliceEncodingAttr>(encoding)) {
      if (auto parent = dyn_cast<BlockedEncodingAttr>(slice.getParent())) {
        auto newParent = BlockedEncodingAttr::get(
            ctx, parent.getSizePerThread(), parent.getThreadsPerWarp(),
            parent.getWarpsPerCTA(), parent.getOrder(), parent.getCTALayout(),
            parent.getIsSme(), true, parent.getSmeWarpsPerCTA());
        return RankedTensorType::get(
            tensorTy.getShape(), tensorTy.getElementType(),
            SliceEncodingAttr::get(ctx, slice.getDim(), newParent));
      }
    }
    if (auto blocked = dyn_cast<BlockedEncodingAttr>(encoding)) {
      auto newEncoding = BlockedEncodingAttr::get(
          ctx, blocked.getSizePerThread(), blocked.getThreadsPerWarp(),
          blocked.getWarpsPerCTA(), blocked.getOrder(), blocked.getCTALayout(),
          blocked.getIsSme(), true, blocked.getSmeWarpsPerCTA());
      return RankedTensorType::get(tensorTy.getShape(),
                                   tensorTy.getElementType(), newEncoding);
    }
    return type;
  };

  IRMapping mapping;
  std::function<Value(Value)> cloneValue = [&](Value value) -> Value {
    if (Value mapped = mapping.lookupOrNull(value))
      return mapped;
    Operation *def = value.getDefiningOp();
    if (!def)
      return value;
    if (def->getNumRegions() != 0 || !mlir::isMemoryEffectFree(def))
      return value;
    for (Value operand : def->getOperands())
      mapping.map(operand, cloneValue(operand));
    Operation *cloned = rewriter.clone(*def, mapping);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(def->getResults(), cloned->getResults())) {
      newResult.setType(withSmeMaskEncoding(newResult.getType()));
      mapping.map(oldResult, newResult);
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(cloned)) {
      if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue())) {
        auto resultTy = cast<ShapedType>(constant.getResult().getType());
        constant->setAttr("value", dense.reshape(resultTy));
      }
    }
    return mapping.lookup(value);
  };
  return cloneValue(root);
}

class BlockedToSME : public mlir::RewritePattern {
  int computeCapability;

public:
  BlockedToSME(MLIRContext *context, int computeCapability)
      : RewritePattern(LoadOp::getOperationName(), 1, context),
        computeCapability(computeCapability) {}
  mlir::LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability <= 70)
      return failure();
    auto loadOp = dyn_cast<LoadOp>(op);
    // Idempotency guard: a load whose result is already SME-encoded was
    // produced by this pattern, so do not re-process it. Explicit
    // tl.load(stride=...) starts as a normal blocked load with inputStride, and
    // should still be converted below.
    if (auto retTy =
            mlir::dyn_cast<RankedTensorType>(loadOp.getResult().getType()))
      if (auto enc = mlir::dyn_cast<BlockedEncodingAttr>(retTy.getEncoding()))
        if (enc.getIsSme())
          return failure();
    // only use sme for dot_load
    if (loadOp.getResult().use_empty())
      return failure();

    Operation *use = *loadOp.getResult().getUsers().begin();
    LocalAllocOp localAlloc = nullptr;
    LocalLoadOp localLoad = nullptr;
    ConvertLayoutOp convertLayout = nullptr;
    RankedTensorType tensorType;
    DotOperandEncodingAttr dotOpEnc;

    // Current v3.6 dot lowering may already have routed the load through shared
    // memory before this pass:
    //   load -> local_alloc -> local_load(dot_operand<useSme>) -> dot
    // In that shape, make local_alloc consume the newly-created SME load below.
    localAlloc = llvm::dyn_cast<LocalAllocOp>(use);
    if (localAlloc) {
      if (!localAlloc.getResult().hasOneUse())
        return failure();
      localLoad = llvm::dyn_cast<LocalLoadOp>(
          *localAlloc.getResult().getUsers().begin());
      if (!localLoad)
        return failure();
      tensorType =
          mlir::dyn_cast<RankedTensorType>(localLoad.getResult().getType());
      if (!tensorType)
        return failure();
      dotOpEnc =
          mlir::dyn_cast<DotOperandEncodingAttr>(tensorType.getEncoding());
    } else {
      while (use) {
        if (use->getNumResults() != 1 || use->getResult(0).use_empty())
          break;
        auto useTensorType =
            mlir::dyn_cast<RankedTensorType>(use->getResult(0).getType());
        if (!useTensorType ||
            !mlir::isa<SwizzledSharedEncodingAttr>(useTensorType.getEncoding()))
          break;
        use = *use->getResult(0).getUsers().begin();
      }

      convertLayout = llvm::dyn_cast<ConvertLayoutOp>(use);
      if (!convertLayout)
        return failure();
      tensorType =
          mlir::dyn_cast<RankedTensorType>(convertLayout.getResult().getType());
      if (!tensorType)
        return failure();
      dotOpEnc =
          mlir::dyn_cast<DotOperandEncodingAttr>(tensorType.getEncoding());
    }

    // Transposed dot operand. After AccelerateMatmul + RemoveLayoutConversions
    // a transposed SME operand shows up as a *register* transpose:
    //   load -> convert(load -> #linear) -> trans(#linear -> dot_op<useSme>)
    // which never reaches SME hardware. Detect it here so the load is routed
    // through SME shared memory and the transpose is applied on the shared
    // memdesc (lowered via LinearLayout), matching the non-transposed SME path.
    TransOp transOp = nullptr;
    if (!localAlloc && !dotOpEnc && loadOp.getResult().hasOneUse() &&
        convertLayout.getResult().hasOneUse()) {
      if (auto t =
              llvm::dyn_cast<TransOp>(*convertLayout->getUsers().begin())) {
        if (t.getOrder() == ArrayRef<int32_t>({1, 0})) {
          if (auto tTy =
                  mlir::dyn_cast<RankedTensorType>(t.getResult().getType())) {
            if (auto de =
                    mlir::dyn_cast<DotOperandEncodingAttr>(tTy.getEncoding())) {
              dotOpEnc = de;
              tensorType = tTy;
              transOp = t;
            }
          }
        }
      }
    }

    // Determine whether sme can be used
    if (!dotOpEnc || (dotOpEnc.getUseSme() == 0))
      return failure();
    auto mmaOpEnc =
        mlir::dyn_cast<IluvatarMmaEncodingAttr>(dotOpEnc.getParent());
    if (!mmaOpEnc)
      return failure();
    auto oldRetType =
        mlir::dyn_cast<RankedTensorType>(loadOp.getResult().getType());
    auto oldRetEncod =
        mlir::dyn_cast<BlockedEncodingAttr>(oldRetType.getEncoding());
    if (!oldRetEncod)
      return failure();
    bool isI8RowXfb8 = oldRetType.getElementType().isInteger(8) &&
                       oldRetEncod.getOrder()[0] != 0;
    if (isI8RowXfb8 && transOp &&
        (loadOp.getMask() || loadOp.getOther() ||
         !loadOp.getBoundaryCheck().empty())) {
      // Transpose sinking below is elementwise, but SME itself does not support
      // non-uniform predication. Keep unsupported boundary cases on the normal
      // load path.
      return failure();
    }

    // Prefer explicit tl.load(stride=...). Auto SME falls back to recovering
    // the row stride from address arithmetic.
    Value in_stride = loadOp.getInputStride();
    if (!in_stride)
      in_stride = getSmeStride(loadOp, rewriter);
    if (!in_stride)
      return failure();
    if (in_stride.getType().isInteger(64))
      in_stride = arith::TruncIOp::create(rewriter, loadOp.getLoc(),
                                          rewriter.getI32Type(), in_stride);

    // Use the load shape (untransposed) for the SME blocked load. For the
    // transposed-operand path tensorType is the transposed dot operand, so use
    // oldRetType instead.
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = lookupNumWarps(mod);
    int numCTAs = TritonGPUDialect::getNumCTAs(mod);

    if (isI8RowXfb8 && transOp) {
      // Sink the value transpose through the load:
      //
      //   trans(load(ptr[N,K] row-major))
      //     -> load(trans(ptr)[K,N] col-major)
      //
      // The transposed pointer is a genuine col-major view of the same
      // row-contiguous storage: addr(k,n) = k + n*K. This lets G2S use the
      // GF(2)-linear colxfb8 layout and avoids exposing the non-linear rowxfb8
      // surrogate through memdesc_trans.
      auto loc = loadOp.getLoc();
      auto *ctx = oldRetType.getContext();
      SmallVector<int32_t> order({1, 0});
      Value transPtr = TransOp::create(rewriter, loc, loadOp.getPtr(), order);
      auto transPtrTy = mlir::cast<RankedTensorType>(transPtr.getType());
      auto transPtrEnc =
          mlir::cast<BlockedEncodingAttr>(transPtrTy.getEncoding());

      // Rebuild the SME encoding from the transposed shape/order so the SME
      // warp distribution is computed for the col tile (64x16 for int8).
      auto transSmeEnc = BlockedEncodingAttr::get(
          ctx, true, numWarps, oldRetType.getElementType(),
          transPtrTy.getShape(), transPtrEnc.getOrder(),
          transPtrEnc.getSizePerThread(), transPtrEnc.getThreadsPerWarp(),
          transPtrEnc.getWarpsPerCTA(), numCTAs);
      auto smePtrTy = RankedTensorType::get(
          transPtrTy.getShape(), transPtrTy.getElementType(), transSmeEnc);
      Value smePtr = ConvertLayoutOp::create(rewriter, loc, smePtrTy, transPtr);
      auto transLoadTy = RankedTensorType::get(
          transPtrTy.getShape(), oldRetType.getElementType(), transSmeEnc);
      auto transLoad =
          LoadOp::create(rewriter, loc, transLoadTy, smePtr, Value(), Value(),
                         loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
                         loadOp.getCache(), loadOp.getEvict(),
                         loadOp.getIsVolatile(), in_stride);

      auto sharedMemorySpace = SharedMemorySpaceAttr::get(ctx);
      auto sharedOrder = getOrderForMemory(transLoadTy);
      auto ctaLayout = getCTALayout(transSmeEnc);
      auto sharedEnc = SwizzledSharedEncodingAttr::get(
          ctx, dotOpEnc, transLoadTy.getShape(), sharedOrder, ctaLayout,
          oldRetType.getElementType(), /*needTrans=*/false);
      auto allocTy =
          MemDescType::get(transLoadTy.getShape(), oldRetType.getElementType(),
                           sharedEnc, sharedMemorySpace);
      auto alloc =
          LocalAllocOp::create(rewriter, loc, allocTy, transLoad.getResult());
      auto localLoad =
          LocalLoadOp::create(rewriter, loc, tensorType, alloc.getResult());

      rewriter.replaceOp(transOp, localLoad.getResult());
      rewriter.eraseOp(convertLayout);
      rewriter.eraseOp(op);
      return success();
    }

    BlockedEncodingAttr smeEnc;
    smeEnc = BlockedEncodingAttr::get(
        oldRetType.getContext(), true, false, numWarps,
        oldRetType.getElementType(), retShape, oldRetEncod.getOrder(),
        oldRetEncod.getSizePerThread(), oldRetEncod.getThreadsPerWarp(),
        oldRetEncod.getWarpsPerCTA(), numCTAs);

    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), smeEnc);
    // loadOp need operand encoding equal result encoding
    // ptr operand
    Value ptr = loadOp.getPtr();
    auto oldPtrType = mlir::dyn_cast<RankedTensorType>(ptr.getType());
    auto newPtrEncoding = BlockedEncodingAttr::get(
        oldPtrType.getContext(), true, false, numWarps,
        oldRetType.getElementType(), oldPtrType.getShape(),
        oldRetEncod.getOrder(), oldRetEncod.getSizePerThread(),
        oldRetEncod.getThreadsPerWarp(), oldRetEncod.getWarpsPerCTA(), numCTAs);
    auto newPtrType = RankedTensorType::get(
        oldPtrType.getShape(), oldPtrType.getElementType(), newPtrEncoding);
    ptr = ConvertLayoutOp::create(rewriter, ptr.getLoc(), newPtrType, ptr);
    // LoadOp requires ptr/mask/other/result encodings to match. The cloned mask
    // DAG carries smeMask=true up to this final conversion; ptr-side ranges
    // carry isSme=true, smeMask=false.
    Value mask = loadOp.getMask();
    if (mask) {
      // A typical K-tail mask and its pointer share the same make_range. Clone
      // the pure mask DAG before assigning the SME encoding so the pointer-only
      // make_range canonicalization cannot also zero the predicate range.
      mask = cloneMaskComputation(mask, rewriter);
      auto oldMaskType = mlir::dyn_cast<RankedTensorType>(mask.getType());
      auto newMaskEncoding = BlockedEncodingAttr::get(
          oldMaskType.getContext(), true, false, numWarps,
          oldRetType.getElementType(), oldMaskType.getShape(),
          oldRetEncod.getOrder(), oldRetEncod.getSizePerThread(),
          oldRetEncod.getThreadsPerWarp(), oldRetEncod.getWarpsPerCTA(),
          numCTAs);
      auto newMaskType =
          RankedTensorType::get(oldMaskType.getShape(),
                                oldMaskType.getElementType(), newMaskEncoding);
      mask =
          ConvertLayoutOp::create(rewriter, mask.getLoc(), newMaskType, mask);
    }
    Value other = loadOp.getOther();
    if (other) {
      auto oldOtherType = mlir::dyn_cast<RankedTensorType>(other.getType());
      auto newOtherEncoding = BlockedEncodingAttr::get(
          oldOtherType.getContext(), true, false, numWarps,
          oldRetType.getElementType(), oldOtherType.getShape(),
          oldRetEncod.getOrder(), oldRetEncod.getSizePerThread(),
          oldRetEncod.getThreadsPerWarp(), oldRetEncod.getWarpsPerCTA(),
          numCTAs);
      auto newOtherType = RankedTensorType::get(oldOtherType.getShape(),
                                                oldOtherType.getElementType(),
                                                newOtherEncoding);
      other = ConvertLayoutOp::create(rewriter, other.getLoc(), newOtherType,
                                      other);
    }

    auto newload =
        LoadOp::create(rewriter, loadOp.getLoc(), newRetType, ptr, mask, other,
                       loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
                       loadOp.getCache(), loadOp.getEvict(),
                       loadOp.getIsVolatile(), in_stride);

    if (localAlloc) {
      localAlloc.getSrcMutable().assign(newload.getResult());
      rewriter.eraseOp(op);
      return success();
    }

    if (transOp) {
      // Route the SME load through shared memory and transpose on the shared
      // memdesc:
      //   local_alloc(sme_load) #shared(useTcu)
      //     -> memdesc_trans -> local_load -> dot_op<useSme>
      // The SME global->shared store fires because local_alloc directly
      // consumes the isSme LoadOp; the transpose is realized by MemDescTransOp,
      // whose useTcu inferTransOpEncoding produces the exact-transpose
      // LinearLayout so local_load reads the transposed data correctly.
      auto loc = loadOp.getLoc();
      auto *ctx = oldRetType.getContext();
      auto sharedMemorySpace = SharedMemorySpaceAttr::get(ctx);
      auto sharedOrder = getOrderForMemory(oldRetType);
      auto ctaLayout = getCTALayout(oldRetType.getEncoding());
      auto sharedEnc = SwizzledSharedEncodingAttr::get(
          ctx, dotOpEnc, oldRetType.getShape(), sharedOrder, ctaLayout,
          oldRetType.getElementType(), /*needTrans=*/true);
      auto allocTy =
          MemDescType::get(oldRetType.getShape(), oldRetType.getElementType(),
                           sharedEnc, sharedMemorySpace);
      auto alloc =
          LocalAllocOp::create(rewriter, loc, allocTy, newload.getResult());
      auto memTrans = MemDescTransOp::create(rewriter, loc, alloc,
                                             ArrayRef<int32_t>({1, 0}));
      auto localLoad =
          LocalLoadOp::create(rewriter, loc, tensorType, memTrans.getResult());
      rewriter.replaceOp(transOp, localLoad.getResult());
      rewriter.eraseOp(convertLayout);
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, oldRetType,
                                                 newload.getResult());
    return success();
  }
};
} // namespace

#define GEN_PASS_DECL_TRITONILUVATARGPUSMELOAD
#define GEN_PASS_DEF_TRITONILUVATARGPUSMELOAD
#include "TritonILUVATARGPUTransforms/Passes.h.inc"

struct TritonILUVATARGPUSmeLoadPass
    : public impl::TritonILUVATARGPUSmeLoadBase<TritonILUVATARGPUSmeLoadPass> {
  using Base = impl::TritonILUVATARGPUSmeLoadBase<TritonILUVATARGPUSmeLoadPass>;

  TritonILUVATARGPUSmeLoadPass() = default;
  explicit TritonILUVATARGPUSmeLoadPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<BlockedToSME>(context, this->computeCapability);
    if (mlir::applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir

namespace mlir {

std::unique_ptr<Pass>
createTritonILUVATARGPUSmeLoadPass(int computeCapability) {
  return std::make_unique<triton::gpu::TritonILUVATARGPUSmeLoadPass>(
      computeCapability);
}

} // namespace mlir
