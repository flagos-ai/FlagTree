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
            if (dataType.isInteger(32)) {
              res = splatOp.getSrc();
              break;
            }
            if (dataType.isInteger(64)) {
              res = arith::TruncIOp::create(rewriter, rewriter.getUnknownLoc(),
                                            rewriter.getI32Type(),
                                            splatOp.getSrc());
              break;
            }
          }
        }
      }
    }
  }
  return res;
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
    // if have stride, can skip
    if (loadOp.getInputStride())
      return failure();
    // if use Mask,  can not use sme
    if (loadOp.getMask())
      return failure();
    // only use sme for dot_load
    if (loadOp.getResult().use_empty())
      return failure();

    Operation *use = *loadOp.getResult().getUsers().begin();
    while (use) {
      if (use->getNumResults() != 1 || use->getResult(0).use_empty())
        break;
      auto tensorType =
          mlir::dyn_cast<RankedTensorType>(use->getResult(0).getType());
      if (!tensorType ||
          !mlir::isa<SwizzledSharedEncodingAttr>(tensorType.getEncoding()))
        break;
      use = *use->getResult(0).getUsers().begin();
    }

    auto convertLayout = llvm::dyn_cast<ConvertLayoutOp>(use);
    if (!convertLayout)
      return failure();
    auto tensorType =
        mlir::dyn_cast<RankedTensorType>(convertLayout.getResult().getType());
    if (!tensorType)
      return failure();
    auto dotOpEnc =
        mlir::dyn_cast<DotOperandEncodingAttr>(tensorType.getEncoding());

    // Transposed dot operand. After AccelerateMatmul + RemoveLayoutConversions
    // a transposed SME operand shows up as a *register* transpose:
    //   load -> convert(load -> #linear) -> trans(#linear -> dot_op<useSme>)
    // which never reaches SME hardware. Detect it here so the load is routed
    // through SME shared memory and the transpose is applied on the shared
    // memdesc (lowered via LinearLayout), matching the non-transposed SME path.
    TransOp transOp = nullptr;
    if (!dotOpEnc && loadOp.getResult().hasOneUse() &&
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

    // find matrix store_major(row or col) stride
    Value in_stride = getSmeStride(loadOp, rewriter);
    if (!in_stride)
      assert(false && "can not find tensor Stride, Please check ttgir or dot "
                      "logic in User code");

    // Use the load shape (untransposed) for the SME blocked load. For the
    // transposed-operand path tensorType is the transposed dot operand, so use
    // oldRetType instead.
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = lookupNumWarps(mod);
    int numCTAs = TritonGPUDialect::getNumCTAs(mod);

    BlockedEncodingAttr smeEnc;
    smeEnc = BlockedEncodingAttr::get(
        oldRetType.getContext(), true, numWarps, oldRetType.getElementType(),
        retShape, oldRetEncod.getOrder(), oldRetEncod.getSizePerThread(),
        oldRetEncod.getThreadsPerWarp(), oldRetEncod.getWarpsPerCTA(), numCTAs);

    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), smeEnc);
    // loadOp need operand encoding equal result encoding
    // ptr operand
    Value ptr = loadOp.getPtr();
    auto oldPtrType = mlir::dyn_cast<RankedTensorType>(ptr.getType());
    auto newPtrEncoding = BlockedEncodingAttr::get(
        oldPtrType.getContext(), true, numWarps, oldRetType.getElementType(),
        oldPtrType.getShape(), oldRetEncod.getOrder(),
        oldRetEncod.getSizePerThread(), oldRetEncod.getThreadsPerWarp(),
        oldRetEncod.getWarpsPerCTA(), numCTAs);
    auto newPtrType = RankedTensorType::get(
        oldPtrType.getShape(), oldPtrType.getElementType(), newPtrEncoding);
    ptr = ConvertLayoutOp::create(rewriter, ptr.getLoc(), newPtrType, ptr);
    // mask operand
    Value mask = loadOp.getMask();
    if (mask) {
      auto oldMaskType = mlir::dyn_cast<RankedTensorType>(mask.getType());
      auto newMaskEncoding = BlockedEncodingAttr::get(
          oldMaskType.getContext(), true, numWarps, oldRetType.getElementType(),
          oldMaskType.getShape(), oldRetEncod.getOrder(),
          oldRetEncod.getSizePerThread(), oldRetEncod.getThreadsPerWarp(),
          oldRetEncod.getWarpsPerCTA(), numCTAs);
      auto newMaskType =
          RankedTensorType::get(oldMaskType.getShape(),
                                oldMaskType.getElementType(), newMaskEncoding);
      mask =
          ConvertLayoutOp::create(rewriter, mask.getLoc(), newMaskType, mask);
    }
    // other operand
    Value other = loadOp.getOther();
    if (other) {
      auto oldOtherType = mlir::dyn_cast<RankedTensorType>(other.getType());
      auto newOtherEncoding = BlockedEncodingAttr::get(
          oldOtherType.getContext(), true, numWarps,
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
