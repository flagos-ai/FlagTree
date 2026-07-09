/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved.
 */
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cmath>
#include <memory>

#define DEBUG_TYPE "TritonMETAXGPUAccelerateMatmulPass"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {
using triton::DotOp;
using triton::gpu::BlockedEncodingAttr;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::SliceEncodingAttr;
using triton::gpu::SwizzledSharedEncodingAttr;

int computeCapabilityToMMAVersion(int computeCapability) {
  if (computeCapability < 70) {
    return 0;
  } else if (computeCapability < 80) {
    return 1;
  } else if (computeCapability < 90) {
    return 2;
  } else if (computeCapability < 100) {
    return 3;
  } else {
    assert(false && "computeCapability > 100 not supported");
    return 3;
  }
}

SmallVector<int64_t, 2> mmaVersionToShapePerWarp(int version) {
  if (version == 1)
    return {16, 16};
  else if (version == 2)
    return {16, 8};
  else if (version == 3)
    return {16, 16};
  else {
    assert(false && "version not supported");
    return {0, 0};
  }
}

SmallVector<unsigned, 2> warpsPerTileV2(triton::DotOp dotOp,
                                        const ArrayRef<int64_t> shape,
                                        int numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, {filter});
  for (Operation *op : slices)
    if (isa<triton::DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, 8};
  bool changed = false;
  do {
    changed = false;
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

SmallVector<unsigned, 2> warpsPerTileMACA(triton::DotOp dotOp,
                                          const ArrayRef<int64_t> shape,
                                          int numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, {filter});
  for (Operation *op : slices)
    if (isa<triton::DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, 16};
  bool changed = false;
  do {
    changed = false;
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

class BlockedToMMA : public mlir::RewritePattern {
  int computeCapability;
  mutable int mmaV1Counter{}; // used to generate ID for MMAv1 encoding
  int dotCnt;
  int numStages;
  int disablePrefetch;
  int storeCoalesce;

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability, int dotCnt,
               int numStages, bool disablePrefetch, bool storeCoalesce)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 2, context),
        computeCapability(computeCapability), dotCnt(dotCnt),
        numStages(numStages), disablePrefetch(disablePrefetch),
        storeCoalesce(storeCoalesce) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    auto dotOp = cast<triton::DotOp>(op);
    // TODO: Check data-types and SM compatibility
    auto oldRetType = cast<RankedTensorType>(dotOp.getResult().getType());
    if (!oldRetType.getEncoding() ||
        isa<triton::gpu::MACAMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // for FMA, should retain the blocked layout.
    int versionMajor = computeCapabilityToMMAVersion(computeCapability);
    if (!supportMMA(dotOp, versionMajor, computeCapability % 10))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    LDBG("mod before BlockedToMMA: " << mod);
    int numWarps = triton::gpu::lookupNumWarps(dotOp);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    // enable opt maca layout or not
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    int m = aTensorTy.getShape()[0];
    int k = aTensorTy.getShape()[1];
    int n = bTensorTy.getShape()[1];
    auto elementTy = aTensorTy.getElementType();
    bool enableTf32 = dotOp.getInputPrecision() == tt::InputPrecision::TF32;

    auto parentOp = dotOp->getParentOp();
    // enable MACA's optimized MMA layout, currently must satisfy:
    // a) only one dot op in the compiled kernel
    // b) num_stage == 2
    // c) dot op contained in a for loop
    // d) not enable TRITON_DISABLE_MACA_OPT_MMA = 1
    bool enableOptMMA =
        this->dotCnt == 1 && (4 >= this->numStages) ||
        (this->numStages >= 2) && isa<scf::ForOp>(parentOp) &&
            std::getenv("TRITON_DISABLE_MACA_OPT_MMA") == nullptr;

    triton::gpu::MACAMmaEncodingAttr mmaEnc;
    SmallVector<unsigned> aorder, border;
    auto elemATy = oldAType.getElementType();
    auto elemBTy = oldBType.getElementType();
    bool enableALdsTrans, enableBLdsTrans;
    aorder = triton::gpu::getOrder(oldAType);
    border = triton::gpu::getOrder(oldBType);
    if (versionMajor == 2) {
      auto elemsPerThread =
          getDefaultElemsPerThread(elementTy, enableTf32, computeCapability);
      auto warpsPerTile = warpsPerTileMACA(dotOp, retShape, numWarps);
      int versionMajor_ = versionMajor;
      int versionMinor_ = computeCapability % 10;
      if (enableOptMMA) {
        Operation *aOp = a.getDefiningOp();
        Operation *bOp = b.getDefiningOp();
        auto aCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(aOp);
        auto bCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(bOp);
        if (aCvtOp && bCvtOp) {
          // when load row major and trans to col major
          //
          // Before AccelerateMatmul Pass:
          //
          //   cvt_pre block(order=(1,0)) -> shared(order=(0,1))
          //   trans shared(order=(0,1)) -> shared1(order=(1,0))
          //   cvt shared1(order=(1,0)) -> dot(order=(1,0))
          //
          // however order=(1,0) is not final order, but (0, 1)
          //
          // After OptimizeDotOperand Pass:
          //
          //   cvt_pre block(order=(1,0)) -> shared(order=(1,0))
          //   trans shared(order=(1,0)) -> shared1(order=(0,1))
          //   cvt shared1(order=(0,1)) -> dot(order=(0,1))
          //
          //
          // So in AccelerateMatmul Pass we reverse layout order
          // of cvt_pre input.
          aorder = getOrder(aCvtOp);
          border = getOrder(bCvtOp);
          if (!(aorder.empty() || border.empty())) {
            llvm::SmallVector<int, 4> tile({m, n, k, numWarps});
            int version = -1;
            bool isOpt =
                updateLayout(elemsPerThread, warpsPerTile, tile, version,
                             numWarps, enableTf32, aTensorTy.getElementType(),
                             aorder, border, this->disablePrefetch, false,
                             this->storeCoalesce, computeCapability);
            if (isOpt) {
              mod->setAttr(
                  "use.opt.maca.mma",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(mod.getContext(), 32), 1));
            }
          }
        }
      }
      enableALdsTrans = getIfLdsTrans(elemsPerThread, versionMajor_,
                                      versionMinor_, aorder, true, elemATy);
      enableBLdsTrans = getIfLdsTrans(elemsPerThread, versionMajor_,
                                      versionMinor_, border, false, elemBTy);
      auto elemStride = triton::gpu::getLdsTransVec(elemATy);
      SmallVector<unsigned> elementsStride = {1, 1};
      if (enableALdsTrans)
        elementsStride[0] = elemStride;
      if (enableBLdsTrans)
        elementsStride[1] = elemStride;
      mmaEnc = triton::gpu::MACAMmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor_, versionMinor_, warpsPerTile,
          elemsPerThread, 0, enableALdsTrans, enableBLdsTrans, elementsStride);
    } else {
      llvm_unreachable("Mma layout only supports versionMajor in {2}");
    }
    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);

    auto oldAOrder =
        cast<triton::gpu::BlockedEncodingAttr>(
            cast<triton::gpu::DotOperandEncodingAttr>(oldAType.getEncoding())
                .getParent())
            .getOrder();
    auto oldBOrder =
        cast<triton::gpu::BlockedEncodingAttr>(
            cast<triton::gpu::DotOperandEncodingAttr>(oldBType.getEncoding())
                .getParent())
            .getOrder();

    auto newAEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(),
        oldAType.getElementType());
    auto newBEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(),
        oldBType.getElementType());

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(), newAEncoding);
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(), newBEncoding);

    a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<triton::DotOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, dotOp.getInputPrecision());

    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, oldRetType, newDot.getResult());
    LDBG("mod after BlockedToMMA: " << mod);
    return success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

class TritonMETAXGPUAccelerateMatmulPass
    : public TritonMETAXGPUAccelerateMatmulBase<
          TritonMETAXGPUAccelerateMatmulPass> {
public:
  TritonMETAXGPUAccelerateMatmulPass() = default;
  TritonMETAXGPUAccelerateMatmulPass(int numStages, bool disablePrefetch,
                                     bool storeCoalesce,
                                     int computeCapability = 80) {
    this->computeCapability = computeCapability;
    this->numStages = numStages;
    this->disablePrefetch = disablePrefetch;
    this->storeCoalesce = storeCoalesce;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    m->setAttr(
        "use.opt.maca.mma",
        mlir::IntegerAttr::get(mlir::IntegerType::get(m.getContext(), 32), 0));

    mlir::RewritePatternSet patterns(context);
    // TODO: support chain dot & multi dot
    patterns.add<::BlockedToMMA>(context, computeCapability, /*dot_cut=*/1,
                                 numStages, disablePrefetch, storeCoalesce);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonMETAXGPUAccelerateMatmulPass(
    int numStages, bool disablePrefetch, bool storeCoalesce,
    int computeCapability) {
  return std::make_unique<TritonMETAXGPUAccelerateMatmulPass>(
      numStages, disablePrefetch, storeCoalesce, computeCapability);
}
