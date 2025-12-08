/*
 * Copyright 2020 - 2022 Enflame.All Rights Reserved.
 *
 */

#include "Conversion/TritonToGCU/TritonToGCUPass.h"

#include "Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
namespace mlir {
#define GEN_PASS_DEF_TRITONGCUDOTLAYOUTOPTIMIZEPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "triton-gcu-dot-layout-optimize"
namespace {
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

struct TritonGCUDotLayoutOptimizePass
    : public mlir::impl::TritonGCUDotLayoutOptimizePassBase<
          TritonGCUDotLayoutOptimizePass> {
  using Base::Base;

  void runOnOperation() override;
  void RefineDotLayout();
  void reWriteDotLayout(triton::DotOp op);
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TritonDialect, mlir::triton::gcu::TritonGCUDialect>();
  }
};

/* fit for gcu dot layout*/
void TritonGCUDotLayoutOptimizePass::reWriteDotLayout(triton::DotOp dot) {
  auto loc = dot.getLoc();
  auto retType = dyn_cast<RankedTensorType>(dot.getType());
  auto retShape = retType.getShape();
  int rank = retShape.size();
  auto retLayout = retType.getEncoding();
  if (!isa<triton::gpu::BlockedEncodingAttr>(retLayout)) {
    LLVM_DEBUG({
      llvm::dbgs() << "bad dot for gcu layout \n";
      dot.dump();
    });
    return;
  }
  auto dOriBlockEncoding =
      dyn_cast<triton::gpu::BlockedEncodingAttr>(retLayout);
  SmallVector<unsigned> warpsPerCTA = dOriBlockEncoding.getWarpsPerCTA();
  assert((static_cast<unsigned>(rank) == warpsPerCTA.size()) &&
         "warpsPerCTA size is not equal to rank!\n");
  // check data type

  auto numWarpsPerCTA = product<unsigned>(warpsPerCTA);
  auto totalElement = product<int64_t>(retShape);
  (void)totalElement;
  (void)numWarpsPerCTA;
  assert((totalElement >= numWarpsPerCTA) &&
         "hi your data is too little, please do't config so large NumberWarps "
         "!\n");
  // try to get alignment to acore
  auto dotOutElementType = retType.getElementType();
  Value inputA = dot.getA();
  Value inputB = dot.getB();
  Value inputC = dot.getC();
  auto tTypeA = dyn_cast<RankedTensorType>(inputA.getType());
  auto k = tTypeA.getShape()[1];
  auto dotM = retShape[rank - 2];
  auto dotN = retShape[rank - 1];
  auto dotSrcElementType = tTypeA.getElementType();
  auto totalMatmulWarp = warpsPerCTA[rank - 2] * warpsPerCTA[rank - 1];
  if (totalMatmulWarp == 1) {
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "refine gcu dot layout \n";
    dot.dump();
  });
  if (dotM < dotN) {
    if (dotN / dotM > totalMatmulWarp) {
      warpsPerCTA[rank - 2] = 1;
      warpsPerCTA[rank - 1] = totalMatmulWarp;
    } else {
      if (totalMatmulWarp == 8) {
        warpsPerCTA[rank - 2] = 2;
        warpsPerCTA[rank - 1] = 4;
      } else if (totalMatmulWarp == 4) {
        warpsPerCTA[rank - 2] = 2;
        warpsPerCTA[rank - 1] = 2;
      } else if (totalMatmulWarp == 2) {
        warpsPerCTA[rank - 2] = 1;
        warpsPerCTA[rank - 1] = 2;
      }
    }
  } else if (dotN < dotM) {
    if (dotM / dotN > totalMatmulWarp) {
      warpsPerCTA[rank - 2] = totalMatmulWarp;
      warpsPerCTA[rank - 1] = 1;
    } else {
      if (totalMatmulWarp == 8) {
        warpsPerCTA[rank - 2] = 4;
        warpsPerCTA[rank - 1] = 2;
      } else if (totalMatmulWarp == 4) {
        warpsPerCTA[rank - 2] = 2;
        warpsPerCTA[rank - 1] = 2;
      } else if (totalMatmulWarp == 2) {
        warpsPerCTA[rank - 2] = 2;
        warpsPerCTA[rank - 1] = 1;
      }
    }
  } else {
    if (totalMatmulWarp == 8) {
      warpsPerCTA[rank - 2] = 4;
      warpsPerCTA[rank - 1] = 2;
    } else if (totalMatmulWarp == 4) {
      warpsPerCTA[rank - 2] = 2;
      warpsPerCTA[rank - 1] = 2;
    } else if (totalMatmulWarp == 2) {
      warpsPerCTA[rank - 2] = 2;
      warpsPerCTA[rank - 1] = 1;
    }
  }
  // dot cost is high priority
  int64_t perWarpM = dotM / warpsPerCTA[rank - 2];
  int64_t perWarpN = dotN / warpsPerCTA[rank - 1];
  if (perWarpM < 1 || perWarpN < 1) {
    LLVM_DEBUG({
      llvm::dbgs() << "bad dot for gcu layout \n";
      dot.dump();
    });
    return;
  }
  // try best to match acore alignment and buffer size balance
  if (dotSrcElementType.isBF16() && dotOutElementType.isBF16()) {
    if (perWarpM < 32 && perWarpN > 64 && warpsPerCTA[rank - 2] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] / 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] * 2;
    } else if (perWarpN < 64 && perWarpM > 32 && warpsPerCTA[rank - 1] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] * 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] / 2;
    }

  } else if (dotSrcElementType.isF16() && dotOutElementType.isF16()) {
    if ((k % 32 == 0) && perWarpM < 32 && perWarpN > 128 &&
        warpsPerCTA[rank - 2] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] / 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] * 2;
    } else if ((k % 64 == 0) && perWarpM < 32 && perWarpN > 64 &&
               warpsPerCTA[rank - 2] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] / 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] * 2;
    } else if ((k % 32 == 0) && perWarpN < 64 && perWarpM > 128 &&
               warpsPerCTA[rank - 1] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] * 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] / 2;
    } else if ((k % 64 == 0) && perWarpN < 64 && perWarpM > 32 &&
               warpsPerCTA[rank - 1] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] * 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] / 2;
    }
  } else if (dotSrcElementType.isF32() && dotOutElementType.isF32()) {
    // acore
    if (perWarpM < 64 && perWarpN > 64 && warpsPerCTA[rank - 2] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] / 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] * 2;
    } else if (perWarpN < 64 && perWarpM > 64 && warpsPerCTA[rank - 1] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] * 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] / 2;
    }
  } else if (dotSrcElementType.isF16() && dotOutElementType.isF32()) {
    if (perWarpM < 32 && (k % 64 == 0) && perWarpN > 64 &&
        warpsPerCTA[rank - 2] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] / 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] * 2;
    } else if (perWarpM < 64 && (k % 32 == 0) && perWarpN > 128 &&
               warpsPerCTA[rank - 2] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] / 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] * 2;
    } else if (perWarpM < 128 && (k % 32 == 0) && perWarpN > 64 &&
               warpsPerCTA[rank - 2] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] / 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] * 2;
    } else if ((k % 64 == 0) && perWarpN < 64 && perWarpM > 32 &&
               warpsPerCTA[rank - 1] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] * 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] / 2;
    } else if ((k % 32 == 0) && perWarpN < 64 && perWarpM > 64 &&
               warpsPerCTA[rank - 1] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] * 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] / 2;
    }
  } else if (dotSrcElementType.isBF16() && dotOutElementType.isF32()) {
    // acore
    if (perWarpM < 32 && perWarpN > 64 && warpsPerCTA[rank - 2] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] / 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] * 2;
    } else if (perWarpN < 64 && perWarpM > 32 && warpsPerCTA[rank - 1] >= 2) {
      warpsPerCTA[rank - 2] = warpsPerCTA[rank - 2] * 2;
      warpsPerCTA[rank - 1] = warpsPerCTA[rank - 1] / 2;
    }
  }
  // TODO(xingxing): refine 400 if need
  SmallVector<unsigned> origonWarpsPerCTA = dOriBlockEncoding.getWarpsPerCTA();
  if (origonWarpsPerCTA == warpsPerCTA) {
    LLVM_DEBUG({
      llvm::dbgs() << "no need or no Opportunity to refine dot layout\n";
      dot.dump();
    });
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "hi  refine dot layout\n";
    dot.dump();
  });
  OpBuilder rewriter(dot);
  Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
      dot.getContext(),
      llvm::ArrayRef<unsigned>(dOriBlockEncoding.getSizePerThread()),
      llvm::ArrayRef<unsigned>(dOriBlockEncoding.getThreadsPerWarp()),
      llvm::ArrayRef<unsigned>(warpsPerCTA),
      llvm::ArrayRef<unsigned>(dOriBlockEncoding.getOrder()),
      dOriBlockEncoding.getCTALayout());
  // new A
  Attribute newAencoding = triton::gpu::DotOperandEncodingAttr::get(
      dot.getContext(), 0, dEncoding, tTypeA.getElementType());
  auto dstAType = RankedTensorType::get(tTypeA.getShape(),
                                        tTypeA.getElementType(), newAencoding);
  Value newA =
      rewriter.create<triton::gpu::ConvertLayoutOp>(loc, dstAType, inputA);
  // new B
  auto tTypeB = dyn_cast<RankedTensorType>(inputB.getType());
  Attribute newBencoding = triton::gpu::DotOperandEncodingAttr::get(
      dot.getContext(), 1, dEncoding, tTypeB.getElementType());
  auto dstBType = RankedTensorType::get(tTypeB.getShape(),
                                        tTypeB.getElementType(), newBencoding);
  Value newB =
      rewriter.create<triton::gpu::ConvertLayoutOp>(loc, dstBType, inputB);
  // new C
  auto tTypeC = dyn_cast<RankedTensorType>(inputC.getType());
  RankedTensorType dstCType = RankedTensorType::get(
      tTypeC.getShape(), tTypeC.getElementType(), dEncoding);
  auto newC =
      rewriter.create<triton::gpu::ConvertLayoutOp>(loc, dstCType, inputC);

  // new retType
  auto newRetType =
      RankedTensorType::get(retShape, retType.getElementType(), dEncoding);
  auto newDot = rewriter.create<triton::DotOp>(loc, newRetType, newA, newB,
                                               newC, dot.getInputPrecision(),
                                               dot.getMaxNumImpreciseAcc());
  auto newOp = newDot.getOperation();
  for (const NamedAttribute attr : dot->getAttrs())
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  auto newResult = rewriter.create<triton::gpu::ConvertLayoutOp>(
      loc, dot.getType(), newDot.getResult());
  dot.getResult().replaceAllUsesWith(newResult.getResult());
  dot.erase();
}

void TritonGCUDotLayoutOptimizePass::RefineDotLayout() {
  auto trionModule = getOperation();
  llvm::SmallVector<DotOp> dotList;
  // if had reduce or Scanop it's result is a slice layout skip which need some
  // case to trace performance
  // bool IsOnlyDotAnchor = true;
  // trionModule.walk([&](mlir::Operation *op) {
  //   if (isa<triton::ReduceOp>(op) || isa<triton::ScanOp>(op)) {
  //     IsOnlyDotAnchor = false;
  //     return;
  //   }
  // });
  // if (!IsOnlyDotAnchor) {
  //   LLVM_DEBUG(llvm::dbgs()
  //              << "had reduce or Scanop it's result is a slice layout skip "
  //                 " need some case to trace performance\n");
  //   return;
  // }
  dotList.clear();
  trionModule.walk([&](DotOp dot) { dotList.push_back(dot); });
  for (auto &dot : dotList) {
    auto retType = dyn_cast<RankedTensorType>(dot.getType());
    int rank = retType.getShape().size();
    if (rank > 2) {
      // need test case for 3D dot
      continue;
    }
    // refine dot layout if need
    reWriteDotLayout(dot);
  }
}

} // namespace
using namespace mlir;
void TritonGCUDotLayoutOptimizePass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "TritonGCUDotLayoutOptimizePass\n");
  RefineDotLayout();
}
