#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "TritonHCUGPUTransforms/MfmaGroup.h"
#include "TritonHCUGPUTransforms/MlsGroup.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "Utility.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

SmallVector<unsigned> getMatrixLoadTensorOrder(triton::MatrixLoadOp matrixOp, int opIdx) {
  auto strides = matrixOp.getStrides();
  assert(strides.size() == 2);

  auto strideKIdx = opIdx == 0 ? 1 : 0;
  auto strideK = strides[strideKIdx];

  bool kMajor = false;
  if (auto constantOp = strideK.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      kMajor = attr.getValue().isOne();
    }
  }

  if (opIdx == 0) {
    return {kMajor, !kMajor};
  }
  else {
    return {!kMajor, kMajor};
  }
}

// Chooses a proper MLS instruction
FailureOr<MlsInsn> chooseMlsInstruction(tt::DotOp dot, int opIdx,
                                        triton::MatrixLoadOp matrixOp, bool kMajor,
                                        int version, int numWarps) {
  // get mfma encoding info
  auto aType = dot.getA().getType();
  auto bType = dot.getB().getType();
  auto dType = dot.getResult().getType();
  auto mfmaEncoding = dyn_cast<ttg::AMDMfmaEncodingAttr>(dType.getEncoding());

  // get the mls shape info
  auto rank = matrixOp.getType().getRank();
  auto elemType = matrixOp.getType().getElementType();
  auto bitwidth = elemType.getIntOrFloatBitWidth();
  auto blockK = matrixOp.getType().getShape()[(rank - 1) - opIdx];
  auto blockNonK = matrixOp.getType().getShape()[(rank - 2) + opIdx];

  // Determine the kDim based on the matrix load and numWarps Info.
  unsigned kTile = 0;
  unsigned nonKTile = opIdx == 0 ? mfmaEncoding.getMfmaTile()[0] : mfmaEncoding.getMfmaTile()[1];
  if (bitwidth == 16) {
    kTile = kMajor ? (blockK >= 64 ? 64 : 32) : 16;
    if (kTile == 64 && kTile * nonKTile * numWarps > blockK * blockNonK)
      kTile = 32;
  } else if (bitwidth == 8) {
    kTile = kMajor ? (blockK >= 128 ? 128 : 64) : 32;
    if (kTile == 128 && kTile * nonKTile * numWarps > blockK * blockNonK)
      kTile = 64;
  }
  auto altKind = MlsInterleaveKind::InterleaveNone;

  // select the mls insn
  auto maybeMlsInsn = MlsInsn::selectOrGetMlsInsn(nonKTile, kTile, bitwidth,
                                                  opIdx, kMajor, altKind, version);
  if (failed(maybeMlsInsn))
    llvm::report_fatal_error("No match found in MLS database\n");

  return maybeMlsInsn;
}

SmallVector<unsigned>
warpsPerCTAMatrixLoad(ArrayRef<int64_t> shape, ArrayRef<unsigned> shapePerWarp, ArrayRef<unsigned> order, int numWarps) {
  unsigned rank = shape.size();
  SmallVector<unsigned> warpsPerCTA(rank);

  unsigned remainingWarps = numWarps;
  unsigned prevWarps = 1;

  // starting from the contiguous dimension
  for (unsigned d = 0; d < rank - 1; ++d) {
    unsigned i = order[d];
    unsigned maxWarpsInDim = std::max<unsigned>(1, static_cast<unsigned>(shape[i]) / shapePerWarp[i]);
    warpsPerCTA[i] = std::clamp<unsigned>(remainingWarps, 1, maxWarpsInDim);
    remainingWarps /= warpsPerCTA[i];
    prevWarps *= warpsPerCTA[i];
  }

  // Expand the last dimension to fill the remaining lanes and warps
  warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

  return warpsPerCTA;
}

Value convertAndCastTensor(OpBuilder &builder, Value value,
                           Attribute newEncoding) {
  auto loc = value.getLoc();
  auto oldType = cast<RankedTensorType>(value.getType());
  auto oldElemType = oldType.getElementType();

  assert(oldElemType.isIntOrFloat());
  auto convertedType =
      RankedTensorType::get(oldType.getShape(), oldElemType, newEncoding);

  Value convertedTensor =
      builder.create<ttg::ConvertLayoutOp>(loc, convertedType, value);

  return convertedTensor;
}

struct MlsEncodingInsertion : public OpRewritePattern<triton::MatrixLoadOp> {
public:
  MlsEncodingInsertion(MLIRContext *context)
      : OpRewritePattern(context, 1) {}

  LogicalResult matchAndRewrite(triton::MatrixLoadOp matrixOp, PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = matrixOp.getType();
    if (!oldRetType.getEncoding() || !isa<ttg::BlockedEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    auto maybeDotOpIdxPair = getDotOpIdxFromMatrixLoad(matrixOp);
    if (succeeded(maybeDotOpIdxPair)) {
      auto dotOp = maybeDotOpIdxPair.value().first;
      auto opIdx = maybeDotOpIdxPair.value().second;

      // 1. choose the mls instruction for matrixOp
      assert(isa<ttg::AMDMfmaEncodingAttr>(dotOp.getType().getEncoding()));
      auto order = getMatrixLoadTensorOrder(matrixOp, opIdx);
      bool kMajor = opIdx == 0 ? order[0] == 1 : order[0] == 0;
      auto numWarps = triton::gpu::lookupNumWarps(matrixOp);
      auto mlsInsn = chooseMlsInstruction(dotOp, opIdx, matrixOp, kMajor, 1, numWarps);

      // 2. warps per tile calculate
      auto mlsTile = mlsInsn->getMlsTile();
      auto shape = matrixOp.getResult().getType().getShape();
      auto elemBitWidth = mlsInsn->getElemBitWidth();
      auto altKind = mlsInsn->getAlt2Kind();
      auto version = mlsInsn->getMlsVersion();
      auto warpsPerCTA = warpsPerCTAMatrixLoad(shape, mlsTile, order, numWarps);

      // 3. create new matrix load op with new encoding and mls attr
      auto newEncoding = opIdx == 0 ? dotOp.getA().getType().getEncoding()
                                               : dotOp.getB().getType().getEncoding();
      if (dyn_cast<ttg::DotOperandEncodingAttr>(newEncoding).getKWidth() != mlsInsn->getDotLayoutKWidth()) {
        newEncoding = ttg::DotOperandEncodingAttr::get(matrixOp.getContext(),
                                                       opIdx,
                                                       dyn_cast<ttg::DotOperandEncodingAttr>(newEncoding).getParent(),
                                                       mlsInsn->getDotLayoutKWidth());
      }

      auto oldResultType = cast<RankedTensorType>(matrixOp.getResult().getType());
      auto newResultType = RankedTensorType::get(oldResultType.getShape(),
                                                                   oldResultType.getElementType(),
                                                                   newEncoding);

      auto newMatrixOp = rewriter.create<triton::MatrixLoadOp>(
                              matrixOp.getLoc(), newResultType,
                              matrixOp.getBase(), matrixOp.getShape(),
                              matrixOp.getStrides(), matrixOp.getTensorShape(), matrixOp.getIndices(),
                              matrixOp.getBoundaryCheck(), matrixOp.getCache(), matrixOp.getEvict(),
                              matrixOp.getIsVolatile());
      auto mlsEncoding = triton::hcugpu::MlsEncodingAttr::get(
                                                          matrixOp.getContext(), opIdx, mlsTile,
                                                          elemBitWidth, altKind, version,
                                                          order, warpsPerCTA);
      newMatrixOp->setAttr(triton::hcugpu::MlsEncodingAttr::getMnemonic(), mlsEncoding);

      // 4. replace the original op
      auto convertedTensor = convertAndCastTensor(rewriter, newMatrixOp.getResult(), matrixOp.getType().getEncoding());
      rewriter.replaceOp(matrixOp, convertedTensor);
    } else {// non-dot case, try to construct a mfma encoding for the matrix load
      // 1. choose the mls instruction for matrixOp
      unsigned opIdx = 0;
      auto order = getMatrixLoadTensorOrder(matrixOp, opIdx);
      bool kMajor = opIdx == 0 ? order[0] == 1 : order[0] == 0;

      auto rank = matrixOp.getType().getRank();
      auto elemType = matrixOp.getType().getElementType();
      auto bitwidth = elemType.getIntOrFloatBitWidth();
      auto blockK = matrixOp.getType().getShape()[(rank - 1) - opIdx];
      auto blockNonK = matrixOp.getType().getShape()[(rank - 2) + opIdx];
      auto numWarps = triton::gpu::lookupNumWarps(matrixOp);

      unsigned kTile = 0;
      unsigned nonKTile = 0;
      if (bitwidth == 16) {
        nonKTile = kMajor ? 16 : (blockNonK >= 64 ? 64 : 32);
        kTile = kMajor ? (blockK >= 64 ? 64 : 32) : 16;
        if (kTile == 64 && kTile * nonKTile * numWarps > blockK * blockNonK)
          kTile = 32;
      } else if (bitwidth == 8) {
        nonKTile = kMajor ? 16 : (blockNonK >= 128 ? 128 : 64);
        kTile = kMajor ? (blockK >= 128 ? 128 : 64) : 32;
        if (kTile == 128 && kTile * nonKTile * numWarps > blockK * blockNonK)
          kTile = 64;
      }

      auto altKind = MlsInterleaveKind::InterleaveNone;
      auto mlsInsn = MlsInsn::selectOrGetMlsInsn(nonKTile, kTile, bitwidth,
                                                                     opIdx, kMajor,
                                                                     altKind, 1);
      if (failed(mlsInsn))
        llvm::report_fatal_error("No match found in MLS database\n");

      assert(blockK % kTile == 0 && blockNonK % nonKTile == 0 && "M or N should be divisible by mDim or nDim");

      // 3. warps per tile calucate
      auto mlsTile = mlsInsn->getMlsTile();
      auto shape = matrixOp.getResult().getType().getShape();
      auto elemBitWidth = mlsInsn->getElemBitWidth();
      auto version = mlsInsn->getMlsVersion();
      auto warpsPerCTA = warpsPerCTAMatrixLoad(shape, mlsTile, order, numWarps);

      // 3. try to construct a mfma encoding for the matrix load
      constexpr unsigned mfmaVersion = 3;
      auto maybeMfmaInsn = MfmaIntrinsic::selectFor(matrixOp.getLoc(),
                                                    mfmaVersion,
                                                    16, 16,
                                                    blockK, elemType,
                                                    elemType,
                                                    false, false,
                                                    HCUISAFeature::MAMC_FP8);
      if (failed(maybeMfmaInsn))
        llvm::report_fatal_error("No match found in MFMA database\n");

      SmallVector<unsigned> tilesPerWarp = {1, 1};
      SmallVector<unsigned> mfmaTiles = {16, 16};
      SmallVector<unsigned> mfmaOrder = {1, 0};
      if (rank == 3) {
        tilesPerWarp.insert(tilesPerWarp.begin(), 1);
        mfmaOrder.insert(mfmaOrder.begin(), 2);
        mfmaTiles.insert(mfmaTiles.begin(), 1);
      }

      auto tileM = nonKTile;
      auto tileN = kTile;

      bool hasBatchDim = rank == 3;
      int mIndex = 0 + hasBatchDim;
      int nIndex = 1 + hasBatchDim;
      tilesPerWarp[mIndex] = tileM/maybeMfmaInsn->mDim;
      tilesPerWarp[nIndex] = tileN/maybeMfmaInsn->nDim;

      SmallVector<unsigned> warpsPerCTAMfma = warpsPerCTAMatrixLoad(shape, mfmaTiles, mfmaOrder, numWarps);
      unsigned mfmaElementBitWidth = elemType.isF64() ? 64 : 32;
      auto mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        matrixOp.getContext(),
        /*versionMajor*/ mfmaVersion,
        warpsPerCTAMfma,
        /*instrShape=*/SmallVector<unsigned>{maybeMfmaInsn->mDim, maybeMfmaInsn->nDim,
                                             maybeMfmaInsn->kDim},
        /*isTransposed=*/false,
        ttg::getCTALayout(matrixOp.getType().getEncoding()),
        tilesPerWarp,
        mfmaElementBitWidth,
        ttg::MmacLayout::INTERLEAVE_TRANSPOSE);
      auto dotAEncoding = ttg::DotOperandEncodingAttr::get(matrixOp.getContext(),
                                                          0, mfmaEnc,
                                                          maybeMfmaInsn->kBase);

      // 4. create new matrix load op with new encoding and mls attr
      auto newEncoding = dotAEncoding;

      auto oldResultType = cast<RankedTensorType>(matrixOp.getResult().getType());
      auto newResultType = RankedTensorType::get(oldResultType.getShape(),
                                                                   oldResultType.getElementType(),
                                                                   newEncoding);

      auto newMatrixOp = rewriter.create<triton::MatrixLoadOp>(
                              matrixOp.getLoc(), newResultType,
                              matrixOp.getBase(), matrixOp.getShape(),
                              matrixOp.getStrides(), matrixOp.getTensorShape(), matrixOp.getIndices(),
                              matrixOp.getBoundaryCheck(), matrixOp.getCache(), matrixOp.getEvict(),
                              matrixOp.getIsVolatile());
      auto mlsEncoding = triton::hcugpu::MlsEncodingAttr::get(
                                                          matrixOp.getContext(), opIdx, mlsTile,
                                                          elemBitWidth, static_cast<unsigned>(altKind), version,
                                                          order, warpsPerCTA);
      newMatrixOp->setAttr(triton::hcugpu::MlsEncodingAttr::getMnemonic(), mlsEncoding);

      // 5. replace the original op
      auto convertedTensor = convertAndCastTensor(rewriter,
                                                        newMatrixOp.getResult(),
                                                        matrixOp.getType().getEncoding());
      rewriter.replaceOp(matrixOp, convertedTensor);
    }

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUGPUMLSENCODINGINSERTION
#include "TritonHCUGPUTransforms/Passes.h.inc"

class TritonHCUGPUMlsEncodingInsertionPass
    : public impl::TritonHCUGPUMlsEncodingInsertionBase<
          TritonHCUGPUMlsEncodingInsertionPass> {
public:
  using impl::TritonHCUGPUMlsEncodingInsertionBase<
      TritonHCUGPUMlsEncodingInsertionPass>::TritonHCUGPUMlsEncodingInsertionBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::RewritePatternSet patterns(context);
    patterns.add<MlsEncodingInsertion>(context);
    if (mlir::applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
