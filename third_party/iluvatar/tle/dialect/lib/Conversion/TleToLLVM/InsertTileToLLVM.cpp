#include "TleTileToLLVMUtils.h"

#include "IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;

namespace {
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::iluvatar_tle;
using namespace mlir::triton::iluvatar_tle;

static std::optional<int64_t> getStaticIndex(InsertTileOp op) {
  if (auto c = op->getOperand(2).getDefiningOp<arith::ConstantOp>())
    return cast<IntegerAttr>(c.getValue()).getInt();
  return std::nullopt;
}

static bool isCTATileAligned(InsertTileOp op, int64_t linearIndex) {
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto tileTy = cast<RankedTensorType>(op.getTile().getType());
  auto srcShape = srcTy.getShape();
  auto tileShape = tileTy.getShape();
  auto ctaTile = getShapePerCTATile(srcTy);
  int rank = srcShape.size();

  SmallVector<int64_t> logicalGrid(rank), tileCoords(rank);
  for (int i = 0; i < rank; ++i)
    logicalGrid[i] = srcShape[i] / tileShape[i];

  int64_t remain = linearIndex;
  for (int i = rank - 1; i >= 0; --i) {
    tileCoords[i] = remain % logicalGrid[i];
    remain /= logicalGrid[i];
  }

  for (int i = 0; i < rank; ++i) {
    int64_t off = tileCoords[i] * tileShape[i];
    if (tileShape[i] % static_cast<int64_t>(ctaTile[i]) != 0)
      return false;
    if (off % static_cast<int64_t>(ctaTile[i]) != 0)
      return false;
  }

  return true;
}

static LogicalResult
lowerInsertTileStatic(InsertTileOp op, InsertTileOp::Adaptor adaptor,
                      ConversionPatternRewriter &rewriter,
                      const LLVMTypeConverter *typeConverter, int64_t index) {
  Location loc = op->getLoc();
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto tileTy = cast<RankedTensorType>(op.getTile().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());

  auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
  auto tileVals = unpackLLElements(loc, adaptor.getTile(), rewriter);

  auto srcShape = srcTy.getShape();
  auto tileShape = tileTy.getShape();

  auto shapePerCTATile = getShapePerCTATile(srcTy);
  auto srcCTAShape = multiDimElementwise<int64_t, unsigned>(
      srcShape, shapePerCTATile, std::divides<unsigned>());
  auto tileCTAShape = multiDimElementwise<int64_t, unsigned>(
      tileShape, shapePerCTATile, std::divides<unsigned>());

  SmallVector<int64_t> logicalTileShape(tileShape.begin(), tileShape.end());
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (logicalTileShape[i] == 0 || srcShape[i] % logicalTileShape[i] != 0)
      return op.emitError("source shape must be divisible by tile shape");
    logicalGridShape[i] = srcShape[i] / logicalTileShape[i];
  }

  SmallVector<int64_t> logicalCoords(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = srcShape.size() - 1; i >= 0; --i) {
    logicalCoords[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  SmallVector<int64_t> elementCoords(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    elementCoords[i] = logicalCoords[i] * logicalTileShape[i];

  auto firstTileCoordinate = multiDimElementwise<int64_t, unsigned>(
      elementCoords, shapePerCTATile, std::divides<unsigned>());

  auto numCTATiles = std::accumulate(tileCTAShape.begin(), tileCTAShape.end(),
                                     1, std::multiplies<>());
  auto srcCTAOrder = getCTATileOrder(srcTy);
  auto tileCTAOrder = getCTATileOrder(tileTy);

  for (size_t d = 0; d < srcCTAShape.size(); ++d) {
    if (firstTileCoordinate[d] + tileCTAShape[d] > srcCTAShape[d])
      return op.emitError("tile write region out of source bounds");
  }

  unsigned totalSrcCTAs = std::accumulate(
      srcCTAShape.begin(), srcCTAShape.end(), 1u, std::multiplies<>());
  unsigned totalTileCTAs = std::accumulate(
      tileCTAShape.begin(), tileCTAShape.end(), 1u, std::multiplies<>());

  unsigned srcElemsPerThreadPerCTA =
      ttg::getTotalElemsPerThread(srcTy) / totalSrcCTAs;
  unsigned tileElemsPerThreadPerCTA =
      ttg::getTotalElemsPerThread(tileTy) / totalTileCTAs;
  if (srcElemsPerThreadPerCTA != tileElemsPerThreadPerCTA)
    return op.emitError("source/tile per-CTA elements per thread mismatch");

  SmallVector<Value> resultVals(srcVals.begin(), srcVals.end());
  for (size_t i = 0; i < numCTATiles; i++) {
    auto coordInTileTensor = tle::delinearize(i, tileCTAShape, tileCTAOrder);
    auto coordInSrcTensor = multiDimElementwise<unsigned, unsigned>(
        coordInTileTensor, firstTileCoordinate, std::plus<unsigned>());
    auto linearIdxInSrcTensor =
        tle::linearize(coordInSrcTensor, srcCTAShape, srcCTAOrder);
    auto linearIdxInTileTensor =
        tle::linearize(coordInTileTensor, tileCTAShape, tileCTAOrder);

    size_t srcStartIdx = linearIdxInSrcTensor * srcElemsPerThreadPerCTA;
    size_t tileStartIdx = linearIdxInTileTensor * tileElemsPerThreadPerCTA;
    if (srcStartIdx + srcElemsPerThreadPerCTA > resultVals.size() ||
        tileStartIdx + tileElemsPerThreadPerCTA > tileVals.size())
      return op.emitError("internal error: register index out of bounds");

    llvm::copy(
        ArrayRef<Value>(tileVals).slice(tileStartIdx, srcElemsPerThreadPerCTA),
        resultVals.begin() + srcStartIdx);
  }

  Value ret = packLLElements(loc, typeConverter, resultVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

static LogicalResult
lowerInsertTileViaSMEMDynamic(InsertTileOp op, InsertTileOp::Adaptor adaptor,
                              ConversionPatternRewriter &rewriter,
                              const LLVMTypeConverter *typeConverter,
                              const TargetInfoBase &targetInfo) {
  Location loc = op->getLoc();
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto tileTy = cast<RankedTensorType>(op.getTile().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto srcShape = srcTy.getShape();
  auto tileShape = tileTy.getShape();
  int rank = srcShape.size();

  MLIRContext *ctx = rewriter.getContext();
  auto i1Ty = rewriter.getIntegerType(1);
  auto i8Ty = rewriter.getIntegerType(8);
  auto i32Ty = rewriter.getIntegerType(32);
  Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
  if (!llvmElemTy)
    return op.emitError("SMEM path: failed to convert element type");
  int64_t elemBytes = llvmElemTy.getIntOrFloatBitWidth() / 8;

  auto srcOffsets = emitOffsetForLayout(srcTy.getEncoding(), srcTy);
  auto tileOffsets = emitOffsetForLayout(tileTy.getEncoding(), tileTy);
  unsigned srcElemsPerThread = ttg::getTotalElemsPerThread(srcTy);
  unsigned tileElemsPerThread = ttg::getTotalElemsPerThread(tileTy);
  if (srcOffsets.size() != srcElemsPerThread)
    return op.emitError("SMEM path: src offsets size mismatch");
  if (tileOffsets.size() != tileElemsPerThread)
    return op.emitError("SMEM path: tile offsets size mismatch");

  auto tileOrder = getCTATileOrder(tileTy);
  SmallVector<int64_t> smemStrides(rank, 0);
  {
    int64_t s = 1;
    for (int i = 0; i < rank; ++i) {
      unsigned dim = tileOrder[i];
      smemStrides[dim] = s;
      s *= tileShape[dim];
    }
  }

  auto srcThreadOffsets = computeThreadOffsets(loc, rewriter, srcTy);
  auto tileThreadOffsets = computeThreadOffsets(loc, rewriter, tileTy);
  auto smemPtrTy =
      LLVM::LLVMPointerType::get(ctx, targetInfo.getSharedAddressSpace());
  Value smemBase =
      LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());

  SmallVector<int64_t> logicalGrid(rank), suffix(rank, 1);
  for (int d = 0; d < rank; ++d)
    logicalGrid[d] = srcShape[d] / tileShape[d];
  for (int d = rank - 2; d >= 0; --d)
    suffix[d] = suffix[d + 1] * logicalGrid[d + 1];

  Value dynIndex = adaptor.getIndex();
  unsigned dynIndexWidth = dynIndex.getType().getIntOrFloatBitWidth();
  if (dynIndexWidth > 32)
    dynIndex = LLVM::TruncOp::create(rewriter, loc, i32Ty, dynIndex);
  else if (dynIndexWidth < 32)
    dynIndex = LLVM::ZExtOp::create(rewriter, loc, i32Ty, dynIndex);

  SmallVector<Value> tileStartVals(rank), tileEndVals(rank);
  for (int d = 0; d < rank; ++d) {
    Value sv = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)suffix[d]));
    Value gv = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr((int32_t)logicalGrid[d]));
    Value tv = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr((int32_t)tileShape[d]));
    Value coord = LLVM::UDivOp::create(rewriter, loc, i32Ty, dynIndex, sv);
    coord = LLVM::URemOp::create(rewriter, loc, i32Ty, coord, gv);
    tileStartVals[d] = LLVM::MulOp::create(rewriter, loc, i32Ty, coord, tv);
    tileEndVals[d] =
        LLVM::AddOp::create(rewriter, loc, i32Ty, tileStartVals[d], tv);
  }

  auto tileVals = unpackLLElements(loc, adaptor.getTile(), rewriter);
  for (unsigned i = 0; i < tileElemsPerThread; ++i) {
    Value smemByteOffsetV = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      Value baseOff = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)tileOffsets[i][d]));
      Value globalCoordV = LLVM::AddOp::create(rewriter, loc, i32Ty, baseOff,
                                               tileThreadOffsets[d]);
      Value tileShapeV = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)tileShape[d]));
      Value tileLocalCoordV =
          LLVM::URemOp::create(rewriter, loc, i32Ty, globalCoordV, tileShapeV);
      Value sb = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)(smemStrides[d] * elemBytes)));
      smemByteOffsetV = LLVM::AddOp::create(
          rewriter, loc, i32Ty, smemByteOffsetV,
          LLVM::MulOp::create(rewriter, loc, i32Ty, tileLocalCoordV, sb));
    }

    Value sp = LLVM::GEPOp::create(rewriter, loc, smemPtrTy, i8Ty, smemBase,
                                   ValueRange{smemByteOffsetV},
                                   LLVM::GEPNoWrapFlags::inbounds);
    LLVM::StoreOp::create(rewriter, loc, tileVals[i], sp, elemBytes);
  }

  NVVM::Barrier0Op::create(rewriter, loc);

  auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> resultVals;
  resultVals.reserve(srcElemsPerThread);
  for (unsigned i = 0; i < srcElemsPerThread; ++i) {
    Value inRange = LLVM::ConstantOp::create(rewriter, loc, i1Ty,
                                             rewriter.getIntegerAttr(i1Ty, 1));
    Value smemByteOffsetV = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      Value baseOff = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)srcOffsets[i][d]));
      Value globalCoordV = LLVM::AddOp::create(rewriter, loc, i32Ty, baseOff,
                                               srcThreadOffsets[d]);
      Value ge = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge,
                                      globalCoordV, tileStartVals[d]);
      Value lt = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult,
                                      globalCoordV, tileEndVals[d]);
      inRange = LLVM::AndOp::create(
          rewriter, loc, LLVM::AndOp::create(rewriter, loc, ge, lt), inRange);

      Value tileShapeV = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)tileShape[d]));
      Value tileLocalSafeV =
          LLVM::URemOp::create(rewriter, loc, i32Ty, globalCoordV, tileShapeV);
      Value sb = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)(smemStrides[d] * elemBytes)));
      smemByteOffsetV = LLVM::AddOp::create(
          rewriter, loc, i32Ty, smemByteOffsetV,
          LLVM::MulOp::create(rewriter, loc, i32Ty, tileLocalSafeV, sb));
    }

    Value lp = LLVM::GEPOp::create(rewriter, loc, smemPtrTy, i8Ty, smemBase,
                                   ValueRange{smemByteOffsetV},
                                   LLVM::GEPNoWrapFlags::inbounds);
    Value tileLoaded =
        LLVM::LoadOp::create(rewriter, loc, llvmElemTy, lp, elemBytes);
    Value merged =
        LLVM::SelectOp::create(rewriter, loc, inRange, tileLoaded, srcVals[i]);
    resultVals.push_back(merged);
  }

  NVVM::Barrier0Op::create(rewriter, loc);

  Value ret = packLLElements(loc, typeConverter, resultVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

struct InsertTileOpConversion : public ConvertOpToLLVMPattern<InsertTileOp> {
  InsertTileOpConversion(LLVMTypeConverter &typeConverter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<InsertTileOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(InsertTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto tileTy = dyn_cast<RankedTensorType>(op.getTile().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if (!srcTy || !tileTy || !dstTy)
      return op.emitError("insert_tile operands must be ranked tensors");

    auto srcEnc = srcTy.getEncoding();
    auto tileEnc = tileTy.getEncoding();
    auto dstEnc = dstTy.getEncoding();
    if (!srcEnc || !tileEnc || !dstEnc)
      return op.emitError("insert_tile requires tensors with encoding");
    if (!isa<ttg::BlockedEncodingAttr>(srcEnc) ||
        !isa<ttg::BlockedEncodingAttr>(tileEnc) ||
        !isa<ttg::BlockedEncodingAttr>(dstEnc))
      return op.emitError("insert_tile only supports BlockedEncodingAttr");

    auto staticIndex = getStaticIndex(op);
    if (staticIndex.has_value() && isCTATileAligned(op, staticIndex.value()))
      return lowerInsertTileStatic(
          op, adaptor, rewriter, this->getTypeConverter(), staticIndex.value());

    return lowerInsertTileViaSMEMDynamic(op, adaptor, rewriter,
                                         this->getTypeConverter(), targetInfo);
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

namespace mlir::triton::iluvatar_tle {

void populateInsertTileOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const TargetInfoBase &targetInfo,
                                        PatternBenefit benefit) {
  patterns.add<InsertTileOpConversion>(typeConverter, targetInfo, benefit);
}

} // namespace mlir::triton::iluvatar_tle
