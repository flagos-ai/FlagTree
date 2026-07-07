#include "TleTileToLLVMUtils.h"

#include "IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::iluvatar_tle;
using namespace mlir::triton::iluvatar_tle;

static SmallVector<int64_t> getTileShape(ExtractTileOp op) {
  SmallVector<int64_t> ts;
  if (auto a = dyn_cast<DenseI64ArrayAttr>(op->getAttr("tile_shape")))
    for (auto v : a.asArrayRef())
      ts.push_back(v);
  return ts;
}

static std::optional<int64_t> getStaticIndex(ExtractTileOp op) {
  if (auto c = op->getOperand(1).getDefiningOp<arith::ConstantOp>())
    return cast<IntegerAttr>(c.getValue()).getInt();
  return std::nullopt;
}

static bool isCTATileAligned(ExtractTileOp op, int64_t linearIndex) {
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto srcShape = srcTy.getShape();
  auto tileShape = getTileShape(op);
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
lowerExtractTileStatic(ExtractTileOp op, ExtractTileOp::Adaptor adaptor,
                       ConversionPatternRewriter &rewriter,
                       const LLVMTypeConverter *typeConverter,
                       int64_t linearIndex) {
  Location loc = op->getLoc();
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();
  auto tileShape = getTileShape(op);
  int rank = srcShape.size();
  auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
  auto shapePerCTATile = getShapePerCTATile(srcTy);
  auto srcCTAShape = multiDimElementwise<int64_t, unsigned>(
      srcShape, shapePerCTATile, std::divides<unsigned>());
  auto dstCTAShape = multiDimElementwise<int64_t, unsigned>(
      dstShape, shapePerCTATile, std::divides<unsigned>());
  SmallVector<int64_t> logicalGrid(rank), logicalCoords(rank),
      elementCoords(rank);
  for (int i = 0; i < rank; ++i)
    logicalGrid[i] = srcShape[i] / tileShape[i];
  int64_t remain = linearIndex;
  for (int i = rank - 1; i >= 0; --i) {
    logicalCoords[i] = remain % logicalGrid[i];
    remain /= logicalGrid[i];
  }
  for (int i = 0; i < rank; ++i)
    elementCoords[i] = logicalCoords[i] * tileShape[i];
  auto firstTileCoord = multiDimElementwise<int64_t, unsigned>(
      elementCoords, shapePerCTATile, std::divides<unsigned>());
  auto srcCTAOrder = getCTATileOrder(srcTy);
  auto dstCTAOrder = getCTATileOrder(dstTy);
  unsigned totalSrcCTAs = std::accumulate(
      srcCTAShape.begin(), srcCTAShape.end(), 1, std::multiplies<>());
  unsigned elemsPerCTA = ttg::getTotalElemsPerThread(srcTy) / totalSrcCTAs;
  unsigned numDstCTAs = std::accumulate(dstCTAShape.begin(), dstCTAShape.end(),
                                        1, std::multiplies<>());
  SmallVector<Value> resultVals;
  resultVals.reserve(ttg::getTotalElemsPerThread(dstTy));
  for (unsigned i = 0; i < numDstCTAs; ++i) {
    auto coordInDst = tle::delinearize(i, dstCTAShape, dstCTAOrder);
    auto coordInSrc = multiDimElementwise<unsigned, unsigned>(
        coordInDst, firstTileCoord, std::plus<unsigned>());
    unsigned linearInSrc = tle::linearize(coordInSrc, srcCTAShape, srcCTAOrder);
    size_t startIdx = linearInSrc * elemsPerCTA;
    if (startIdx + elemsPerCTA > vals.size())
      return op.emitError("static path: register index out of bounds");
    llvm::append_range(resultVals,
                       llvm::ArrayRef(vals).slice(startIdx, elemsPerCTA));
  }
  Value ret = packLLElements(loc, typeConverter, resultVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

static LogicalResult
lowerExtractTileViaSMEM(ExtractTileOp op, ExtractTileOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter,
                        const LLVMTypeConverter *typeConverter,
                        const TargetInfoBase &targetInfo) {
  Location loc = op->getLoc();
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();
  auto tileShape = getTileShape(op);
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
  auto dstOffsets = emitOffsetForLayout(dstTy.getEncoding(), dstTy);
  unsigned totalElemsPerThread = ttg::getTotalElemsPerThread(srcTy);
  unsigned dstElemsPerThread = ttg::getTotalElemsPerThread(dstTy);
  if (srcOffsets.size() != totalElemsPerThread)
    return op.emitError("SMEM path: src offsets size mismatch");
  if (dstOffsets.size() != dstElemsPerThread)
    return op.emitError("SMEM path: dst offsets size mismatch");

  auto dstOrder = getCTATileOrder(dstTy);
  SmallVector<int64_t> smemStrides(rank, 0);
  {
    int64_t s = 1;
    for (int i = 0; i < rank; ++i) {
      unsigned dim = dstOrder[i];
      smemStrides[dim] = s;
      s *= dstShape[dim];
    }
  }

  auto srcThreadOffsets = computeThreadOffsets(loc, rewriter, srcTy);
  auto dstThreadOffsets = computeThreadOffsets(loc, rewriter, dstTy);

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

  auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
  for (unsigned i = 0; i < totalElemsPerThread; ++i) {
    Value inRange = LLVM::ConstantOp::create(rewriter, loc, i1Ty,
                                             rewriter.getIntegerAttr(i1Ty, 1));
    Value smemByteOffset = LLVM::ConstantOp::create(
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

      Value localInTile = LLVM::SubOp::create(rewriter, loc, i32Ty,
                                              globalCoordV, tileStartVals[d]);
      Value sb = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)(smemStrides[d] * elemBytes)));
      smemByteOffset = LLVM::AddOp::create(
          rewriter, loc, i32Ty, smemByteOffset,
          LLVM::MulOp::create(rewriter, loc, i32Ty, localInTile, sb));
    }

    Block *cur = rewriter.getInsertionBlock();
    Block *thenBlock = rewriter.splitBlock(cur, rewriter.getInsertionPoint());
    Block *merge = rewriter.splitBlock(thenBlock, thenBlock->begin());

    rewriter.setInsertionPointToEnd(cur);
    LLVM::CondBrOp::create(rewriter, loc, inRange, thenBlock, merge);

    rewriter.setInsertionPointToStart(thenBlock);
    Value sp = LLVM::GEPOp::create(rewriter, loc, smemPtrTy, i8Ty, smemBase,
                                   ValueRange{smemByteOffset},
                                   LLVM::GEPNoWrapFlags::inbounds);
    LLVM::StoreOp::create(rewriter, loc, srcVals[i], sp, elemBytes);
    LLVM::BrOp::create(rewriter, loc, merge);

    rewriter.setInsertionPointToStart(merge);
  }

  NVVM::Barrier0Op::create(rewriter, loc);

  SmallVector<Value> dstVals;
  dstVals.reserve(dstElemsPerThread);
  for (unsigned i = 0; i < dstElemsPerThread; ++i) {
    Value smemByteOffsetV = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      Value baseOff = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)dstOffsets[i][d]));
      Value globalCoordV = LLVM::AddOp::create(rewriter, loc, i32Ty, baseOff,
                                               dstThreadOffsets[d]);
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

    Value lp = LLVM::GEPOp::create(rewriter, loc, smemPtrTy, i8Ty, smemBase,
                                   ValueRange{smemByteOffsetV},
                                   LLVM::GEPNoWrapFlags::inbounds);
    dstVals.push_back(
        LLVM::LoadOp::create(rewriter, loc, llvmElemTy, lp, elemBytes));
  }

  NVVM::Barrier0Op::create(rewriter, loc);

  Value ret = packLLElements(loc, typeConverter, dstVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

struct ExtractTileOpConversion : public ConvertOpToLLVMPattern<ExtractTileOp> {
  ExtractTileOpConversion(LLVMTypeConverter &typeConverter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern<ExtractTileOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ExtractTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if (!srcTy || !dstTy)
      return op.emitError("extract_tile operands must be ranked tensors");
    if (!srcTy.getEncoding() || !dstTy.getEncoding())
      return op.emitError("extract_tile requires tensors with encoding");
    if (!isa<ttg::BlockedEncodingAttr>(srcTy.getEncoding()))
      return op.emitError("extract_tile only supports BlockedEncodingAttr");

    auto staticIndex = getStaticIndex(op);
    if (staticIndex.has_value() && isCTATileAligned(op, staticIndex.value()))
      return lowerExtractTileStatic(
          op, adaptor, rewriter, this->getTypeConverter(), staticIndex.value());
    return lowerExtractTileViaSMEM(op, adaptor, rewriter,
                                   this->getTypeConverter(), targetInfo);
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

namespace mlir::triton::iluvatar_tle {
void populateExtractTileOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         const TargetInfoBase &targetInfo,
                                         PatternBenefit benefit) {
  patterns.add<ExtractTileOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::iluvatar_tle
