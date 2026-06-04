#ifdef __TLE__

#include "TileOpUtils.h"

#include "Conversion/MUSATLEToLLVM/LocalPointersOpToLLVM.h"
#include "Dialect/MUSATLE/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::triton;

namespace {

namespace ttg = mlir::triton::gpu;
namespace musa_tle = mlir::triton::musa_tle;

static SmallVector<int64_t> getTileShape(ArrayRef<int64_t> tileShape) {
  SmallVector<int64_t> result;
  llvm::append_range(result, tileShape);
  return result;
}

static SmallVector<int64_t> getTileShape(musa_tle::ExtractTileOp op) {
  return getTileShape(op.getTileShape());
}

static SmallVector<int64_t> getTileShape(musa_tle::InsertTileOp op) {
  return getTileShape(op.getTileShape());
}

template <typename OpT> static std::optional<int64_t> getStaticIndex(OpT op) {
  auto constOp = op.getIndex().template getDefiningOp<arith::ConstantOp>();
  if (constOp) {
    auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
    if (intAttr)
      return intAttr.getInt();
  }
  auto llvmConstOp = op.getIndex().template getDefiningOp<LLVM::ConstantOp>();
  if (llvmConstOp) {
    auto intAttr = dyn_cast<IntegerAttr>(llvmConstOp.getValue());
    if (intAttr)
      return intAttr.getInt();
  }
  return std::nullopt;
}

static Value convertIndexToI32(Location loc, Value index,
                               ConversionPatternRewriter &rewriter) {
  auto i32Ty = rewriter.getI32Type();
  if (index.getType() == i32Ty)
    return index;
  if (auto intTy = dyn_cast<IntegerType>(index.getType())) {
    if (intTy.getWidth() > 32)
      return LLVM::TruncOp::create(rewriter, loc, i32Ty, index);
    return LLVM::ZExtOp::create(rewriter, loc, i32Ty, index);
  }
  if (index.getType().isIndex())
    return arith::IndexCastOp::create(rewriter, loc, i32Ty, index);
  return index;
}

static int64_t getByteWidth(Type type) {
  int bitWidth = mlir::getIntOrFloatOrPtrBitWidth(type);
  return (bitWidth + 7) / 8;
}

static SmallVector<int64_t> getRowMajorSuffix(ArrayRef<int64_t> logicalGrid) {
  SmallVector<int64_t> suffix(logicalGrid.size(), 1);
  for (int d = static_cast<int>(logicalGrid.size()) - 2; d >= 0; --d)
    suffix[d] = suffix[d + 1] * logicalGrid[d + 1];
  return suffix;
}

static void computeTileStartEnd(Location loc,
                                ConversionPatternRewriter &rewriter,
                                Value linearIndex, ArrayRef<int64_t> srcShape,
                                ArrayRef<int64_t> tileShape,
                                SmallVectorImpl<Value> &tileStartVals,
                                SmallVectorImpl<Value> &tileEndVals) {
  auto i32Ty = rewriter.getI32Type();
  int rank = srcShape.size();
  SmallVector<int64_t> logicalGrid(rank);
  for (int d = 0; d < rank; ++d)
    logicalGrid[d] = srcShape[d] / tileShape[d];
  SmallVector<int64_t> suffix = getRowMajorSuffix(logicalGrid);

  Value index = convertIndexToI32(loc, linearIndex, rewriter);
  tileStartVals.resize(rank);
  tileEndVals.resize(rank);
  for (int d = 0; d < rank; ++d) {
    Value suffixV = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(suffix[d]));
    Value gridV = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(logicalGrid[d]));
    Value tileV = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(tileShape[d]));
    Value coord = LLVM::UDivOp::create(rewriter, loc, i32Ty, index, suffixV);
    coord = LLVM::URemOp::create(rewriter, loc, i32Ty, coord, gridV);
    tileStartVals[d] = LLVM::MulOp::create(rewriter, loc, i32Ty, coord, tileV);
    tileEndVals[d] =
        LLVM::AddOp::create(rewriter, loc, i32Ty, tileStartVals[d], tileV);
  }
}

template <typename OpT>
static bool isCTATileAligned(OpT op, int64_t linearIndex,
                             RankedTensorType srcTy,
                             ArrayRef<int64_t> tileShape) {
  auto srcShape = srcTy.getShape();
  auto ctaTile = musa_tle::getShapePerCTATile(srcTy);
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

static LogicalResult lowerExtractTileStaticCTAAligned(
    musa_tle::ExtractTileOp op, musa_tle::ExtractTileOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, const LLVMTypeConverter *typeConverter,
    int64_t linearIndex) {
  Location loc = op.getLoc();
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();
  auto tileShape = getTileShape(op);
  int rank = srcShape.size();

  SmallVector<Value> vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
  auto shapePerCTATile = musa_tle::getShapePerCTATile(srcTy);
  auto srcCTAShape = musa_tle::multiDimElementwise<int64_t, unsigned>(
      srcShape, shapePerCTATile, std::divides<unsigned>());
  auto dstCTAShape = musa_tle::multiDimElementwise<int64_t, unsigned>(
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

  auto firstTileCoord = musa_tle::multiDimElementwise<int64_t, unsigned>(
      elementCoords, shapePerCTATile, std::divides<unsigned>());
  auto srcCTAOrder = musa_tle::getCTATileOrder(srcTy);
  auto dstCTAOrder = musa_tle::getCTATileOrder(dstTy);

  unsigned totalSrcCTAs = std::accumulate(
      srcCTAShape.begin(), srcCTAShape.end(), 1u, std::multiplies<>());
  unsigned elemsPerCTA = ttg::getTotalElemsPerThread(srcTy) / totalSrcCTAs;
  unsigned numDstCTAs = std::accumulate(dstCTAShape.begin(), dstCTAShape.end(),
                                        1u, std::multiplies<>());

  SmallVector<Value> resultVals;
  resultVals.reserve(ttg::getTotalElemsPerThread(dstTy));
  for (unsigned i = 0; i < numDstCTAs; ++i) {
    auto coordInDst = musa_tle::delinearize(i, dstCTAShape, dstCTAOrder);
    auto coordInSrc = musa_tle::multiDimElementwise<unsigned, unsigned>(
        coordInDst, firstTileCoord, std::plus<unsigned>());
    unsigned linearInSrc =
        musa_tle::linearize(coordInSrc, srcCTAShape, srcCTAOrder);
    size_t startIdx = linearInSrc * elemsPerCTA;
    if (startIdx + elemsPerCTA > vals.size())
      return op.emitError("static extract_tile register index out of bounds");
    llvm::append_range(resultVals,
                       ArrayRef<Value>(vals).slice(startIdx, elemsPerCTA));
  }

  Value ret = packLLElements(loc, typeConverter, resultVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

static LogicalResult lowerExtractTileViaSMEM(
    musa_tle::ExtractTileOp op, musa_tle::ExtractTileOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, const LLVMTypeConverter *typeConverter,
    const TargetInfoBase &targetInfo) {
  Location loc = op.getLoc();
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();
  auto tileShape = getTileShape(op);
  int rank = srcShape.size();

  auto i1Ty = rewriter.getI1Type();
  auto i8Ty = rewriter.getI8Type();
  auto i32Ty = rewriter.getI32Type();
  Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
  if (!llvmElemTy)
    return op.emitError("failed to convert extract_tile element type");
  int64_t elemBytes = getByteWidth(llvmElemTy);

  auto srcOffsets = mlir::emitOffsetForLayout(srcTy.getEncoding(), srcTy);
  auto dstOffsets = mlir::emitOffsetForLayout(dstTy.getEncoding(), dstTy);
  unsigned srcElemsPerThread = ttg::getTotalElemsPerThread(srcTy);
  unsigned dstElemsPerThread = ttg::getTotalElemsPerThread(dstTy);
  if (srcOffsets.size() != srcElemsPerThread)
    return op.emitError("extract_tile source offsets size mismatch");
  if (dstOffsets.size() != dstElemsPerThread)
    return op.emitError("extract_tile result offsets size mismatch");

  auto dstOrder = musa_tle::getCTATileOrder(dstTy);
  SmallVector<int64_t> smemStrides(rank, 0);
  int64_t stride = 1;
  for (int i = 0; i < rank; ++i) {
    unsigned dim = dstOrder[i];
    smemStrides[dim] = stride;
    stride *= dstShape[dim];
  }

  auto srcThreadOffsets = musa_tle::computeThreadOffsets(loc, rewriter, srcTy);
  auto dstThreadOffsets = musa_tle::computeThreadOffsets(loc, rewriter, dstTy);

  auto smemPtrTy = LLVM::LLVMPointerType::get(
      rewriter.getContext(), targetInfo.getSharedAddressSpace());
  Value smemBase =
      LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());

  SmallVector<Value> tileStartVals, tileEndVals;
  computeTileStartEnd(loc, rewriter, adaptor.getIndex(), srcShape, tileShape,
                      tileStartVals, tileEndVals);

  SmallVector<Value> srcVals =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  for (unsigned i = 0; i < srcElemsPerThread; ++i) {
    Value inRange = LLVM::ConstantOp::create(rewriter, loc, i1Ty,
                                             rewriter.getIntegerAttr(i1Ty, 1));
    Value smemByteOffset = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      Value baseOff = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(srcOffsets[i][d]));
      Value globalCoord = LLVM::AddOp::create(rewriter, loc, i32Ty, baseOff,
                                              srcThreadOffsets[d]);

      Value ge = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge,
                                      globalCoord, tileStartVals[d]);
      Value lt = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult,
                                      globalCoord, tileEndVals[d]);
      Value dimInRange = LLVM::AndOp::create(rewriter, loc, ge, lt);
      inRange = LLVM::AndOp::create(rewriter, loc, dimInRange, inRange);

      Value localCoord = LLVM::SubOp::create(rewriter, loc, i32Ty, globalCoord,
                                             tileStartVals[d]);
      Value strideBytes = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(smemStrides[d] * elemBytes));
      Value dimOffset =
          LLVM::MulOp::create(rewriter, loc, i32Ty, localCoord, strideBytes);
      smemByteOffset =
          LLVM::AddOp::create(rewriter, loc, i32Ty, smemByteOffset, dimOffset);
    }

    Block *current = rewriter.getInsertionBlock();
    Block *thenBlock =
        rewriter.splitBlock(current, rewriter.getInsertionPoint());
    Block *mergeBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());

    rewriter.setInsertionPointToEnd(current);
    LLVM::CondBrOp::create(rewriter, loc, inRange, thenBlock, mergeBlock);

    rewriter.setInsertionPointToStart(thenBlock);
    Value ptr = LLVM::GEPOp::create(rewriter, loc, smemPtrTy, i8Ty, smemBase,
                                    ValueRange{smemByteOffset},
                                    LLVM::GEPNoWrapFlags::inbounds);
    LLVM::StoreOp::create(rewriter, loc, srcVals[i], ptr, elemBytes);
    LLVM::BrOp::create(rewriter, loc, mergeBlock);

    rewriter.setInsertionPointToStart(mergeBlock);
  }

  targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

  SmallVector<Value> dstVals;
  dstVals.reserve(dstElemsPerThread);
  for (unsigned i = 0; i < dstElemsPerThread; ++i) {
    Value smemByteOffset = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      Value baseOff = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(dstOffsets[i][d]));
      Value globalCoord = LLVM::AddOp::create(rewriter, loc, i32Ty, baseOff,
                                              dstThreadOffsets[d]);
      Value tileShapeV = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(tileShape[d]));
      Value tileLocal =
          LLVM::URemOp::create(rewriter, loc, i32Ty, globalCoord, tileShapeV);
      Value strideBytes = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(smemStrides[d] * elemBytes));
      Value dimOffset =
          LLVM::MulOp::create(rewriter, loc, i32Ty, tileLocal, strideBytes);
      smemByteOffset =
          LLVM::AddOp::create(rewriter, loc, i32Ty, smemByteOffset, dimOffset);
    }

    Value ptr = LLVM::GEPOp::create(rewriter, loc, smemPtrTy, i8Ty, smemBase,
                                    ValueRange{smemByteOffset},
                                    LLVM::GEPNoWrapFlags::inbounds);
    dstVals.push_back(
        LLVM::LoadOp::create(rewriter, loc, llvmElemTy, ptr, elemBytes));
  }

  targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

  Value ret = packLLElements(loc, typeConverter, dstVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

static LogicalResult lowerInsertTileStaticCTAAligned(
    musa_tle::InsertTileOp op, musa_tle::InsertTileOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, const LLVMTypeConverter *typeConverter,
    int64_t linearIndex) {
  Location loc = op.getLoc();
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto tileTy = cast<RankedTensorType>(op.getTile().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());

  SmallVector<Value> srcVals =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> tileVals =
      unpackLLElements(loc, adaptor.getTile(), rewriter);

  auto srcShape = srcTy.getShape();
  auto tileShape = getTileShape(op);
  auto shapePerCTATile = musa_tle::getShapePerCTATile(srcTy);
  auto srcCTAShape = musa_tle::multiDimElementwise<int64_t, unsigned>(
      srcShape, shapePerCTATile, std::divides<unsigned>());
  auto tileCTAShape = musa_tle::multiDimElementwise<int64_t, unsigned>(
      tileShape, shapePerCTATile, std::divides<unsigned>());

  SmallVector<int64_t> logicalGrid(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    logicalGrid[i] = srcShape[i] / tileShape[i];

  SmallVector<int64_t> logicalCoords(srcShape.size(), 0);
  int64_t remain = linearIndex;
  for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; --i) {
    logicalCoords[i] = remain % logicalGrid[i];
    remain /= logicalGrid[i];
  }

  SmallVector<int64_t> elementCoords(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    elementCoords[i] = logicalCoords[i] * tileShape[i];

  auto firstTileCoord = musa_tle::multiDimElementwise<int64_t, unsigned>(
      elementCoords, shapePerCTATile, std::divides<unsigned>());
  auto srcCTAOrder = musa_tle::getCTATileOrder(srcTy);
  auto tileCTAOrder = musa_tle::getCTATileOrder(tileTy);

  unsigned totalSrcCTAs = std::accumulate(
      srcCTAShape.begin(), srcCTAShape.end(), 1u, std::multiplies<>());
  unsigned totalTileCTAs = std::accumulate(
      tileCTAShape.begin(), tileCTAShape.end(), 1u, std::multiplies<>());
  unsigned srcElemsPerCTA = ttg::getTotalElemsPerThread(srcTy) / totalSrcCTAs;
  unsigned tileElemsPerCTA =
      ttg::getTotalElemsPerThread(tileTy) / totalTileCTAs;
  if (srcElemsPerCTA != tileElemsPerCTA)
    return op.emitError("insert_tile source/tile per-CTA element mismatch");

  unsigned numTileCTAs = totalTileCTAs;
  SmallVector<Value> resultVals(srcVals.begin(), srcVals.end());
  for (unsigned i = 0; i < numTileCTAs; ++i) {
    auto tileCoord = musa_tle::delinearize(i, tileCTAShape, tileCTAOrder);
    auto srcCoord = musa_tle::multiDimElementwise<unsigned, unsigned>(
        tileCoord, firstTileCoord, std::plus<unsigned>());
    unsigned srcLinear =
        musa_tle::linearize(srcCoord, srcCTAShape, srcCTAOrder);
    unsigned tileLinear =
        musa_tle::linearize(tileCoord, tileCTAShape, tileCTAOrder);
    size_t srcStart = srcLinear * srcElemsPerCTA;
    size_t tileStart = tileLinear * tileElemsPerCTA;
    if (srcStart + srcElemsPerCTA > resultVals.size() ||
        tileStart + tileElemsPerCTA > tileVals.size())
      return op.emitError("static insert_tile register index out of bounds");
    llvm::copy(ArrayRef<Value>(tileVals).slice(tileStart, srcElemsPerCTA),
               resultVals.begin() + srcStart);
  }

  Value ret = packLLElements(loc, typeConverter, resultVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

static LogicalResult lowerInsertTileViaSMEM(
    musa_tle::InsertTileOp op, musa_tle::InsertTileOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, const LLVMTypeConverter *typeConverter,
    const TargetInfoBase &targetInfo) {
  Location loc = op.getLoc();
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto tileTy = cast<RankedTensorType>(op.getTile().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto srcShape = srcTy.getShape();
  auto tileShape = getTileShape(op);
  int rank = srcShape.size();

  auto i1Ty = rewriter.getI1Type();
  auto i8Ty = rewriter.getI8Type();
  auto i32Ty = rewriter.getI32Type();
  Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
  if (!llvmElemTy)
    return op.emitError("failed to convert insert_tile element type");
  int64_t elemBytes = getByteWidth(llvmElemTy);

  auto srcOffsets = mlir::emitOffsetForLayout(srcTy.getEncoding(), srcTy);
  auto tileOffsets = mlir::emitOffsetForLayout(tileTy.getEncoding(), tileTy);
  unsigned srcElemsPerThread = ttg::getTotalElemsPerThread(srcTy);
  unsigned tileElemsPerThread = ttg::getTotalElemsPerThread(tileTy);
  if (srcOffsets.size() != srcElemsPerThread)
    return op.emitError("insert_tile source offsets size mismatch");
  if (tileOffsets.size() != tileElemsPerThread)
    return op.emitError("insert_tile tile offsets size mismatch");

  auto tileOrder = musa_tle::getCTATileOrder(tileTy);
  SmallVector<int64_t> smemStrides(rank, 0);
  int64_t stride = 1;
  for (int i = 0; i < rank; ++i) {
    unsigned dim = tileOrder[i];
    smemStrides[dim] = stride;
    stride *= tileShape[dim];
  }

  auto srcThreadOffsets = musa_tle::computeThreadOffsets(loc, rewriter, srcTy);
  auto tileThreadOffsets =
      musa_tle::computeThreadOffsets(loc, rewriter, tileTy);

  auto smemPtrTy = LLVM::LLVMPointerType::get(
      rewriter.getContext(), targetInfo.getSharedAddressSpace());
  Value smemBase =
      LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());

  SmallVector<Value> tileStartVals, tileEndVals;
  computeTileStartEnd(loc, rewriter, adaptor.getIndex(), srcShape, tileShape,
                      tileStartVals, tileEndVals);

  SmallVector<Value> tileVals =
      unpackLLElements(loc, adaptor.getTile(), rewriter);
  for (unsigned i = 0; i < tileElemsPerThread; ++i) {
    Value smemByteOffset = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      Value baseOff = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(tileOffsets[i][d]));
      Value globalCoord = LLVM::AddOp::create(rewriter, loc, i32Ty, baseOff,
                                              tileThreadOffsets[d]);
      Value tileShapeV = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(tileShape[d]));
      Value tileLocal =
          LLVM::URemOp::create(rewriter, loc, i32Ty, globalCoord, tileShapeV);
      Value strideBytes = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(smemStrides[d] * elemBytes));
      Value dimOffset =
          LLVM::MulOp::create(rewriter, loc, i32Ty, tileLocal, strideBytes);
      smemByteOffset =
          LLVM::AddOp::create(rewriter, loc, i32Ty, smemByteOffset, dimOffset);
    }

    Value ptr = LLVM::GEPOp::create(rewriter, loc, smemPtrTy, i8Ty, smemBase,
                                    ValueRange{smemByteOffset},
                                    LLVM::GEPNoWrapFlags::inbounds);
    LLVM::StoreOp::create(rewriter, loc, tileVals[i], ptr, elemBytes);
  }

  targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

  SmallVector<Value> srcVals =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> resultVals;
  resultVals.reserve(srcElemsPerThread);
  for (unsigned i = 0; i < srcElemsPerThread; ++i) {
    Value inRange = LLVM::ConstantOp::create(rewriter, loc, i1Ty,
                                             rewriter.getIntegerAttr(i1Ty, 1));
    Value smemByteOffset = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      Value baseOff = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(srcOffsets[i][d]));
      Value globalCoord = LLVM::AddOp::create(rewriter, loc, i32Ty, baseOff,
                                              srcThreadOffsets[d]);

      Value ge = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge,
                                      globalCoord, tileStartVals[d]);
      Value lt = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult,
                                      globalCoord, tileEndVals[d]);
      Value dimInRange = LLVM::AndOp::create(rewriter, loc, ge, lt);
      inRange = LLVM::AndOp::create(rewriter, loc, dimInRange, inRange);

      Value tileShapeV = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(tileShape[d]));
      Value tileLocal =
          LLVM::URemOp::create(rewriter, loc, i32Ty, globalCoord, tileShapeV);
      Value strideBytes = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(smemStrides[d] * elemBytes));
      Value dimOffset =
          LLVM::MulOp::create(rewriter, loc, i32Ty, tileLocal, strideBytes);
      smemByteOffset =
          LLVM::AddOp::create(rewriter, loc, i32Ty, smemByteOffset, dimOffset);
    }

    Value ptr = LLVM::GEPOp::create(rewriter, loc, smemPtrTy, i8Ty, smemBase,
                                    ValueRange{smemByteOffset},
                                    LLVM::GEPNoWrapFlags::inbounds);
    Value tileLoaded =
        LLVM::LoadOp::create(rewriter, loc, llvmElemTy, ptr, elemBytes);
    resultVals.push_back(
        LLVM::SelectOp::create(rewriter, loc, inRange, tileLoaded, srcVals[i]));
  }

  targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

  Value ret = packLLElements(loc, typeConverter, resultVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

struct ExtractTileOpConversion
    : public ConvertOpToLLVMPattern<musa_tle::ExtractTileOp> {
  ExtractTileOpConversion(LLVMTypeConverter &typeConverter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(musa_tle::ExtractTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if (!srcTy || !dstTy)
      return op.emitError("extract_tile operands must be ranked tensors");
    if (!srcTy.getEncoding() || !dstTy.getEncoding())
      return op.emitError("extract_tile requires tensors with encoding");
    if (!isa<ttg::BlockedEncodingAttr>(srcTy.getEncoding()) ||
        !isa<ttg::BlockedEncodingAttr>(dstTy.getEncoding()))
      return op.emitError("extract_tile only supports BlockedEncodingAttr");

    auto staticIndex = getStaticIndex(op);
    auto tileShape = getTileShape(op);
    if (staticIndex && isCTATileAligned(op, *staticIndex, srcTy, tileShape))
      return lowerExtractTileStaticCTAAligned(op, adaptor, rewriter,
                                              getTypeConverter(), *staticIndex);
    return lowerExtractTileViaSMEM(op, adaptor, rewriter, getTypeConverter(),
                                   targetInfo);
  }

private:
  const TargetInfoBase &targetInfo;
};

struct InsertTileOpConversion
    : public ConvertOpToLLVMPattern<musa_tle::InsertTileOp> {
  InsertTileOpConversion(LLVMTypeConverter &typeConverter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(musa_tle::InsertTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto tileTy = dyn_cast<RankedTensorType>(op.getTile().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if (!srcTy || !tileTy || !dstTy)
      return op.emitError("insert_tile operands must be ranked tensors");
    if (!srcTy.getEncoding() || !tileTy.getEncoding() || !dstTy.getEncoding())
      return op.emitError("insert_tile requires tensors with encoding");
    if (!isa<ttg::BlockedEncodingAttr>(srcTy.getEncoding()) ||
        !isa<ttg::BlockedEncodingAttr>(tileTy.getEncoding()) ||
        !isa<ttg::BlockedEncodingAttr>(dstTy.getEncoding()))
      return op.emitError("insert_tile only supports BlockedEncodingAttr");

    auto staticIndex = getStaticIndex(op);
    auto tileShape = getTileShape(op);
    if (staticIndex && isCTATileAligned(op, *staticIndex, srcTy, tileShape))
      return lowerInsertTileStaticCTAAligned(op, adaptor, rewriter,
                                             getTypeConverter(), *staticIndex);
    return lowerInsertTileViaSMEM(op, adaptor, rewriter, getTypeConverter(),
                                  targetInfo);
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

namespace mlir::triton::musa_tle {

void populateMUSATLETileOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         const TargetInfoBase &targetInfo,
                                         RewritePatternSet &patterns,
                                         PatternBenefit benefit) {
  patterns.add<ExtractTileOpConversion, InsertTileOpConversion>(
      typeConverter, targetInfo, benefit);
}

} // namespace mlir::triton::musa_tle

#endif // __TLE__
