#include "TleTileToLLVMUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;

namespace {

namespace ttg = mlir::triton::gpu;
using namespace mlir::triton::tle;
// Extract compile-time constant index if available, otherwise return nullopt.
static std::optional<int64_t> getStaticIndex(InsertTileOp op) {
  if (auto c = op->getOperand(2).getDefiningOp<mlir::arith::ConstantOp>())
    return mlir::cast<mlir::IntegerAttr>(c.getValue()).getInt();
  return std::nullopt;
}
// Check if the tile to be inserted is CTA-aligned (for register shuffle path).
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
// Lowering for static, CTA-aligned insert_tile (register shuffle path).
static LogicalResult
lowerInsertTileStatic(InsertTileOp op, InsertTileOp::Adaptor adaptor,
                      ConversionPatternRewriter &rewriter,
                      const LLVMTypeConverter *typeConverter, int64_t index) {
  Location loc = op->getLoc();
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto tileTy = cast<RankedTensorType>(op.getTile().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());

  // Unpack source and tile values.
  auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
  auto tileVals = unpackLLElements(loc, adaptor.getTile(), rewriter);

  auto srcShape = srcTy.getShape();
  auto tileShape = tileTy.getShape();

  // Compute CTA tile shapes and grid info.
  auto shapePerCTATile = getShapePerCTATile(srcTy);
  auto srcCTAShape = multiDimElementwise<int64_t, unsigned>(
      srcShape, shapePerCTATile, std::divides<unsigned>());
  auto tileCTAShape = multiDimElementwise<int64_t, unsigned>(
      tileShape, shapePerCTATile, std::divides<unsigned>());

  SmallVector<int64_t> logicalTileShape(tileShape.begin(), tileShape.end());
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  // Check divisibility and compute logical grid shape.
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (logicalTileShape[i] == 0 || srcShape[i] % logicalTileShape[i] != 0)
      return op.emitError("source shape must be divisible by tile shape");
    logicalGridShape[i] = srcShape[i] / logicalTileShape[i];
  }
  // Compute logical coordinates from linear index.
  SmallVector<int64_t> logicalCoords(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = srcShape.size() - 1; i >= 0; --i) {
    logicalCoords[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  SmallVector<int64_t> elementCoords(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    elementCoords[i] = logicalCoords[i] * logicalTileShape[i];

  // Compute first tile coordinate in CTA space.
  auto firstTileCoordinate = multiDimElementwise<int64_t, unsigned>(
      elementCoords, shapePerCTATile, std::divides<unsigned>());

  auto numCTATiles = std::accumulate(tileCTAShape.begin(), tileCTAShape.end(),
                                     1, std::multiplies<>());

  // Get CTA order for source and tile.
  auto srcCTAOrder = getCTATileOrder(srcTy);
  auto tileCTAOrder = getCTATileOrder(tileTy);
  // Check tile write region is within bounds.
  for (size_t d = 0; d < srcCTAShape.size(); ++d) {
    if (firstTileCoordinate[d] + tileCTAShape[d] > srcCTAShape[d])
      return op.emitError("tile write region out of source bounds");
  }

  unsigned totalSrcCTAs = std::accumulate(
      srcCTAShape.begin(), srcCTAShape.end(), 1u, std::multiplies<>());
  unsigned totalTileCTAs = std::accumulate(
      tileCTAShape.begin(), tileCTAShape.end(), 1u, std::multiplies<>());

  // Compute per-CTA elements per thread for source and tile.
  unsigned srcElemsPerThreadPerCTA =
      ttg::getTotalElemsPerThread(srcTy) / totalSrcCTAs;
  unsigned tileElemsPerThreadPerCTA =
      ttg::getTotalElemsPerThread(tileTy) / totalTileCTAs;

  if (srcElemsPerThreadPerCTA != tileElemsPerThreadPerCTA)
    return op.emitError("source/tile per-CTA elements per thread mismatch");

  // Copy tile values into the correct region of the result.
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
        tileStartIdx + tileElemsPerThreadPerCTA > tileVals.size()) {
      return op.emitError("Internal error: register index out of bounds")
             << " srcStartIdx=" << srcStartIdx
             << " tileStartIdx=" << tileStartIdx
             << " srcVals.size()=" << resultVals.size()
             << " tileVals.size()=" << tileVals.size();
    }

    llvm::copy(
        ArrayRef<Value>(tileVals).slice(tileStartIdx, srcElemsPerThreadPerCTA),
        resultVals.begin() + srcStartIdx);
  }

  Value ret = packLLElements(loc, typeConverter, resultVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

// Dynamic or non-CTA-aligned lowering: use SMEM relay path.
static LogicalResult
lowerInsertTileViaSMEMDynamic(InsertTileOp op, InsertTileOp::Adaptor adaptor,
                              ConversionPatternRewriter &rewriter,
                              const LLVMTypeConverter *typeConverter) {
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
  auto elemTy = srcTy.getElementType();
  Type llvmElemTy = typeConverter->convertType(elemTy);
  if (!llvmElemTy)
    return op.emitError("SMEM path: failed to convert element type");
  int64_t elemBytes = llvmElemTy.getIntOrFloatBitWidth() / 8;

  // Compute total number of elements in the tile.
  int64_t totalTileElems = 1;
  for (auto dim : tileShape)
    totalTileElems *= dim;

  auto srcOffsets = mlir::emitOffsetForLayout(srcTy.getEncoding(), srcTy);
  auto tileOffsets = mlir::emitOffsetForLayout(tileTy.getEncoding(), tileTy);
  unsigned srcElemsPerThread = ttg::getTotalElemsPerThread(srcTy);
  unsigned tileElemsPerThread = ttg::getTotalElemsPerThread(tileTy);

  // Check offset sizes match per-thread element counts.
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

  // Compute per-thread offsets for source and tile.
  auto srcThreadOffsets = computeThreadOffsets(loc, rewriter, srcTy);
  auto tileThreadOffsets = computeThreadOffsets(loc, rewriter, tileTy);

  auto smemPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
  auto smemArrTy = LLVM::LLVMArrayType::get(
      i8Ty, static_cast<uint64_t>(totalTileElems * elemBytes));
  std::string smemName =
      "__smem_insert_tile_" +
      std::to_string(reinterpret_cast<uintptr_t>(op.getOperation()));
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    rewriter.create<LLVM::GlobalOp>(loc, smemArrTy, false,
                                    LLVM::Linkage::Internal, smemName,
                                    Attribute{}, 16, 3);
  }
  // Allocate global shared memory buffer for tile relay.
  Value smemBufArr =
      rewriter.create<LLVM::AddressOfOp>(loc, smemPtrTy, smemName);
  Value zero32 = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(0));
  Value smemBase = rewriter.create<LLVM::GEPOp>(
      loc, smemPtrTy, smemArrTy, smemBufArr, ValueRange{zero32, zero32},
      LLVM::GEPNoWrapFlags::inbounds);

  SmallVector<int64_t> logicalGrid(rank), suffix(rank, 1);
  for (int d = 0; d < rank; ++d)
    logicalGrid[d] = srcShape[d] / tileShape[d];
  for (int d = rank - 2; d >= 0; --d)
    suffix[d] = suffix[d + 1] * logicalGrid[d + 1];

  // Compute tile start/end coordinates for dynamic index.
  Value dynIndex = adaptor.getIndex();
  unsigned dynIndexWidth = dynIndex.getType().getIntOrFloatBitWidth();
  if (dynIndexWidth > 32) {
    dynIndex = rewriter.create<LLVM::TruncOp>(loc, i32Ty, dynIndex);
  } else if (dynIndexWidth < 32) {
    dynIndex = rewriter.create<LLVM::ZExtOp>(loc, i32Ty, dynIndex);
  }
  SmallVector<Value> tileStartVals(rank), tileEndVals(rank);
  for (int d = 0; d < rank; ++d) {
    Value sv = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)suffix[d]));
    Value gv = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)logicalGrid[d]));
    Value tv = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)tileShape[d]));
    Value coord = rewriter.create<LLVM::UDivOp>(loc, i32Ty, dynIndex, sv);
    coord = rewriter.create<LLVM::URemOp>(loc, i32Ty, coord, gv);
    tileStartVals[d] = rewriter.create<LLVM::MulOp>(loc, i32Ty, coord, tv);
    tileEndVals[d] =
        rewriter.create<LLVM::AddOp>(loc, i32Ty, tileStartVals[d], tv);
  }

  auto tileVals = unpackLLElements(loc, adaptor.getTile(), rewriter);
  for (unsigned i = 0; i < tileElemsPerThread; ++i) {
    Value smemByteOffsetV = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      Value baseOff = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)tileOffsets[i][d]));
      Value globalCoordV = rewriter.create<LLVM::AddOp>(loc, i32Ty, baseOff,
                                                        tileThreadOffsets[d]);

      Value tileShapeV = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)tileShape[d]));
      Value tileLocalCoordV =
          rewriter.create<LLVM::URemOp>(loc, i32Ty, globalCoordV, tileShapeV);

      Value sb = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)(smemStrides[d] * elemBytes)));
      smemByteOffsetV = rewriter.create<LLVM::AddOp>(
          loc, i32Ty, smemByteOffsetV,
          rewriter.create<LLVM::MulOp>(loc, i32Ty, tileLocalCoordV, sb));
    }

    Value sp = rewriter.create<LLVM::GEPOp>(loc, smemPtrTy, i8Ty, smemBase,
                                            ValueRange{smemByteOffsetV},
                                            LLVM::GEPNoWrapFlags::inbounds);
    // Store tile value to shared memory buffer.
    rewriter.create<LLVM::StoreOp>(loc, tileVals[i], sp, elemBytes);
  }
  // Synchronize threads after tile store.
  rewriter.create<NVVM::Barrier0Op>(loc);

  auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> resultVals;
  resultVals.reserve(srcElemsPerThread);

  // For each source element, check if it should be overwritten by tile.
  for (unsigned i = 0; i < srcElemsPerThread; ++i) {
    Value inRange = rewriter.create<LLVM::ConstantOp>(
        loc, i1Ty, rewriter.getIntegerAttr(i1Ty, 1));
    Value smemByteOffsetV = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      Value baseOff = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)srcOffsets[i][d]));
      Value globalCoordV = rewriter.create<LLVM::AddOp>(loc, i32Ty, baseOff,
                                                        srcThreadOffsets[d]);

      Value ge = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge,
                                               globalCoordV, tileStartVals[d]);
      Value lt = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult,
                                               globalCoordV, tileEndVals[d]);
      inRange = rewriter.create<LLVM::AndOp>(
          loc, rewriter.create<LLVM::AndOp>(loc, ge, lt), inRange);

      Value tileShapeV = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)tileShape[d]));
      Value tileLocalSafeV =
          rewriter.create<LLVM::URemOp>(loc, i32Ty, globalCoordV, tileShapeV);

      Value sb = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)(smemStrides[d] * elemBytes)));
      smemByteOffsetV = rewriter.create<LLVM::AddOp>(
          loc, i32Ty, smemByteOffsetV,
          rewriter.create<LLVM::MulOp>(loc, i32Ty, tileLocalSafeV, sb));
    }

    Value lp = rewriter.create<LLVM::GEPOp>(loc, smemPtrTy, i8Ty, smemBase,
                                            ValueRange{smemByteOffsetV},
                                            LLVM::GEPNoWrapFlags::inbounds);
    Value tileLoaded =
        rewriter.create<LLVM::LoadOp>(loc, llvmElemTy, lp, elemBytes);
    // Overwrite source value with tile value if in range.
    Value merged =
        rewriter.create<LLVM::SelectOp>(loc, inRange, tileLoaded, srcVals[i]);
    resultVals.push_back(merged);
  }

  rewriter.create<NVVM::Barrier0Op>(loc);
  // Synchronize threads after tile load.
  Value ret = packLLElements(loc, typeConverter, resultVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

// ============================================================================
// InsertTileOp -> LLVM conversion
// ============================================================================
struct InsertTileOpConversion : public ConvertOpToLLVMPattern<InsertTileOp> {

  using ConvertOpToLLVMPattern<InsertTileOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(InsertTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();

    // Basic type checks.
    // Check operand and result types and encodings.
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto tileTy = dyn_cast<RankedTensorType>(op.getTile().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());

    if (!srcTy || !tileTy || !dstTy) {
      return op.emitError("insert_tile operands must be ranked tensors");
    }

    auto srcEnc = srcTy.getEncoding();
    auto tileEnc = tileTy.getEncoding();
    auto dstEnc = dstTy.getEncoding();

    if (!srcEnc || !tileEnc || !dstEnc) {
      return op.emitError("insert_tile requires tensors with encoding");
    }

    if (!isa<ttg::BlockedEncodingAttr>(srcEnc) ||
        !isa<ttg::BlockedEncodingAttr>(tileEnc) ||
        !isa<ttg::BlockedEncodingAttr>(dstEnc)) {
      return op.emitError("insert_tile only supports BlockedEncodingAttr");
    }

    auto staticIndex = getStaticIndex(op);
    if (staticIndex.has_value() && isCTATileAligned(op, staticIndex.value())) {
      int64_t index = staticIndex.value();
      return lowerInsertTileStatic(op, adaptor, rewriter,
                                   this->getTypeConverter(), index);
    }

    return lowerInsertTileViaSMEMDynamic(op, adaptor, rewriter,
                                         this->getTypeConverter());
  }
};

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================
namespace mlir::triton::tle {

void populateInsertTileOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        unsigned benefit) {
  patterns.add<InsertTileOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::tle
