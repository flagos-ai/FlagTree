#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "TleTileToLLVMUtils.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;

namespace {

namespace ttg = mlir::triton::gpu;
using namespace mlir::triton::tle;

// ============================================================================
// InsertTileOp -> LLVM conversion (AMD-style)
// ============================================================================
struct InsertTileOpConversion 
    : public ConvertOpToLLVMPattern<InsertTileOp> {

  using ConvertOpToLLVMPattern<InsertTileOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      InsertTileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();

    // ======================================================================
    // Step 1: type checks
    // ======================================================================
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

    // ======================================================================
    // Step 2: unpack input register values
    // ======================================================================
    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto tileVals = unpackLLElements(loc, adaptor.getTile(), rewriter);

    // ======================================================================
    // Step 3 & 4: map logical tile grid to physical CTA grid
    // ======================================================================
    auto srcShape = srcTy.getShape();
    auto tileShape = tileTy.getShape();

    // 1) Get physical CTA tile shape in element space (e.g. [8, 16]).
    auto shapePerCTATile = getShapePerCTATile(srcTy);

    // 2) Compute physical CTA-grid shapes for src and tile tensors.
    auto srcCTAShape = multiDimElementwise<int64_t, unsigned>(
      srcShape, shapePerCTATile, std::divides<unsigned>());
    auto tileCTAShape = multiDimElementwise<int64_t, unsigned>(
      tileShape, shapePerCTATile, std::divides<unsigned>());

    // 3) Read scalar index passed by the frontend.
    int64_t index = 0;
    if (auto constOp = op->getOperand(2).getDefiningOp<mlir::arith::ConstantOp>()) {
        index = mlir::cast<mlir::IntegerAttr>(constOp.getValue()).getInt();
    }

    // 4) Logical tile shape is defined by the tile tensor shape.
    SmallVector<int64_t> logicalTileShape(tileShape.begin(), tileShape.end());

    // 5) Compute logical tile-grid shape.
    SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
    for (size_t i = 0; i < srcShape.size(); ++i) {
        if (logicalTileShape[i] == 0 || srcShape[i] % logicalTileShape[i] != 0)
          return op.emitError("source shape must be divisible by tile shape");
        logicalGridShape[i] = srcShape[i] / logicalTileShape[i];
    }

    // 6) Delinearize index into logical tile coordinates (e.g. 1 -> [1, 0]).
    SmallVector<int64_t> logicalCoords(srcShape.size(), 0);
    int64_t remain = index;
    for (int i = srcShape.size() - 1; i >= 0; --i) {
        logicalCoords[i] = remain % logicalGridShape[i];
        remain /= logicalGridShape[i];
    }

    // 7) Convert logical tile coords to absolute element coordinates.
    SmallVector<int64_t> elementCoords(srcShape.size(), 0);
    for (size_t i = 0; i < srcShape.size(); ++i) {
        elementCoords[i] = logicalCoords[i] * logicalTileShape[i];
    }

    // 8) Convert insertion start to physical CTA coordinates.
    auto firstTileCoordinate = multiDimElementwise<int64_t, unsigned>(
      elementCoords, shapePerCTATile, std::divides<unsigned>());


    // Total number of CTA tiles to overwrite (determined by tile tensor).
    auto numCTATiles = std::accumulate(tileCTAShape.begin(), tileCTAShape.end(),
                                       1, std::multiplies<>());
                                      
    auto srcCTAOrder = getCTATileOrder(srcTy);
    auto tileCTAOrder = getCTATileOrder(tileTy);

    // Bounds check in CTA space.
    for (size_t d = 0; d < srcCTAShape.size(); ++d) {
      if (firstTileCoordinate[d] + tileCTAShape[d] > srcCTAShape[d]) {
        return op.emitError("tile write region out of source bounds");
      }
    }

    // ======================================================================
    // Step 6: compute per-CTA elements-per-thread
    // ======================================================================
    unsigned totalSrcCTAs = std::accumulate(srcCTAShape.begin(), 
                                           srcCTAShape.end(),
                                           1u, std::multiplies<>());

    unsigned totalTileCTAs = std::accumulate(tileCTAShape.begin(),
                                             tileCTAShape.end(),
                                             1u, std::multiplies<>());

    unsigned srcElemsPerThreadPerCTA = 
        ttg::getTotalElemsPerThread(srcTy) / totalSrcCTAs;

    unsigned tileElemsPerThreadPerCTA =
        ttg::getTotalElemsPerThread(tileTy) / totalTileCTAs;

    if (srcElemsPerThreadPerCTA != tileElemsPerThreadPerCTA) {
      return op.emitError("source/tile per-CTA elements per thread mismatch");
    }

    // ======================================================================
    // Step 7: copy src and overwrite target region with tile (core loop)
    // ======================================================================
    SmallVector<Value> resultVals(srcVals.begin(), srcVals.end());

    for (size_t i = 0; i < numCTATiles; i++) {
      // 7.1 CTA coordinate of current sub-tile inside tile tensor.
      auto coordInTileTensor = tle::delinearize(i, tileCTAShape, tileCTAOrder);

      // 7.2 Map to CTA coordinate in source tensor.
          auto coordInSrcTensor = multiDimElementwise<unsigned, unsigned>(
            coordInTileTensor, firstTileCoordinate, std::plus<unsigned>());

      // 7.3 Linearize CTA coordinates to linear indices.
      auto linearIdxInSrcTensor = linearize(
          coordInSrcTensor, srcCTAShape, srcCTAOrder);

      auto linearIdxInTileTensor = linearize(
          coordInTileTensor, tileCTAShape, tileCTAOrder);

      // 7.4 Compute register slice start offsets for src and tile.
      size_t srcStartIdx = linearIdxInSrcTensor * srcElemsPerThreadPerCTA;
      size_t tileStartIdx = linearIdxInTileTensor * tileElemsPerThreadPerCTA;

      // 7.5 Safety bounds check for computed register slices.
      if (srcStartIdx + srcElemsPerThreadPerCTA > resultVals.size() ||
          tileStartIdx + tileElemsPerThreadPerCTA > tileVals.size()) {
        return op.emitError("Internal error: register index out of bounds")
               << " srcStartIdx=" << srcStartIdx
               << " tileStartIdx=" << tileStartIdx
               << " srcVals.size()=" << resultVals.size()
               << " tileVals.size()=" << tileVals.size();
      }

      // 7.6 Overwrite src slice with corresponding tile slice.
      llvm::copy(ArrayRef<Value>(tileVals).slice(tileStartIdx,
                     srcElemsPerThreadPerCTA),
             resultVals.begin() + srcStartIdx);
    }

    // ======================================================================
    // Step 8: pack final result
    // ======================================================================
    Value ret = packLLElements(
        loc, this->getTypeConverter(), resultVals, rewriter, dstTy
    );

    rewriter.replaceOp(op, ret);
    return success();
  }
};

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================
namespace mlir::triton::tle {

void populateInsertTileOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    unsigned benefit) {
  patterns.add<InsertTileOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::tle