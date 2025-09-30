#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace SharedToDotOperandMMAv1 {

using getMNCoordsFunc = SmallVector<CoordTy> (*)(
    Value, Location, ConversionPatternRewriter &, ArrayRef<unsigned int>,
    const IluvatarMmaEncodingAttr &, ArrayRef<int64_t>, int, int, bool);
DEFINE_LOAD_FUNC(getMNCoords)

} // namespace SharedToDotOperandMMAv1

namespace mlir {
namespace LLVM {
using namespace mlir::triton;
using mlir::triton::gpu::getOrder;
using mlir::triton::gpu::getSizePerThread;

Value createIndexConstant(OpBuilder &builder, Location loc,
                          TypeConverter *converter, int64_t value) {
  Type ty = converter->convertType(builder.getIndexType());
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

SmallVector<Value> getMultiDimOffset(Attribute layout, Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     const TargetInfoBase &targetInfo,
                                     unsigned elemId, RankedTensorType type,
                                     ArrayRef<unsigned> multiDimCTAInRepId,
                                     ArrayRef<unsigned> shapePerCTATile,
                                     bool isTrans, bool stNotRd) {
  auto shape = type.getShape();
  unsigned rank = shape.size();
  if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout)) {
    auto multiDimOffsetFirstElem = emitBaseIndexForLayout(
        loc, rewriter, targetInfo, blockedLayout, type, false);
    SmallVector<Value> multiDimOffset(rank);
    SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
        elemId, getSizePerThread(layout), getOrder(layout));
    for (unsigned d = 0; d < rank; ++d) {
      multiDimOffset[d] =
          add(multiDimOffsetFirstElem[d],
              i32_val(multiDimCTAInRepId[d] * shapePerCTATile[d] +
                      multiDimElemId[d]));
    }
    return multiDimOffset;
  }
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    unsigned dim = sliceLayout.getDim();
    auto parentEncoding = sliceLayout.getParent();
    auto parentSizePerThread = getSizePerThread(parentEncoding);
    auto parentShape = sliceLayout.paddedShape(shape);
    auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                          parentEncoding);
    auto offsets = emitOffsetForLayout(layout, type);
    auto parentOffset = emitOffsetForLayout(parentEncoding, parentTy);
    SmallVector<int> idxs;
    for (SmallVector<unsigned> off : offsets) {
      off.insert(off.begin() + dim, 0);
      auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
      idxs.push_back(std::distance(parentOffset.begin(), it));
    }
    auto multiDimOffsetParent = getMultiDimOffset(
        parentEncoding, loc, rewriter, targetInfo, idxs[elemId], parentTy,
        sliceLayout.paddedShape(multiDimCTAInRepId),
        sliceLayout.paddedShape(shapePerCTATile));
    SmallVector<Value> multiDimOffset(rank);
    for (unsigned d = 0; d < rank + 1; ++d) {
      if (d == dim)
        continue;
      unsigned slicedD = d < dim ? d : (d - 1);
      multiDimOffset[slicedD] = multiDimOffsetParent[d];
    }
    return multiDimOffset;
  }
  if (auto mmaLayout = mlir::dyn_cast<NvidiaMmaEncodingAttr>(layout)) {
    assert(rank == 2 ||
           (rank == 3 && mmaLayout.isAmpere()) && "Unexpected rank");
    auto shapePerCTA = getShapePerCTA(mmaLayout, shape);
    auto instrShape = mmaLayout.getInstrShape();
    SmallVector<Value> mmaColIdx(2);
    SmallVector<Value> mmaRowIdx(2);
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    // TODO: fix the bug in MMAEncodingAttr document
    SmallVector<Value> multiDimWarpId(2);
    auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
    auto warpOrder = triton::gpu::getWarpOrder(mmaLayout);
    multiDimWarpId = delinearize(rewriter, loc, warpId, warpsPerCTA, warpOrder);
    Value _1 = i32_val(1);
    Value _2 = i32_val(2);
    Value _4 = i32_val(4);
    Value _8 = i32_val(8);
    Value _16 = i32_val(16);
    if (mmaLayout.isAmpere() || mmaLayout.isHopper()) {
      multiDimWarpId[rank - 1] = urem(
          multiDimWarpId[rank - 1],
          i32_val(ceil<unsigned>(shapePerCTA[rank - 1], instrShape[rank - 1])));
      multiDimWarpId[rank - 2] = urem(
          multiDimWarpId[rank - 2],
          i32_val(ceil<unsigned>(shapePerCTA[rank - 2], instrShape[rank - 2])));

      Value mmaGrpId = udiv(laneId, _4);
      Value mmaGrpIdP8 = add(mmaGrpId, _8);
      Value mmaThreadIdInGrp = urem(laneId, _4);
      Value mmaThreadIdInGrpM2 = mul(mmaThreadIdInGrp, _2);
      Value mmaThreadIdInGrpM2P1 = add(mmaThreadIdInGrpM2, _1);
      Value rowWarpOffset =
          mul(multiDimWarpId[rank - 2], i32_val(instrShape[rank - 2]));
      mmaRowIdx[0] = add(mmaGrpId, rowWarpOffset);
      mmaRowIdx[1] = add(mmaGrpIdP8, rowWarpOffset);
      Value colWarpOffset =
          mul(multiDimWarpId[rank - 1], i32_val(instrShape[rank - 1]));
      mmaColIdx[0] = add(mmaThreadIdInGrpM2, colWarpOffset);
      mmaColIdx[1] = add(mmaThreadIdInGrpM2P1, colWarpOffset);
    } else if (mmaLayout.isVolta()) {
      // Volta doesn't follow the pattern here.
    } else {
      llvm_unreachable("Unexpected MMALayout version");
    }

    SmallVector<Value> multiDimOffset(rank);
    if (mmaLayout.isHopper()) {
      unsigned elemIdRem4 = elemId % 4;
      unsigned nGrpId = elemId / 4;
      multiDimOffset[0] = elemIdRem4 < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
      multiDimOffset[1] = elemIdRem4 % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
      multiDimOffset[1] = add(multiDimOffset[1], i32_val(8 * nGrpId));
      multiDimOffset[0] = add(multiDimOffset[0], i32_val(multiDimCTAInRepId[0] *
                                                         shapePerCTATile[0]));
      multiDimOffset[1] = add(multiDimOffset[1], i32_val(multiDimCTAInRepId[1] *
                                                         shapePerCTATile[1]));
    } else if (mmaLayout.isAmpere()) {
      if (rank == 3)
        multiDimOffset[0] =
            add(multiDimWarpId[0],
                i32_val(multiDimCTAInRepId[0] * shapePerCTATile[0]));
      multiDimOffset[rank - 2] = elemId < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
      multiDimOffset[rank - 1] = elemId % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
      multiDimOffset[rank - 2] =
          add(multiDimOffset[rank - 2], i32_val(multiDimCTAInRepId[rank - 2] *
                                                shapePerCTATile[rank - 2]));
      multiDimOffset[rank - 1] =
          add(multiDimOffset[rank - 1], i32_val(multiDimCTAInRepId[rank - 1] *
                                                shapePerCTATile[rank - 1]));
    } else if (mmaLayout.isVolta()) {
      auto [isARow, isBRow, isAVec4, isBVec4, _] =
          mmaLayout.decodeVoltaLayoutStates();
      auto coords = SharedToDotOperandMMAv1::getMNCoords(
          threadId, loc, rewriter, mmaLayout.getWarpsPerCTA(), mmaLayout, shape,
          isARow, isBRow, isAVec4, isBVec4);
      return coords[elemId];
    } else {
      llvm_unreachable("Unexpected MMALayout version");
    }
    return multiDimOffset;
  }
  if (auto mmaLayout = mlir::dyn_cast<IluvatarMmaEncodingAttr>(layout)) {
    assert(rank == 2 && "Unexpected rank");
    SmallVector<Value> multiDimOffset(rank);
    Value threadId = getThreadId(rewriter, loc);
    if (mmaLayout.isVolta()) {
      int bitwidth = type.getElementType().getIntOrFloatBitWidth();
      int elemVecSize = stNotRd ? (32 / bitwidth) : 1;
      static auto func = SharedToDotOperandMMAv1::load_getMNCoords_func(
          "iluvatar", "getMNCoords");
      auto coords = func(threadId, loc, rewriter, mmaLayout.getWarpsPerCTA(),
                         mmaLayout, shape, bitwidth, elemVecSize, isTrans);
      return coords[elemId];
    } else {
      llvm_unreachable("Unexpected MMALayout version");
    }
  }
  if (isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>(layout)) {
    auto multiDimBase =
        emitBaseIndexForLayout(loc, rewriter, targetInfo, layout, type, false);
    SmallVector<SmallVector<unsigned>> offsets;
    assert(rank == 2);
    SmallVector<Value> multiDimOffset(rank);
    if (auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(layout)) {
      emitMfmaOffsetForCTA(mfmaLayout, offsets, 0, multiDimCTAInRepId[0],
                           multiDimCTAInRepId[1]);
    } else if (auto wmmaLayout = dyn_cast<AMDWmmaEncodingAttr>(layout)) {
      emitWmmaOffsetForCTA(wmmaLayout, offsets, 0, multiDimCTAInRepId[0],
                           multiDimCTAInRepId[1]);
    }
    multiDimOffset[0] = add(multiDimBase[0], i32_val(offsets[elemId][0]));
    multiDimOffset[1] = add(multiDimBase[1], i32_val(offsets[elemId][1]));
    return multiDimOffset;
  }
  llvm_unreachable("unexpected layout in getMultiDimOffset");
}

} // namespace LLVM
} // namespace mlir