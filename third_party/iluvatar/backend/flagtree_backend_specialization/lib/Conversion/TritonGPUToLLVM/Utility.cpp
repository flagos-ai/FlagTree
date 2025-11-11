#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "python/src/plugin.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace SharedToDotOperandMMAv1 {

using getMNCoordsFunc = SmallVector<CoordTy> (*)(
    Value, Location, ConversionPatternRewriter &, ArrayRef<unsigned int>,
    const IluvatarMmaEncodingAttr &, ArrayRef<int64_t>, int, int, bool);
DEFINE_LOAD_FUNC(getMNCoords)

} // namespace SharedToDotOperandMMAv1

using namespace mlir;
using namespace mlir::triton;

using emitOffsetForTCULayoutFunc = SmallVector<SmallVector<unsigned>> (*)(
    const triton::gpu::IluvatarMmaEncodingAttr &, RankedTensorType);
DEFINE_LOAD_FUNC(emitOffsetForTCULayout)

using emitBaseIndexForTCULayoutFunc = SmallVector<Value> (*)(
    Location, RewriterBase &, const triton::gpu::IluvatarMmaEncodingAttr &,
    RankedTensorType);
DEFINE_LOAD_FUNC(emitBaseIndexForTCULayout)

using remapOffsetFunc = Value (*)(Value, Value, RankedTensorType, bool,
                                  Location, RewriterBase &, int, bool);
DEFINE_LOAD_FUNC(remapOffset)

namespace mlir {

SmallVector<Value> emitBaseIndexForLayoutImpl_BackendMmaEncodingAttr(
    Location loc, RewriterBase &rewriter, const Attribute &layout,
    RankedTensorType type) {
  SmallVector<Value> result;
  if (auto mmaLayout = mlir::dyn_cast<IluvatarMmaEncodingAttr>(layout)) {
    if (mmaLayout.isVolta()) {
      DEFINE_CALL_LOAD_FUNC(iluvatar, emitBaseIndexForTCULayout);
      result = func(loc, rewriter, mmaLayout, type);
    }
  }
  return result;
}

SmallVector<SmallVector<unsigned>> emitOffsetForLayout_BackendMmaEncodingAttr(
    const IluvatarMmaEncodingAttr &mmaLayout, RankedTensorType type) {
  if (mmaLayout.isVolta()) {
    DEFINE_CALL_LOAD_FUNC(iluvatar, emitOffsetForTCULayout)
    return func(mmaLayout, type);
  }
  llvm_unreachable("unsupported emitOffsetForLayout");
}

Value getSwizzledSharedPtrs_backend(
    Location loc, RewriterBase &rewriter, RankedTensorType srcTy,
    ArrayRef<Value> idx, triton::gpu::SharedEncodingAttr resSharedLayout,
    Type resElemTy, SharedMemoryObject smemObj, Type dstPtrTy, Value dstPtrBase,
    Value idxRow, Value idxCol, ArrayRef<unsigned> outOrder, unsigned perPhase,
    Value strideRow, Value strideCol) {
  bool isRow = outOrder[0] == 1;
  Value off = NULL;
  auto capability = getNVIDIAComputeCapability(
      smemObj.base.getDefiningOp()->getParentOfType<ModuleOp>());
  if (resSharedLayout.getUseTcu() && outOrder.size() == 2) {
    DEFINE_CALL_LOAD_FUNC(iluvatar, remapOffset)
    off = func(idx[0], idx[1], srcTy, isRow, loc, rewriter, capability,
               !perPhase);
  } else {
    off = add(mul(idxCol, strideCol), mul(idxRow, strideRow));
  }
  return gep(dstPtrTy, resElemTy, dstPtrBase, off);
}

unsigned
storeDistributedToShared_outVec(triton::gpu::SharedEncodingAttr layout) {
  return layout.getVec();
}

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

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(ctx), globalType,
        /*isConstant=*/true, LLVM::Linkage::Private, stringConstName,
        rewriter.getStringAttr(contentStr), 1, 4);
  }

  Value zero = i32_val(0);
  Type globalPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value localPtr = addrspacecast(ptr_ty(ctx), globalPtr);
  Value stringStart =
      gep(ptr_ty(ctx), i8_ty, localPtr, SmallVector<Value>({zero}));
  return stringStart;
}

} // namespace LLVM

SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(Location loc, RewriterBase &rewriter,
                  const LinearLayout &layout,
                  ArrayRef<std::pair<StringAttr, Value>> indices) {
  assert(layout.getNumInDims() == indices.size());
  for (auto [inDimName, idx] : indices) {
    assert(layout.hasInDim(inDimName) && "Invalid inDimName");
  }

  // This function can emit a lot of MLIR code, which ultimately makes
  // compilation slow.  (We think this shouldn't be the case -- it's not *that*
  // much code -- but we're not clear on how to fix the slowness, which happens
  // in the bowels of MLIR.)
  //
  // As a result we go through some contortions to avoid emitting code where
  // possible.

  // Manually constant-fold the layout where possible.
  SmallVector<std::pair<StringAttr, int32_t>> constantIns;
  for (auto [inDimName, idx] : indices) {
    if (auto constant = dyn_cast<LLVM::ConstantOp>(idx.getDefiningOp())) {
      constantIns.push_back(
          {inDimName, constant.getValue().cast<IntegerAttr>().getInt()});
    } else {
      constantIns.push_back({inDimName, 0});
    }
  }
  SmallVector<int32_t> constantComponent =
      llvm::to_vector(llvm::make_second_range(layout.apply(constantIns)));

  Value zero = i32_val(0);
  SmallVector<std::pair<StringAttr, Value>> outIndices;
  for (auto [i, outDimName] : llvm::enumerate(layout.getOutDimNames())) {
    if (constantComponent[i] == 0)
      outIndices.push_back({outDimName, zero});
    else
      outIndices.push_back({outDimName, i32_val(constantComponent[i])});
  }

  for (auto [inDimName, idx] : indices) {
    if (isa<LLVM::ConstantOp>(idx.getDefiningOp())) {
      continue;
    }

    int nBits = layout.getInDimSizeLog2(inDimName);
    for (int i = 0; i < nBits; i++) {
      Value bit = and_(idx, i32_val(1 << i));
      Value bit_is_zero = icmp_eq(bit, zero);
      for (auto &[outDimName, outIdx] : outIndices) {
        int32_t basis = layout.getBasis(inDimName, i, outDimName);
        if (basis == 0)
          continue;
        outIdx = xor_(outIdx, select(bit_is_zero, zero, i32_val(basis)));
      }
    }
  }

  return outIndices;
}

} // namespace mlir
