/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir {
namespace LLVM {
namespace PPU {

SmallVector<unsigned> getOrderForShape(ArrayRef<int64_t> shape,
                                       ArrayRef<unsigned> layoutOrder) {
  SmallVector<unsigned> order(shape.size());
  // Default minor-to-major order
  std::iota(order.rbegin(), order.rend(), 0);
  if (layoutOrder.size() > 0) {
    // If a layout order is provided, we assume it specifies the order in
    // which the dimensions are first accessed, and unspecified dimensions
    // retain the minor-to-major order. For example, if order = [2, 1, 0] and
    // layoutOrder = [0, 1], we need to shift `layoutOrder`
    // by -1 (move them right). The resulting order will then be [1, 2, 0].
    int rankDiff = layoutOrder.size() - shape.size();
    auto minRank = std::min<size_t>(shape.size(), layoutOrder.size());
    for (size_t i = 0; i < minRank; ++i)
      order[i] = layoutOrder[i] - rankDiff;
  }
  assert(isPermutationOfIota(order) && "Invalid order");
  return order;
}

SmallVector<Value> getStrides(const SharedMemoryObject &smemObj,
                              triton::gpu::MemDescType memDesc, Location loc,
                              RewriterBase &rewriter) {
  auto allocShape = memDesc.getAllocShape();
  auto allocShapePerCTA =
      triton::gpu::getAllocationShapePerCTA(memDesc.getEncoding(), allocShape);
  auto layoutOrder = triton::gpu::getOrder(memDesc);
  SmallVector<Value> allocStrides(allocShapePerCTA.size());
  auto order = getOrderForShape(allocShapePerCTA, layoutOrder);

  int64_t stride = 1;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  for (auto idx : order) {
    allocStrides[idx] = b.i32_val(stride);
    stride *= allocShapePerCTA[idx];
  }
  return SmallVector<Value>(allocStrides.end() - smemObj.getOffsets().size(),
                            allocStrides.end());
}

template <typename T>
SmallVector<T> insertValue(ArrayRef<T> vec, unsigned index, T value) {
  SmallVector<T> res(vec.begin(), vec.end());
  res.insert(res.begin() + index, value);
  return res;
}
template <typename T>
SmallVector<T> insertValue(const SmallVector<T> &vec, unsigned index, T value) {
  SmallVector<T> res(vec.begin(), vec.end());
  res.insert(res.begin() + index, value);
  return res;
}

Attribute getExpandedEncoding(Attribute encoding) {
  auto ctx = encoding.getContext();
  if (auto sharedEncoding =
          mlir::dyn_cast<triton::gpu::PPUAIUSharedEncodingAttr>(encoding)) {
    auto order = sharedEncoding.getOrder();
    auto rank = order.size();
    if (rank == 3) {
      return encoding;
    }
    auto expandedOrder = SmallVector<unsigned>(3, 0);
    expandedOrder[0] = order[0] + 1;
    expandedOrder[1] = order[1] + 1;
    ArrayRef<unsigned> expandedOrderArr(expandedOrder);
    auto expandedEncoding = triton::gpu::PPUAIUSharedEncodingAttr::get(
        ctx, sharedEncoding.getVersionMajor(), sharedEncoding.getAIUStrategy(),
        expandedOrderArr, sharedEncoding.getCTALayout());
    return expandedEncoding;
  } else if (auto mmaEncoding = mlir::dyn_cast<PPUMmaEncodingAttr>(encoding)) {
    // auto warpsPerCTA = triton::gpu::getWarpsPerCTA(mmaEncoding);
    auto warpsPerCTA = mmaEncoding.getWarpsPerCTA();
    auto rank = warpsPerCTA.size();
    if (rank == 3) {
      return encoding;
    }
    auto expandedWarpsPerCTA = insertValue<unsigned>(warpsPerCTA, 0, 1);
    auto instrShape = mmaEncoding.getInstrShape();
    auto expandedInstrShape = insertValue<unsigned>(instrShape, 0, 1);
    auto expandedMmaEncoding = PPUMmaEncodingAttr::get(
        ctx, mmaEncoding.getVersionMajor(), mmaEncoding.getVersionMinor(),
        expandedWarpsPerCTA, mmaEncoding.getCTALayout(), expandedInstrShape, 2);
    return expandedMmaEncoding;
  } else if (auto dotOperandEncoding =
                 mlir::dyn_cast<DotOperandEncodingAttr>(encoding)) {
    auto mmaEncoding =
        mlir::cast<PPUMmaEncodingAttr>(dotOperandEncoding.getParent());
    auto expandedMMAEncoding = getExpandedEncoding(mmaEncoding);
    auto expandedEncoding = DotOperandEncodingAttr::get(
        ctx, dotOperandEncoding.getOpIdx(), expandedMMAEncoding,
        dotOperandEncoding.getKWidth());
    return expandedEncoding;
  } else
    llvm_unreachable("unsupported encoding");
}

triton::gpu::MemDescType getExpandedDesc(triton::gpu::MemDescType descTy) {
  auto shapePerCTA = getShapePerCTA(descTy);
  auto rank = shapePerCTA.size();
  if (rank == 3)
    return descTy;

  auto elTy = descTy.getElementType();
  auto shape = descTy.getShape();
  auto expandedShape = SmallVector<int64_t>(3, 1);
  expandedShape[1] = shape[0];
  expandedShape[2] = shape[1];
  auto encoding = descTy.getEncoding();
  auto expandedEncoding = getExpandedEncoding(encoding);
  auto expandedDesc = triton::gpu::MemDescType::get(
      expandedShape, elTy, expandedEncoding, descTy.getMemorySpace());
  return expandedDesc;
}

SharedMemoryObject
getExpandedSharedMemoryObject(ConversionPatternRewriter &rewriter, Location loc,
                              SharedMemoryObject smemObj,
                              ArrayRef<int64_t> shape) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(shape.size() == 2 || shape.size() == 3);
  auto offsets = smemObj.getOffsets();
  auto rank = offsets.size();
  assert(rank == shape.size());
  if (rank == 3)
    return smemObj;
  offsets.insert(offsets.begin(), b.i32_val(0));
  auto expandedSmemObj =
      SharedMemoryObject(smemObj.getBase(), smemObj.getBaseElemType(), offsets);
  return expandedSmemObj;
}

Value getSliceKOffset(ConversionPatternRewriter &rewriter, Location loc,
                      Value tensor, int opIdx) {
  auto builder = TritonLLVMOpBuilder(loc, rewriter);
  if (auto subView =
          dyn_cast<triton::gpu::MemDescIndexOp>(tensor.getDefiningOp())) {
    // channel offset is always the last dimension of builder.sub memory slice
    auto memTy = subView.getType();
    auto aiuEnc =
        cast<triton::gpu::PPUAIUSharedEncodingAttr>(memTy.getEncoding());
    auto kOffset = aiuEnc.getKOffset();
    Value baseOff = builder.i32_val(kOffset);
    return baseOff;
  } else {
    return builder.i32_val(0);
  }
}

DenseMap<unsigned, Value> getPPUAIUV1SwizzledSharedPtrs(
    Location loc, const TargetInfoBase &target, unsigned inVec,
    RankedTensorType srcTy, triton::gpu::MemDescType memTy,
    triton::gpu::PPUAIUSharedEncodingAttr resSharedLayout, Type resElemTy,
    SharedMemoryObject smemObj, RewriterBase &rewriter,
    SmallVectorImpl<Value> &offsetVals) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto dstPtrTy = ptr_ty(rewriter.getContext(), 3);
  auto dstOffset =
      dot(rewriter, loc, offsetVals, getStrides(smemObj, memTy, loc, rewriter));
  Value dstPtrBase = b.gep(dstPtrTy, resElemTy, smemObj.getBase(), dstOffset);

  auto srcEncoding = srcTy.getEncoding();
  auto srcShape = srcTy.getShape();
  auto srcShapePerCTA = triton::gpu::getShapePerCTA(srcTy);
  unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
  // swizzling params as described in TritonGPUAttrDefs.td
  unsigned outVec = 8;
  auto outOrder = resSharedLayout.getOrder();

  // Return values
  DenseMap<unsigned, Value> ret;
  // Tensor indices held by the current thread, as LLVM values
  auto srcIndices = emitIndices(loc, rewriter, target, srcEncoding, srcTy,
                                /*withCTAOffset=*/false);

  auto aiuLoad = resSharedLayout.getAIUStrategy();
  unsigned cubeC = aiuLoad[0];
  unsigned cubeW = aiuLoad[1];
  unsigned aiuWarpCopyC = aiuLoad[2];
  unsigned aiuWarpCopyW = aiuLoad[3];
  unsigned minVec = std::min(outVec, inVec);
  unsigned elemsPerSlice =
      16 * cubeW; // 32/(resElemTy.getIntOrFloatBitWidth()/8) * cubeW;
  unsigned elemsPerCopy = cubeC * cubeW * aiuWarpCopyC * aiuWarpCopyW;
  unsigned elemsPerCube = cubeC * cubeW;

  // for AIU load the outVec size should always be 8
  assert(outVec == 8 && "vec size of AIU load should always be 8");

  for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
    auto idx = srcIndices[elemIdx];
    Value idxCol = idx[outOrder[0]]; // contiguous dimension
    Value idxRow;
    if (outOrder.size() >= 2) {
      idxRow = idx[outOrder[1]]; // discontiguous dimension
    } else {
      idxRow = b.i32_val(0);
    }

    // element row index inside cube
    Value idxRowInnerCube = b.urem(idxRow, b.i32_val(cubeW));
    // cube row index inside tensor
    Value idxRowCube = b.udiv(idxRow, b.i32_val(cubeW));
    // element column index inside cube
    Value idxColInnerCube = b.urem(idxCol, b.i32_val(cubeC));
    // cube col index inside tensor
    Value idxColCube = b.udiv(idxCol, b.i32_val(cubeC));
    // warp channel index inside copy
    Value warpCIdx = b.urem(idxColCube, b.i32_val(aiuWarpCopyC));
    // warp width index insice copy
    Value warpWIdx = b.urem(idxRowCube, b.i32_val(aiuWarpCopyW));
    // copy index inside tensor
    Value copyIdx = b.udiv(idxColCube, b.i32_val(aiuWarpCopyC));
    // slice ID inside cube
    Value sliceID = b.udiv(idxColInnerCube, b.i32_val(16));
    // slice offset inside cube
    Value sliceOffset = b.mul(sliceID, b.i32_val(elemsPerSlice));
    // warp offset inside copy
    Value warpOffset =
        b.add(b.mul(warpWIdx, b.i32_val(elemsPerCube * aiuWarpCopyC)),
              b.mul(warpCIdx, b.i32_val(elemsPerCube)));
    // copy offset insice tensor
    Value copyOffset = b.mul(copyIdx, b.i32_val(elemsPerCopy));

    Value sliceStartOffset = b.add(copyOffset, b.add(warpOffset, sliceOffset));
    Value sliceStartPtr =
        b.gep(dstPtrTy, resElemTy, dstPtrBase, sliceStartOffset);

    // column index inside slice, slice shape is (cubeW, 16)
    Value idxColInnerSlice = b.urem(idxColInnerCube, b.i32_val(16));
    // new swizzled row index inside slice, swizzled slice shape is (cubeW/4,
    // 64)
    Value rowSwizzleID = b.udiv(idxRowInnerCube, b.i32_val(4));
    // new linear slice index inside slice, slice shape is (cubeW/4, 64)
    Value idxColSlicelinear =
        b.urem(b.add(b.mul(idxRowInnerCube, b.i32_val(16)), idxColInnerSlice),
               b.i32_val(64));
    // new column slice ID, fp16, vec=8
    // Value colSliceID = lshr(idxColSlicelinear, i32_val(3));
    Value colSliceID = b.udiv(idxColSlicelinear, b.i32_val(8));

    // rotated length: (((sliceID>1)|(sliceID<1))&0x3) << 1
    // sliceID 0, 1, 2, 3 ---> rotated length: 0, 4, 2, 6
    Value rotateLen = b.and_(
        b.or_(b.lshr(sliceID, b.i32_val(1)), b.shl(sliceID, b.i32_val(1))),
        b.i32_val(3));
    rotateLen = b.shl(rotateLen, b.i32_val(1));
    // rotate position bit
    Value colBitPos = b.sub(b.i32_val(7), colSliceID);
    Value ROrL = b.icmp_uge(colBitPos, rotateLen);
    Value rotR = b.sub(colBitPos, rotateLen);
    Value rotL = b.add(colBitPos, b.sub(b.i32_val(8), rotateLen));
    Value colRotBitPos = b.select(ROrL, rotR, rotL);
    Value colRotID = b.sub(b.i32_val(7), colRotBitPos);
    Value colSwizzleID = b.xor_(colRotID, b.urem(rowSwizzleID, b.i32_val(2)));

    Value swizzleOffset = b.add(b.mul(rowSwizzleID, b.i32_val(64)),
                                b.mul(colSwizzleID, b.i32_val(8)));

    // for minVec is not equal to outVec
    swizzleOffset = b.or_(swizzleOffset, b.urem(idxCol, b.i32_val(8)));
    ret[elemIdx] = b.gep(dstPtrTy, resElemTy, sliceStartPtr, swizzleOffset);
  }

  return ret;
}

DenseMap<unsigned, Value> getPPUAIUV2SwizzledSharedPtrs(
    Location loc, const TargetInfoBase &target, unsigned inVec,
    RankedTensorType srcTy, triton::gpu::MemDescType memTy,
    triton::gpu::PPUAIUSharedEncodingAttr resSharedLayout, Type resElemTy,
    SharedMemoryObject smemObj, RewriterBase &rewriter,
    SmallVectorImpl<Value> &offsetVals) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto dstPtrTy = ptr_ty(rewriter.getContext(), 3);
  auto dstOffset =
      dot(rewriter, loc, offsetVals, getStrides(smemObj, memTy, loc, rewriter));

  Value dstPtrBase = b.gep(dstPtrTy, resElemTy, smemObj.getBase(), dstOffset);
  auto srcEncoding = srcTy.getEncoding();
  unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
  auto outOrder = resSharedLayout.getOrder();
  auto elemBytes = resElemTy.getIntOrFloatBitWidth() / 8;

  // Return values
  DenseMap<unsigned, Value> ret;
  // Tensor indices held by the current thread, as LLVM values
  auto srcIndices = emitIndices(loc, rewriter, target, srcEncoding, srcTy,
                                /*withCTAOffset=*/false);

  auto aiuLoad = resSharedLayout.getAIUStrategy();
  unsigned cubeC = aiuLoad[0];
  unsigned cubeW = aiuLoad[1];
  unsigned aiuWarpCopyC = aiuLoad[2];
  unsigned aiuWarpCopyW = aiuLoad[3];
  unsigned swizzledBytes = aiuLoad[4];
  unsigned perPhase = 1;
  unsigned maxPhase = 8;
  if (swizzledBytes == 64) {
    perPhase = 2;
    maxPhase = 4;
  }
  unsigned outVec = 8; // the outVec size for AIU load should always be 8
  unsigned minVec = std::min(outVec, inVec);
  unsigned swizzledElems = swizzledBytes / elemBytes;

  // When cubeC is smaller than swizzledElems(i.e. channel bytes < swizzled
  // bytes), we use aiuFactor to skip invalid data in shared memory
  unsigned aiuFactor = swizzledElems / cubeC;
  assert((cubeC == 16 && aiuFactor == 2) || (cubeC != 16 && aiuFactor == 1));

  unsigned elemsPerCube = cubeC * cubeW * aiuFactor;
  unsigned elemsPerCopy = elemsPerCube * aiuWarpCopyC * aiuWarpCopyW;

  for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
    auto idx = srcIndices[elemIdx];
    Value idxCol = idx[outOrder[0]]; // contiguous dimension
    Value idxRow;
    if (outOrder.size() >= 2) {
      idxRow = idx[outOrder[1]]; // discontiguous dimension
    } else {
      idxRow = b.i32_val(0);
    }

    // cube row index inside tensor
    Value idxRowCube = b.udiv(idxRow, b.i32_val(cubeW));
    // cube col index inside tensor
    Value idxColCube = b.udiv(idxCol, b.i32_val(cubeC));
    // warp channel index inside copy
    Value warpCIdx = b.urem(idxColCube, b.i32_val(aiuWarpCopyC));
    // warp width index inside copy
    Value warpWIdx = b.urem(idxRowCube, b.i32_val(aiuWarpCopyW));
    // copy index inside tensor
    Value copyIdx = b.udiv(idxColCube, b.i32_val(aiuWarpCopyC));
    // warp offset inside copy
    Value warpOffset =
        b.add(b.mul(warpWIdx, b.i32_val(elemsPerCube * aiuWarpCopyC)),
              b.mul(warpCIdx, b.i32_val(elemsPerCube)));
    // copy offset inside tensor
    Value copyOffset = b.mul(copyIdx, b.i32_val(elemsPerCopy));
    Value cubeStartOffset = b.add(copyOffset, warpOffset);

    // Computes the pointers for accessing the provided swizzled
    // shared memory layout `resSharedLayout`. More specifically, it computes,
    // for all indices (row, col) of `srcEncoding` such that idx % inVec = 0,
    // the pointer: ptr[(row, col)] = base + (rowOff * strideRow + colOff)
    // where:
    //   phase = (row // perPhase) % maxPhase
    //   rowOff = row
    //   colOff = colOffSwizzled + colOffOrdered
    //     colOffSwizzled = ((col // outVec) ^ phase) * outVec
    //     colOffOrdered = (col % outVec) // minVec * minVec

    // element row index inside cube
    Value idxRowInnerCube = b.urem(idxRow, b.i32_val(cubeW));
    // element column index inside cube
    Value idxColInnerCube = b.urem(idxCol, b.i32_val(cubeC * aiuFactor));

    // To support cubeC smaller than swizzledElems(i.e. cubeC = 16), we need to
    // update indices (row, col) as follows:
    //   row = row + col // cubeC
    //   col = col % cubeC
    idxRowInnerCube =
        b.add(idxRowInnerCube, b.udiv(idxColInnerCube, b.i32_val(cubeC)));
    idxColInnerCube = b.urem(idxColInnerCube, b.i32_val(cubeC));

    // phase = (row // perPhase) % maxPhase
    Value phase = b.urem(b.udiv(idxRowInnerCube, b.i32_val(perPhase)),
                         b.i32_val(maxPhase));

    // row offset is simply idxRowInnerCube * numElemsPerSwizzlingRow
    Value rowOff = b.mul(idxRowInnerCube, b.i32_val(cubeC * aiuFactor));

    // Because swizzling happens at a granularity of outVec, we need to
    // decompose colOff into a swizzled factor and a non-swizzled
    // (ordered) factor: colOffSwizzled = ((col // outVec) ^ phase) * outVec
    // colOffOrdered = (col % outVec) // minVec * minVec
    Value colOffSwizzled =
        b.xor_(b.udiv(idxColInnerCube, b.i32_val(outVec)), phase);
    colOffSwizzled = b.mul(colOffSwizzled, b.i32_val(outVec));
    Value colOffOrdered = b.urem(idxColInnerCube, b.i32_val(outVec));
    colOffOrdered = b.udiv(colOffOrdered, b.i32_val(minVec));
    colOffOrdered = b.mul(colOffOrdered, b.i32_val(minVec));
    Value colOff = b.add(colOffSwizzled, colOffOrdered);

    Value offset = b.add(cubeStartOffset, b.add(rowOff, colOff));
    ret[elemIdx] = b.gep(dstPtrTy, resElemTy, dstPtrBase, offset);
  }

  return ret;
}

DenseMap<unsigned, Value> getAIUSwizzledSharedPtrs(
    Location loc, const TargetInfoBase &target, unsigned inVec,
    RankedTensorType srcTy, triton::gpu::MemDescType memTy,
    triton::gpu::PPUAIUSharedEncodingAttr resSharedLayout, Type resElemTy,
    SharedMemoryObject smemObj, RewriterBase &rewriter,
    SmallVectorImpl<Value> &offsetVals) {
  if (resSharedLayout.isPPU0010()) {
    return getPPUAIUV1SwizzledSharedPtrs(loc, target, inVec, srcTy, memTy,
                                         resSharedLayout, resElemTy, smemObj,
                                         rewriter, offsetVals);
  } else {
    assert(resSharedLayout.isPPU0015());
    return getPPUAIUV2SwizzledSharedPtrs(loc, target, inVec, srcTy, memTy,
                                         resSharedLayout, resElemTy, smemObj,
                                         rewriter, offsetVals);
  }
}

} // namespace PPU
} // namespace LLVM
} // namespace mlir
