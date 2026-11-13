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

#ifndef TRITON_CONVERSION_TRITONPPUGPU_TO_LLVM_AIU_UTILITY_H
#define TRITON_CONVERSION_TRITONPPUGPU_TO_LLVM_AIU_UTILITY_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir {
namespace LLVM {
namespace PPU {
SmallVector<Value> getStrides(const SharedMemoryObject &smemObj,
                              triton::gpu::MemDescType memDesc, Location loc,
                              RewriterBase &rewriter);

triton::gpu::CTAEncodingAttr
getExpandedCTALayout(MLIRContext *ctx, triton::gpu::CTAEncodingAttr ctaLayout);

Attribute getExpandedEncoding(Attribute encoding);

triton::gpu::MemDescType getExpandedDesc(triton::gpu::MemDescType descTy);

SharedMemoryObject
getExpandedSharedMemoryObject(ConversionPatternRewriter &rewriter, Location loc,
                              SharedMemoryObject smemObj,
                              ArrayRef<int64_t> shape);

Value getSliceKOffset(ConversionPatternRewriter &rewriter, Location loc,
                      Value tensor, int opIdx);

DenseMap<unsigned, Value> getAIUSwizzledSharedPtrs(
    Location loc, const TargetInfoBase &target, unsigned inVec,
    RankedTensorType srcTy, triton::gpu::MemDescType memTy,
    triton::gpu::PPUAIUSharedEncodingAttr resSharedLayout, Type resElemTy,
    SharedMemoryObject smemObj, RewriterBase &rewriter,
    SmallVectorImpl<Value> &offsetVals);

// The address in a PPU0010 AIU buffer of the element at `idxCol` of the
// contiguous dimension and `idxRow` of the other one, which are i32 values.
Value getAIUSwizzledSharedPtr(Location loc, triton::gpu::MemDescType memTy,
                              Type resElemTy, SharedMemoryObject smemObj,
                              RewriterBase &rewriter, Value idxCol,
                              Value idxRow);

inline llvm::SmallVector<unsigned>
AIULoadStrategy(unsigned numWarps, unsigned xElems, unsigned channelElems,
                unsigned elemBytes, unsigned version = 1) {
  if (version == 1) {
    unsigned sliceByte = 32;
    unsigned maxSliceBytes = 128;
    unsigned channelBytes = channelElems * elemBytes;
    unsigned minXElems = 16;
    unsigned maxCubeW = 2048;
    auto sliceTotal = channelBytes / sliceByte;
    unsigned numSlice, cubeC, cubeW, warpC, warpW;

    assert(channelBytes >= sliceByte &&
           "channelBytes must be greater than or equal to sliceByte(32B)");

    if (channelBytes <= maxSliceBytes) {
      numSlice = channelBytes / sliceByte;
    } else {
      if (sliceTotal % 4 == 0) {
        numSlice = 4;
      } else if (sliceTotal % 2 == 0) {
        numSlice = 2;
      } else {
        numSlice = 1;
      }
    }
    unsigned channelCopy = sliceTotal / numSlice;
    warpC = 1;
    warpW = 1;
    unsigned maxWarpW = xElems / 16;

    if (numWarps % channelCopy == 0) {
      warpC = channelCopy;
      warpW = numWarps / channelCopy;
      warpW = std::min<unsigned>(warpW, maxWarpW);
    } else {
      if (channelCopy % numWarps == 0) {
        warpC = numWarps;
        warpW = 1;
      } else {
        warpC = 1;
        warpW = std::min<unsigned>(numWarps, maxWarpW);
      }
    }
    cubeW = xElems / warpW;
    cubeC = numSlice * sliceByte / elemBytes;
    // TODO: should split copy in W
    assert(cubeW <= maxCubeW && "cubeW exceed the limitation");
    return {cubeC, cubeW, warpC, warpW, numSlice};
  } else if (version == 2) {
    assert(channelElems % 64 == 0 || channelElems % 32 == 0 ||
           channelElems % 16 == 0);

    unsigned swizzledBytes;
    if (channelElems * elemBytes <= 64) {
      swizzledBytes = 64;
    } else {
      swizzledBytes = 128;
    }

    unsigned channelBytes = elemBytes * channelElems;
    unsigned cubeC;
    if (channelBytes < swizzledBytes) {
      cubeC = channelElems;
    } else {
      cubeC = swizzledBytes / elemBytes;
    }
    unsigned warpCMax = (channelElems + cubeC - 1) / cubeC;
    unsigned warpC = std::min<unsigned>(numWarps, warpCMax);
    unsigned warpWMax = xElems / 16;
    unsigned warpW = std::min<unsigned>(numWarps / warpC, warpWMax);
    unsigned cubeW = xElems / warpW;
    unsigned maxCubeW = 2048;
    // TODO: should split copy in W
    assert(cubeW <= maxCubeW && "cubeW exceed the limitation");

    return {cubeC, cubeW, warpC, warpW, swizzledBytes};
  } else {
    assert(false && "Unknown PPU AIU version");
  }
  return {};
}

// The PPU0010 AIU descriptor has no row stride: the 2d form copies from a dense
// [X, Z] tensor and the 3d form from a dense [Y, X, Z] tensor, and both zero
// fill out of bounds. Rows that are S elements apart are copied as the 2d
// tensor [rows, S] when nothing past the row length V has to be zero filled,
// and as the 3d tensor [rows, S / V, V] at x = 0 otherwise, which needs S to
// be a multiple of V. A layout with neither form stays a plain load when S and
// V are constants, and is copied by the threads when it is found at run time.
//
// The widths of the size fields are not documented. On a PPU-ZW810E, a 2d row
// length (Z) of 40000, a 3d row count (Y) of 32768 (40000 rows, clamped by the
// lowering) and a 3d stride factor (X) of 32768 copy correctly. kAIUMaxDim,
// inclusive, is the limit the compiler applies to the fields a form is chosen
// from: the row stride a strided 2d copy holds in its row-length field, and
// both factors of a 3d copy. The row count of a 2d copy is passed through, the
// measurement above having found that field wider than the limit.
constexpr int64_t kAIUMaxDim = 1 << 15;

// Looks through the extensions that widen block-pointer shapes and strides to
// i64.
inline Value stripAIUIntExtensions(Value v) {
  while (isa_and_nonnull<arith::ExtSIOp, arith::ExtUIOp>(v.getDefiningOp()))
    v = v.getDefiningOp()->getOperand(0);
  return v;
}

inline std::optional<int64_t> getAIUConstantInt(Value v) {
  return getConstantIntValue(stripAIUIntExtensions(v));
}

// How the rows of a 2-D AIU tile lie in memory, from the strides and shape of
// its block pointer. `rowDim` is the dimension that order[0] does not name.
enum class AIURowLayout {
  Dense,   // the row stride is the row length
  Strided, // the row stride and the row length are different constants
  Runtime, // only known at run time
};

inline AIURowLayout getAIURowLayout(ValueRange strides, ValueRange shape,
                                    unsigned rowDim) {
  unsigned colDim = 1 - rowDim;
  // Without a unit stride on the dimension order[0] names, the copy reads the
  // tile in the dense layout `order` gives, which the PPU AIU tests expect
  // (test_bp_gemm_order01).
  if (strides.empty() || getAIUConstantInt(strides[colDim]) != 1)
    return AIURowLayout::Dense;
  Value rowStride = strides[rowDim];
  Value rowLength = shape[colDim];
  if (stripAIUIntExtensions(rowStride) == stripAIUIntExtensions(rowLength))
    return AIURowLayout::Dense;
  std::optional<int64_t> stride = getAIUConstantInt(rowStride);
  std::optional<int64_t> length = getAIUConstantInt(rowLength);
  if (!stride || !length)
    return AIURowLayout::Runtime;
  return *stride == *length ? AIURowLayout::Dense : AIURowLayout::Strided;
}

// Whether rows of `length` elements that are a constant `stride` apart can be
// copied, zero filling past the row length if `zeroFillColumns`: the 2d form
// holds the stride as its row length, the 3d form holds the row length and
// the stride as a multiple of it, each within kAIUMaxDim.
inline bool isAIUStrideExpressible(int64_t stride, int64_t length,
                                   bool zeroFillColumns) {
  if (length <= 0 || stride < length)
    return false;
  if (!zeroFillColumns)
    return stride <= kAIUMaxDim;
  return stride % length == 0 && length <= kAIUMaxDim &&
         stride / length <= kAIUMaxDim;
}

} // namespace PPU
} // namespace LLVM
} // namespace mlir

#endif
