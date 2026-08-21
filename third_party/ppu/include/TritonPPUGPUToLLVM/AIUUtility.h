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

} // namespace PPU
} // namespace LLVM
} // namespace mlir

#endif
