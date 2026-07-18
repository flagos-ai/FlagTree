// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::nvidia_gpu {

constexpr inline int TMA_SIZE_BYTES = 128;
constexpr inline int TMA_ALIGN = 128;

inline bool isFp4Padded(Attribute encoding) {
  auto mmaEnc = dyn_cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return mmaEnc && mmaEnc.getFp4Padded();
}

SmallVector<Value> translateTMAIndices(OpBuilder &builder, Location loc,
                                       Attribute encoding,
                                       SmallVector<Value> indices);

gpu::CTAEncodingAttr updateCTALayoutForShape(gpu::CTAEncodingAttr ctaLayout,
                                             ArrayRef<int64_t> shape);

gpu::SharedEncodingTrait
updateEncodingForShape(Operation *op, gpu::SharedEncodingTrait encoding,
                       RankedTensorType tensorType);

triton::gpu::SharedEncodingTrait
getEncodingFromDescriptor(Operation *op, RankedTensorType tensorType,
                          Value desc);

SmallVector<int64_t> getTMABlockShape(ArrayRef<int64_t> shapePerCTA,
                                      int elementBitWidth, int swizzleBytes,
                                      bool fp4Padded, bool transposed,
                                      bool packedSize);

inline SmallVector<int64_t> getTMABlockShape(Attribute encoding,
                                             ArrayRef<int64_t> shapePerCTA,
                                             bool packedSize) {
  auto mmaEnc = cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return getTMABlockShape(shapePerCTA, mmaEnc.getElementBitWidth(),
                          mmaEnc.getSwizzlingByteWidth(), mmaEnc.getFp4Padded(),
                          mmaEnc.getTransposed(), packedSize);
}

inline SmallVector<int64_t> getTMABlockShape(RankedTensorType ty,
                                             bool packedSize) {
  auto shapePerCTA = gpu::getShapePerCTA(ty);
  return getTMABlockShape(ty.getEncoding(), shapePerCTA, packedSize);
}

inline SmallVector<int64_t> getTMABlockShape(triton::gpu::MemDescType ty,
                                             bool packedSize) {
  auto shapePerCTA = gpu::getShapePerCTA(ty);
  return getTMABlockShape(ty.getEncoding(), shapePerCTA, packedSize);
}

std::optional<int> getTMASwizzleMode(Operation *op, TensorDescType ty);

std::optional<int> getTMAElementType(Operation *op, TensorDescType ty);

LogicalResult createTMADesc(Value tmaPtr, MakeTensorDescOp op,
                            OpBuilder &builder);

} // namespace mlir::triton::nvidia_gpu
