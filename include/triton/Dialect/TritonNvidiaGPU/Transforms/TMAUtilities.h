/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
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

#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::nvidia_gpu {

constexpr inline int TMA_SIZE_BYTES = 128;
constexpr inline int TMA_ALIGN = 128;

inline bool isFp4Padded(Attribute encoding) {
  auto mmaEnc = dyn_cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return mmaEnc && mmaEnc.getFp4Padded();
}

triton::gpu::SharedEncodingTrait
getEncodingFromDescriptor(Operation *op, RankedTensorType tensorType,
                          Value desc);

bool hasCGABroadcast(gpu::MemDescType memDescType);

Value sextI16ToI32Indices(Value indices, OpBuilder &builder, Location loc);

inline SmallVector<int64_t> getTMABlockShape(Attribute encoding,
                                             ArrayRef<int64_t> shapePerCTA,
                                             bool packedSize,
                                             gpu::TMAMode mode) {
  auto mmaEnc = cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return triton::gpu::getTMABlockShape(
      shapePerCTA, mmaEnc.getElementBitWidth(), mmaEnc.getSwizzlingByteWidth(),
      mmaEnc.getFp4Padded(), mmaEnc.getTransposed(), packedSize, mode);
}

inline SmallVector<int64_t> getTMABlockShape(triton::gpu::MemDescType ty,
                                             bool packedSize,
                                             gpu::TMAMode mode) {
  auto shapePerCTA = gpu::getShapePerCTA(ty);
  return getTMABlockShape(ty.getEncoding(), shapePerCTA, packedSize, mode);
}

inline SmallVector<int64_t> getTMABlockShape(triton::TensorDescInterface ty,
                                             bool packedSize,
                                             gpu::TMAMode mode) {
  auto shapePerCTA = gpu::getShapePerCTA(ty.getSharedLayout(), ty.getShape());
  return getTMABlockShape(ty.getSharedLayout(), shapePerCTA, packedSize, mode);
}

FailureOr<int> getTMASwizzleMode(Location loc, triton::TensorDescInterface ty);
FailureOr<int> getTMAElementType(Location loc, triton::TensorDescInterface ty);

LogicalResult createTMADesc(Value tmaPtr, MakeTensorDescOp op,
                            OpBuilder &builder);

} // namespace mlir::triton::nvidia_gpu
