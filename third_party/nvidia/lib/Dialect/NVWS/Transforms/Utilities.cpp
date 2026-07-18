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

#include "Utilities.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir::triton::nvws {

Operation *createAlloc(OpBuilder &builder, Location loc,
                       MemDescType memDescType, Value src) {
  if (isa<SharedMemorySpaceAttr>(memDescType.getMemorySpace())) {
    return LocalAllocOp::create(builder, loc, memDescType, src);
  } else {
    assert(isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace()));
    return TMEMAllocOp::create(builder, loc, memDescType, src);
  }
}

ArefCreateOp createArefCreateOp(OpBuilder &builder, ArrayRef<Type> arefTypes,
                                ValueRange allocOps, Location loc) {
  auto ctx = builder.getContext();
  auto arefTy = ArefType::get(ctx, TypeArrayAttr::get(ctx, arefTypes));
  return ArefCreateOp::create(builder, loc, arefTy, allocOps);
}

int getArefDepth(MemDescType bufTy) {
  auto shape = bufTy.getShape();
  return isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding())
             ? 1
             : shape[0];
}

MemDescType getArefViewBufferType(MemDescType bufTy) {
  auto isScalesEnc =
      isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding());
  auto shape = bufTy.getShape();
  return gpu::MemDescType::get(isScalesEnc ? shape : shape.drop_front(),
                               bufTy.getElementType(), bufTy.getEncoding(),
                               bufTy.getMemorySpace(),
                               /*mutableMemory*/ true,
                               /*allocShape=*/bufTy.getAllocShape());
}

MemDescType getArefMultiBufferedType(MemDescType bufTy, int depth) {
  auto shape = bufTy.getShape();
  SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
  if (!isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding()))
    bufferShape.insert(bufferShape.begin(), depth);
  return gpu::MemDescType::get(bufferShape, bufTy.getElementType(),
                               bufTy.getEncoding(), bufTy.getMemorySpace(),
                               /*mutableMemory*/ true);
}

} // namespace mlir::triton::nvws
