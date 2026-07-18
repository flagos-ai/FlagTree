// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef NVIDIA_NVWS_TRANSFORMS_UTILITY_H_
#define NVIDIA_NVWS_TRANSFORMS_UTILITY_H_

#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::nvws {

Operation *createAlloc(OpBuilder &builder, Location loc,
                       gpu::MemDescType memDescType, Value src);

ArefCreateOp createArefCreateOp(OpBuilder &builder, ArrayRef<Type> arefTypes,
                                ValueRange allocOps, Location loc);

template <typename Range>
inline std::optional<int> findValuePosInRange(const Range &range,
                                              mlir::Value v) {
  for (auto [pos, arg] : llvm::enumerate(range)) {
    if (arg == v)
      return pos;
  }
  return {};
}

#if 0
struct PartitionId : std::pair<int, int> {
  PartitionId(int index, int tag) : std::pair<int, int>(index, tag) {}
  int &index() { return first; }
  int &tag() { return second; }
};

std::optional<PartitionId> getPartitionId(Operation *op);
#endif

gpu::MemDescType getArefViewBufferType(gpu::MemDescType arefBufType);
gpu::MemDescType getArefMultiBufferedType(gpu::MemDescType arefBufType,
                                          int depth);
int getArefDepth(gpu::MemDescType bufTy);

} // namespace mlir::triton::nvws

#endif // NVIDIA_NVWS_TRANSFORMS_UTILITY_H_
