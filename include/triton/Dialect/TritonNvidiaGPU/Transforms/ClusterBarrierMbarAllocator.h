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

#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERMBARALLOCATOR_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERMBARALLOCATOR_H_

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace triton {
namespace nvidia_gpu {

inline constexpr llvm::StringLiteral kClusterBarrierMbarOffsetAttrName =
    "ttg.mbar_offset";
inline constexpr llvm::StringLiteral kWSClusterBarrierCountAttrName =
    "ttg.ws_cluster_barrier_count";
inline constexpr int64_t kClusterBarrierMbarSlotSize = 16;
inline constexpr int64_t kClusterBarrierMbarBufferCount = 2;
inline constexpr int64_t kClusterBarrierMbarAllocationSize =
    kClusterBarrierMbarSlotSize * kClusterBarrierMbarBufferCount;

inline void copyClusterBarrierMbarOffset(Operation *src, Operation *dst) {
  if (Attribute attr = src->getAttr(kClusterBarrierMbarOffsetAttrName))
    dst->setAttr(kClusterBarrierMbarOffsetAttrName, attr);
}

bool needsClusterBarrier(Operation *op);

void runClusterBarrierMbarAllocator(ModuleOp mod);

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERMBARALLOCATOR_H_
