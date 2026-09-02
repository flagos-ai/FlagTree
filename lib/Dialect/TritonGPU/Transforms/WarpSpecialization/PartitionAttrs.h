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

#ifndef TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_PARTITIONATTRS_H_
#define TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_PARTITIONATTRS_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace mlir {
class Operation;
class OpOperand;
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

namespace mlir::triton::gpu {

inline constexpr char kPartitionAttrName[] = "ttg.partition";
inline constexpr char kPartitionOutputsAttrName[] = "ttg.partition.outputs";
inline constexpr char kPartitionStagesAttrName[] = "ttg.partition.stages";
inline constexpr char kWarpSpecializeTagAttrName[] = "ttg.warp_specialize.tag";

SetVector<int> getPartitionIds(Operation *op);
SmallVector<SetVector<int>, 4> getPartitionOutputs(Operation *op);
SetVector<int> getPartitionIds(OpOperand *use);
bool hasPartition(Operation *op);
bool hasWarpSpecializeTag(Operation *op);
std::optional<int> getWarpSpecializeTag(Operation *op);

LogicalResult verifyPartitionedLoop(scf::ForOp loop);

} // namespace mlir::triton::gpu

#endif // TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_PARTITIONATTRS_H_
