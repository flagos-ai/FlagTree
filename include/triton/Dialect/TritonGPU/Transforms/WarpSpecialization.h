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

#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_WARPSPECIALIZATION_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_WARPSPECIALIZATION_H_

#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace scf {
class ForOp;
} // namespace scf
namespace triton::gpu {
// This is the final step to prepare a loop for warp specialization. This takes
// a loop with a partition schedule and rewrites the loop such that all SSA
// dependencies between partitions are passed through shared memory and
// multibuffers them according to partition stages.
LogicalResult rewritePartitionDependencies(scf::ForOp &loop);
// Given a loop where the partitions' inputs and outputs have been fully
// rewritten to be reference semantic, partitiong the loop into a
// `ttg.warp_specialize` by duplicating the loop for each partition and
// rematerializing, as necessary, operations in the root partition.
LogicalResult partitionLoop(scf::ForOp loop);
} // namespace triton::gpu
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_WARPSPECIALIZATION_H_
