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

#ifndef TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_
#define TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

class OpBuilder;
class DominanceInfo;

namespace scf {
class ForOp;
} // namespace scf
namespace triton::nvidia_gpu {

//===----------------------------------------------------------------------===//
// MMA Pipeline Analysis
//===----------------------------------------------------------------------===//

// Given an MMAv5 operation in a loop, determine if its accumulator can be
// multibuffered.
bool isAccMultibufferingPossible(MMAv5OpInterface mma, scf::ForOp forOp);

// Returns true if the MMA operation requires acc multi-buffering when
// pipelined.
bool requiresAccMultiBuffering(MMAv5OpInterface mma, scf::ForOp forOp);

// Returns true if there are loads from tmem after the MMA operation.
bool hasLoadsAfterMMA(MMAv5OpInterface mma, scf::ForOp forOp);

// Helper class to determine if the operands of an MMA operation are
// pipelineable.
class MMAv5PipelineableOperandsHelper {
public:
  MMAv5PipelineableOperandsHelper(
      MMAv5OpInterface mmaOp, scf::ForOp forOp,
      std::function<bool(Operation *)> isLoadToBePipelined)
      : mmaOp(mmaOp), forOp(forOp), isLoadToBePipelined(isLoadToBePipelined) {
    run();
  }

  bool isPipelineable = false;
  // If true, the existing operand loads are all been found and their
  // pipelineability has been determined.
  bool isOperandsStateDetermined = false;
  SmallVector<Operation *> unpipelineableOperandDefs;

private:
  MMAv5OpInterface mmaOp;
  scf::ForOp forOp;
  std::function<bool(Operation *)> isLoadToBePipelined;
  void run();
  bool isOperandPipelineable(Value v, Operation *&foundDef);
};

bool areScalesPipelineable(TCGen5MMAScaledOp scaledOp, scf::ForOp forOp);
bool isOperandPipelineableBase(
    Value v, scf::ForOp forOp, Operation *&foundDef,
    std::function<bool(Operation *)> isPipelineable =
        [](Operation *) { return false; },
    std::function<bool(Operation *)> isLoadToBePipelined =
        [](Operation *) { return false; });

//===----------------------------------------------------------------------===//
// MMA Pipeline Rewriters
//===----------------------------------------------------------------------===//

// Create a new TMEMAllocOp to use for the pipelined MMA operation. It is
// optionally multi-buffered based on the number of stages.
TMEMAllocOp createTMemAlloc(OpBuilder &builder, TMEMAllocOp oldTMemAllocOp,
                            bool multiBufferred, int numStages);

// Return true if the accumulator of an mma in subsequent iterations is either
// independent from the previous iteration (overwritten) or completely reused,
// without read-modify-write.
// Otherwise, we can not pipeline the MMA, as we need to insert a wait after the
// mma to read back the accumulator for RMW.
bool hasAccReadModifyWrite(MMAv5OpInterface mma, scf::ForOp forOp);

} // namespace triton::nvidia_gpu
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_
