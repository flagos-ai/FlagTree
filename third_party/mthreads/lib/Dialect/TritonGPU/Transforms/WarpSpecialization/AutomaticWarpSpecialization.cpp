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

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUAUTOMATICWARPSPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct AutomaticWarpSpecialization
    : triton::gpu::impl::TritonGPUAutomaticWarpSpecializationBase<
          AutomaticWarpSpecialization> {
  using TritonGPUAutomaticWarpSpecializationBase::
      TritonGPUAutomaticWarpSpecializationBase;

  void runOnOperation() override;
};

void multiBufferTMADescriptors(ModuleOp mod, int numStages) {
  SetVector<scf::ForOp> descUpdateLoops;
  mod.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName)) {
      loop.walk([&](triton::MakeTensorDescOp op) {
        if (auto forOp = op->getParentOfType<scf::ForOp>()) {
          descUpdateLoops.insert(forOp);
        }
      });
    }
  });

  // +1 to make sure that overlapping of the next desc update and the oldest
  // inflight TMA load is safe
  const int numDescs = numStages + 1;
  // CoarseSchedule's notion of numStages is the maximuim loop-pipelining
  // stage + 1, see CoarseSchedule::deSerialize(). So if we want n buffers,
  // we need to pass n + 1 as numStages.
  triton::CoarseSchedule schedule(numDescs + 1);

  for (auto loop : descUpdateLoops) {
    triton::lowerTMADescriptors(loop, schedule);
  }
}

} // namespace

void AutomaticWarpSpecialization::runOnOperation() {
  OpPassManager pm;
  pm.addPass(createTritonGPUPartitionScheduling());
  pm.addPass(createNVWSHoistTmemStore());
  pm.addPass(createNVWSInsertAref());
  pm.addPass(createNVWSInsertTmemAref());
  // `int-range-optimizations` and SCCP are good at cleaning up loop arithmetic.
  // FIXME: Re-enable integer range analysis once it is fixed.
  // pm.addPass(arith::createIntRangeOptimizationsPass());
  pm.addPass(createSCCPPass());
  pm.addPass(createCSEPass());
  pm.addPass(createNVWSLowerAref({numStages}));
  pm.addPass(createTritonGPUPartitionLoops());
  pm.addPass(createNVWSLowerWarpGroup());
  pm.addPass(createTritonGPUScheduleLoops());
  if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();

  // Multi-buffer TMA descriptors. We cannot rely on SWP to do it, to support
  // desc updates in nested loops.
  multiBufferTMADescriptors(getOperation(), numStages);
}
