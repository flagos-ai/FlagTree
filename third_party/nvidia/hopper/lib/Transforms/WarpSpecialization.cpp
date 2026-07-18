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

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "nvgpu-warp-specialization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;
namespace ttnvws = mlir::triton::nvws;

namespace mlir {

void doTaskPartition(triton::FuncOp &funcOp, unsigned numWarpGroups);
int doTaskIdPropagate(triton::FuncOp &funcOp);
bool doDataPartition(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers);
#ifdef __TLE__
LogicalResult doTokenLowering(triton::FuncOp &funcOp,
                              unsigned numConsumerGroups);
#else
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);
#endif

#define GEN_PASS_DEF_NVGPUWARPSPECIALIZATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUWarpSpecializationPass
    : public impl::NVGPUWarpSpecializationBase<NVGPUWarpSpecializationPass> {
public:
  using impl::NVGPUWarpSpecializationBase<
      NVGPUWarpSpecializationPass>::NVGPUWarpSpecializationBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    SmallVector<scf::ForOp> loops;
    funcOp->walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(mlir::triton::kWarpSpecializeAttrName))
        loops.push_back(forOp);
    });
    if (loops.empty()) {
      bool hasExplicitWarpSpecialize = false;
      bool hasNvwsToken = false;
      funcOp->walk(
          [&](ttg::WarpSpecializeOp) { hasExplicitWarpSpecialize = true; });
      funcOp->walk([&](ttnvws::CreateTokenOp) { hasNvwsToken = true; });
#ifdef __TLE__
      if (hasExplicitWarpSpecialize && hasNvwsToken &&
          failed(doTokenLowering(funcOp, /*numConsumerGroups=*/1)))
        signalPassFailure();
#else
      if (hasExplicitWarpSpecialize && hasNvwsToken)
        doTokenLowering(funcOp, /*numConsumerGroups=*/1);
#endif
      return;
    }

    int numWarps = mlir::triton::gpu::lookupNumWarps(funcOp);
    if (numWarps != 4)
      return;

    // FIXME: skip warpspec if there is else block. Need to improve
    // CodePartitioning to correctly handle channels in else block.
    bool hasElse = false;
    funcOp->walk([&](scf::IfOp ifOp) {
      if (ifOp.elseBlock()) {
        for (Operation &op : ifOp.elseBlock()->getOperations()) {
          hasElse = true;
        }
      }
    });
    if (hasElse)
      return;

    OpBuilder builder(funcOp);
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    unsigned numWarpGroups = 3;
    // FIXME: skip data partitioning with on-host TMA.
    bool success = false;
    for (; numWarpGroups >= 2; numWarpGroups--) {
      // Partition key ops into multiple async tasks.
      doTaskPartition(funcOp, numWarpGroups);
      if (dumpIntermediateSteps) {
        llvm::dbgs()
            << "// -----// WarpSpec internal IR Dump After: doTaskPartition\n"
            << moduleOp << "\n\n\n";
      }
      // Propagate taskId.
      int retCode = doTaskIdPropagate(funcOp);
      if (retCode == -1)
        continue;
      if (dumpIntermediateSteps) {
        llvm::dbgs()
            << "// -----// WarpSpec internal IR Dump After: doTaskIdPropagate\n"
            << moduleOp << "\n\n\n";
      }

      // Partition ops into parallel sub ops.
      if (doDataPartition(funcOp, numWarpGroups - 1)) {
        if (dumpIntermediateSteps) {
          llvm::dbgs()
              << "// -----// WarpSpec internal IR Dump After: doDataPartition\n"
              << moduleOp << "\n\n\n";
        }
        success = true;
        break;
      }
      // Clear async_task.
    }
    if (!success)
      signalPassFailure();

    doCodePartition(funcOp, numStages);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doCodePartition\n"
          << moduleOp << "\n\n\n";
    }
#ifdef __TLE__
    if (failed(doTokenLowering(funcOp, numWarpGroups - 1))) {
      signalPassFailure();
      return;
    }
#else
    doTokenLowering(funcOp, numWarpGroups - 1);
#endif
    // Clear num_stages to disable SWP.
    funcOp->walk([&](scf::ForOp forOp) {
      forOp->setAttr(mlir::triton::kNumStagesAttrName,
                     builder.getI32IntegerAttr(0));
    });
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

#define GEN_PASS_DEF_NVGPUTESTWSLOWERTOKEN
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSLowerTokenPass
    : public impl::NVGPUTestWSLowerTokenBase<NVGPUTestWSLowerTokenPass> {
public:
  using impl::NVGPUTestWSLowerTokenBase<
      NVGPUTestWSLowerTokenPass>::NVGPUTestWSLowerTokenBase;

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) {
#ifdef __TLE__
      if (failed(doTokenLowering(funcOp, numConsumerGroups)))
        signalPassFailure();
#else
      doTokenLowering(funcOp, numConsumerGroups);
#endif
    });
  }
};

} // namespace mlir
