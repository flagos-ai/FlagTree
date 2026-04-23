/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "Dialect/GCUWS/IR/Dialect.h"
#include "Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "gcu-warp-specialization"
#define NUM_STAGE
#define WARP_SPECIALIZE
#include "WarpSpecialization/AttrName.h"

namespace mlir::triton::gcu {
int doTaskPartition(mlir::triton::FuncOp &funcOp, unsigned numWarpGroups);
int doTaskIdPropagate(triton::FuncOp &funcOp);
void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers,
                     unsigned numWarps, bool innerBarrier);
} // namespace mlir::triton::gcu

namespace mlir {
#define GEN_PASS_DEF_GCUWARPSPECIALIZATION
#include "Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgcu = mlir::triton::gcu;

namespace {

class GCUWarpSpecializationPass
    : public impl::GCUWarpSpecializationBase<GCUWarpSpecializationPass> {
public:
  using impl::GCUWarpSpecializationBase<
      GCUWarpSpecializationPass>::GCUWarpSpecializationBase;

  void runOnFuncOp(tt::FuncOp funcOp) {
    SmallVector<mlir::scf::ForOp> loops;
    funcOp->walk([&](mlir::scf::ForOp forOp) {
      if (forOp->hasAttr(tt::kWarpSpecializeAttrName))
        loops.push_back(forOp);
    });
    if (loops.empty())
      return;

    // Only support num_warp is 4
    int numWarps = ttg::lookupNumWarps(funcOp);
    if (numWarps != 4) {
      llvm::dbgs() << "WarpSpec currently only supports num_warp = 4.\n";
      return;
    }

    // FIXME: skip warpspec if there is else block. Need to improve
    // CodePartitioning to correctly handle channels in else block.
    bool hasElse = false;
    funcOp->walk([&](mlir::scf::IfOp ifOp) {
      if (ifOp.elseBlock()) {
        hasElse = !ifOp.elseBlock()->getOperations().empty();
      }
    });
    if (hasElse) {
      llvm::dbgs() << "WarpSpec currently does not support else block.\n";
      return;
    }

    OpBuilder builder(funcOp);
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();

    // Assign async task id
    unsigned numWarpGroups = 2;
    if (ttgcu::doTaskPartition(funcOp, numWarpGroups) != 0) {
      llvm::dbgs() << "// -----// Unsupported WarpSpec\n\n\n";
      return;
    }
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doTaskPartition\n"
          << moduleOp << "\n\n\n";
    }

    // Propagate taskId.
    int retCode = ttgcu::doTaskIdPropagate(funcOp);
    if (retCode == -1) {
      llvm_unreachable("Fail to propagate task id.");
      return;
    }
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doTaskIdPropagate\n"
          << moduleOp << "\n\n\n";
    }

    // Code partition
    ttgcu::doCodePartition(funcOp, numStages, numWarps, innerBarrier);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doCodePartition\n"
          << moduleOp << "\n\n\n";
    }

    // Clear num_stages to disable SWP.
    funcOp->walk([&](mlir::scf::ForOp forOp) {
      forOp->setAttr(tt::kNumStagesAttrName, builder.getI32IntegerAttr(0));
    });
  }

  void runOnOperation() override {
    getOperation()->walk([&](tt::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace
