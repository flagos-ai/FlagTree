/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/CloneOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/CreateIfOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/InitDependentMap.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/ProcessArgs.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimes.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopOps.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/Debug.h"
#include <memory>

static constexpr const char *DEBUG_TYPE = "AddControlFlowCondition";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

// Check if the module should be skipped for control flow condition processing
static LogicalResult verifyControlFlowPrerequisites(ModuleOp module) {
  // Check if scopeOp has ssbuffer.skip
  bool hasSkipAttr = false;
  module.walk([&](Operation *op) {
    auto scopeOp = dyn_cast<scope::ScopeOp>(op);
    if (!scopeOp) {
      return;
    }
    if (scopeOp->hasAttr("ssbuffer.skip")) {
      hasSkipAttr = true;
    }
  });
  if (hasSkipAttr) {
    LDBG("scopeOp has ssbuffer.skip, skip processing.");
    return failure();
  }

  return success();
}

void AddControlFlowConditionPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Enter add controlflow condition pass.\n");
  LDBG("before AddControlFlowCondition:");
  LLVM_DEBUG(module.dump());

  if (failed(verifyControlFlowPrerequisites(module))) {
    return;
  }

  PassManager pm(&getContext(), module.getOperationName());
  ControlFlowConditionInfo info;

  // Step0: Clone ops in vector/cube to ensure that each block_id has its own
  // ops without sharing
  pm.addPass(createCloneOpsPass());

  // Step1: Process shared iter_args in for ops to eliminate arg sharing across
  // block_ids
  std::unique_ptr<ProcessArgsPass> processArgsPass(new ProcessArgsPass());
  processArgsPass->setConditionInfo(&info);
  pm.addPass(std::move(processArgsPass));

  // Step2: Create if ops based on block_id
  std::unique_ptr<CreateIfOpsPass> createIfOpsPass(new CreateIfOpsPass());
  createIfOpsPass->setConditionInfo(&info);
  pm.addPass(std::move(createIfOpsPass));

  // Step3: Initialize crossCoreDependentMap, intraCoreDependentMap, and build
  // if block DAG
  std::unique_ptr<InitDependentMapPass> initDependentMapPass(
      new InitDependentMapPass());
  initDependentMapPass->setConditionInfo(&info);
  pm.addPass(std::move(initDependentMapPass));

  // Step4: Update for ops with block counters and inner dependency conditions,
  // and insert PIPE_S inter-core synchronization
  std::unique_ptr<UpdateLoopOpsPass> updateLoopOpsPass(new UpdateLoopOpsPass());
  updateLoopOpsPass->setConditionInfo(&info);
  pm.addPass(std::move(updateLoopOpsPass));

  // Step5:Update the conditions of ifOp based on the intraCoreDependentMap and
  // crossCoreDependentMap

  constexpr llvm::StringLiteral legacyKernel[] = {
      "chunk_gated_delta_rule_fwd_kernel_h_blockdim128"};
  bool useLegacyConditions = false;
  for (auto &op : module.getOps()) {
    auto funcOp = llvm::dyn_cast<func::FuncOp>(&op);
    if (funcOp && llvm::is_contained(legacyKernel, funcOp.getSymName())) {
      useLegacyConditions = true;
      break;
    }
  }

  auto updatePass = std::make_unique<UpdateConditionInfoPass>();
  updatePass->setConditionInfo(&info);
  pm.addPass(std::move(updatePass));

  // Step6: Update for loop iteration times based on intraCoreDependentMap
  std::unique_ptr<UpdateLoopIterTimesPass> updateLoopIterTimesPass(
      new UpdateLoopIterTimesPass());
  updateLoopIterTimesPass->setConditionInfo(&info);
  pm.addPass(std::move(updateLoopIterTimesPass));

  if (failed(runPipeline(pm, module))) {
    LDBG("Pass failed!");
    if (!CVPipeline::hasFallbackAttr(module)) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
  }

  LDBG("Exit add controlflow condition pass.");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAddControlFlowConditionPass() {
  return std::make_unique<AddControlFlowConditionPass>();
}

void registerAddControlFlowConditionPasses() {
  registerPass(createCloneOpsPass);
  registerPass(createCreateIfOpsPass);
  registerPass(createProcessArgsPass);
  registerPass(createUpdateLoopOpsPass);
  registerPass(createAddControlFlowConditionPass);
}
} // namespace triton
} // namespace mlir
