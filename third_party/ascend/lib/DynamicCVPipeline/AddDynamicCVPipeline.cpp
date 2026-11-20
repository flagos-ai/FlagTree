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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <optional>

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache.h"
#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlockPass.h"
#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"
#include "ascend/include/DynamicCVPipeline/RemoveAttributes.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromComputePass.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflowPass.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"

#include "DynamicCVPipeline/Common/FallbackHelper.h"

static constexpr const char *DEBUG_TYPE = "add-dynamic-cv-pipeline";
static constexpr unsigned MAX_RETRY_TIMES = 2;
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ADDDYNAMICCVPIPELINE
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"
} // namespace triton
} // namespace mlir

static std::optional<int64_t> getErrorCode(ModuleOp moduleOp) {
  auto errCodeAttr =
      moduleOp->getAttrOfType<IntegerAttr>(CVPipeline::ERRCODE_ATTR);
  return errCodeAttr ? std::optional<int64_t>(errCodeAttr.getInt())
                     : std::nullopt;
}

static inline void addPasses(OpPassManager &pm) {
  pm.addPass(createPreCheckAvailablePass());
  pm.addPass(createStandardizeOpPass());
  pm.addPass(createPlanComputeBlockPass());
  pm.addPass(createComputeBlockOptPass());
  pm.addPass(createSplitDataflowPass());
  pm.addPass(createAnalyzeDataFlowPass());
  pm.addPass(createAllocMultiCachePass());
  pm.addPass(createAddControlFlowConditionPass());
  pm.addPass(createSeparateMemoryFromComputePass());
  pm.addPass(createRemoveSsbufAttrPass());
}

// must collect all sub-passes since they now are not added to pipeline
void AddDynamicCVPipelinePass::getDependentDialects(
    DialectRegistry &registry) const {
  Base::getDependentDialects(registry);
  OpPassManager tempPM(ModuleOp::getOperationName());
  addPasses(tempPM);
  tempPM.getDependentDialects(registry);
}

AddDynamicCVPipelinePass::AddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options)
    : AddDynamicCVPipelineBase(options) {}

void AddDynamicCVPipelinePass::runOnOperation() {
  auto moduleOp = getOperation();
  OpBuilder builder(moduleOp.getContext());
  compileOn91095Flag = this->compileOn91095;

  LDBG("Enter pass");
  moduleOp->removeAttr(CVPipeline::ERRCODE_ATTR);

  if (!compileOn91095Flag) {
    llvm::errs() << "Add-dynamic-cv-pipeline is only supported on 91095 now.\n";
    return;
  }

  ScopedDiagnosticHandler handler(&getContext(), [&](Diagnostic &diag) {
    LLVM_DEBUG({
      // In debug mode, continue to other handlers, i.e. print to stderr
      return llvm::failure();
    });

    // otherwise prohibit any other handler
    return llvm::success();
  });

  bool tuplePreloadFailed = false;
  for (unsigned attempt = 0; attempt < MAX_RETRY_TIMES; ++attempt) {
    // restore() consumes the saved region bodies. Each attempt needs its own
    // snapshot, taken before changing buffer counts for the retry.
    CVPipeline::FallbackHelper fallback(moduleOp);
    if (attempt > 0) {
      BufferCountManager bufferCountManager(moduleOp);
      bufferCountManager.setBufferCount(BufferCountManager::DepType::IntraCore,
                                        2);
      bufferCountManager.setBufferCount(BufferCountManager::DepType::InterCore,
                                        1);
    }

    // Do not reuse pass instances or partially transformed IR on retry.
    PassManager pm(&getContext(), moduleOp.getOperationName());
    addPasses(pm);

    // run passes in separate pm, instead of the pipeline to suppress reproducer
    auto result = pm.run(moduleOp);
    auto errCode = getErrorCode(moduleOp);
    if (succeeded(result) && !errCode.has_value()) {
      if (tuplePreloadFailed) {
        CVPipeline::setFallbackAttr(moduleOp,
                                    CVPipeline::ERRCODE_TUPLE_PRELOAD_FAILED);
      }
      LDBG("Process successfully");
      return;
    }

    if (errCode == CVPipeline::ERRCODE_TUPLE_PRELOAD_FAILED) {
      if (attempt + 1 < MAX_RETRY_TIMES) {
        LDBG("Tuple-buffer failed; Retrying with tuple preload disabled.");
        tuplePreloadFailed = true;
        fallback.restore();
        moduleOp->removeAttr(CVPipeline::ERRCODE_ATTR);
        continue;
      }
      // Consume repeated retry requests instead of leaking code 3 to callers.
      errCode = CVPipeline::ERRCODE_FAILED;
    }

    if (!errCode.has_value()) {
      moduleOp->emitWarning() << "[" << DEBUG_TYPE << "] "
                              << "Unexpected pass failure (no fallback attr "
                                 "set); fallback to compilation without "
                                 "dynamic CV pipeline.";
    } else if (errCode == CVPipeline::ERRCODE_IGNORED) {
      // This is an expected fallback: do not attach the full module IR.
      mlir::emitWarning(moduleOp->getLoc())
          << "[" << DEBUG_TYPE << "] "
          << "Kernel not applicable for dynamic CV pipeline "
             "(no matmul / already scope-optimized / unsupported "
             "pattern); falling back to standard compilation.";
    } else {
      moduleOp->emitWarning()
          << "[" << DEBUG_TYPE << "] " << "Pass failed (errcode=" << *errCode
          << "); "
             "falling back to compilation without "
             "dynamic CV pipeline.";
    }

    fallback.restore();
    moduleOp->setAttr(CVPipeline::ERRCODE_ATTR,
                      builder.getI32IntegerAttr(
                          errCode.value_or(CVPipeline::ERRCODE_FAILED)));
    return;
  }

  LDBG("Process successfully");
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createAddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options) {
  return std::make_unique<AddDynamicCVPipelinePass>(options);
}
