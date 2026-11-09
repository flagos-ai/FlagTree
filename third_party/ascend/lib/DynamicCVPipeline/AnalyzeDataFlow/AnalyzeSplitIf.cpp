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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "analyze-split-if";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace CVPipeline;

namespace {

static bool isVectorScope(scope::ScopeOp scopeOp) {
  auto coreTypeAttr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  return coreTypeAttr && coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

static bool isCubeScope(scope::ScopeOp scopeOp) {
  auto coreTypeAttr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  return coreTypeAttr && coreTypeAttr.getTcoretype() == hivm::TCoreType::CUBE;
}

// A main-loop restricted to scf.for (not scf.while): the op must both carry
// ssbuffer.main_loop and be an scf.for.
static bool isForMainLoopOp(Operation *op) {
  return isa<scf::ForOp>(op) && op->hasAttr(kMainLoop);
}

struct MainLoopScan {
  bool hasComputeOutsideSplittedIf = false;
  llvm::DenseSet<int> splittedIfTags;
};

// Walk `mainLoopOp` and:
//   1. Collect every ssbuffer.splitted_if tag attached to a nested scf.if.
//   2. Detect any "interesting" op (per `isInteresting`) that lives outside a
//      splitted_if. Such an op means the split-if is NOT the only compute, so
//      the rollback scenario does not apply.
static MainLoopScan
scanMainLoop(Operation *mainLoopOp,
             llvm::function_ref<bool(Operation *)> isInteresting) {
  MainLoopScan scan;

  mainLoopOp->walk([&](Operation *op) -> WalkResult {
    if (op == mainLoopOp) {
      return WalkResult::advance();
    }
    if (op->getParentOp() != mainLoopOp) {
      return WalkResult::advance();
    }
    if (isa<scf::YieldOp>(op)) {
      return WalkResult::advance();
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (auto tag = ifOp->getAttrOfType<IntegerAttr>(kSplittedIf)) {
        scan.splittedIfTags.insert(static_cast<int>(tag.getInt()));
      } else {
        scan.hasComputeOutsideSplittedIf = true;
      }
      return WalkResult::skip();
    }

    if (isInteresting(op)) {
      scan.hasComputeOutsideSplittedIf = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return scan;
}

static bool isMatmulOp(Operation *op) { return isa<linalg::MatmulOp>(op); }

// "Real" tensor compute: arithmetic/reduction compute on tensors, excluding
// pure data-movement / control-flow-glue ops that are not overlap-able
// compute. isTensorComputeOp already drops copy/broadcast/fill; we further
// drop linalg.transpose (a data permutation) and arith.select (control-flow
// glue selecting between a splitted_if result and the carried value), which
// the split-if itself emits around the splitted_ifs.
static bool isRealTensorComputeOp(Operation *op) {
  if (!CVPipeline::isTensorComputeOp(op)) {
    return false;
  }
  if (isa<linalg::TransposeOp, arith::SelectOp>(op)) {
    return false;
  }
  return true;
}

// Roll back when ALL of the following hold:
//   - VECTOR scf.for main loops contain tensor compute ops ONLY inside
//     splitted_ifs.
//   - CUBE scf.for main loops contain matmuls ONLY inside splitted_ifs.
//   - All splitted_if ops on VECTOR share a single tag value, all on CUBE
//     share a single tag value, and the two values are the same (i.e. there
//     is exactly one split-if pair, appearing on both cores).
static bool shouldRollbackSplitIf(ModuleOp module) {
  llvm::DenseSet<int> allTags;
  bool vectorHasExtra = false;
  bool cubeHasExtra = false;

  module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    bool isVector = isVectorScope(scopeOp);
    bool isCube = isCubeScope(scopeOp);
    if (!isVector && !isCube) {
      return WalkResult::advance();
    }

    scopeOp.walk([&](Operation *op) -> WalkResult {
      if (!isForMainLoopOp(op)) {
        return WalkResult::advance();
      }
      // Only consider main loops that live directly in this scope, so a main
      // loop nested in a child scope is not misattributed to this one.
      if (op->getParentOfType<scope::ScopeOp>() != scopeOp) {
        return WalkResult::advance();
      }

      if (isVector) {
        auto scan = scanMainLoop(op, isRealTensorComputeOp);
        if (scan.hasComputeOutsideSplittedIf) {
          vectorHasExtra = true;
          return WalkResult::interrupt();
        }
        allTags.insert(scan.splittedIfTags.begin(), scan.splittedIfTags.end());
      } else {
        auto scan = scanMainLoop(op, isMatmulOp);
        if (scan.hasComputeOutsideSplittedIf) {
          cubeHasExtra = true;
          return WalkResult::interrupt();
        }
        allTags.insert(scan.splittedIfTags.begin(), scan.splittedIfTags.end());
      }
      return WalkResult::advance();
    });

    if (vectorHasExtra || cubeHasExtra) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (vectorHasExtra || cubeHasExtra) {
    return false;
  }
  // Each core must carry exactly one splitted_if value, shared by both. This
  // also rejects the no-split-if case (sizes 0) and any multi-split-if case.
  if (allTags.size() != 1) {
    return false;
  }
  return true;
}

} // namespace

void AnalyzeSplitIfPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Before AnalyzeSplitIf:\n" << module << "\n");

  if (shouldRollbackSplitIf(module)) {
    LDBG("split-if is the only compute in both VECTOR and CUBE "
         "main loops; rollback to original workflow.");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }

  LDBG("After AnalyzeSplitIf:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeSplitIfPass() {
  return std::make_unique<AnalyzeSplitIfPass>();
}

} // namespace triton
} // namespace mlir
