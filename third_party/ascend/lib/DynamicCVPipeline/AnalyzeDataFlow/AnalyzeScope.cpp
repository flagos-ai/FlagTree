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

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "analyze-scope";
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
  if (!coreTypeAttr) {
    return false;
  }
  return coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

static bool checkTransferInteraction(mlir::Operation *op) {
  bool hasCVInteraction = false;
  // check v->c data interaction
  if (isa<hivm::CopyOp>(op)) {
    hasCVInteraction = true;
  }
  // check c->v data interaction
  // user with "ssbuffer.add_from_matmul" is not a real c->v data interaction
  if (isa<bufferization::ToTensorOp>(op)) {
    for (Operation *user : op->getUsers()) {
      if (!user->hasAttr(CVPipeline::kAddFromMatmul)) {
        hasCVInteraction = true;
        break;
      }
      for (Operation *userUser : user->getUsers()) {
        if (!isa<scf::YieldOp>(userUser)) {
          hasCVInteraction = true;
          break;
        }
      }
    }
  }

  return hasCVInteraction;
}

static bool checkVecScopeMainLoop(ModuleOp module) {
  bool hasMainLoop = false;
  bool allMainLoopsSatisfy = true;

  module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    if (!isVectorScope(scopeOp)) {
      return WalkResult::advance();
    }

    scopeOp.walk([&](Operation *op) -> WalkResult {
      if (!isMainLoopOp(op)) {
        return WalkResult::advance();
      }

      hasMainLoop = true;
      bool hasCVInteraction = false;

      op->walk([&](mlir::Operation *innerOp) -> WalkResult {
        if (innerOp == op) {
          return WalkResult::advance();
        }

        // ops with "ssbuffer.transfer_id" are injected in SplitDataflowPass for
        // data transfer
        if (innerOp->hasAttr(CVPipeline::kTransferId)) {
          hasCVInteraction = checkTransferInteraction(innerOp);
          if (hasCVInteraction) {
            return WalkResult::interrupt();
          }
        }

        return WalkResult::advance();
      });

      // As long as there is a for/while with "ssbuffer.main_loop" that is not a
      // real mainloop, the processing conditions are not met, need to skip.
      if (!hasCVInteraction) {
        allMainLoopsSatisfy = false;
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (!allMainLoopsSatisfy) {
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return hasMainLoop && allMainLoopsSatisfy;
}

static bool isFixpipeToUB(Operation *op) {
  auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op);
  if (!fixpipeOp)
    return false;
  auto dstType = dyn_cast<MemRefType>(fixpipeOp.getDst().getType());
  if (!dstType)
    return false;
  auto addrSpaceAttr =
      dyn_cast_or_null<hivm::AddressSpaceAttr>(dstType.getMemorySpace());
  if (!addrSpaceAttr)
    return false;

  return addrSpaceAttr.getAddressSpace() == hivm::AddressSpace::UB;
}

// For every main_loop id, gather all for/while ops sharing that id and count
// the hivm.hir.copy ops and the hivm.hir.fixpipe ops writing to UB within
// them. Only when ALL main_loop ids have either count equal to zero (every id
// has only copy or only fixpipe, none has both), the dynamic CV pipeline
// cannot be applied and we fall back to the original workflow.
//   - hivm::CopyOp    typically appears in VECTOR scope main_loops
//   - hivm::FixpipeOp typically appears in CUBE scope main_loops
// Nested regions inside the main_loop op are also walked, and scf.yield
// terminators are skipped.
static bool isMainLoopOnlyCopyOrFixpipe(ModuleOp module) {
  // main_loop id -> (countCopy, countFixpipe)
  llvm::DenseMap<int, std::pair<int, int>> idToCounts;

  module.walk([&](Operation *op) -> WalkResult {
    if (!isa<scf::ForOp, scf::WhileOp>(op)) {
      return WalkResult::advance();
    }
    auto mainLoopAttr = op->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
    if (!mainLoopAttr) {
      return WalkResult::advance();
    }

    int id = mainLoopAttr.getInt();
    auto &counts = idToCounts[id];

    op->walk([&](mlir::Operation *innerOp) -> WalkResult {
      // Skip the loop op itself (the walk visits it first)
      if (innerOp == op) {
        return WalkResult::advance();
      }
      // Skip yield terminators (they are not real ops)
      if (isa<scf::YieldOp>(innerOp)) {
        return WalkResult::advance();
      }
      if (isa<hivm::CopyOp>(innerOp)) {
        ++counts.first;
      } else if (isFixpipeToUB(innerOp)) {
        ++counts.second;
      }
      return WalkResult::advance();
    });

    return WalkResult::advance();
  });

  // Only trigger fallback when EVERY main_loop id has only copy or only
  // fixpipe (one count is zero). If at least one id has both copy and
  // fixpipe ops, the dynamic CV pipeline still be applicable.
  if (idToCounts.empty()) {
    return false;
  }

  for (const auto &entry : idToCounts) {
    if (entry.second.first != 0 && entry.second.second != 0) {
      return false;
    }
  }

  return true;
}

static LogicalResult verifyMainLoop(ModuleOp module) {
  bool hasMainLoopOp = false;
  module.walk([&](Operation *op) {
    if (isMainLoopOp(op)) {
      hasMainLoopOp = true;
    }
  });

  if (!hasMainLoopOp) {
    LDBG("[INFO]: No cycle of multiple iterations, the DynamicCVPipeline pass "
         "will be interrupted, and resumed to the original workflow.");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return failure();
  }

  if (!checkVecScopeMainLoop(module)) {
    LDBG("[INFO]: No op beside matmul add in vector main loop.");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return failure();
  };

  if (isMainLoopOnlyCopyOrFixpipe(module)) {
    LDBG("[INFO]: One-way CV interaction for fallback.");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return failure();
  }

  return success();
}

} // namespace

void AnalyzeScopePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Before AnalyzeScope:\n" << module << "\n");

  if (failed(verifyMainLoop(module))) {
    return;
  }

  LDBG("After AnalyzeScope:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeScopePass() {
  return std::make_unique<AnalyzeScopePass>();
}

} // namespace triton
} // namespace mlir
