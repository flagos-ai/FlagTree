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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

static constexpr const char *DEBUG_TYPE =
    "analyze-cube-control-flow-input-chain";
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

static bool isCubeScope(scope::ScopeOp scopeOp) {
  auto coreTypeAttr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!coreTypeAttr) {
    return false;
  }
  return coreTypeAttr.getTcoretype() == hivm::TCoreType::CUBE;
}

static bool isControlFlowOp(Operation *op) {
  return llvm::isa<scf::SCFDialect>(op->getDialect());
}

static bool hasIncompatibleOpForCondition(Value val,
                                          llvm::DenseSet<Value> &visited);

static inline bool hasIncompatibleUpstream(ValueRange operands,
                                           llvm::DenseSet<Value> &visited) {
  return llvm::any_of(operands, [&](Value operand) {
    return hasIncompatibleOpForCondition(operand, visited);
  });
}

static bool hasIncompatibleOpForCondition(Value val,
                                          llvm::DenseSet<Value> &visited) {
  if (visited.contains(val)) {
    return false;
  }
  visited.insert(val);

  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    return false;
  }
  if (isVectorOnlyOp(defOp)) {
    LDBG("Fallback reason: incompatible upstream op for control flow: "
         << *defOp);
    return true;
  }
  return hasIncompatibleUpstream(defOp->getOperands(), visited);
}

static bool checkControlFlowOpInputs(Operation *cfOp) {
  llvm::SmallVector<Value> scalarOperands;
  llvm::TypeSwitch<Operation *>(cfOp)
      .Case([&](scf::IfOp ifOp) {
        scalarOperands.push_back(ifOp.getCondition());
      })
      .Case([&](scf::ForOp forOp) {
        scalarOperands.append(
            {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()});
      })
      .Case([&](scf::WhileOp whileOp) {
        // while op is very complicated, all loop-carried vars may influence
        // conditions, conservatively take all args
        auto operands = whileOp->getOperands();
        scalarOperands.append(operands.begin(), operands.end());
      })
      .Default([&](Operation *op) {
        // user passed in unknown op, conservatively take all operands
        auto operands = op->getOperands();
        scalarOperands.append(operands.begin(), operands.end());
      });

  llvm::DenseSet<Value> visited;
  return hasIncompatibleUpstream(scalarOperands, visited);
}

bool checkCubeControlFlowInputChain(ModuleOp module) {
  bool shouldReturn = false;

  module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    if (!isCubeScope(scopeOp)) {
      return WalkResult::advance();
    }
    LDBG("Found CUBE scope");

    scopeOp.walk([&](Operation *op) -> WalkResult {
      if (!isControlFlowOp(op)) {
        return WalkResult::advance();
      }
      LDBG("Found control flow op in CUBE scope");

      if (checkControlFlowOpInputs(op)) {
        shouldReturn = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (shouldReturn) {
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return shouldReturn;
}

} // namespace

void AnalyzeCubeControlFlowInputChainPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Enter AnalyzeCubeControlFlowInputChainPass.");

  if (checkCubeControlFlowInputChain(module)) {
    setFallbackAttr(module, ERRCODE_IGNORED);
    return;
  }

  LDBG("Exit AnalyzeCubeControlFlowInputChainPass.");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createAnalyzeCubeContolFLowInputChainPass() {
  return std::make_unique<AnalyzeCubeControlFlowInputChainPass>();
}

} // namespace triton
} // namespace mlir
