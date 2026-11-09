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

#include <memory>

#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"

#include "DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflowPass.h"

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "ssbuf-check-unsupported-scenario";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

// Returns the result of the enclosing scf.if if loadOp is a masked load
// directly yielded by the scf.if.
static Value getMaskedLoadIfResult(memref::LoadOp loadOp, scf::IfOp &parentIf) {
  if (!loadOp->hasOneUse()) {
    return nullptr;
  }
  parentIf = llvm::dyn_cast_if_present<scf::IfOp>(loadOp->getParentOp());
  auto yieldOp = llvm::dyn_cast<scf::YieldOp>(*loadOp->getUsers().begin());
  if (!parentIf || !yieldOp || yieldOp->getParentOp() != parentIf) {
    return nullptr;
  }

  for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
    if (operand == loadOp.getResult() && idx < parentIf.getNumResults()) {
      return parentIf.getResult(idx);
    }
  }
  return nullptr;
}

// Returns true if root transitively depends on target within scope.
static bool dependsOnValue(Value root, Value target, Operation *scope) {
  if (root == target) {
    return true;
  }
  llvm::SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited{root};

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (cur == target) {
      return true;
    }
    Operation *defOp = cur.getDefiningOp();
    if (!defOp || !scope->isProperAncestor(defOp)) {
      continue;
    }
    for (Value operand : defOp->getOperands()) {
      if (visited.insert(operand).second) {
        worklist.push_back(operand);
      }
    }
    for (Region &region : defOp->getRegions()) {
      for (Block &block : region) {
        if (Operation *term = block.getTerminator()) {
          for (Value operand : term->getOperands()) {
            if (visited.insert(operand).second) {
              worklist.push_back(operand);
            }
          }
        }
      }
    }
  }
  return false;
}

// Returns true if val has a user within scope with a core type different from
// refCoreType.
static bool hasDifferentCoreTypeUser(Value val, CoreType refCoreType,
                                     Operation *scope) {
  llvm::SmallVector<Value> worklist{val};
  llvm::DenseSet<Value> visited{val};

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (Operation *user : cur.getUsers()) {
      if (!scope->isProperAncestor(user)) {
        continue;
      }
      if (user->hasTrait<OpTrait::IsTerminator>() &&
          user->getParentOp() == scope) {
        continue;
      }
      CoreType userCoreType = getOpCoreType(user);
      if (userCoreType != CoreType::UNDETERMINED &&
          userCoreType != refCoreType &&
          userCoreType != CoreType::CUBE_AND_VECTOR) {
        return true;
      }
      for (Value res : user->getResults()) {
        if (visited.insert(res).second) {
          worklist.push_back(res);
        }
      }
    }
  }
  return false;
}

// Checks if a loop carries an iter_arg updated by ifResult and used by a
// different core type.
static bool isLoopCarryingConflictingIterArg(Operation *loopOp, Value ifResult,
                                             CoreType loadCoreType) {
  if (!isa<scf::ForOp, scf::WhileOp>(loopOp)) {
    return false;
  }
  MainLoop mainLoop(loopOp);
  scf::YieldOp loopYield = MainLoop::getLoopYieldOp(loopOp);
  if (!loopYield) {
    return false;
  }

  SmallVector<Value> iterArgs = mainLoop.getIterArgs();
  SmallVector<Value> beforeIterArgs = mainLoop.getBeforeIterArgs();

  for (auto [argIdx, yieldVal] : llvm::enumerate(loopYield.getOperands())) {
    if (!dependsOnValue(yieldVal, ifResult, loopOp)) {
      continue;
    }
    if (argIdx < iterArgs.size() &&
        hasDifferentCoreTypeUser(iterArgs[argIdx], loadCoreType, loopOp)) {
      return true;
    }
    if (argIdx < beforeIterArgs.size() &&
        hasDifferentCoreTypeUser(beforeIterArgs[argIdx], loadCoreType,
                                 loopOp)) {
      return true;
    }
  }
  return false;
}

// Pattern:
// for iter_arg(%arg21){
//   y = *operates on %arg21* {ssbuffer.core_type = "CUBE"}
//   masked_load = if {
//     a = memref.load
//     yield a
//   } else {
//     yield b
//   } { ssbuffer.core_type = "VECTOR" }
//   yield masked_load
// }
static llvm::LogicalResult
checkNoMaskedMemrefLoadOfDifferentCoreTypes(ModuleOp module) {
  auto result = module->walk([](memref::LoadOp loadOp) -> WalkResult {
    scf::IfOp parentIf;
    Value ifResult = getMaskedLoadIfResult(loadOp, parentIf);
    if (!ifResult) {
      return WalkResult::advance();
    }

    CoreType loadCoreType = getValueCoreType(ifResult);
    if (loadCoreType == CoreType::UNDETERMINED) {
      loadCoreType = getOpCoreType(parentIf);
    }
    if (loadCoreType == CoreType::UNDETERMINED) {
      loadCoreType = getOpCoreType(loadOp);
    }
    if (loadCoreType == CoreType::UNDETERMINED) {
      return WalkResult::advance();
    }

    for (Operation *currOp = parentIf->getParentOp(); currOp;
         currOp = currOp->getParentOp()) {
      if (isLoopCarryingConflictingIterArg(currOp, ifResult, loadCoreType)) {
        LOG_DEBUG("Found unsupported masked memref load feeding loop iter_arg "
                  "used by different core type");
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
  return llvm::failure(result.wasInterrupted());
}

namespace {

class CheckUnsupportedScenario
    : public PassWrapper<CheckUnsupportedScenario, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckUnsupportedScenario)

  // Constructor
  CheckUnsupportedScenario() = default;

  // Run the pass
  void runOnOperation() override {
    auto module = getOperation();
    if (hasFallbackAttr(module)) {
      return;
    }
    if (checkNoMaskedMemrefLoadOfDifferentCoreTypes(module).failed()) {
      setFallbackAttr(module, ERRCODE_IGNORED);
      return;
    }
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-check-unsupported-scenario";
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createCheckUnsupportedScenarioPass() {
  return std::make_unique<CheckUnsupportedScenario>();
}

} // namespace mlir::triton
