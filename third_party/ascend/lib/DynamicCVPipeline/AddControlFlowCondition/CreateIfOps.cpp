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

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/CreateIfOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

static constexpr const char *DEBUG_TYPE = "CreateIfOps";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

// Check if a value is used outside the given region ops
static bool isUsedOutsideRegion(Value v,
                                const llvm::DenseSet<Operation *> &regionOps) {
  for (OpOperand &use : v.getUses()) {
    if (!regionOps.contains(use.getOwner())) {
      return true;
    }
  }

  return false;
}

// Find iter_arg in main loop for `v` (yield operand idx = body arg position).
static Value findIterArgInMainLoop(Value v, mlir::Type t) {
  for (Operation *user : v.getUsers()) {
    auto yieldOp = dyn_cast<scf::YieldOp>(user);
    if (!yieldOp) {
      continue;
    }

    SmallVector<Value> iterArgs =
        MainLoop(yieldOp->getParentOp()).getIterArgs();
    if (iterArgs.empty()) {
      continue;
    }

    for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
      if (operand.getAsOpaquePointer() == v.getAsOpaquePointer()) {
        if (idx >= iterArgs.size()) {
          LDBG("[Error]: yield operand index "
               << idx << " out of range for iter_args size "
               << iterArgs.size());
          continue;
        }
        Value iterArg = iterArgs[idx];
        if (iterArg.getType() == t) {
          return iterArg;
        }
      }
    }
  }

  LDBG("[Error]: else yield value not found in forOp/whileOp iter_args: " << v);
  return nullptr;
}

// Replace uses of old values that are outside the ifOp with the new values from
// ifOp
static LogicalResult replaceExternalIfOpUses(scf::IfOp ifOp,
                                             ArrayRef<Value> oldYieldValues) {
  for (size_t i = 0; i < oldYieldValues.size(); ++i) {
    Value oldVal = oldYieldValues[i];

    if (!oldVal) {
      LDBG("[Error]: oldVal is null at index " << i);
      return failure();
    }

    if (i >= ifOp.getNumResults()) {
      LDBG("[Error]: index " << i << " exceeds ifOp results count "
                             << ifOp.getNumResults());
      return failure();
    }

    Value newVal = ifOp.getResult(i);
    if (oldVal.getType() != newVal.getType()) {
      LDBG("[Error]: type mismatch at index " << i << ": " << oldVal.getType()
                                              << " vs " << newVal.getType());
      return failure();
    }

    SmallVector<OpOperand *> usesToReplace;

    for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {
      Operation *user = use.getOwner();

      // Skip uses inside the ifOp
      if (ifOp->isAncestor(user)) {
        continue;
      }

      // Skip uses after the ifOp in the same block
      if (user->getBlock() == ifOp->getBlock() &&
          !ifOp->isBeforeInBlock(user)) {
        continue;
      }

      usesToReplace.push_back(&use);
    }

    for (OpOperand *use : usesToReplace) {
      use->set(newVal);
    }
  }

  return success();
}

// Computes yield values per block (values yielded from the if); op-agnostic as
// findIterArgInMainLoop dispatches on the yield's parent op (forOp/whileOp).
LogicalResult CreateIfOpsPass::computeYieldValues(
    Operation *loopOp,
    const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
    llvm::DenseMap<int, SmallVector<Value>> &thenYieldValues,
    llvm::DenseMap<int, SmallVector<Value>> &elseYieldValues) {
  (void)loopOp; // unused; findIterArgInMainLoop walks the yield's parent.
  for (auto &p : blockOps) {
    int id = p.first;
    const SmallVector<Operation *> &ops = p.second;

    // Collect all ops in this block including nested ones
    llvm::DenseSet<Operation *> regionOps;
    for (Operation *op : ops) {
      if (failed(collectAllNestedOps(op, regionOps))) {
        return failure();
      }
    }

    // Find values used outside the region
    SmallVector<Value> yieldValues;
    for (Operation *op : ops) {
      for (Value res : op->getResults()) {
        if (isUsedOutsideRegion(res, regionOps)) {
          yieldValues.push_back(res);
        }
      }
    }

    thenYieldValues[id] = yieldValues;

    // For else branch, use iteration args from main loop
    elseYieldValues[id].clear();
    elseYieldValues[id].reserve(yieldValues.size());
    for (Value v : yieldValues) {
      Value foundArg = findIterArgInMainLoop(v, v.getType());
      if (!foundArg) {
        return failure();
      } else {
        elseYieldValues[id].push_back(foundArg);
      }
    }
  }

  return success();
}

// Create result types from yield values
static SmallVector<mlir::Type>
getResultTypes(const SmallVector<Value> &values) {
  SmallVector<mlir::Type> types;

  for (Value v : values) {
    types.push_back(v.getType());
  }

  return types;
}

// Create ifOp for a single block
static scf::IfOp createIfOpForBlock(OpBuilder &builder, Location loc,
                                    int blockId,
                                    const SmallVector<Value> &thenValues,
                                    const SmallVector<Value> &elseValues) {
  bool needsYield = !thenValues.empty();
  // Check size consistency
  if (needsYield && thenValues.size() != elseValues.size()) {
    LDBG("[Error]: then/else yield count mismatch: "
         << thenValues.size() << " vs " << elseValues.size());
    return scf::IfOp();
  }

  // Check type consistency
  if (needsYield) {
    for (size_t i = 0; i < thenValues.size(); ++i) {
      if (thenValues[i].getType() != elseValues[i].getType()) {
        LDBG("[Error]: then/else yield type mismatch at index "
             << i << ": " << thenValues[i].getType() << " vs "
             << elseValues[i].getType());
        return scf::IfOp();
      }
    }
  }

  SmallVector<mlir::Type> resultTypes = getResultTypes(thenValues);

  Value trueVal = builder.create<arith::ConstantOp>(loc, builder.getI1Type(),
                                                    builder.getBoolAttr(true));

  scf::IfOp ifOp;
  if (needsYield) {
    ifOp = builder.create<scf::IfOp>(loc, resultTypes, trueVal, true);
  } else {
    ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, trueVal, false);
  }

  ifOp->setAttr(CVPipeline::kIf, builder.getI32IntegerAttr(blockId));

  // notify npuir that of the scenario
  ifOp->setAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr,
                builder.getUnitAttr());

  return ifOp;
}

// Move ops to then branch and create yield
static LogicalResult moveOpsToThenBranch(scf::IfOp ifOp,
                                         SmallVector<Operation *> &ops,
                                         const SmallVector<Value> &thenValues,
                                         const SmallVector<Value> &elseValues,
                                         Location loc) {
  if (ops.empty() && !thenValues.empty()) {
    LDBG("[Error]: moving empty ops but thenValues not empty");
    return failure();
  }

  Block &thenBlock = ifOp.getThenRegion().front();

  for (Operation *op : llvm::reverse(ops)) {
    op->moveBefore(&thenBlock, thenBlock.begin());
  }

  if (!thenValues.empty()) {
    OpBuilder thenBuilder(&thenBlock, thenBlock.end());
    thenBuilder.create<scf::YieldOp>(loc, thenValues);

    Block &elseBlock = ifOp.getElseRegion().front();
    OpBuilder elseBuilder(&elseBlock, elseBlock.end());
    elseBuilder.create<scf::YieldOp>(loc, elseValues);
  }

  return success();
}

// Create if ops (scf.if %true) for each block_id in the main loop.
// `op` is the main-loop op (scf.for or scf.while).
LogicalResult CreateIfOpsPass::createIfInMainLoop(
    Operation *op,
    const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
    const llvm::DenseMap<int, SmallVector<Value>> &thenYieldValues,
    const llvm::DenseMap<int, SmallVector<Value>> &elseYieldValues) {
  SmallVector<int> ids = getBlockIdsInOrder(op);
  if (ids.empty() && !MainLoop(op).getBody()) {
    LDBG("[Error]: op with ssbuffer.main_loop is neither scf::ForOp nor "
         "scf::WhileOp");
    return failure();
  }

  for (int id : ids) {
    const SmallVector<Operation *> &ops = blockOps.lookup(id);
    if (ops.empty()) {
      continue;
    }

    OpBuilder builder(ops.front());
    Location loc = ops.front()->getLoc();

    scf::IfOp ifOp =
        createIfOpForBlock(builder, loc, id, thenYieldValues.lookup(id),
                           elseYieldValues.lookup(id));
    if (!ifOp) {
      return failure();
    }

    // Clone ops list since we'll modify it
    SmallVector<Operation *> opsToMove = ops;
    if (failed(moveOpsToThenBranch(ifOp, opsToMove, thenYieldValues.lookup(id),
                                   elseYieldValues.lookup(id), loc))) {
      return failure();
    }

    if (failed(replaceExternalIfOpUses(ifOp, thenYieldValues.lookup(id)))) {
      return failure();
    }
  }

  return success();
}

void CreateIfOpsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("before createIfOps:\n" << module);

  auto walkResult = module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
      return WalkResult::advance();
    }

    llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
    if (failed(collectOpsByBlockId(op, blockOps))) {
      LDBG("[Error]: op with ssbuffer.main_loop is neither scf::ForOp nor "
           "scf::WhileOp, or a body op is missing ssbuffer.block_id\n");
      return WalkResult::interrupt();
    }

    // blockCounterNums is keyed on the main-loop op (scf.for or scf.while).
    // Record it for both forOp and whileOp.
    if (info) {
      info->blockCounterNums[op] = blockOps.size();
    }

    llvm::DenseMap<int, SmallVector<Value>> thenYieldValues;
    llvm::DenseMap<int, SmallVector<Value>> elseYieldValues;

    if (failed(computeYieldValues(op, blockOps, thenYieldValues,
                                  elseYieldValues))) {
      return WalkResult::interrupt();
    }

    if (failed(createIfInMainLoop(op, blockOps, thenYieldValues,
                                  elseYieldValues))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LDBG("after createIfOps:\n" << module);
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createCreateIfOpsPass() {
  return std::make_unique<CreateIfOpsPass>();
}

} // namespace triton
} // namespace mlir
