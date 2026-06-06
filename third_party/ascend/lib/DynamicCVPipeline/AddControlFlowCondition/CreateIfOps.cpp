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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/CreateIfOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

static constexpr const char *DEBUG_TYPE = "CreateIfOps";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::outs() << __VA_ARGS__; \
  llvm::outs() << "\n"; \
})

using namespace llvm;
using namespace mlir;
using namespace triton;

// Check if a value is used outside the given region ops
static bool isUsedOutsideRegion(Value v, const llvm::DenseSet<Operation *> &regionOps)
{
  for (OpOperand &use : v.getUses()) {
    if (!regionOps.contains(use.getOwner())) {
      return true;
    }
  }

  return false;
}

// Find the iteration argument in main loop that corresponds to the given value
static Value findIterArgInMainLoop(Value v, mlir::Type t)
{
  for (Operation *user : v.getUsers()) {
    auto yieldOp = dyn_cast<scf::YieldOp>(user);
    if (!yieldOp) {
      continue;
    }

    auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
    if (!forOp) {
      continue;
    }

    for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
      if (operand.getAsOpaquePointer() == v.getAsOpaquePointer()) {
        Value iterArg = forOp.getRegionIterArgs()[idx];
        if (iterArg.getType() == t) {
          return iterArg;
        }
      }
    }
  }

  LDBG("[Error]: else yield value not found in forOp iter_args: " << v << "\n");
  return nullptr;
}

// Replace uses of old values that are outside the ifOp with the new values from ifOp
static LogicalResult replaceExternalIfOpUses(scf::IfOp ifOp, ArrayRef<Value> oldYieldValues)
{
  for (size_t i = 0; i < oldYieldValues.size(); ++i) {
    Value oldVal = oldYieldValues[i];

    if (!oldVal) {
      LDBG("[Error]: oldVal is null at index " << i << "\n");
      return failure();
    }

    if (i >= ifOp.getNumResults()) {
      LDBG("[Error]: index " << i << " exceeds ifOp results count " << ifOp.getNumResults() << "\n");
      return failure();
    }

    Value newVal = ifOp.getResult(i);
    if (oldVal.getType() != newVal.getType()) {
      LDBG("[Error]: type mismatch at index " << i << ": " << oldVal.getType() << " vs " << newVal.getType() << "\n");
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
      if (user->getBlock() == ifOp->getBlock() && !ifOp->isBeforeInBlock(user)) {
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

// Compute yield values for each block: values that need to be yielded from the if
LogicalResult CreateIfOpsPass::computeYieldValues(scf::ForOp forOp,
                                                  const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
                                                  llvm::DenseMap<int, SmallVector<Value>> &thenYieldValues,
                                                  llvm::DenseMap<int, SmallVector<Value>> &elseYieldValues)
{
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
static SmallVector<mlir::Type> getResultTypes(const SmallVector<Value> &values)
{
  SmallVector<mlir::Type> types;

  for (Value v : values) {
    types.push_back(v.getType());
  }

  return types;
}

// Create ifOp for a single block
static scf::IfOp createIfOpForBlock(OpBuilder &builder, Location loc, int blockId,
                                    const SmallVector<Value> &thenValues,
                                    const SmallVector<Value> &elseValues)
{
  bool needsYield = !thenValues.empty();
  // Check size consistency
  if (needsYield && thenValues.size() != elseValues.size()) {
    LDBG("[Error]: then/else yield count mismatch: " << thenValues.size()
          << " vs " << elseValues.size() << "\n");
    return scf::IfOp();
  }

  // Check type consistency
  if (needsYield) {
    for (size_t i = 0; i < thenValues.size(); ++i) {
      if (thenValues[i].getType() != elseValues[i].getType()) {
        LDBG("[Error]: then/else yield type mismatch at index " << i << ": "
              << thenValues[i].getType() << " vs " << elseValues[i].getType() << "\n");
        return scf::IfOp();
      }
    }
  }

  SmallVector<mlir::Type> resultTypes = getResultTypes(thenValues);

  Value trueVal = builder.create<arith::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));

  scf::IfOp ifOp;
  if (needsYield) {
    ifOp = builder.create<scf::IfOp>(loc, resultTypes, trueVal, true);
  } else {
    ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, trueVal, false);
  }

  ifOp->setAttr("ssbuffer.if", builder.getI32IntegerAttr(blockId));

  return ifOp;
}

// Move ops to then branch and create yield
static LogicalResult moveOpsToThenBranch(scf::IfOp ifOp, SmallVector<Operation *> &ops,
                                         const SmallVector<Value> &thenValues,
                                         const SmallVector<Value> &elseValues, Location loc)
{
  if (ops.empty() && !thenValues.empty()) {
    LDBG("[Error]: moving empty ops but thenValues not empty\n");
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

// Create if ops (scf.if %true) for each block_id in the main loop
LogicalResult CreateIfOpsPass::createIfInMainLoop(scf::ForOp forOp,
                                                  const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
                                                  const llvm::DenseMap<int, SmallVector<Value>> &thenYieldValues,
                                                  const llvm::DenseMap<int, SmallVector<Value>> &elseYieldValues)
{
  SmallVector<int> ids = getBlockIdsInOrder(forOp);

  for (int id : ids) {
    const SmallVector<Operation *> &ops = blockOps.lookup(id);
    if (ops.empty()) {
      continue;
    }

    OpBuilder builder(ops.front());
    Location loc = ops.front()->getLoc();

    scf::IfOp ifOp = createIfOpForBlock(builder, loc, id,
                                        thenYieldValues.lookup(id),
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

void CreateIfOpsPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("before createIfOps:\n" << module << "\n");

  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      signalPassFailure();
      return WalkResult::interrupt();
    }

    // Create if ops (scf.if %true) by block_id
    llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
    if (failed(collectOpsByBlockId(forOp, blockOps))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }

    if (info) {
      info->blockCounterNums[forOp] = blockOps.size();
    }

    llvm::DenseMap<int, SmallVector<Value>> thenYieldValues;
    llvm::DenseMap<int, SmallVector<Value>> elseYieldValues;

    if (failed(computeYieldValues(forOp, blockOps, thenYieldValues, elseYieldValues))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }

    if (failed(createIfInMainLoop(forOp, blockOps, thenYieldValues, elseYieldValues))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  LDBG("after createIfOps:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createCreateIfOpsPass()
{
  return std::make_unique<CreateIfOpsPass>();
}

} // namespace triton
} // namespace mlir
