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

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AsyncLoadHoistingPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "async-load-hoisting"

using namespace mlir;
using namespace triton;


namespace mlir {
namespace triton {

// Check if op is a mask processing operation (broadcast, cmpf, select, etc.)
// Mask values are not directly fed into Cube
static bool isMaskProcessingOp(Operation* op)
{
  return mlir::isa<arith::CmpFOp, arith::SelectOp, arith::MaximumFOp, arith::MinimumFOp>(op);
}

// Recursively check if all downstream users of op are mask processing operations
static bool isOnlyUsedByMaskProcessingOpsRecursively(Operation* op, llvm::DenseSet<Operation*>& visited)
{
  if (visited.count(op)) {
    return true; // Already visited
  }
  visited.insert(op);

  if (op->getNumResults() == 0) {
    return false;
  }

  bool hasAnyUsers = false;
  for (auto result : op->getResults()) {
    for (auto* user : result.getUsers()) {
      hasAnyUsers = true;
      if (!isMaskProcessingOp(user) && !isOnlyUsedByMaskProcessingOpsRecursively(user, visited)) {
        return false;
      }
    }
  }
  if (!hasAnyUsers) {
    return false;
  }
  return true;
}

// Check if all downstream users of op are mask processing operations
static bool isOnlyUsedByMaskProcessingOps(Operation* op)
{
  llvm::DenseSet<Operation*> visited;
  return isOnlyUsedByMaskProcessingOpsRecursively(op, visited);
}

static bool isLoadOp(Operation* op)
{
  auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectInterface) {
    return false;
  }

  // Skip memref.copy (has exactly 2 operands: src and dst)
  if (mlir::isa<memref::CopyOp>(op)) {
    return false;
  }

  // Skip memref.load with all scalar indices
  if (mlir::isa<memref::LoadOp>(op)) {
    bool allScalarIndices = true;
    for (size_t i = 1; i < op->getNumOperands(); ++i) {
      auto* defOp = op->getOperand(i).getDefiningOp();
      if (!defOp || !llvm::isa<arith::ConstantOp>(defOp)) {
        allScalarIndices = false;
        break;
      }
    }
    if (allScalarIndices && op->getNumOperands() > 1) {
      return false;
    }
  }

  if (isOnlyUsedByMaskProcessingOps(op)) {
    return false;
  }

  return effectInterface.hasEffect<MemoryEffects::Read>();
}

// Check if a Value's computation chain is linear and predictable.
// Returns false if the chain contains remsi (modulo), load (indirect addressing),
// or complex loops.
static bool isLinearAndPredictable(mlir::Value value, llvm::DenseSet<mlir::Value>& visited)
{
  if (value.getDefiningOp<mlir::arith::ConstantOp>()) return true;
  if (mlir::isa<mlir::BlockArgument>(value)) return true;
  if (!visited.insert(value).second) return true;

  mlir::Operation* defOp = value.getDefiningOp();
  if (!defOp) return true;

  // Block nonlinear ops: memref.load (indirect addressing), scf.while/for (complex control flow)
  if (mlir::isa<mlir::memref::LoadOp, mlir::scf::WhileOp, mlir::scf::ForOp>(defOp)) {
    return false;
  }

  // Recursively check operands
  for (mlir::Value operand : defOp->getOperands()) {
    mlir::Operation* operandDefOp = operand.getDefiningOp();
    if (!operandDefOp || operandDefOp->getParentRegion() != defOp->getParentRegion()) {
      return true; // Cross-region boundary, treat as linear
    }
    if (!isLinearAndPredictable(operand, visited)) {
      return false;
    }
  }

  return true;
}

// Check if all reinterpret_cast offsets in chain are linearly predictable
static bool filterNonLinearReinterpretCast(const std::vector<mlir::Operation*>& chain)
{
  for (mlir::Operation* op : chain) {
    auto castOp = llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(op);
    if (!castOp) continue;

    llvm::DenseSet<mlir::Value> visited;

    for (auto offset : castOp.getOffsets()) {
      if (!isLinearAndPredictable(offset, visited)) {
        return false;
      }
    }
  }
  return true;
}

// Get ssbuffer.block_id attribute value, returns -1 if not present
static int64_t getBlockId(Operation* op)
{
  if (auto blockIdAttr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
    return blockIdAttr.getInt();
  }
  return -1;
}

static void collectAddressGenerationOps(Operation* op, int64_t expectedBlockId, llvm::DenseSet<Operation*>& visited, std::vector<Operation*>& chain)
{
  for (auto operand : op->getOperands()) {
    auto* defOp = operand.getDefiningOp();

    if (!defOp || visited.count(defOp) || llvm::isa<arith::ConstantOp>(defOp)) {
      continue;
    }

    if (isLoadOp(defOp)) continue;

    if (mlir::isa<MemRefType>(operand.getType())) {
      for (auto* user : operand.getUsers()) {
        if (auto copyOp = llvm::dyn_cast<memref::CopyOp>(user);
            copyOp && copyOp.getTarget() == operand && visited.insert(copyOp).second) {
          chain.push_back(copyOp);
          collectAddressGenerationOps(copyOp, expectedBlockId, visited, chain);
          continue;
        }
        if (mlir::isa<memref::SubViewOp>(user) && visited.insert(user).second) {
          chain.push_back(user);
          for (auto result : user->getResults()) {
            for (auto* subviewUser : result.getUsers()) {
              if (auto copyOp = llvm::dyn_cast<memref::CopyOp>(subviewUser);
                  visited.insert(copyOp).second) {
                chain.push_back(copyOp);
                collectAddressGenerationOps(copyOp, expectedBlockId, visited, chain);
              }
            }
          }
          collectAddressGenerationOps(user, expectedBlockId, visited, chain);
        }
      }
    }

    visited.insert(defOp);
    chain.push_back(defOp);

    collectAddressGenerationOps(defOp, expectedBlockId, visited, chain);
  }
}
// Returns full chain and filtered chain (only ops with same block_id)
static std::pair<std::vector<Operation*>, std::vector<Operation*>> getAddressGenerationChainWithFilter(Operation* loadOp)
{
  std::vector<Operation*> chain;
  chain.push_back(loadOp);

  llvm::DenseSet<Operation*> visitedOp;
  visitedOp.insert(loadOp);

  int64_t expectedBlockId = getBlockId(loadOp);

  collectAddressGenerationOps(loadOp, expectedBlockId, visitedOp, chain);

  if (chain.size() > 1) {
    auto* block = loadOp->getBlock();
    auto sortStartIter = chain.begin() + 1;
    std::stable_sort(sortStartIter, chain.end(), [block](Operation* a, Operation* b) {
      for (auto& op : *block) {
        if (&op == a) return true;
        if (&op == b) return false;
      }
      return false;
    });
    auto loadOpIter = chain.begin();
    std::rotate(loadOpIter, loadOpIter + 1, chain.end());
  }

  std::vector<Operation*> filteredChain;
  for (Operation* op : chain) {
    if (getBlockId(op) == expectedBlockId) {
      filteredChain.push_back(op);
    }
  }

  return {chain, filteredChain};
}

// Check if chain contains any block argument
static bool chainContainsBlockArg(Operation* loadOp)
{
  auto [fullChain, filteredChain] = getAddressGenerationChainWithFilter(loadOp);

  for (Operation* op : fullChain) {
    for (auto operand : op->getOperands()) {
      if (!operand.getDefiningOp()) {
        return true;
      }
    }
  }
  return false;
}


// Process all load ops with the same block_id
static void scanAndHoistBlock(llvm::SmallVector<Operation*> &cache)
{
    for (Operation* opPtr : cache) {
      bool isLoad = isLoadOp(opPtr);
      bool hasBlockArg = chainContainsBlockArg(opPtr);
      if (isLoad) {
        auto [fullChain, filteredChain] = getAddressGenerationChainWithFilter(opPtr);
        (void)fullChain;
        bool passNonLinear = filterNonLinearReinterpretCast(filteredChain);
        if (passNonLinear && hasBlockArg) {
          opPtr->setAttr("gm_load_bufferable", UnitAttr::get(opPtr->getContext()));
        }
      }
    }
}

// Traverse all ops in region, grouped by ssbuffer.block_id
static void asyncLoadHoistingImpl(Region& region)
{
  llvm::SmallVector<Operation*> cache;

  for (Block& block : llvm::make_early_inc_range(region)) {
    for (Operation& op : llvm::make_early_inc_range(block)) {
      int64_t opBlockId = getBlockId(&op);

      if (op.getNumRegions() > 0) {
        for (Region& childRegion : op.getRegions()) {
          if (childRegion.empty()) continue;
          asyncLoadHoistingImpl(childRegion);
        }
        continue;
      }

      if (cache.empty()) {
        cache.push_back(&op);
        continue;
      }

      int64_t cacheBlockId = getBlockId(cache.front());
      if (opBlockId == cacheBlockId && opBlockId != -1) {
        cache.push_back(&op);
      } else {
        scanAndHoistBlock(cache);
        cache.clear();
        cache.push_back(&op);
      }
    }
  }

  if (!cache.empty()) {
    scanAndHoistBlock(cache);
    cache.clear();
  }
}

} // namespace triton
} // namespace mlir

void AsyncLoadHoistingPass::runOnOperation()
{
  auto module = getOperation();

  LLVM_DEBUG({
    llvm::dbgs() << "[async-load-hoisting] Before AsyncLoadHoistingPass:\n" << module << "\n";
  });

  module->walk([&](func::FuncOp func) {
    asyncLoadHoistingImpl(func.getBody());
  });

  LLVM_DEBUG({
    llvm::dbgs() << "[async-load-hoisting] After AsyncLoadHoistingPass:\n" << module << "\n";
  });
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createAsyncLoadHoistingPass()
{
  return std::make_unique<AsyncLoadHoistingPass>();
}

void mlir::triton::registerAsyncLoadHoistingPasses()
{
  registerPass([]() -> std::unique_ptr<mlir::Pass> { return createAsyncLoadHoistingPass(); });
}
