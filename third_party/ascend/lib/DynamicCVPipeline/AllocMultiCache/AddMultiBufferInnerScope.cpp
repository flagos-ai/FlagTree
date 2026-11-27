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

#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <climits>

static constexpr const char *DEBUG_TYPE = "AddMultiBufferInnerScope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace hivm;
using namespace annotation;
using namespace triton;
using namespace CVPipeline;

using BufferPair = std::pair<Value, Value>;
using BufferMap = DenseMap<Value, SmallVector<BufferPair>>;

// Buffer count constants
constexpr int kBufferCountOne = 1;

namespace mlir {
namespace triton {

// Collect main_loop loops (forOp / whileOp) in a single block
static int collectMainLoopsInBlock(Block &block,
                                   SmallVector<Operation *> &mainLoops) {
  int count = 0;
  for (Operation &op : block) {
    if (isMainLoopOp(&op)) {
      mainLoops.push_back(&op);
      count++;
    }
  }
  return count;
}

// Recursively collect main_loop loops, returns count of collected items
static int collectMainLoopsRecursively(Region &region,
                                       SmallVector<Operation *> &mainLoops) {
  int totalCount = 0;
  for (Block &block : region) {
    totalCount += collectMainLoopsInBlock(block, mainLoops);
    for (Operation &op : block) {
      for (auto &nestedRegion : op.getRegions())
        totalCount += collectMainLoopsRecursively(nestedRegion, mainLoops);
    }
  }
  return totalCount;
}

struct InnerBlockInfo {
  Value blockId;
  SmallVector<Operation *> ops;
};

static std::optional<int64_t> getOutermostSsbufferId(Operation *op) {
  std::optional<int64_t> result;
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (current->hasAttr(kMainLoop))
      return result.has_value() ? result : -1;

    if (current->getNumRegions() >= 2)
      return getOpBlockId(current);

    // Otherwise remember the deepest id seen; the parent walk will
    // overwrite it if a closer-to-boundary op carries one.
    if (auto curId = getOpBlockId(current); curId.has_value())
      result = curId;
  }
  return result;
}

void collectNestedOps(Block *block, SmallVector<Operation *> &ops) {
  for (auto &op : *block) {
    ops.push_back(&op);
    for (auto &region : op.getRegions()) {
      for (auto &innerBlock : region) {
        collectNestedOps(&innerBlock, ops);
      }
    }
  }
}

// Get priority of forOp (lower value means higher priority)
// Priority order: main_loop (1) > block_id (2) > iter_args (3) > none (0)
// This is used to select the most relevant main loop when multiple candidates
// exist
static int getForOpPriority(scf::ForOp f) {
  constexpr int priorityMainLoop = 1;
  constexpr int priorityBlockId = 2;
  constexpr int priorityIterArgs = 3;

  // Check if forOp itself has main_loop attribute
  bool hasMainloop = f->hasAttr(kMainLoop);
  bool bodyHasMainloop = false;
  bool bodyHasBlockId = false;

  // Check terminator for main_loop and block_id attributes
  if (auto *term = f.getBody()->getTerminator()) {
    bodyHasMainloop = term->hasAttr(kMainLoop);
    bodyHasBlockId = term->getAttrOfType<IntegerAttr>(kBlockId) != nullptr;
  }

  bool opHasBlockId = f->getAttrOfType<IntegerAttr>(kBlockId) != nullptr;
  bool hasIterArgs = f.getNumResults() > 0 || !f.getInitArgs().empty();

  if (hasMainloop || bodyHasMainloop) {
    return priorityMainLoop;
  }
  if (opHasBlockId || bodyHasBlockId) {
    return priorityIterArgs;
  }
  if (hasIterArgs) {
    return priorityIterArgs;
  }
  return 0;
}

// Collect a single dependency value to depValueMap. Same-block check uses
// outermost id so inner ops of a multi-region op (e.g. subview at block 3
// inside ifOp at block 4) are not treated as cross-block consumers of a
// same-block producer.
//
// i1Found is set to true when the operand is a tensor with element type i1,
// signaling the caller to fall back (set ERRCODE_IGNORED + signalPassFailure)
// rather than process the dep through the multi-buffer pipeline. The operand
// is intentionally NOT added to depValueMap in that case.
// i1 return is done temporarily.

static void collectDepValue(Value operand, Block *body, Operation *currentOp,
                            DenseMap<Value, int> &outputToBlockId,
                            DenseMap<Value, SmallVector<Value>> &depValueMap,
                            Value groupKey, bool &i1Found) {
  if (auto barg = dyn_cast<BlockArgument>(operand)) {
    if (barg.getOwner() == body &&
        !llvm::is_contained(depValueMap[groupKey], barg))
      depValueMap[groupKey].push_back(barg);
    return;
  }

  if (!outputToBlockId.count(operand))
    return;

  auto currentOutermost = getOutermostSsbufferId(currentOp);
  auto operandOutermost = getOutermostSsbufferId(operand.getDefiningOp());

  if (currentOutermost.has_value() && currentOutermost == operandOutermost)
    return;

  // i1 tensor deps: trigger fallback only for cross-block deps that are
  // actually about to be multi-buffered. Same-block i1 tensors (e.g. a
  // condition operand of an arith.select inside the same block) are
  // filtered out by the same-block check above and never enter the
  // multi-buffer pipeline, so they do not need the fallback.
  if (auto shapedType = dyn_cast<ShapedType>(operand.getType())) {
    if (shapedType.getElementType().isInteger(1)) {
      i1Found = true;
      return;
    }
  }

  if (!llvm::is_contained(depValueMap[groupKey], operand))
    depValueMap[groupKey].push_back(operand);
}

// Recursively find a nested main_loop (forOp / whileOp) inside `loop`'s body
static Operation *findNestedMainloop(const MainLoop &loop) {
  SmallVector<Operation *> allOps;
  collectNestedOps(loop.getBody(), allOps);

  for (Operation *op : allOps) {
    if (isa<scf::ForOp, scf::WhileOp>(op) && op->hasAttr(kMainLoop))
      return op;
  }
  return {};
}

bool isInsideMainLoopForOp(Operation *op) {
  return isMainLoopOp(op->getParentOp());
}

bool isInsideMainLoopForOpTraverse(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isMainLoopOp(parent))
      return true;
  }
  return false;
}

// Collect all ops with ssbuffer.id from allOps, grouped by id
// Returns 0=success, -1=invalid negative block ID from upstream pass
static int
groupOpsBySsbufferId(SmallVector<Operation *> &allOps,
                     llvm::MapVector<int, SmallVector<Operation *>> &opsById) {
  llvm::MapVector<Value, Operation *> opsByValue;
  for (Operation *op : allOps) {
    auto id = getOpBlockId(op);
    if (!id.has_value()) {
      continue;
    }
    for (auto res : op->getResults()) {
      opsByValue[res] = op;
    }
  }
  // Deduplicate: a multi-result op (e.g. scf.if) is inserted N times in
  // opsByValue and would produce duplicated dep_marks like [1, 1] otherwise.
  DenseSet<Operation *> seen;
  for (auto &p : opsByValue) {
    Operation *op = p.second;
    if (!seen.insert(op).second)
      continue;
    auto id = getOpBlockId(op);
    if (!id.has_value()) {
      continue;
    }
    opsById[*id].push_back(op);
  }
  // Also register ops that carry ssbuffer.block_id but have no results
  for (Operation *op : allOps) {
    auto id = getOpBlockId(op);
    if (!id.has_value() || !op->getResults().empty())
      continue;
    if (seen.insert(op).second)
      opsById[*id].push_back(op);
  }
  return 0;
}

// True when operand is produced by an op with a block_id and lives in a
// different logical block from the consumer (mirrors the same-block check
// in collectDepValue).
static bool
isCrossBlockDepOperand(Operation *consumerOp, Value operand,
                       const DenseMap<Value, int> &outputToBlockId) {
  if (!outputToBlockId.count(operand))
    return false;
  auto consumerOutermost = getOutermostSsbufferId(consumerOp);
  auto operandOutermost = getOutermostSsbufferId(operand.getDefiningOp());
  return !(consumerOutermost.has_value() &&
           consumerOutermost == operandOutermost);
}

// Invoke callback for each cross-block dep operand yielded by a multi-region
// op (e.g. scf.if, scf.while). Skips ops with fewer than 2 regions, empty
// regions, and regions whose terminator is not scf.yield.
static void
forEachYieldedCrossBlockDep(Operation *op,
                            const DenseMap<Value, int> &outputToBlockId,
                            llvm::function_ref<void(Value)> callback) {
  if (op->getNumRegions() < 2)
    return;
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    auto yieldOp = dyn_cast<scf::YieldOp>(region.back().getTerminator());
    if (!yieldOp)
      continue;
    for (Value operand : yieldOp->getOperands()) {
      if (isCrossBlockDepOperand(op, operand, outputToBlockId))
        callback(operand);
    }
  }
}

// Returns 0=success (including normal skip when blocks empty), -1=invalid
// negative block ID Returns 0=success (including normal skip when blocks
// empty), -1=invalid negative block ID. i1Found is set to true when any tensor
// dep collected here has element type i1; the caller is expected to abort and
// trigger fallback in that case.
static int
collectInnerBlockInfo(const MainLoop &loop,
                      DenseMap<Value, InnerBlockInfo> &blocks,
                      DenseMap<Value, SmallVector<Value>> &depValueMap,
                      SmallVector<Operation *> &allOps, bool &i1Found) {
  depValueMap.clear();
  Block *body = loop.getBody();
  if (!body)
    return 0;

  collectNestedOps(body, allOps);

  llvm::MapVector<int, SmallVector<Operation *>> opsById;
  if (groupOpsBySsbufferId(allOps, opsById) != 0)
    return -1;
  if (opsById.empty())
    return 0;

  // Build mapping from output to block id
  DenseMap<Value, int> outputToBlockId;
  for (auto &p : opsById)
    for (Operation *op : p.second)
      for (auto res : op->getResults())
        outputToBlockId[res] = p.first;

  // Collect dependency values for each block. Inner ops of multi-region ops
  // (e.g. scf.if) are included so their scalar deps get tracked; cross-block
  // judgment still attributes them to the ifOp via getOutermostSsbufferId.
  for (auto &p : opsById) {
    Operation *keyOp = nullptr;
    for (Operation *op : p.second) {
      if (!op->getResults().empty()) {
        keyOp = op;
        break;
      }
    }
    if (!keyOp)
      continue;
    Value groupKey = keyOp->getResult(0);
    InnerBlockInfo bi;
    bi.blockId = groupKey;
    bi.ops = p.second;
    blocks[groupKey] = bi;

    for (Operation *op : bi.ops)
      for (Value operand : op->getOperands())
        collectDepValue(operand, body, op, outputToBlockId, depValueMap,
                        groupKey, i1Found);
  }

  // Additional pass: collect deps from yield ops of multi-region consumers
  // (e.g. scf.if), treating the multi-region op as the dep consumer.
  for (auto &blockPair : blocks) {
    Value blockKey = blockPair.first;
    for (Operation *op : blockPair.second.ops) {
      forEachYieldedCrossBlockDep(op, outputToBlockId, [&](Value operand) {
        if (!llvm::is_contained(depValueMap[blockKey], operand))
          depValueMap[blockKey].push_back(operand);
      });
    }
  }

  return 0;
}

// Check if a yieldOp is already processed in blocks
static bool isYieldAlreadyProcessed(scf::YieldOp yieldOp,
                                    DenseMap<Value, InnerBlockInfo> &blocks) {
  for (auto &p : blocks) {
    if (llvm::is_contained(p.second.ops, yieldOp.getOperation())) {
      return true;
    }
  }
  return false;
}

// Process yield op that is not in blocks: add parent multi-region op as
// consumer Generic version: supports any op with >= 2 regions (scf.if,
// scf.while, etc.)
static void
processYieldNotInBlocks(scf::YieldOp yieldOp,
                        DenseMap<Value, InnerBlockInfo> &blocks,
                        DenseMap<Value, SmallVector<Operation *>> &depUserMap) {
  if (isYieldAlreadyProcessed(yieldOp, blocks))
    return;

  Operation *parentOp = yieldOp->getParentOp();
  // Generic: check if parent op has >= 2 regions
  if (!parentOp || parentOp->getNumRegions() < 2)
    return;

  for (Value operand : yieldOp->getOperands()) {
    // Add parent multi-region op as consumer for yield operands
    // This handles cases where depVal is only used in yield (not as direct
    // operand)
    depUserMap[operand].push_back(parentOp);
  }
}

DenseMap<Value, SmallVector<Operation *>>
buildDepUserMap(DenseMap<Value, InnerBlockInfo> &blocks,
                SmallVector<Operation *> &allOps,
                DenseMap<Value, SmallVector<Value>> &depValueMap) {
  DenseMap<Value, SmallVector<Operation *>> depUserMap;

  // First pass: process operations in blocks
  for (auto &p : blocks)
    for (Operation *op : p.second.ops)
      for (Value operand : op->getOperands())
        depUserMap[operand].push_back(op);

  // Second pass: process yield operations that are not in blocks (e.g., INT_MIN
  // block_id) Generic version: supports any multi-region op's yield
  for (Operation *op : allOps) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      processYieldNotInBlocks(yieldOp, blocks, depUserMap);
    }
  }

  return depUserMap;
}

// Check if depVal matches the special pattern with linalg::FillOp
static bool isEmptyFillPattern(Value depVal) {
  Operation *defOp = depVal.getDefiningOp();
  auto fillOp = dyn_cast<linalg::FillOp>(defOp);
  if (!fillOp)
    return false;

  if (fillOp.getOutputs().empty())
    return false;

  Value outs = fillOp.getOutputs()[0];
  if (!outs)
    return false;
  Operation *outsDef = outs.getDefiningOp();
  if (!outsDef)
    return false;
  return isa<tensor::EmptyOp>(outsDef) ||
         isa<bufferization::AllocTensorOp>(outsDef);
}

// Check if depVal is the result of a bufferization.alloc_tensor
static bool isAllocTensorPattern(Value depVal) {
  return isa_and_nonnull<bufferization::AllocTensorOp>(depVal.getDefiningOp());
}

// Check if depVal is the result of a bufferization.to_tensor wrapping a
// freshly-allocated memref that has no data copy landing on it before the
// to_tensor
static bool isAllocToTensorPattern(Value depVal) {
  auto toTensorOp =
      dyn_cast_or_null<bufferization::ToTensorOp>(depVal.getDefiningOp());
  if (!toTensorOp)
    return false;
  Value memref = toTensorOp.getOperand();
  auto allocOp = dyn_cast_or_null<memref::AllocOp>(memref.getDefiningOp());
  if (!allocOp)
    return false;

  // Scan from allocOp down to (but not including) toTensorOp
  bool seenAlloc = false;
  for (Operation &op : *allocOp->getBlock()) {
    if (&op == allocOp) {
      seenAlloc = true;
      continue;
    }
    if (&op == toTensorOp)
      break;
    if (!seenAlloc)
      continue;
    for (OpOperand &use : memref.getUses()) {
      if (use.getOwner() == &op)
        return false;
    }
  }
  return true;
}

SmallVector<Value>
collectBufferValues(DenseMap<Value, SmallVector<Value>> &depValueMap,
                    const DenseSet<Value> &clonedDepVals) {
  SmallVector<Value> valueList;
  llvm::DenseSet<Value> seenVals;

  for (auto &p : depValueMap) {
    for (Value depVal : p.second) {
      Operation *op = depVal.getDefiningOp();
      if (!op)
        continue;
      if (!seenVals.insert(depVal).second)
        continue;

      auto shapedType = dyn_cast<ShapedType>(depVal.getType());
      if (!shapedType)
        continue;

      // Skip tensor::EmptyOp - it should only get dep_mark, not buffer
      // allocation
      if (isa<tensor::EmptyOp>(op))
        continue;

      // Skip depVals that were already cloned by cloneEmptyFillsInBlocks
      if (clonedDepVals.contains(depVal))
        continue;

      // Skip bufferization.alloc_tensor
      if (isa<bufferization::AllocTensorOp>(op))
        continue;

      // Skip to_tensor whose operand is a memref.alloc — handled by
      // cloneAllocToTensorsInBlocks in Phase 2.
      if (isAllocToTensorPattern(depVal))
        continue;

      valueList.push_back(depVal);
    }
  }

  return valueList;
}

SmallVector<Value>
collectScalarDeps(DenseMap<Value, SmallVector<Value>> &depValueMap,
                  DenseMap<Value, SmallVector<Operation *>> &depUserMap) {
  SmallVector<Value> scalarValueList;

  for (auto &p : depValueMap) {
    for (Value depVal : p.second) {
      if (isa<BlockArgument>(depVal))
        continue;

      Operation *depDefinedOp = depVal.getDefiningOp();
      if (!depDefinedOp)
        continue;

      if (isa<ShapedType>(depVal.getType())) {
        // tensor::EmptyOp should be treated like scalar, add dep_mark
        if (!isa<tensor::EmptyOp>(depDefinedOp))
          continue;
        // Check if definingOp's parentOp is a main_loop forOp
        auto *parentOp = depDefinedOp->getParentOp();
        if (!parentOp || !parentOp->hasAttr(kMainLoop))
          continue;
      }

      auto userIt = depUserMap.find(depVal);
      if (userIt == depUserMap.end())
        continue;

      auto producerId = getOpBlockId(depDefinedOp);
      if (!producerId.has_value()) {
        continue;
      }

      SmallVector<Operation *> depUsers = userIt->second;
      bool hasCrossBlockUser = false;
      for (Operation *depUser : depUsers) {
        auto userId = getOutermostSsbufferId(depUser);
        if (!userId.has_value() || *userId != *producerId) {
          hasCrossBlockUser = true;
          break;
        }
      }

      if (hasCrossBlockUser)
        scalarValueList.push_back(depVal);
    }
  }

  return scalarValueList;
}

// True if op is nested strictly inside the main loop.
static bool isOpInMainLoop(Operation *op, const MainLoop &mainLoop) {
  return op && mainLoop.getOperation()->isProperAncestor(op);
}

// Collect the values an op depends on: its direct operands plus the values its
// nested regions capture from above.
static void collectOpDependencies(Operation *op, SmallVector<Value> &deps) {
  for (Value v : op->getOperands()) {
    deps.push_back(v);
  }
  if (op->getNumRegions() > 0) {
    llvm::SetVector<Value> above;
    mlir::getUsedValuesDefinedAbove(op->getRegions(), above);
    for (Value v : above) {
      deps.push_back(v);
    }
  }
}

// Depth-first build of the scalar op slice feeding `root`. Recursion stops at
// tensor operands
static void buildScalarSlice(Value root, const MainLoop &mainLoop,
                             SmallVector<Operation *> &sliceInOrder,
                             DenseSet<Operation *> &visited,
                             llvm::SetVector<Value> &boundaryTensors) {
  Operation *def = root.getDefiningOp();
  if (!def || !isOpInMainLoop(def, mainLoop)) {
    return;
  }
  if (!visited.insert(def).second) {
    return;
  }

  SmallVector<Value> deps;
  collectOpDependencies(def, deps);
  for (Value dep : deps) {
    if (isa<TensorType>(dep.getType())) {
      // Tensor boundary: let it travel through the normal tensor path.
      boundaryTensors.insert(dep);
      continue;
    }
    Operation *depDef = dep.getDefiningOp();
    if (!depDef || !isOpInMainLoop(depDef, mainLoop)) {
      continue; // block arg or loop-invariant value: reference it directly
    }
    buildScalarSlice(dep, mainLoop, sliceInOrder, visited, boundaryTensors);
  }
  sliceInOrder.push_back(def);
}

// Find the ancestor of `op` that is a direct child of `block`.
static Operation *getAncestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block) {
    op = op->getParentOp();
  }
  return op;
}

// Rematerialize the scalar slice of `root` into each of its cross-block
// consumer blocks and rewire those consumers to the local copy. Returns true on
// rewrite.
static bool
rematerializeScalarDep(Value root, int producerId, const MainLoop &mainLoop,
                       const SmallVector<Operation *> &sliceInOrder) {
  Block *body = mainLoop.getBody();

  // Group cross-block users by their block id.
  llvm::MapVector<int, SmallVector<Operation *>> usersByBlock;
  for (Operation *user : root.getUsers()) {
    Operation *bodyAnc = getAncestorInBlock(user, body);
    if (!bodyAnc) {
      continue;
    }
    auto userId = getOpBlockId(user);
    if (!userId.has_value()) {
      userId = getOpBlockId(bodyAnc);
    }
    if (!userId.has_value() || *userId == producerId) {
      continue;
    }
    usersByBlock[*userId].push_back(user);
  }
  if (usersByBlock.empty()) {
    return false;
  }

  bool changed = false;
  for (auto &entry : usersByBlock) {
    int userBlockId = entry.first;
    SmallVector<Operation *> &users = entry.second;

    // Insert the rematerialized slice before the earliest consumer.
    Operation *insertPt = nullptr;
    for (Operation *user : users) {
      Operation *anc = getAncestorInBlock(user, body);
      if (!anc) {
        continue;
      }
      if (!insertPt || anc->isBeforeInBlock(insertPt)) {
        insertPt = anc;
      }
    }
    if (!insertPt) {
      continue;
    }

    OpBuilder builder(insertPt);
    IRMapping map;
    for (Operation *op : sliceInOrder) {
      Operation *cloned = builder.clone(*op, map);
      cloned->walk([&](Operation *o) {
        o->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
      });
    }

    Value clonedRoot = map.lookupOrDefault(root);
    if (clonedRoot == root) {
      continue;
    }
    for (Operation *user : users) {
      user->replaceUsesOfWith(root, clonedRoot);
    }
    changed = true;
  }
  return changed;
}

// Scan the main loop for cross-block scalar dependencies whose data originates
// from a tensor, and rematerialize the scalar portion into each consumer block
// so the tensor part can use the normal tensor-dependency buffering.
static void rematerializeTensorRootedScalarDeps(const MainLoop &mainLoop) {
  Block *body = mainLoop.getBody();
  if (!body) {
    return;
  }

  SmallVector<Operation *> allOps;
  collectNestedOps(body, allOps);

  // Collect candidate roots (deduplicated) before mutating the IR.
  llvm::SetVector<Value> roots;
  for (Operation *op : allOps) {
    auto userId = getOpBlockId(op);
    if (!userId.has_value()) {
      continue;
    }
    for (Value operand : op->getOperands()) {
      if (isa<ShapedType>(operand.getType())) {
        continue; // only scalar operands can be cross-block scalar deps
      }
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || !isOpInMainLoop(defOp, mainLoop)) {
        continue;
      }
      auto producerId = getOpBlockId(defOp);
      if (!producerId.has_value() || *producerId == *userId) {
        continue;
      }
      roots.insert(operand);
    }
  }

  for (Value root : roots) {
    // If this .value() failed, it must be a bug in above codes.
    auto producerId = getOpBlockId(root.getDefiningOp()).value();

    SmallVector<Operation *> sliceInOrder;
    DenseSet<Operation *> visited;
    llvm::SetVector<Value> boundaryTensors;
    buildScalarSlice(root, mainLoop, sliceInOrder, visited, boundaryTensors);

    // Pure scalar/memref chains keep the existing dep_mark handling.
    if (boundaryTensors.empty()) {
      continue;
    }

    rematerializeScalarDep(root, producerId, mainLoop, sliceInOrder);
  }
}

static Value getIterCount(OpBuilder &builder, const MainLoop &loop,
                          Location loc, SmallVector<Operation *> *newOps,
                          int blockId = -1) {
  auto i32Type = builder.getI32Type();

  if (loop.isWhile()) {
    assert(loop.iterCounter &&
           "whileOp main_loop requires a global iteration counter");
    return loop.iterCounter;
  }

  auto forOp = cast<scf::ForOp>(loop.getOperation());
  Value iv = forOp.getInductionVar();
  Value lb = forOp.getLowerBound();
  Value step = forOp.getStep();
  Type ivType = iv.getType();

  // Check if lower bound is a constant zero
  bool lbIsZero = false;
  if (auto constOp = lb.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
      lbIsZero = (intAttr.getInt() == 0);
  }

  Value iterIdx;
  if (lbIsZero) {
    // Optimization: if lb is 0, use iv directly (or iv/step if step != 1)
    bool stepIsOne = false;
    if (auto constOp = step.getDefiningOp<mlir::arith::ConstantOp>())
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
        stepIsOne = intAttr.getInt() == 1;
    if (stepIsOne) {
      iterIdx = iv;
    } else {
      iterIdx = builder.create<mlir::arith::DivUIOp>(loc, iv, step);
      if (newOps)
        newOps->push_back(iterIdx.getDefiningOp());
      if (blockId >= 0) {
        iterIdx.getDefiningOp()->setAttr(kBlockId,
                                         builder.getI32IntegerAttr(blockId));
      }
    }
  } else {
    // General case: (iv - lb) / step
    Value diff = builder.create<mlir::arith::SubIOp>(loc, iv, lb);
    iterIdx = builder.create<mlir::arith::DivUIOp>(loc, diff, step);
    if (newOps) {
      newOps->push_back(diff.getDefiningOp());
      newOps->push_back(iterIdx.getDefiningOp());
    }
    if (blockId >= 0) {
      diff.getDefiningOp()->setAttr(kBlockId,
                                    builder.getI32IntegerAttr(blockId));
      iterIdx.getDefiningOp()->setAttr(kBlockId,
                                       builder.getI32IntegerAttr(blockId));
    }
  }

  // Cast to i32 if necessary
  if (ivType == i32Type)
    return iterIdx;

  Value result;
  constexpr int bits32 = 32;
  if (ivType.isIndex()) {
    result = builder.create<mlir::arith::IndexCastOp>(loc, i32Type, iterIdx);
  } else if (auto intType = dyn_cast<mlir::IntegerType>(ivType)) {
    // Extend or truncate integer types to i32
    if (intType.getWidth() < bits32)
      result = builder.create<mlir::arith::ExtSIOp>(loc, i32Type, iterIdx);
    else if (intType.getWidth() > bits32)
      result = builder.create<mlir::arith::TruncIOp>(loc, i32Type, iterIdx);
    else
      return iterIdx;
  } else {
    result = builder.create<mlir::arith::IndexCastOp>(loc, i32Type, iterIdx);
  }
  if (newOps)
    newOps->push_back(result.getDefiningOp());
  if (blockId >= 0) {
    result.getDefiningOp()->setAttr(kBlockId,
                                    builder.getI32IntegerAttr(blockId));
  }
  return result;
}

// yieldFn is used for normal op case, getNestedResults is used for nested if
// case
static void createConditionalYield(
    OpBuilder &builder, Location loc, bool hasResults,
    function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
    Operation *op,
    std::function<SmallVector<Value>()> getNestedResults = nullptr) {
  if (hasResults && getNestedResults) {
    builder.create<mlir::scf::YieldOp>(loc, getNestedResults());
  } else if (hasResults && yieldFn && op) {
    builder.create<mlir::scf::YieldOp>(loc, yieldFn(builder, loc, op));
  } else {
    builder.create<mlir::scf::YieldOp>(loc);
  }
}

// Build if-else chain for N==2 (simple nested structure)
static int buildIfChainTwoBuffers(
    OpBuilder &builder, Location loc, Value indexVal,
    SmallVector<BufferPair> &buffers, SmallVector<Operation *> &newOps,
    SmallVector<Operation *> &outIfOps,
    function_ref<Operation *(OpBuilder &, Location, Value)> createOpFn,
    function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
    mlir::TypeRange types, bool hasResults, int blockId) {
  // Create condition: index == 0
  Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  Value firstCond = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, indexVal, zero);
  auto firstIf =
      builder.create<mlir::scf::IfOp>(loc, types, firstCond, true, true);

  newOps.push_back(zero.getDefiningOp());
  newOps.push_back(firstCond.getDefiningOp());
  newOps.push_back(firstIf);
  outIfOps.push_back(firstIf);

  // Tag counter operations with block_id
  if (blockId >= 0) {
    zero.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
    firstCond.getDefiningOp()->setAttr(kBlockId,
                                       builder.getI32IntegerAttr(blockId));
  }

  // Then branch: use buffer[0]
  builder.setInsertionPointToStart(&firstIf.getThenRegion().front());
  Operation *op0 = createOpFn(builder, loc, buffers[0].second);
  if (!op0)
    return -1;
  newOps.push_back(op0);
  createConditionalYield(builder, loc, hasResults, yieldFn, op0, nullptr);

  // Else branch: use buffer[1]
  builder.setInsertionPointToStart(&firstIf.getElseRegion().front());
  Operation *op1 = createOpFn(builder, loc, buffers[1].second);
  if (!op1)
    return -1;
  newOps.push_back(op1);
  createConditionalYield(builder, loc, hasResults, yieldFn, op1, nullptr);

  builder.setInsertionPointAfter(firstIf);
  return 0;
}

// Build if-else chain for N>2 (if-else-if-else chain with proper yield
// passthrough)
static int buildIfChainMultiBuffers(
    OpBuilder &builder, Location loc, Value indexVal,
    SmallVector<BufferPair> &buffers, SmallVector<Operation *> &newOps,
    SmallVector<Operation *> &outIfOps,
    function_ref<Operation *(OpBuilder &, Location, Value)> createOpFn,
    function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
    mlir::TypeRange types, bool hasResults, int blockId) {
  int N = buffers.size();

  // Create rootIf (idx == 0)
  Value zeroVal = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  Value firstCond = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, indexVal, zeroVal);
  if (blockId >= 0) {
    zeroVal.getDefiningOp()->setAttr(kBlockId,
                                     builder.getI32IntegerAttr(blockId));
    firstCond.getDefiningOp()->setAttr(kBlockId,
                                       builder.getI32IntegerAttr(blockId));
  }
  newOps.push_back(zeroVal.getDefiningOp());
  newOps.push_back(firstCond.getDefiningOp());

  auto rootIf =
      builder.create<mlir::scf::IfOp>(loc, types, firstCond, true, true);
  if (!rootIf)
    return -1;
  newOps.push_back(rootIf);
  outIfOps.push_back(rootIf);

  // Then branch of rootIf: use buffer[0]
  builder.setInsertionPointToStart(&rootIf.getThenRegion().front());
  Operation *op0 = createOpFn(builder, loc, buffers[0].second);
  if (!op0)
    return -1;
  newOps.push_back(op0);
  createConditionalYield(builder, loc, hasResults, yieldFn, op0, nullptr);

  // Build the nested if chain in rootIf's else region
  Block *currentElseBlock = &rootIf.getElseRegion().front();

  for (int i = 1; i < N - 1; ++i) {
    // Set insertion to current else block
    builder.setInsertionPointToStart(currentElseBlock);

    // Create condition for idx == i
    Value iVal = builder.create<mlir::arith::ConstantIntOp>(loc, i, 32);
    Value cond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, indexVal, iVal);
    if (blockId >= 0) {
      iVal.getDefiningOp()->setAttr(kBlockId,
                                    builder.getI32IntegerAttr(blockId));
      cond.getDefiningOp()->setAttr(kBlockId,
                                    builder.getI32IntegerAttr(blockId));
    }
    newOps.push_back(iVal.getDefiningOp());
    newOps.push_back(cond.getDefiningOp());

    // Create nested if for this level
    auto nestedIf =
        builder.create<mlir::scf::IfOp>(loc, types, cond, true, true);
    if (!nestedIf)
      return -1;
    newOps.push_back(nestedIf);

    // Then branch: use buffer[i]
    builder.setInsertionPointToStart(&nestedIf.getThenRegion().front());
    Operation *op = createOpFn(builder, loc, buffers[i].second);
    if (!op)
      return -1;
    newOps.push_back(op);
    createConditionalYield(builder, loc, hasResults, yieldFn, op, nullptr);

    // Set insertion to end of currentElseBlock and add yield
    builder.setInsertionPointToEnd(currentElseBlock);
    createConditionalYield(builder, loc, hasResults, yieldFn, nullptr,
                           [&nestedIf]() { return nestedIf.getResults(); });

    // Move to nestedIf's else block for next iteration
    currentElseBlock = &nestedIf.getElseRegion().front();
  }

  // Final else: use buffer[N-1]
  builder.setInsertionPointToStart(currentElseBlock);
  Operation *opLast = createOpFn(builder, loc, buffers[N - 1].second);
  if (!opLast)
    return -1;
  newOps.push_back(opLast);
  createConditionalYield(builder, loc, hasResults, yieldFn, opLast, nullptr);

  builder.setInsertionPointAfter(rootIf);
  return 0;
}

// Build if-else chain for buffer selection: if (idx==0) -> buf[0] else ... else
// -> buf[N-1]
static int
buildIfChain(OpBuilder &builder, Location loc, Value indexVal,
             SmallVector<BufferPair> &buffers, SmallVector<Operation *> &newOps,
             SmallVector<Operation *> &outIfOps,
             function_ref<Operation *(OpBuilder &, Location, Value)> createOpFn,
             function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
             std::optional<mlir::TypeRange> resultTypes = std::nullopt,
             int blockId = -1) {
  int N = buffers.size();
  auto types = resultTypes.value_or(mlir::TypeRange{});
  bool hasResults = !types.empty();

  if (N == 2) {
    return buildIfChainTwoBuffers(builder, loc, indexVal, buffers, newOps,
                                  outIfOps, createOpFn, yieldFn, types,
                                  hasResults, blockId);
  }
  return buildIfChainMultiBuffers(builder, loc, indexVal, buffers, newOps,
                                  outIfOps, createOpFn, yieldFn, types,
                                  hasResults, blockId);
}

// Compute buffer index: iterCount % N
static Value computeBufferIndex(OpBuilder &builder, const MainLoop &loop,
                                Location loc, int N,
                                SmallVector<Operation *> *newOps,
                                int blockId = -1) {
  Value iterCount = getIterCount(builder, loop, loc, newOps, blockId);
  Value Nval = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  Value bufIdx = builder.create<mlir::arith::RemSIOp>(loc, iterCount, Nval);
  if (newOps) {
    newOps->push_back(Nval.getDefiningOp());
    newOps->push_back(bufIdx.getDefiningOp());
  }
  // Tag counter operations with block_id
  if (blockId >= 0) {
    MLIRContext *ctx = builder.getContext();
    Nval.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
    bufIdx.getDefiningOp()->setAttr(kBlockId,
                                    builder.getI32IntegerAttr(blockId));
  }
  return bufIdx;
}

static SmallVector<Operation *>
insertProducerLogic(OpBuilder &builder, Value depVal,
                    SmallVector<BufferPair> &buffers, const MainLoop &loop,
                    int groupId = -1) {
  SmallVector<Operation *> newOps;
  int N = buffers.size();
  Location loc = depVal.getLoc();
  // Single buffer producer logic
  if (N == kBufferCountOne) {
    Operation *producerOp = builder.create<hivm::CopyOp>(
        loc, mlir::TypeRange{}, depVal, buffers[0].second);
    if (!producerOp)
      return newOps;
    if (groupId >= 0) {
      producerOp->setAttr(
          kIntraDeps, builder.getI32ArrayAttr({groupId, crossCoreProducerId}));
    }
    newOps.push_back(producerOp);
    return newOps;
  }

  Value bufIdx = computeBufferIndex(builder, loop, loc, N, &newOps);
  SmallVector<Operation *> outIfOps;
  if (buildIfChain(
          builder, loc, bufIdx, buffers, newOps, outIfOps,
          [&](OpBuilder &b, Location l, Value buffer) -> Operation * {
            return b.create<hivm::CopyOp>(l, mlir::TypeRange{}, depVal, buffer);
          },
          nullptr) != 0) {
    return {};
  }
  // Multi-buffer: tag each inner-branch hivm.copy with [gid, 1]. The scf.if
  // wrapper itself is just `remsi % N` dispatch and is not a write op.
  if (groupId >= 0) {
    for (Operation *op : newOps) {
      if (isa<hivm::CopyOp>(op)) {
        op->setAttr(kIntraDeps,
                    builder.getI32ArrayAttr({groupId, crossCoreProducerId}));
      }
    }
  }
  return newOps;
}

// Handle consumer when N==1 (directly return buffer)
static Operation *handleSingleBufferConsumer(OpBuilder &builder, Location loc,
                                             SmallVector<BufferPair> &buffers) {
  auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
  auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(),
                                                memrefType.getElementType());
  return builder.create<mlir::bufferization::ToTensorOp>(
      loc, tensorType, buffers[0].second,
      mlir::UnitAttr::get(builder.getContext()),
      mlir::UnitAttr::get(builder.getContext()));
}

// Helper function to create ToTensorOp
static mlir::bufferization::ToTensorOp createToTensorOp(OpBuilder &builder,
                                                        Location loc,
                                                        mlir::Type tensorType,
                                                        Value buffer) {
  return builder.create<mlir::bufferization::ToTensorOp>(
      loc, tensorType, buffer, mlir::UnitAttr::get(builder.getContext()),
      mlir::UnitAttr::get(builder.getContext()));
}

static int insertConsumerLogic(OpBuilder &builder, Value depVal,
                               SmallVector<BufferPair> &buffers,
                               const MainLoop &loop,
                               SmallVector<Operation *> &outIfOps,
                               int groupId = -1, int blockId = -1) {
  SmallVector<Operation *> newOps;
  int N = buffers.size();
  Location loc = builder.getInsertionPoint()->getLoc();

  if (N == kBufferCountOne) {
    Operation *consumerOp = handleSingleBufferConsumer(builder, loc, buffers);
    outIfOps.push_back(consumerOp);
    if (groupId >= 0) {
      consumerOp->setAttr(kIntraDeps, builder.getI32ArrayAttr({groupId, 0}));
    }
    return 0;
  }

  Value readIdx = computeBufferIndex(builder, loop, loc, N, &newOps, blockId);
  auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
  auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(),
                                                memrefType.getElementType());
  mlir::TypeRange resultTypes(tensorType);
  int ret = buildIfChain(
      builder, loc, readIdx, buffers, newOps, outIfOps,
      [&](OpBuilder &b, Location l, Value buffer) -> Operation * {
        return createToTensorOp(b, l, tensorType, buffer);
      },
      [&](OpBuilder &b, Location l, Operation *op) -> Value {
        return cast<mlir::bufferization::ToTensorOp>(op).getResult();
      },
      resultTypes, blockId);
  if (ret != 0) {
    return ret;
  }
  if (groupId >= 0 && !outIfOps.empty()) {
    outIfOps.front()->setAttr(kIntraDeps,
                              builder.getI32ArrayAttr({groupId, 0}));
  }
  return 0;
}

static void addBlockAttrForOps(SmallVector<Operation *> &newOps, int blockId,
                               OpBuilder &builder) {
  auto attr = builder.getI32IntegerAttr(blockId);
  for (auto *op : newOps)
    op->setAttr(kBlockId, attr);
}

// Add dep_mark attribute to operation
static void addDepMarkAttr(Operation *op, int depMark, OpBuilder &builder) {
  if (auto existingAttr = op->getAttrOfType<mlir::ArrayAttr>(kDepMark)) {
    SmallVector<int> marks;
    for (auto attr : existingAttr)
      marks.push_back(cast<mlir::IntegerAttr>(attr).getInt());
    marks.push_back(depMark);
    op->setAttr(kDepMark, builder.getI32ArrayAttr(marks));
  } else {
    op->setAttr(kDepMark, builder.getI32ArrayAttr({depMark}));
  }
}

static void addIntraBufferAttr(SmallVector<Operation *> &ops,
                               OpBuilder &builder) {
  for (auto *op : ops) {
    if (isa<scf::IfOp>(op) || isa<hivm::CopyOp>(op) ||
        isa<bufferization::ToTensorOp>(op)) {
      op->setAttr(kIntraBuffer, builder.getUnitAttr());
    }
  }
}

// Collect cross-block user operations
static SmallVector<Operation *>
collectCrossBlockUsers(Value depVal, int producerId,
                       DenseMap<Value, SmallVector<Operation *>> &depUserMap) {
  SmallVector<Operation *> crossBlockUsers;

  auto userIt = depUserMap.find(depVal);
  if (userIt == depUserMap.end())
    return crossBlockUsers;

  for (Operation *depUser : userIt->second) {
    auto userId = getOutermostSsbufferId(depUser);
    if ((!userId.has_value() || *userId != producerId) &&
        isInsideMainLoopForOpTraverse(depUser))
      crossBlockUsers.push_back(depUser);
  }
  return crossBlockUsers;
}

// Insert buffer selection logic (scf.if + to_tensor) at the start of the given
// region. Returns the scf::IfOp that performs the buffer selection, or nullptr
// on failure.
static Operation *
insertBufferSelectionInRegion(OpBuilder &builder, Region &region, Location loc,
                              Value depVal, SmallVector<BufferPair> &buffers,
                              const MainLoop &loop, int blockId) {
  auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
  auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(),
                                                memrefType.getElementType());

  // Insert at the start of the region
  builder.setInsertionPointToStart(&region.front());

  // Compute buffer index
  Value readIdx =
      computeBufferIndex(builder, loop, loc, buffers.size(), nullptr, blockId);

  // Build buffer selection if-else chain
  SmallVector<Operation *> newIfOps;
  SmallVector<Operation *> outIfOps;
  int ret = buildIfChain(
      builder, loc, readIdx, buffers, newIfOps, outIfOps,
      [&](OpBuilder &b, Location l, Value buffer) -> Operation * {
        return createToTensorOp(b, l, tensorType, buffer);
      },
      [&](OpBuilder &b, Location l, Operation *op) -> Value {
        return cast<mlir::bufferization::ToTensorOp>(op).getResult();
      },
      tensorType, blockId);
  if (ret != 0)
    return nullptr;

  if (newIfOps.empty())
    return nullptr;

  // Tag all operations in newIfOps with block_id
  for (auto *op : newIfOps) {
    op->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
    op->setAttr(kIntraBuffer, builder.getUnitAttr());
  }

  // The main scf.if is the first one in outIfOps
  return outIfOps.front();
}

static void
markScalarDeps(SmallVector<Value> &scalarValueList,
               DenseMap<Value, SmallVector<Operation *>> &depUserMap,
               OpBuilder &builder, int startDepMark) {
  int nextDepMark = startDepMark;

  for (Value depVal : scalarValueList) {
    Operation *depDefinedOp = depVal.getDefiningOp();
    if (!depDefinedOp)
      continue;

    if (!isInsideMainLoopForOp(depDefinedOp))
      continue;

    auto producerId = getOpBlockId(depDefinedOp);
    if (!producerId.has_value())
      continue;
    auto crossBlockUsers =
        collectCrossBlockUsers(depVal, *producerId, depUserMap);
    if (crossBlockUsers.empty())
      continue;

    int depMark = nextDepMark++;
    // Add depmark to producer
    addDepMarkAttr(depDefinedOp, depMark, builder);
    // Add depmark to consumer
    for (Operation *depUser : crossBlockUsers) {
      addDepMarkAttr(depUser, depMark, builder);
    }
  }
}

// Check if depUser is a multi-region op and depVal is not a direct operand
// Returns true when op was added as consumer due to yield using depVal
// Generic version: supports any op with >= 2 regions (scf.if, scf.while, etc.)
static bool isMultiRegionConsumerFromYield(Operation *depUser, Value depVal) {
  // Check if op has >= 2 regions (generic for any multi-region op)
  if (depUser->getNumRegions() < 2)
    return false;

  for (OpOperand &operand : depUser->getOpOperands()) {
    if (operand.get() == depVal)
      return false; // depVal is a direct operand
  }
  return true; // depVal comes from yield
}

// Process normal consumer for a block: generate one buffer selection and share
// among all ops in the block
static int processNormalConsumerBlock(OpBuilder &consumedBuilder, Value depVal,
                                      SmallVector<BufferPair> &buffers,
                                      const MainLoop &loop,
                                      SmallVector<Operation *> &opsInBlock,
                                      int userBlockId, int groupId,
                                      OpBuilder &globalBuilder) {
  SmallVector<Operation *> resultIfOps;
  int ret = insertConsumerLogic(consumedBuilder, depVal, buffers, loop,
                                resultIfOps, groupId, userBlockId);
  if (ret != 0)
    return -1;

  if (resultIfOps.empty())
    return 0;

  addBlockAttrForOps(resultIfOps, userBlockId, globalBuilder);
  if (buffers.size() > kBufferCountOne) {
    for (auto *op : resultIfOps) {
      if (isa<scf::IfOp>(op)) {
        op->setAttr(kIntraBuffer, globalBuilder.getUnitAttr());
      }
    }
  } else {
    addIntraBufferAttr(resultIfOps, globalBuilder);
  }

  Operation *resultIf = resultIfOps.back();
  Value selectedBuffer = resultIf->getResult(0);

  // Replace operands of all ops in this block
  for (Operation *opInBlock : opsInBlock) {
    for (OpOperand &use : opInBlock->getOpOperands()) {
      if (use.get() == depVal)
        use.set(selectedBuffer);
    }
  }
  return 0;
}

// Handle all regions (except region 0) when depVal is used in that region's
// yield Generic version: supports any op with >= 2 regions (scf.if, scf.while,
// etc.) For scf.if: handles else region (region index 1) For scf.while: handles
// after region (region index 1) For any op with 3+ regions: handles all regions
// from index 1 onwards
static int processMultiRegionAllYields(OpBuilder &consumedBuilder, Value depVal,
                                       SmallVector<BufferPair> &buffers,
                                       const MainLoop &loop, Operation *depUser,
                                       int userBlockId, int groupId) {
  // Generic: check if op has >= 2 regions
  if (depUser->getNumRegions() < 2)
    return 0;

  // Iterate through all regions (from region 1 onwards, excluding region 0)
  for (size_t i = 1; i < depUser->getNumRegions(); ++i) {
    Region &region = depUser->getRegion(i);
    if (region.empty())
      continue;

    auto yieldOp = dyn_cast<scf::YieldOp>(region.back().getTerminator());
    if (!yieldOp)
      continue;

    for (OpOperand &operand : yieldOp->getOpOperands()) {
      if (operand.get() != depVal)
        continue;

      Operation *selectIf = insertBufferSelectionInRegion(
          consumedBuilder, region, yieldOp.getLoc(), depVal, buffers, loop,
          userBlockId);
      if (!selectIf)
        return -1;

      if (groupId >= 0) {
        selectIf->setAttr(kIntraDeps,
                          consumedBuilder.getI32ArrayAttr({groupId, 0}));
      }

      operand.set(selectIf->getResult(0));
      return 0; // Only handle one
    }
  }
  return 0;
}

// Find the last op in `anchorOp->getBlock()` whose `ssbuffer.block_id`
// attribute matches `blockId`. Returns nullptr if anchorOp is null or has no
// block, or if no such op is found.
static Operation *findLastOpWithBlockIdInBlock(Operation *anchorOp,
                                               int blockId) {
  if (!anchorOp)
    return nullptr;
  Block *block = anchorOp->getBlock();
  if (!block)
    return nullptr;
  Operation *result = nullptr;
  for (Operation &op : *block) {
    if (auto id = getOpBlockId(&op); id.has_value() && *id == blockId)
      result = &op;
  }
  return result;
}

// Find the first op in `anchorOp->getBlock()` whose `ssbuffer.block_id`
// attribute matches `blockId`. Returns nullptr if anchorOp is null or has no
// block, or if no such op is found.
static Operation *findFirstOpWithBlockIdInBlock(Operation *anchorOp,
                                                int blockId) {
  if (!anchorOp)
    return nullptr;
  Block *block = anchorOp->getBlock();
  if (!block)
    return nullptr;
  for (Operation &op : *block) {
    if (auto id = getOpBlockId(&op); id.has_value() && *id == blockId)
      return &op;
  }
  return nullptr;
}

// Process producer and consumer for a single dependency value
static int processDepVal(Value depVal, const MainLoop &loop,
                         BufferMap &bufferMap,
                         DenseMap<Value, SmallVector<Operation *>> &depUserMap,
                         OpBuilder &globalBuilder, int producerId,
                         int groupId) {
  Operation *depDefinedOp = depVal.getDefiningOp();
  if (!depDefinedOp)
    return 0;

  SmallVector<BufferPair> &buffers = bufferMap[depVal];

  auto userIt = depUserMap.find(depVal);
  if (userIt == depUserMap.end())
    return 0;
  SmallVector<Operation *> depUsers = userIt->second;

  // Read the module-level `ssbuffer.insertionOptimization` attribute inline so
  // processDepVal can be called multiple times in the same pass run and stay
  // in sync with whatever the Python caller last wrote onto the ModuleOp.
  bool enableOpt = true;
  if (mlir::ModuleOp mod = loop->getParentOfType<mlir::ModuleOp>())
    enableOpt = mod->hasAttr(CVPipeline::kInsertionOptimization);

  // Create producer
  OpBuilder producedBuffers(loop.getContext());
  // When enable_buffer_insert_optimization is on, place the producer chain at
  // the end of depDefinedOp's block_id=X region (after the last op with that
  // block_id). Otherwise keep the original "right after depDefinedOp" anchor.
  Operation *producerAnchor = depDefinedOp;
  if (enableOpt) {
    if (auto prodId = getOpBlockId(depDefinedOp); prodId.has_value()) {
      if (Operation *lastInRegion =
              findLastOpWithBlockIdInBlock(depDefinedOp, *prodId))
        producerAnchor = lastInRegion;
    }
  }
  producedBuffers.setInsertionPointAfter(producerAnchor);
  SmallVector<Operation *> producerNewOps =
      insertProducerLogic(producedBuffers, depVal, buffers, loop, groupId);
  addBlockAttrForOps(producerNewOps, producerId, globalBuilder);
  if (buffers.size() > kBufferCountOne) {
    for (auto *op : producerNewOps) {
      if (isa<scf::IfOp>(op)) {
        op->setAttr(kIntraBuffer, globalBuilder.getUnitAttr());
      }
    }
  } else {
    addIntraBufferAttr(producerNewOps, globalBuilder);
  }

  // Single pass: process both normal ops and multi-region ops
  // For normal ops: generate one buffer selection per block_id and share among
  // all ops in that block For multi-region ops: process independently
  DenseMap<int, SmallVector<Operation *>> opsByBlockId;
  DenseMap<int, Value> processedBlockSelections;

  for (Operation *depUser : depUsers) {
    auto userBlockId = getOutermostSsbufferId(depUser);
    if (!userBlockId.has_value() || *userBlockId == producerId)
      continue;

    if (isMultiRegionConsumerFromYield(depUser, depVal)) {
      // Multi-region op: process independently
      OpBuilder consumedBuilder(loop.getContext());
      // When enable_buffer_insert_optimization is on, place the consumer chain
      // at the start of depUser's block_id=X region (before the first op with
      // that block_id). Otherwise keep "right before depUser".
      Operation *consumerAnchor = depUser;
      if (enableOpt) {
        if (auto userId = getOpBlockId(depUser); userId.has_value()) {
          if (Operation *firstInRegion =
                  findFirstOpWithBlockIdInBlock(depUser, *userId))
            consumerAnchor = firstInRegion;
        }
      }
      consumedBuilder.setInsertionPoint(consumerAnchor);

      if (int ret =
              processMultiRegionAllYields(consumedBuilder, depVal, buffers,
                                          loop, depUser, *userBlockId, groupId))
        return ret;
    } else {
      // Normal op: collect by block_id for batch processing
      opsByBlockId[*userBlockId].push_back(depUser);
    }
  }

  // Re-group by Block* before processing: getOutermostSsbufferId may merge
  // ops from different Regions (e.g. then/else of an scf.if) into the same
  // block_id, but a single buffer selection's result is only visible in its
  // own Region, so sharing across Regions would break SSA dominance.
  for (auto &blockPair : opsByBlockId) {
    int userBlockId = blockPair.first;
    SmallVector<Operation *> &opsInBlock = blockPair.second;
    if (opsInBlock.empty())
      continue;

    DenseMap<Block *, SmallVector<Operation *>> opsByBlock;
    for (Operation *op : opsInBlock) {
      Block *blk = op->getBlock();
      if (!blk)
        continue;
      opsByBlock[blk].push_back(op);
    }

    for (auto &regionPair : opsByBlock) {
      SmallVector<Operation *> &opsInRegion = regionPair.second;
      if (opsInRegion.empty())
        continue;

      Operation *firstOp = opsInRegion.front();
      OpBuilder consumedBuilder(loop.getContext());
      // When enable_buffer_insert_optimization is on, place the consumer chain
      // at the start of the dep user's block_id=X region (before the first op
      // with that block_id). Otherwise keep "right before firstOp".
      Operation *consumerAnchor = firstOp;
      if (enableOpt) {
        if (auto userId = getOpBlockId(firstOp); userId.has_value()) {
          if (Operation *firstInRegion =
                  findFirstOpWithBlockIdInBlock(firstOp, *userId))
            consumerAnchor = firstInRegion;
        }
      }
      consumedBuilder.setInsertionPoint(consumerAnchor);

      if (int ret = processNormalConsumerBlock(consumedBuilder, depVal, buffers,
                                               loop, opsInRegion, userBlockId,
                                               groupId, globalBuilder))
        return ret;
    }
  }

  return 0;
}

// If `clonedDepVals` is non-null, depVals that were actually cloned
static int cloneDepsToConsumers(
    const MainLoop &loop, DenseMap<Value, InnerBlockInfo> &blocks,
    DenseMap<Value, SmallVector<Value>> &depValueMap,
    DenseMap<Value, SmallVector<Operation *>> &depUserMap, OpBuilder &builder,
    llvm::function_ref<bool(Value)> patternCheck,
    llvm::function_ref<Value(IRMapping &, OpBuilder &, Value depVal,
                             int userBlockId, ArrayRef<Operation *> users)>
        cloneFn,
    DenseSet<Value> *clonedDepVals = nullptr) {
  llvm::DenseSet<Value> seenVals;

  for (auto &blockPair : blocks) {
    auto depIt = depValueMap.find(blockPair.first);
    if (depIt == depValueMap.end())
      continue;

    for (Value depVal : depIt->second) {
      Operation *defOp = depVal.getDefiningOp();
      if (!defOp)
        continue;
      if (!seenVals.insert(depVal).second)
        continue;
      if (!patternCheck(depVal))
        continue;
      if (defOp->getParentOp() != loop.getOperation())
        continue;

      auto producerId = getOpBlockId(defOp);
      if (!producerId.has_value())
        continue;

      auto userIt = depUserMap.find(depVal);
      if (userIt == depUserMap.end())
        continue;

      // Group users by their consumer block_id, skipping users in the
      // producer's own block and users that no longer reference depVal.
      DenseMap<int, SmallVector<Operation *>> opsByBlockId;
      for (Operation *user : userIt->second) {
        auto userBlockId = getOpBlockId(user);
        if (!userBlockId.has_value() || *userBlockId == producerId)
          continue;
        bool stillUses = false;
        for (OpOperand &opnd : user->getOpOperands()) {
          if (opnd.get() == depVal) {
            stillUses = true;
            break;
          }
        }
        if (!stillUses)
          continue;
        opsByBlockId[*userBlockId].push_back(user);
      }

      for (auto &p : opsByBlockId) {
        int userBlockId = p.first;
        auto &users = p.second;
        if (users.empty())
          continue;

        Operation *firstUser = users.front();
        builder.setInsertionPoint(firstUser);
        IRMapping mapper;
        Value newVal = cloneFn(mapper, builder, depVal, userBlockId, users);
        if (!newVal)
          continue;
        if (clonedDepVals)
          clonedDepVals->insert(depVal);
        for (Operation *user : users) {
          user->replaceUsesOfWith(depVal, newVal);
        }
      }
    }
  }
  return 0;
}

// Phase 1: clone empty+fill (and any `ins` defining ops sharing the empty's
// parentOp) to consumer blocks; runs before dep collection because the cloned
// fill's `ins` chain may reach a producer-side tensor that Phase 2 must see.
static int cloneEmptyFillsInBlocks(
    const MainLoop &loop, DenseMap<Value, InnerBlockInfo> &blocks,
    DenseMap<Value, SmallVector<Value>> &depValueMap,
    DenseMap<Value, SmallVector<Operation *>> &depUserMap,
    OpBuilder &globalBuilder, DenseSet<Value> *clonedDepVals = nullptr) {
  return cloneDepsToConsumers(
      loop, blocks, depValueMap, depUserMap, globalBuilder, isEmptyFillPattern,
      [](IRMapping &mapper, OpBuilder &builder, Value depVal, int userBlockId,
         ArrayRef<Operation *> users) -> Value {
        auto fillOp = cast<linalg::FillOp>(depVal.getDefiningOp());
        // outs may come from either tensor::EmptyOp (the original case) or
        // bufferization.alloc_tensor (the new case). Both are "fresh tensor"
        // sources and treated identically below.
        Operation *origAllocLike = fillOp.getOutputs()[0].getDefiningOp();

        // Collect the `ins` operands whose defining op shares the parentOp with
        // the empty/alloc_tensor.
        SmallVector<Value> insToClone;
        Operation *allocParent = origAllocLike->getParentOp();
        for (Value insVal : fillOp.getInputs()) {
          Operation *insDef = insVal.getDefiningOp();
          if (!insDef || insDef->getParentOp() != allocParent)
            continue;
          insToClone.push_back(insVal);
        }

        Operation *newAllocLike = builder.clone(*origAllocLike, mapper);
        newAllocLike->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
        mapper.map(origAllocLike->getResult(0), newAllocLike->getResult(0));

        for (Value insVal : insToClone) {
          Operation *insDef = insVal.getDefiningOp();
          Operation *newIns = builder.clone(*insDef, mapper);
          newIns->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
          mapper.map(insVal, newIns->getResult(0));
        }

        Operation *newFill = builder.clone(*fillOp, mapper);
        newFill->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
        return newFill->getResult(0);
      },
      clonedDepVals);
}

// Clone bufferization.alloc_tensor to each consumer block
static int
cloneAllocTensorsInBlocks(const MainLoop &loop,
                          DenseMap<Value, InnerBlockInfo> &blocks,
                          DenseMap<Value, SmallVector<Value>> &depValueMap,
                          DenseMap<Value, SmallVector<Operation *>> &depUserMap,
                          OpBuilder &globalBuilder) {
  return cloneDepsToConsumers(
      loop, blocks, depValueMap, depUserMap, globalBuilder,
      isAllocTensorPattern,
      [](IRMapping &mapper, OpBuilder &builder, Value depVal, int userBlockId,
         ArrayRef<Operation *> users) -> Value {
        auto origAlloc =
            cast<bufferization::AllocTensorOp>(depVal.getDefiningOp());
        Operation *newAlloc = builder.clone(*origAlloc, mapper);
        newAlloc->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
        return newAlloc->getResult(0);
      });
}

// Clone a memref.alloc + bufferization.to_tensor chain to each consumer block.
static int cloneAllocToTensorsInBlocks(
    const MainLoop &loop, DenseMap<Value, InnerBlockInfo> &blocks,
    DenseMap<Value, SmallVector<Value>> &depValueMap,
    DenseMap<Value, SmallVector<Operation *>> &depUserMap,
    OpBuilder &globalBuilder) {
  return cloneDepsToConsumers(
      loop, blocks, depValueMap, depUserMap, globalBuilder,
      isAllocToTensorPattern,
      [](IRMapping &mapper, OpBuilder &builder, Value depVal, int userBlockId,
         ArrayRef<Operation *> users) -> Value {
        auto toTensor = cast<bufferization::ToTensorOp>(depVal.getDefiningOp());
        Operation *origAlloc = toTensor.getOperand().getDefiningOp();

        Operation *newAlloc = builder.clone(*origAlloc, mapper);
        newAlloc->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
        mapper.map(origAlloc->getResult(0), newAlloc->getResult(0));

        Operation *newToTensor = builder.clone(*toTensor, mapper);
        newToTensor->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
        return newToTensor->getResult(0);
      });
}

// Process cross-block tensor dependencies for double buffering
static int
processTensorDependencies(const MainLoop &loop,
                          DenseMap<Value, InnerBlockInfo> &blocks,
                          DenseMap<Value, SmallVector<Value>> &depValueMap,
                          DenseMap<Value, SmallVector<Operation *>> &depUserMap,
                          BufferMap &bufferMap, OpBuilder &globalBuilder,
                          int &groupId, const DenseSet<Value> &clonedDepVals) {
  // Dedupe by depVal instead of op
  llvm::DenseSet<Value> seenVals;

  for (auto &blockPair : blocks) {
    Value blockKey = blockPair.first;
    auto depIt = depValueMap.find(blockKey);
    if (depIt == depValueMap.end())
      continue;

    SmallVector<Value> &depValues = depIt->second;

    for (Value depVal : depValues) {
      // Skip if already processed
      if (!seenVals.insert(depVal).second)
        continue;

      // Validate dependency value (skip BlockArgument, null definingOp,
      // non-ShapedType)
      if (isa<BlockArgument>(depVal) || !depVal.getDefiningOp() ||
          !isa<ShapedType>(depVal.getType()))
        continue;

      // Skip tensor::EmptyOp - it should only get dep_mark, not buffer
      // allocation
      if (isa<tensor::EmptyOp>(depVal.getDefiningOp()))
        continue;

      // Skip bufferization.alloc_tensor
      if (isa<bufferization::AllocTensorOp>(depVal.getDefiningOp()))
        continue;

      // Skip to_tensor whose operand is a memref.alloc — cloned to consumer
      // blocks by cloneAllocToTensorsInBlocks in Phase 2.
      if (isAllocToTensorPattern(depVal))
        continue;

      auto *parentOp = depVal.getDefiningOp()->getParentOp();
      if (parentOp != loop.getOperation())
        continue;

      // Skip depVals that were already cloned by cloneEmptyFillsInBlocks.
      if (clonedDepVals.contains(depVal))
        continue;

      auto userIt = depUserMap.find(depVal);
      if (userIt == depUserMap.end())
        continue;
      SmallVector<Operation *> depUsers = userIt->second;

      auto producerId = getOpBlockId(depVal.getDefiningOp());
      if (!producerId.has_value()) {
        continue;
      }

      // Check if all users are in the same block
      bool allUsersSameBlock = true;
      for (Operation *depUser : depUsers) {
        auto userId = getOutermostSsbufferId(depUser);
        if (!userId.has_value() || *userId != *producerId) {
          allUsersSameBlock = false;
          break;
        }
      }
      if (allUsersSameBlock)
        continue;

      // Process cross-block dependency with double buffering
      if (processDepVal(depVal, loop, bufferMap, depUserMap, globalBuilder,
                        *producerId, groupId) != 0)
        return -1;
      groupId++;
    }
  }
  return 0;
}

static BufferMap insertBuffersBeforeLoop(const MainLoop &loop,
                                         SmallVector<Value> &valueList,
                                         OpBuilder &builder, int groupId) {
  BufferMap bufferMap;
  Block *parentBlock = loop.getBlock();
  OpBuilder insertedBuffers(builder.getContext());
  insertedBuffers.setInsertionPoint(parentBlock, loop.getIterator());

  BufferCountManager bufferCountMgr(loop.getOperation());
  int bufNum = bufferCountMgr.getBufferCountByType(
      BufferCountManager::DepType::IntraCore);

  for (Value depVal : valueList) {
    ShapedType shapedType = cast<ShapedType>(depVal.getType());
    Type elemType = shapedType.getElementType();
    AddressSpace addrSpace = AddressSpace::UB;

    SmallVector<BufferPair> buffers;
    for (int i = 0; i < bufNum; ++i) {
      MemRefType memrefType = MemRefType::get(
          shapedType.getShape(), elemType, MemRefLayoutAttrInterface{},
          AddressSpaceAttr::get(insertedBuffers.getContext(), addrSpace));

      auto allocOp =
          insertedBuffers.create<memref::AllocOp>(loop.getLoc(), memrefType);

      auto genericType = MemRefType::get(shapedType.getShape(), elemType,
                                         MemRefLayoutAttrInterface{}, 0u);

      auto casted = insertedBuffers.create<memref::MemorySpaceCastOp>(
          loop.getLoc(), genericType, allocOp.getResult());

      buffers.push_back({casted.getResult(), casted.getResult()});
    }

    bufferMap[depVal] = buffers;
    groupId++;
  }

  return bufferMap;
}

static bool
hasMemrefDepValue(DenseMap<Value, SmallVector<Value>> &depValueMap) {
  for (auto &p : depValueMap) {
    Value groupKey = p.first;
    Operation *producer = groupKey.getDefiningOp();
    for (Value depVal : p.second) {
      if (isa<MemRefType>(depVal.getType())) {
        Operation *depOp = depVal.getDefiningOp();
        LDBG("MEMREF DEP: producer=<" +
             std::string(producer ? producer->getName().getStringRef().str()
                                  : "(block-arg)") +
             ">, depVal=<" +
             std::string(depOp ? depOp->getName().getStringRef().str()
                               : "(block-arg)") +
             ">");
        return true;
      }
    }
  }
  return false;
}

// Build the before-region of the new whileOp
static void buildBeforeRegion(scf::WhileOp oldWhile, OpBuilder &bb, Location bl,
                              ValueRange iterArgs) {
  Block *oldBefore = oldWhile.getBeforeBody();
  IRMapping mapper;
  unsigned numOrig = oldBefore->getNumArguments();
  for (unsigned i = 0; i < numOrig; ++i)
    mapper.map(oldBefore->getArgument(i), iterArgs[i]);

  Operation *oldCond = nullptr;
  for (Operation &op : *oldBefore) {
    if (isa<scf::ConditionOp>(&op)) {
      oldCond = &op;
      continue;
    }
    bb.clone(op, mapper);
  }
  if (!oldCond)
    return;

  SmallVector<Value> newCondOps;
  for (Value operand : oldCond->getOperands()) {
    Value mapped = mapper.lookupOrNull(operand);
    newCondOps.push_back(mapped ? mapped : operand);
  }
  newCondOps.push_back(iterArgs[numOrig]);
  Value condValue = newCondOps.front();
  ArrayRef<Value> carriedValues = ArrayRef<Value>(newCondOps).drop_front();
  bb.create<scf::ConditionOp>(bl, condValue, carriedValues);
}

// Build the after-region of the new whileOp. The counter add-1 is not
// created here; it is inserted later by insertWhileCounterOps at the right
// position. counterIterArgOut is yielded as a placeholder and gets replaced.
static void buildAfterRegion(scf::WhileOp oldWhile, OpBuilder &ab, Location al,
                             ValueRange iterArgs, Value &counterIterArgOut) {
  Block *oldAfter = oldWhile.getAfterBody();
  unsigned numOrig = oldAfter->getNumArguments();
  counterIterArgOut = iterArgs[numOrig];

  IRMapping mapper;
  for (unsigned i = 0; i < numOrig; ++i)
    mapper.map(oldAfter->getArgument(i), iterArgs[i]);

  Operation *oldYield = nullptr;
  for (Operation &op : *oldAfter) {
    if (isa<scf::YieldOp>(&op)) {
      oldYield = &op;
      continue;
    }
    ab.clone(op, mapper);
  }
  if (!oldYield)
    return;

  SmallVector<Value> newYieldOps;
  for (Value operand : oldYield->getOperands()) {
    Value mapped = mapper.lookupOrNull(operand);
    newYieldOps.push_back(mapped ? mapped : operand);
  }
  newYieldOps.push_back(counterIterArgOut);
  ab.create<scf::YieldOp>(al, newYieldOps);
}

// Create a pass-managed global iteration counter for a whileOp main_loop
static std::pair<Value, scf::WhileOp>
setupWhileIterArgCounter(const MainLoop &loop, OpBuilder &builder) {
  auto oldWhile = cast<scf::WhileOp>(loop.getOperation());
  Location loc = loop.getLoc();
  MLIRContext *ctx = loop.getContext();
  Type i32Type = builder.getI32Type();

  // Init 0 for the new counter iter_arg, inserted before oldWhile so the
  // new whileOp can replace it in-place.
  OpBuilder preBuilder(ctx);
  preBuilder.setInsertionPoint(oldWhile);
  Value zero = preBuilder.create<arith::ConstantIntOp>(loc, 0, 32);

  // Old inits/result-types + i32 counter appended at the end.
  SmallVector<Value> newInits(oldWhile.getInits().begin(),
                              oldWhile.getInits().end());
  newInits.push_back(zero);
  SmallVector<Type> newResultTypes(oldWhile.getResultTypes().begin(),
                                   oldWhile.getResultTypes().end());
  newResultTypes.push_back(i32Type);

  // Captured by the after-builder when the do-region is constructed;
  // returned to the caller as the live iteration count.
  Value counterIterArg;

  OpBuilder cb(ctx);
  cb.setInsertionPoint(oldWhile);
  auto newWhile = cb.create<scf::WhileOp>(
      loc, newResultTypes, newInits,
      [&](OpBuilder &bb, Location bl, ValueRange iterArgs) {
        buildBeforeRegion(oldWhile, bb, bl, iterArgs);
      },
      [&](OpBuilder &ab, Location al, ValueRange iterArgs) {
        buildAfterRegion(oldWhile, ab, al, iterArgs, counterIterArg);
      });

  // Move attrs
  for (auto attr : oldWhile->getAttrs())
    newWhile->setAttr(attr.getName(), attr.getValue());
  newWhile->setAttr(kIterCounter, cb.getUnitAttr());

  for (unsigned i = 0, e = oldWhile.getNumResults(); i < e; ++i)
    oldWhile.getResult(i).replaceAllUsesWith(newWhile.getResult(i));
  oldWhile.erase();

  return {counterIterArg, newWhile};
}

// Insert the iterCounter add-1 (and its constant 1) right after the last op
// sharing the block_id of the first op that consumes mainLoop.iterCounter.
// Must run after processTensorDependencies (dispatch ops land there later).
// Fallback when no counter user exists: last op's block_id, mirroring the
// legacy buildAfterRegion placement.
static void insertWhileCounterOps(const MainLoop &mainLoop) {
  if (!mainLoop.isWhile() || !mainLoop.iterCounter)
    return;

  auto whileOp = cast<scf::WhileOp>(mainLoop.getOperation());
  Block *doBlock = whileOp.getAfterBody();
  if (!doBlock)
    return;

  std::optional<int> counterBlockId;
  for (Operation &op : *doBlock) {
    if (llvm::is_contained(op.getOperands(), mainLoop.iterCounter)) {
      counterBlockId = getOpBlockId(&op);
      break;
    }
  }
  if (!counterBlockId) {
    for (Operation &op : llvm::reverse(*doBlock)) {
      if (auto id = getOpBlockId(&op); id.has_value()) {
        counterBlockId = id;
        break;
      }
    }
    if (!counterBlockId)
      return;
  }

  Operation *lastWithBlockId = nullptr;
  for (Operation &op : *doBlock) {
    auto id = getOpBlockId(&op);
    if (id.has_value() && *id == *counterBlockId)
      lastWithBlockId = &op;
  }

  Location loc = lastWithBlockId->getLoc();
  OpBuilder builder(mainLoop.getContext());
  builder.setInsertionPointAfter(lastWithBlockId);
  IntegerAttr blockIdAttr = builder.getI32IntegerAttr(*counterBlockId);

  Value one = builder.create<arith::ConstantIntOp>(
      loc, 1, mainLoop.iterCounter.getType().getIntOrFloatBitWidth());
  one.getDefiningOp()->setAttr(kBlockId, blockIdAttr);

  Value nextCounter =
      builder.create<arith::AddIOp>(loc, mainLoop.iterCounter, one);
  Operation *iterAddOp = nextCounter.getDefiningOp();
  iterAddOp->setAttr(kBlockId, blockIdAttr);
  iterAddOp->setAttr(kIterCounter, builder.getUnitAttr());

  // Replace the placeholder counterIterArgOut operand in the yield with
  // nextCounter so the iter_arg increments each iteration.
  Operation *yieldOp = doBlock->getTerminator();
  SmallVector<Value> newOperands(yieldOp->getOperands().begin(),
                                 yieldOp->getOperands().end());
  bool replaced = false;
  for (Value &operand : newOperands) {
    if (operand == mainLoop.iterCounter) {
      operand = nextCounter;
      replaced = true;
      break;
    }
  }
  if (replaced)
    yieldOp->setOperands(newOperands);
}

static int addInnerMultiBuffer(MainLoop mainLoop, OpBuilder &builder,
                               scope::ScopeOp vectorScope, int &groupId,
                               bool &i1Found, bool &memrefFound) {
  OpBuilder globalBuilder(mainLoop.getContext());

  // Two-phase dep collection for empty+fill cloning:
  //   Phase 1 (initial): collect deps, build user map, then clone the
  //                      tensor::EmptyOp + linalg::FillOp pattern into each
  //                      consumer's block. The cloned fill's `ins` chain can
  //                      introduce fresh cross-block references (e.g. a
  //                      producer-side tensor referenced by the cloned
  //                      tensor.extract / arith.extf).
  //   Scalar rematerialize: trace each cloned fill's `ins` chain. If the
  //                        chain reaches a producer-side tensor (via
  //                        tensor.extract / arith.extf), build a fresh
  //                        block-local chain that reads from a new
  //                        multi-buffer added in Phase 2.
  //   Phase 2 (re-collect): re-run collectInnerBlockInfo / buildDepUserMap
  //                        so the new cross-block refs (from clone + scalar
  //                        rematerialize) are visible to the multi-buffer
  //                        pipeline. The multi-buffer transfer structure
  //                        (allocs, scf.if, intraDeps, intra_buffer) is
  //                        created here and is not affected by the reorder.
  DenseMap<Value, InnerBlockInfo> blocks;
  DenseMap<Value, SmallVector<Value>> depValueMap;
  SmallVector<Operation *> allOps;

  // whileOp: bufNum>1 needs pre-created counter (no implicit iter count);
  // bufNum==1 skips to avoid dead iter_arg.
  if (mainLoop.isWhile()) {
    BufferCountManager bufferCountMgr(mainLoop.getOperation());
    int bufNum = bufferCountMgr.getBufferCountByType(
        BufferCountManager::DepType::IntraCore);
    if (bufNum > kBufferCountOne) {
      auto [counter, newWhile] =
          setupWhileIterArgCounter(mainLoop, globalBuilder);
      // Update mainLoop so subsequent collections target the new whileOp
      // (the old one was erased inside setupWhileIterArgCounter).
      mainLoop.op = newWhile;
      mainLoop.body = newWhile.getAfterBody();
      mainLoop.iterCounter = counter;
    }
  }

  if (collectInnerBlockInfo(mainLoop, blocks, depValueMap, allOps, i1Found) !=
      0)
    return -1;

  if (blocks.empty())
    return -1;

  // Memref-type dep values are not supported here.
  if (hasMemrefDepValue(depValueMap)) {
    LDBG("ERROR: Memref type dependent values found in user IR, fallback");
    memrefFound = true;
    return -1;
  }

  // Phase 1: build initial depUserMap and clone empty+fill patterns. We use
  // a fresh user map built from the initial allOps so the clone can find
  // consumer-block users; the cloned fills will rewrite those users' uses.
  DenseMap<Value, SmallVector<Operation *>> initialDepUserMap =
      buildDepUserMap(blocks, allOps, depValueMap);
  DenseSet<Value> phase1ClonedDepVals;
  if (cloneEmptyFillsInBlocks(mainLoop, blocks, depValueMap, initialDepUserMap,
                              globalBuilder, &phase1ClonedDepVals) != 0)
    return -1;

  rematerializeTensorRootedScalarDeps(mainLoop);

  // Phase 2: re-collect deps now that cloned ops (and rematerialized scalar
  // chains) have created new cross-block references. depValueMap and allOps
  // are cleared inside collectInnerBlockInfo, so we re-discover everything
  // from scratch.
  blocks.clear();
  depValueMap.clear();
  allOps.clear();
  if (collectInnerBlockInfo(mainLoop, blocks, depValueMap, allOps, i1Found) !=
      0)
    return -1;

  // Phase 2 may surface i1 tensor deps that the clone introduced (e.g. a
  // cloned scalar chain reaching a producer-side i1 tensor). Abort here too.
  if (i1Found) {
    LDBG("i1 tensor dep found in Phase 2, falling back");
    return -1;
  }

  if (blocks.empty())
    return -1;

  // Drop memref-typed deps from Phase 2's collection
  int droppedMemrefDeps = 0;
  for (auto &p : depValueMap) {
    llvm::erase_if(p.second, [&droppedMemrefDeps](Value v) {
      if (isa<MemRefType>(v.getType())) {
        ++droppedMemrefDeps;
        return true;
      }
      return false;
    });
  }
  if (droppedMemrefDeps > 0)
    LDBG("Dropped " + std::to_string(droppedMemrefDeps) +
         " clone-induced memref deps from Phase 2 depValueMap");

  auto depUserMap = buildDepUserMap(blocks, allOps, depValueMap);

  // Clone bufferization.alloc_tensor deps to each consumer's block.
  if (cloneAllocTensorsInBlocks(mainLoop, blocks, depValueMap, depUserMap,
                                globalBuilder) != 0)
    return -1;

  // Clone memref.alloc + bufferization.to_tensor deps to each consumer's block
  if (cloneAllocToTensorsInBlocks(mainLoop, blocks, depValueMap, depUserMap,
                                  globalBuilder) != 0)
    return -1;
  auto valueList = collectBufferValues(depValueMap, phase1ClonedDepVals);
  LLVM_DEBUG(
      llvm::dbgs()
      << "[addInnerMultiBuffer] before insertBuffersBeforeLoop, valueList.size="
      << valueList.size() << "\n");

  auto bufferMap =
      insertBuffersBeforeLoop(mainLoop, valueList, builder, groupId);

  LLVM_DEBUG(
      llvm::dbgs() << "[addInnerMultiBuffer] before collectScalarDeps\n");
  auto scalarValueList = collectScalarDeps(depValueMap, depUserMap);

  LLVM_DEBUG(llvm::dbgs() << "[addInnerMultiBuffer] before markScalarDeps\n");
  markScalarDeps(scalarValueList, depUserMap, globalBuilder, 1);

  LLVM_DEBUG(llvm::dbgs()
             << "[addInnerMultiBuffer] before processTensorDependencies\n");
  if (processTensorDependencies(mainLoop, blocks, depValueMap, depUserMap,
                                bufferMap, globalBuilder, groupId,
                                phase1ClonedDepVals) != 0) {
    return -1;
  }

  // WhileOp only: insert the counter add-1 now that the dispatch ops that
  // use mainLoop.iterCounter have been emitted.
  insertWhileCounterOps(mainLoop);

  LLVM_DEBUG(llvm::dbgs() << "[addInnerMultiBuffer] DONE\n");
  return 0;
}

void AddMultiBufferInnerScopePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry
      .insert<mlir::annotation::AnnotationDialect, mlir::memref::MemRefDialect,
              mlir::bufferization::BufferizationDialect,
              mlir::arith::ArithDialect, mlir::scf::SCFDialect,
              mlir::hivm::HIVMDialect, mlir::scope::ScopeDialect>();
}

void AddMultiBufferInnerScopePass::runOnOperation() {
  auto module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  OpBuilder builder(module.getContext());

  // Scan existing kIntraDeps to start groupId after the max already used
  // (e.g. by InterCoreTransferAndSyncPass for C2C fixpipe transfers).
  int groupId = 0;
  module.walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<ArrayAttr>(CVPipeline::kIntraDeps)) {
      if (!attr.empty()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr[0])) {
          groupId = std::max(groupId, static_cast<int>(intAttr.getInt()) + 1);
        }
      }
    }
  });

  LDBG("Enter pass.");

  auto walkResult = module.walk([&](scope::ScopeOp scope) -> WalkResult {
    // Step 1: Check if scope has coreType attribute
    auto coreTypeAttr =
        scope->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr)
      return WalkResult::advance();

    // Step 2: Check if core type is VECTOR
    hivm::TCoreType coreType = coreTypeAttr.getTcoretype();
    if (coreType != hivm::TCoreType::VECTOR) {
      LDBG("Not vector scope");
      return WalkResult::advance();
    }

    // Step 3: Collect all main_loop loops (forOp / whileOp) in the scope
    SmallVector<Operation *> mainLoops;
    int foundCount =
        collectMainLoopsRecursively(scope.getBodyRegion(), mainLoops);
    if (foundCount < 0) {
      LDBG("collectMainLoopsRecursively failed");
      return WalkResult::interrupt();
    }
    if (foundCount == 0)
      return WalkResult::advance();

    // Step 4: Process each main_loop
    for (Operation *loopOp : mainLoops) {
      MainLoop mainLoop(loopOp);
      if (findNestedMainloop(mainLoop)) {
        LDBG("Nested main_loop found, this is not allowed");
        return WalkResult::interrupt();
      }
      // i1Found / memrefFound are reset per main_loop so they only trigger
      // fallback for the current scope's deps.
      bool i1Found = false;
      bool memrefFound = false;
      int ret = addInnerMultiBuffer(mainLoop, builder, scope, groupId, i1Found,
                                    memrefFound);
      if (i1Found) {
        LDBG("i1 tensor dep found, setting fallback attribute");
        CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
        return WalkResult::interrupt();
      }
      if (memrefFound) {
        LDBG("memref dep found, setting fallback attribute to IGNORED");
        CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
        return WalkResult::interrupt();
      }
      if (ret != 0) {
        LDBG(
            "addInnerMultiBuffer failed for main_loop; signaling pass failure");
        CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    if (!CVPipeline::hasFallbackAttr(module)) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
    return;
  }

  LDBG("Process successfully");
}

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferInnerScopePass() {
  return std::make_unique<AddMultiBufferInnerScopePass>();
}

void registerAddMultiBufferInnerScopePasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAddMultiBufferInnerScopePass();
  });
}

} // namespace triton
} // namespace mlir
