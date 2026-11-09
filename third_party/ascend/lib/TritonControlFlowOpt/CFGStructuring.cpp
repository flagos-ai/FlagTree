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

#include "TritonControlFlowOpt/CFGStructuring.h"
#include "Utils/Utils.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Visitors.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

using namespace mlir;
using namespace triton;

namespace {

// This transformation handles acyclic CFGs made from cf.br/cf.cond_br and
// return-like terminators. A branch with a common successor becomes an scf.if
// whose results model that successor's block arguments. A branch whose arms
// terminate independently is handled by the terminal-path builder below.

//===----------------------------------------------------------------------===//
// CFG discovery and join selection
//===----------------------------------------------------------------------===//

static bool isSupportedReturn(Operation *op) {
  return isa<triton::ReturnOp, func::ReturnOp, triton::MapElementwiseReturnOp>(
      op);
}

static SmallVector<Block *> getCfgSuccessors(Block *block) {
  Operation *term = block->getTerminator();
  if (auto br = dyn_cast<cf::BranchOp>(term))
    return {br.getDest()};
  if (auto condBr = dyn_cast<cf::CondBranchOp>(term))
    return {condBr.getTrueDest(), condBr.getFalseDest()};
  return {};
}

static DenseMap<Block *, unsigned> computeDistances(Block *start) {
  DenseMap<Block *, unsigned> distances;
  SmallVector<Block *> worklist;

  distances[start] = 0;
  worklist.push_back(start);

  for (unsigned i = 0; i < worklist.size(); ++i) {
    Block *block = worklist[i];
    unsigned nextDistance = distances[block] + 1;
    for (Block *successor : getCfgSuccessors(block)) {
      if (successor->getParent() != start->getParent())
        continue;
      if (distances.count(successor))
        continue;
      distances[successor] = nextDistance;
      worklist.push_back(successor);
    }
  }

  return distances;
}

/// Finds the closest block reachable from both branch arms. Minimizing the
/// maximum arm distance prefers the earliest balanced convergence point; the
/// total distance provides deterministic tie-breaking.
static FailureOr<Block *> findNearestCommonBlock(Block *lhs, Block *rhs,
                                                 Location loc,
                                                 bool emitDiagnostic = true) {
  DenseMap<Block *, unsigned> lhsDistances = computeDistances(lhs);
  DenseMap<Block *, unsigned> rhsDistances = computeDistances(rhs);

  Block *best = nullptr;
  unsigned bestMaxDistance = std::numeric_limits<unsigned>::max();
  unsigned bestTotalDistance = std::numeric_limits<unsigned>::max();

  for (auto &entry : lhsDistances) {
    Block *candidate = entry.first;
    auto rhsIt = rhsDistances.find(candidate);
    if (rhsIt == rhsDistances.end())
      continue;

    unsigned lhsDistance = entry.second;
    unsigned rhsDistance = rhsIt->second;
    unsigned maxDistance = std::max(lhsDistance, rhsDistance);
    unsigned totalDistance = lhsDistance + rhsDistance;
    if (maxDistance < bestMaxDistance ||
        (maxDistance == bestMaxDistance && totalDistance < bestTotalDistance)) {
      best = candidate;
      bestMaxDistance = maxDistance;
      bestTotalDistance = totalDistance;
    }
  }

  if (!best && emitDiagnostic) {
    emitError(loc) << "unsupported non-tree control flow: branch arms do not "
                      "reach a common convergence block";
    return failure();
  }
  if (!best)
    return failure();

  return best;
}

/// Replaces a destination block's SSA arguments with the values carried by the
/// incoming branch. Callers erase the original CFG blocks after their bodies
/// have been moved or cloned.
static LogicalResult replaceBlockArguments(Block *block, ValueRange incoming,
                                           Location loc) {
  if (block->getNumArguments() != incoming.size()) {
    emitError(loc) << "invalid branch operand count while structuring "
                      "control flow: "
                   << incoming.size() << " operands for "
                   << block->getNumArguments() << " block arguments";
    return failure();
  }

  for (auto [arg, value] : llvm::zip(block->getArguments(), incoming))
    arg.replaceAllUsesWith(value);
  return success();
}

/// Moves non-terminator operations into the currently constructed SCF region.
/// The original terminator remains available to drive recursive CFG traversal.
static void moveBlockBodyBefore(Block *block, OpBuilder &builder) {
  SmallVector<Operation *> movedOps = llvm::map_to_vector(
      block->without_terminator(), [](Operation &op) { return &op; });
  for (Operation *op : movedOps)
    op->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
}

struct ReturnPathResult {
  // A terminal path cannot place a return inside an scf.if region. Bubble the
  // return operands outward so the caller can yield them and emit one return
  // after the structured conditional.
  SmallVector<Value> operands;
};

//===----------------------------------------------------------------------===//
// Converging branch construction
//===----------------------------------------------------------------------===//

static FailureOr<SmallVector<Value>> buildRegionPath(Block *block,
                                                     ValueRange incoming,
                                                     Block *stopBlock,
                                                     OpBuilder &builder);

static FailureOr<ReturnPathResult>
buildReturnPath(Block *block, ValueRange incoming, OpBuilder &builder);

static FailureOr<scf::IfOp> buildTerminalValueIf(cf::CondBranchOp condBr,
                                                 OpBuilder &builder);

static FailureOr<scf::IfOp> buildStructuredIf(cf::CondBranchOp condBr,
                                              Block *joinBlock,
                                              OpBuilder &builder) {
  // The join block arguments define the exact result contract of the new if.
  // Each arm is recursively consumed up to that join and yields its incoming
  // values in the same order.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(joinBlock->getNumArguments());
  for (BlockArgument arg : joinBlock->getArguments())
    resultTypes.push_back(arg.getType());

  auto ifOp = builder.create<scf::IfOp>(condBr.getLoc(), resultTypes,
                                        condBr.getCondition(),
                                        /*withElseRegion=*/true);

  {
    OpBuilder::InsertionGuard guard(builder);
    Operation *autoYield =
        resultTypes.empty() ? ifOp.thenBlock()->getTerminator() : nullptr;
    if (autoYield)
      builder.setInsertionPoint(autoYield);
    else
      builder.setInsertionPointToStart(ifOp.thenBlock());
    FailureOr<SmallVector<Value>> thenYield = buildRegionPath(
        condBr.getTrueDest(), condBr.getTrueDestOperands(), joinBlock, builder);
    if (failed(thenYield))
      return failure();
    if (thenYield->size() != resultTypes.size()) {
      condBr.emitError("then branch yields ")
          << thenYield->size() << " values, expected " << resultTypes.size();
      return failure();
    }
    if (!autoYield)
      builder.create<scf::YieldOp>(condBr.getLoc(), *thenYield);
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Operation *autoYield =
        resultTypes.empty() ? ifOp.elseBlock()->getTerminator() : nullptr;
    if (autoYield)
      builder.setInsertionPoint(autoYield);
    else
      builder.setInsertionPointToStart(ifOp.elseBlock());
    FailureOr<SmallVector<Value>> elseYield =
        buildRegionPath(condBr.getFalseDest(), condBr.getFalseDestOperands(),
                        joinBlock, builder);
    if (failed(elseYield))
      return failure();
    if (elseYield->size() != resultTypes.size()) {
      condBr.emitError("else branch yields ")
          << elseYield->size() << " values, expected " << resultTypes.size();
      return failure();
    }
    if (!autoYield)
      builder.create<scf::YieldOp>(condBr.getLoc(), *elseYield);
  }

  return ifOp;
}

static Operation *createReturnLike(OpBuilder &builder, Location loc,
                                   Operation *sampleReturn,
                                   ValueRange operands) {
  // Preserve whether the containing callable uses tt.return or func.return,
  // along with any dialect-specific attributes on that terminator.
  OperationState state(loc, sampleReturn->getName());
  state.addOperands(operands);
  state.addAttributes(sampleReturn->getAttrs());
  return builder.create(state);
}

//===----------------------------------------------------------------------===//
// Terminal branch construction
//===----------------------------------------------------------------------===//

/// Computes the value types returned by a terminal path without mutating the
/// CFG. Both arms of a terminal conditional must return the same signature so
/// they can become the results of a value-producing scf.if.
static FailureOr<SmallVector<Type>>
collectReturnPathTypes(Block *block, SmallPtrSetImpl<Block *> &visiting) {
  if (!visiting.insert(block).second)
    return block->getTerminator()->emitError()
           << "unsupported cyclic terminal control flow";

  Operation *term = block->getTerminator();
  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    FailureOr<SmallVector<Type>> result =
        collectReturnPathTypes(br.getDest(), visiting);
    visiting.erase(block);
    return result;
  }

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    FailureOr<Block *> nestedJoin = findNearestCommonBlock(
        condBr.getTrueDest(), condBr.getFalseDest(), condBr.getLoc(),
        /*emitDiagnostic=*/false);
    if (succeeded(nestedJoin)) {
      FailureOr<SmallVector<Type>> result =
          collectReturnPathTypes(*nestedJoin, visiting);
      visiting.erase(block);
      return result;
    }

    FailureOr<SmallVector<Type>> thenTypes =
        collectReturnPathTypes(condBr.getTrueDest(), visiting);
    FailureOr<SmallVector<Type>> elseTypes =
        collectReturnPathTypes(condBr.getFalseDest(), visiting);
    visiting.erase(block);
    if (failed(thenTypes) || failed(elseTypes))
      return failure();
    if (!haveSameTypes(TypeRange{*thenTypes}, TypeRange{*elseTypes})) {
      condBr.emitError("terminal branch return types do not match");
      return failure();
    }
    return *thenTypes;
  }

  if (isSupportedReturn(term)) {
    SmallVector<Type> types;
    for (Value operand : term->getOperands())
      types.push_back(operand.getType());
    visiting.erase(block);
    return types;
  }

  visiting.erase(block);
  return term->emitError()
         << "unsupported terminator while analyzing terminal control flow";
}

static Operation *findReturnOnPath(Block *block,
                                   SmallPtrSetImpl<Block *> &visited) {
  // The operation name/attributes of one reachable return are used as the
  // template after terminal paths have been converted to yielded values.
  if (!visited.insert(block).second)
    return nullptr;

  Operation *term = block->getTerminator();
  if (isSupportedReturn(term))
    return term;
  for (Block *successor : getCfgSuccessors(block)) {
    if (successor->getParent() != block->getParent())
      continue;
    if (Operation *returnOp = findReturnOnPath(successor, visited))
      return returnOp;
  }
  return nullptr;
}

static SmallVector<Value> mapValues(ValueRange values, IRMapping &mapping) {
  // lookupOrDefault is intentional for values captured from outside the cloned
  // path; only block arguments and locally cloned results require mappings.
  SmallVector<Value> mapped;
  mapped.reserve(values.size());
  for (Value value : values)
    mapped.push_back(mapping.lookupOrDefault(value));
  return mapped;
}

static FailureOr<SmallVector<Value>>
buildClonedTerminalPath(Block *block, ValueRange incoming, OpBuilder &builder,
                        IRMapping mapping, SmallPtrSetImpl<Block *> &visiting);

static FailureOr<SmallVector<Value>>
buildClonedTerminalTerminator(Operation *term, OpBuilder &builder,
                              IRMapping mapping,
                              SmallPtrSetImpl<Block *> &visiting) {
  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    SmallVector<Value> incoming = mapValues(br.getDestOperands(), mapping);
    return buildClonedTerminalPath(br.getDest(), incoming, builder, mapping,
                                   visiting);
  }

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    SmallPtrSet<Block *, 16> thenVisiting;
    FailureOr<SmallVector<Type>> thenTypes =
        collectReturnPathTypes(condBr.getTrueDest(), thenVisiting);
    SmallPtrSet<Block *, 16> elseVisiting;
    FailureOr<SmallVector<Type>> elseTypes =
        collectReturnPathTypes(condBr.getFalseDest(), elseVisiting);
    if (failed(thenTypes) || failed(elseTypes))
      return failure();
    if (!haveSameTypes(TypeRange{*thenTypes}, TypeRange{*elseTypes})) {
      condBr.emitError("terminal branch return types do not match");
      return failure();
    }

    auto ifOp = builder.create<scf::IfOp>(
        condBr.getLoc(), *thenTypes,
        mapping.lookupOrDefault(condBr.getCondition()),
        /*withElseRegion=*/true);

    {
      OpBuilder::InsertionGuard guard(builder);
      Operation *autoYield =
          thenTypes->empty() ? ifOp.thenBlock()->getTerminator() : nullptr;
      if (autoYield)
        builder.setInsertionPoint(autoYield);
      else
        builder.setInsertionPointToStart(ifOp.thenBlock());
      SmallVector<Value> incoming =
          mapValues(condBr.getTrueDestOperands(), mapping);
      FailureOr<SmallVector<Value>> thenReturn = buildClonedTerminalPath(
          condBr.getTrueDest(), incoming, builder, mapping, visiting);
      if (failed(thenReturn))
        return failure();
      if (!haveSameTypes(TypeRange{ValueRange{*thenReturn}},
                         TypeRange{*thenTypes})) {
        condBr.emitError("then terminal branch returns incompatible values");
        return failure();
      }
      if (!autoYield)
        builder.create<scf::YieldOp>(condBr.getLoc(), *thenReturn);
    }

    {
      OpBuilder::InsertionGuard guard(builder);
      Operation *autoYield =
          thenTypes->empty() ? ifOp.elseBlock()->getTerminator() : nullptr;
      if (autoYield)
        builder.setInsertionPoint(autoYield);
      else
        builder.setInsertionPointToStart(ifOp.elseBlock());
      SmallVector<Value> incoming =
          mapValues(condBr.getFalseDestOperands(), mapping);
      FailureOr<SmallVector<Value>> elseReturn = buildClonedTerminalPath(
          condBr.getFalseDest(), incoming, builder, mapping, visiting);
      if (failed(elseReturn))
        return failure();
      if (!haveSameTypes(TypeRange{ValueRange{*elseReturn}},
                         TypeRange{*thenTypes})) {
        condBr.emitError("else terminal branch returns incompatible values");
        return failure();
      }
      if (!autoYield)
        builder.create<scf::YieldOp>(condBr.getLoc(), *elseReturn);
    }

    return SmallVector<Value>(ifOp->getResults().begin(),
                              ifOp->getResults().end());
  }

  if (isSupportedReturn(term))
    return mapValues(term->getOperands(), mapping);

  return term->emitError()
         << "unsupported terminator while structuring terminal control flow";
}

static FailureOr<SmallVector<Value>>
buildClonedTerminalPath(Block *block, ValueRange incoming, OpBuilder &builder,
                        IRMapping mapping, SmallPtrSetImpl<Block *> &visiting) {
  // Terminal paths are cloned rather than moved. Analysis and construction can
  // therefore fail without partially consuming the original CFG; the old
  // blocks are erased only after the complete replacement return is built.
  if (!visiting.insert(block).second)
    return block->getTerminator()->emitError()
           << "unsupported cyclic terminal control flow";

  if (block->getNumArguments() != incoming.size()) {
    visiting.erase(block);
    return block->getTerminator()->emitError()
           << "invalid branch operand count while structuring terminal "
              "control flow";
  }

  for (auto [arg, value] : llvm::zip(block->getArguments(), incoming))
    mapping.map(arg, value);

  for (Operation &op : block->without_terminator())
    builder.clone(op, mapping);

  FailureOr<SmallVector<Value>> result = buildClonedTerminalTerminator(
      block->getTerminator(), builder, mapping, visiting);
  visiting.erase(block);
  return result;
}

static bool hasNonTreeCondBranch(Region &body) {
  // A branch without a join cannot be consumed by the move-based path. If any
  // such branch exists, clone the complete terminal tree atomically instead.
  for (Block &block : body) {
    auto condBr = dyn_cast<cf::CondBranchOp>(block.getTerminator());
    if (!condBr)
      continue;
    if (failed(findNearestCommonBlock(condBr.getTrueDest(),
                                      condBr.getFalseDest(), condBr.getLoc(),
                                      /*emitDiagnostic=*/false)))
      return true;
  }
  return false;
}

static LogicalResult structureTerminalReturnBody(Operation *funcOp,
                                                 Region &body) {
  // This path is used when branch arms do not reconverge but both end in
  // compatible returns. The cloned scf.if produces the return operands, after
  // which every original non-entry block can be removed together.
  Block &entryBlock = body.front();
  Operation *entryTerm = entryBlock.getTerminator();
  SmallPtrSet<Block *, 16> visited;
  Operation *sampleReturn = findReturnOnPath(&entryBlock, visited);
  if (!sampleReturn) {
    return funcOp->emitError()
           << "unsupported non-tree control flow: no terminal return found";
  }

  OpBuilder builder(entryTerm);
  IRMapping mapping;
  SmallPtrSet<Block *, 16> visiting;
  FailureOr<SmallVector<Value>> returnOperands =
      buildClonedTerminalTerminator(entryTerm, builder, mapping, visiting);
  if (failed(returnOperands))
    return failure();

  createReturnLike(builder, entryTerm->getLoc(), sampleReturn, *returnOperands);

  SmallVector<Block *> eraseBlocks;
  for (Block &block : llvm::drop_begin(body.getBlocks()))
    eraseBlocks.push_back(&block);

  entryTerm->erase();
  for (Block *block : eraseBlocks) {
    for (Operation &op : *block)
      op.dropAllReferences();
  }
  for (Block *block : llvm::reverse(eraseBlocks))
    block->erase();

  return success();
}

static FailureOr<scf::IfOp> buildTerminalValueIf(cf::CondBranchOp condBr,
                                                 OpBuilder &builder) {
  SmallPtrSet<Block *, 16> thenVisiting;
  FailureOr<SmallVector<Type>> thenTypes =
      collectReturnPathTypes(condBr.getTrueDest(), thenVisiting);
  SmallPtrSet<Block *, 16> elseVisiting;
  FailureOr<SmallVector<Type>> elseTypes =
      collectReturnPathTypes(condBr.getFalseDest(), elseVisiting);
  if (failed(thenTypes) || failed(elseTypes))
    return failure();
  if (!haveSameTypes(TypeRange{*thenTypes}, TypeRange{*elseTypes})) {
    condBr.emitError("terminal branch return types do not match");
    return failure();
  }

  auto ifOp = builder.create<scf::IfOp>(condBr.getLoc(), *thenTypes,
                                        condBr.getCondition(),
                                        /*withElseRegion=*/true);

  {
    OpBuilder::InsertionGuard branchGuard(builder);
    Operation *autoYield =
        thenTypes->empty() ? ifOp.thenBlock()->getTerminator() : nullptr;
    if (autoYield)
      builder.setInsertionPoint(autoYield);
    else
      builder.setInsertionPointToStart(ifOp.thenBlock());
    FailureOr<ReturnPathResult> thenReturn = buildReturnPath(
        condBr.getTrueDest(), condBr.getTrueDestOperands(), builder);
    if (failed(thenReturn))
      return failure();
    if (!haveSameTypes(TypeRange{ValueRange{thenReturn->operands}},
                       TypeRange{*thenTypes})) {
      condBr.emitError("then terminal branch returns incompatible values");
      return failure();
    }
    if (!autoYield)
      builder.create<scf::YieldOp>(condBr.getLoc(), thenReturn->operands);
  }

  {
    OpBuilder::InsertionGuard branchGuard(builder);
    Operation *autoYield =
        thenTypes->empty() ? ifOp.elseBlock()->getTerminator() : nullptr;
    if (autoYield)
      builder.setInsertionPoint(autoYield);
    else
      builder.setInsertionPointToStart(ifOp.elseBlock());
    FailureOr<ReturnPathResult> elseReturn = buildReturnPath(
        condBr.getFalseDest(), condBr.getFalseDestOperands(), builder);
    if (failed(elseReturn))
      return failure();
    if (!haveSameTypes(TypeRange{ValueRange{elseReturn->operands}},
                       TypeRange{*thenTypes})) {
      condBr.emitError("else terminal branch returns incompatible values");
      return failure();
    }
    if (!autoYield)
      builder.create<scf::YieldOp>(condBr.getLoc(), elseReturn->operands);
  }

  return ifOp;
}

static FailureOr<ReturnPathResult>
buildReturnPath(Block *block, ValueRange incoming, OpBuilder &builder) {
  Operation *term = block->getTerminator();
  if (failed(replaceBlockArguments(block, incoming, term->getLoc())))
    return failure();
  moveBlockBodyBefore(block, builder);

  if (auto br = dyn_cast<cf::BranchOp>(term))
    return buildReturnPath(br.getDest(), br.getDestOperands(), builder);

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    FailureOr<Block *> nestedJoin = findNearestCommonBlock(
        condBr.getTrueDest(), condBr.getFalseDest(), condBr.getLoc(),
        /*emitDiagnostic=*/false);
    if (succeeded(nestedJoin)) {
      FailureOr<scf::IfOp> nestedIf =
          buildStructuredIf(condBr, *nestedJoin, builder);
      if (failed(nestedIf))
        return failure();

      SmallVector<Value> nestedResults((*nestedIf)->getResults().begin(),
                                       (*nestedIf)->getResults().end());
      return buildReturnPath(*nestedJoin, nestedResults, builder);
    }

    FailureOr<scf::IfOp> terminalIf = buildTerminalValueIf(condBr, builder);
    if (failed(terminalIf)) {
      condBr.emitError() << "unsupported non-tree control flow: branch arms do "
                            "not both terminate with compatible returns";
      return failure();
    }

    ReturnPathResult result;
    result.operands.assign((*terminalIf)->getResults().begin(),
                           (*terminalIf)->getResults().end());
    return result;
  }

  if (isSupportedReturn(term)) {
    ReturnPathResult result;
    result.operands.assign(term->getOperands().begin(),
                           term->getOperands().end());
    return result;
  }

  return term->emitError()
         << "unsupported terminator while structuring terminal control flow";
}

//===----------------------------------------------------------------------===//
// Top-level CFG consumption
//===----------------------------------------------------------------------===//

/// Consumes one branch arm until `stopBlock`. Nested conditionals are converted
/// recursively, and their results become the incoming values of the next join.
static FailureOr<SmallVector<Value>> buildRegionPath(Block *block,
                                                     ValueRange incoming,
                                                     Block *stopBlock,
                                                     OpBuilder &builder) {
  if (block == stopBlock)
    return SmallVector<Value>(incoming.begin(), incoming.end());

  Operation *term = block->getTerminator();
  if (failed(replaceBlockArguments(block, incoming, term->getLoc())))
    return failure();
  moveBlockBodyBefore(block, builder);

  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    SmallVector<Value> operands(br.getDestOperands().begin(),
                                br.getDestOperands().end());
    if (br.getDest() == stopBlock)
      return operands;
    return buildRegionPath(br.getDest(), operands, stopBlock, builder);
  }

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    FailureOr<Block *> nestedJoin = findNearestCommonBlock(
        condBr.getTrueDest(), condBr.getFalseDest(), condBr.getLoc());
    if (failed(nestedJoin))
      return failure();

    FailureOr<scf::IfOp> nestedIf =
        buildStructuredIf(condBr, *nestedJoin, builder);
    if (failed(nestedIf))
      return failure();

    SmallVector<Value> nestedResults((*nestedIf)->getResults().begin(),
                                     (*nestedIf)->getResults().end());
    if (*nestedJoin == stopBlock)
      return nestedResults;
    return buildRegionPath(*nestedJoin, nestedResults, stopBlock, builder);
  }

  if (isSupportedReturn(term)) {
    return term->emitError()
           << "unsupported early return while structuring control flow";
  }

  return term->emitError()
         << "unsupported terminator while structuring control flow";
}

static LogicalResult appendStructuredBlock(Block *block, ValueRange incoming,
                                           OpBuilder &builder,
                                           Operation *anchorTerminator);

static LogicalResult appendStructuredTerminator(Operation *term,
                                                OpBuilder &builder,
                                                Operation *anchorTerminator) {
  // Walk the entry CFG in execution order. A converging conditional is emitted
  // as scf.if and traversal resumes at its join; a terminal conditional emits
  // the final return and finishes the function body.
  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    return appendStructuredBlock(br.getDest(), br.getDestOperands(), builder,
                                 anchorTerminator);
  }

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    FailureOr<Block *> joinBlock = findNearestCommonBlock(
        condBr.getTrueDest(), condBr.getFalseDest(), condBr.getLoc(),
        /*emitDiagnostic=*/false);
    if (failed(joinBlock)) {
      SmallPtrSet<Block *, 16> visited;
      Operation *sampleReturn = findReturnOnPath(condBr.getTrueDest(), visited);
      if (!sampleReturn) {
        visited.clear();
        sampleReturn = findReturnOnPath(condBr.getFalseDest(), visited);
      }
      if (!sampleReturn) {
        return condBr.emitError()
               << "unsupported non-tree control flow: branch arms do not "
                  "reach a common convergence block";
      }

      FailureOr<scf::IfOp> terminalIf = buildTerminalValueIf(condBr, builder);
      if (failed(terminalIf))
        return failure();

      SmallVector<Value> returnOperands((*terminalIf)->getResults().begin(),
                                        (*terminalIf)->getResults().end());
      createReturnLike(builder, condBr.getLoc(), sampleReturn, returnOperands);
      return success();
    }

    FailureOr<scf::IfOp> ifOp = buildStructuredIf(condBr, *joinBlock, builder);
    if (failed(ifOp))
      return failure();

    return appendStructuredBlock(*joinBlock, (*ifOp)->getResults(), builder,
                                 anchorTerminator);
  }

  if (isSupportedReturn(term)) {
    term->moveBefore(anchorTerminator);
    return success();
  }

  return term->emitError()
         << "unsupported entry terminator while structuring control flow";
}

static LogicalResult appendStructuredBlock(Block *block, ValueRange incoming,
                                           OpBuilder &builder,
                                           Operation *anchorTerminator) {
  Operation *term = block->getTerminator();
  if (failed(replaceBlockArguments(block, incoming, term->getLoc())))
    return failure();

  moveBlockBodyBefore(block, builder);
  return appendStructuredTerminator(term, builder, anchorTerminator);
}

static LogicalResult validateSupportedCfg(Region &body) {
  // Validate every reachable block before move-based construction starts, so
  // unsupported terminators cannot leave a partially consumed function body.
  for (Block &block : body) {
    Operation *term = block.getTerminator();
    if (!isa<cf::BranchOp, cf::CondBranchOp>(term) && !isSupportedReturn(term))
      return term->emitError()
             << "unsupported terminator in multi-block function";
  }
  return success();
}

/// Collects blocks reachable from the function entry through supported CFG
/// successors. Restrict successor traversal to the current region because a
/// branch-like operation must not make a nested or enclosing block reachable.
static void collectReachableBlocks(Block *block,
                                   SmallPtrSetImpl<Block *> &reachable) {
  if (!reachable.insert(block).second)
    return;

  for (Block *successor : getCfgSuccessors(block)) {
    if (successor->getParent() == block->getParent())
      collectReachableBlocks(successor, reachable);
  }
}

/// Removes blocks that cannot be reached from the entry block before CFG
/// validation and structuring. Frontend lowering may leave detached return
/// blocks behind; they are not part of the function's executable CFG and must
/// not affect join discovery, cycle checks or the entry-terminator decision.
static void eraseUnreachableBlocks(Region &body) {
  if (body.empty() || body.hasOneBlock())
    return;

  SmallPtrSet<Block *, 16> reachable;
  collectReachableBlocks(&body.front(), reachable);

  SmallVector<Block *> eraseBlocks;
  for (Block &block : body) {
    if (!reachable.contains(&block))
      eraseBlocks.push_back(&block);
  }

  // Break successor and operand references before erasing. This also handles
  // unreachable blocks that refer to one another, including unreachable
  // cycles.
  for (Block *block : eraseBlocks) {
    for (Operation &op : *block)
      op.dropAllReferences();
  }
  for (Block *block : llvm::reverse(eraseBlocks))
    block->erase();
}

/// Rejects backedges before any destructive block movement occurs. General
/// restructuring of cyclic CFGs is not supported.
static LogicalResult rejectCyclicCfg(Block *block,
                                     SmallPtrSetImpl<Block *> &visiting,
                                     SmallPtrSetImpl<Block *> &visited) {
  if (visited.contains(block))
    return success();
  if (!visiting.insert(block).second)
    return block->getTerminator()->emitError()
           << "unsupported cyclic control flow in multi-block function";

  for (Block *successor : getCfgSuccessors(block)) {
    if (successor->getParent() == block->getParent() &&
        failed(rejectCyclicCfg(successor, visiting, visited)))
      return failure();
  }

  visiting.erase(block);
  visited.insert(block);
  return success();
}

static LogicalResult structureFunctionBody(Operation *funcOp, Region &body) {
  // Validation is deliberately completed before the move-based path starts.
  // From that point onward the function is rewritten as one SCF entry block and
  // the consumed CFG blocks are erased only after construction succeeds.
  if (body.empty() || body.hasOneBlock())
    return success();

  eraseUnreachableBlocks(body);
  if (body.hasOneBlock())
    return success();

  if (failed(validateSupportedCfg(body)))
    return failure();

  SmallPtrSet<Block *, 16> visiting;
  SmallPtrSet<Block *, 16> visited;
  if (failed(rejectCyclicCfg(&body.front(), visiting, visited)))
    return failure();

  if (hasNonTreeCondBranch(body))
    return structureTerminalReturnBody(funcOp, body);

  Block &entryBlock = body.front();
  Operation *entryTerm = entryBlock.getTerminator();
  if (isSupportedReturn(entryTerm)) {
    return funcOp->emitError()
           << "multi-block function entry cannot terminate with return";
  }

  SmallVector<Block *> eraseBlocks;
  for (Block &block : llvm::drop_begin(body.getBlocks()))
    eraseBlocks.push_back(&block);

  OpBuilder builder(entryTerm);
  if (failed(appendStructuredTerminator(entryTerm, builder, entryTerm)))
    return failure();

  entryTerm->erase();
  for (Block *block : eraseBlocks) {
    for (Operation &op : *block)
      op.dropAllReferences();
  }
  for (Block *block : llvm::reverse(eraseBlocks))
    block->erase();

  return success();
}

} // namespace

namespace mlir::triton::controlflow {

LogicalResult structureCFG(ModuleOp module) {
  // Collect functions first because structureFunctionBody mutates their nested
  // regions. This keeps the module walk independent of those mutations.
  SmallVector<Operation *> functions;
  module.walk([&](Operation *op) {
    if (isa<triton::FuncOp, func::FuncOp, triton::MapElementwiseOp>(op))
      functions.push_back(op);
  });

  // Handle both Triton callables and ordinary func.func wrappers; declarations
  // and functions erased by an enclosing transformation are skipped.
  for (Operation *op : functions) {
    if (!op->getParentOp())
      continue;

    if (auto funcOp = dyn_cast<triton::FuncOp>(op)) {
      if (!funcOp.isDeclaration() &&
          failed(structureFunctionBody(funcOp, funcOp.getBody())))
        return failure();
      continue;
    }

    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (!funcOp.isDeclaration() &&
          failed(structureFunctionBody(funcOp, funcOp.getBody())))
        return failure();
      continue;
    }

    auto mapOp = cast<triton::MapElementwiseOp>(op);
    if (failed(structureFunctionBody(mapOp, mapOp.getRegion())))
      return failure();
  }

  return success();
}

} // namespace mlir::triton::controlflow
