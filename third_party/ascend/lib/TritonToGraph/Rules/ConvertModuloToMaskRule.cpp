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

#include "TritonToGraph/GraphOptimizationRule.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

// A modulo whose result is only ever used to compute load addresses, and whose
// wrapped lanes are already discarded by a store mask.  Such a modulo can be
// dropped so the addresses become linear again, provided the loads that consume
// them get a boundary mask so the wrapped lanes read in-bounds zeros instead of
// the data the wrap used to fetch.
struct ModuloCandidate {
  // The op defining the modulo result: either arith.remsi, or the subi of the
  // divsi/muli/subi expansion that canonicalization leaves behind.
  Operation *anchor = nullptr;
  Value dividend;
  Value divisor;
  Value boundScalar;
  Value result;
  // anchor plus, for the expanded form, the muli and divsi that die with it.
  SmallVector<Operation *, 3> patternOps;
  SmallVector<triton::LoadOp, 4> loads;
  int64_t tileSize = 0;
  // The axis that expand_dims inserted on the modulo result.  The mask has to
  // be expanded on the same axis to vary along the same dimension of the load.
  int64_t axis = 0;
};

Value getSplatScalar(Value value) {
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    return splat.getSrc();
  return nullptr;
}

// Reports the extent of the tt.make_range that the tile offset is built from.
// The offset is `splat(pid * BLOCK) + make_range(0, BLOCK)`, so the range is
// reached through the addi.
int64_t getMakeRangeExtent(Value value) {
  if (auto range = value.getDefiningOp<triton::MakeRangeOp>())
    return static_cast<int64_t>(range.getEnd()) -
           static_cast<int64_t>(range.getStart());
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    if (int64_t lhs = getMakeRangeExtent(add.getLhs()))
      return lhs;
    return getMakeRangeExtent(add.getRhs());
  }
  return 0;
}

bool isCompileTimeConstant(Value value) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant));
}

// Reports whether `mask` reaches the mask operand of a tt.store.  Shape ops and
// arith.andi are transparent because a store mask is normally the conjunction
// of one comparison per output axis, each broadcast to the output shape.
bool feedsStoreMask(Value mask, SmallPtrSetImpl<Operation *> &visited) {
  for (OpOperand &use : mask.getUses()) {
    Operation *user = use.getOwner();
    if (auto store = dyn_cast<triton::StoreOp>(user)) {
      if (store.getMask() == mask)
        return true;
      continue;
    }
    if (!isa<triton::BroadcastOp, triton::ExpandDimsOp, arith::AndIOp>(user))
      continue;
    if (!visited.insert(user).second)
      continue;
    if (feedsStoreMask(user->getResult(0), visited))
      return true;
  }
  return false;
}

// Reports whether some `cmpi slt` of the un-wrapped tile offset against the
// same bound guards a store.  This is the proof that the kernel itself treats
// offsets at or beyond the bound as absent, so the values the wrap used to
// fetch for those lanes never reach memory.
//
// It also pins the modulo to an axis of the output.  A modulo on a reduction
// axis could not be rewritten this way, because zeros injected there would
// change every output element rather than only the discarded lanes, and such a
// modulo has no store-mask guard precisely because the axis has no output
// coordinate.
bool hasStoreMaskGuard(Value offset, Value boundScalar) {
  SmallVector<Value, 2> forms = {offset};
  for (OpOperand &use : offset.getUses()) {
    if (isa<triton::ExpandDimsOp>(use.getOwner()))
      forms.push_back(use.getOwner()->getResult(0));
  }

  for (Value form : forms) {
    for (OpOperand &use : form.getUses()) {
      auto compare = dyn_cast<arith::CmpIOp>(use.getOwner());
      if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt ||
          compare.getLhs() != form)
        continue;
      Value bound = getSplatScalar(compare.getRhs());
      if (bound != boundScalar)
        continue;
      SmallPtrSet<Operation *, 16> visited;
      if (feedsStoreMask(compare.getResult(), visited))
        return true;
    }
  }
  return false;
}

// Returns true when two distinct loop-carried paths cross nested scf.for ops.
// An address passed through one loop may still be lowered as a direct access,
// but the current structured lowering does not preserve every component of a
// nested loop-carried pointer state.  Keep the modulo in that shape until the
// lowerer can prove the relay lossless.
bool crossesNestedCarriedFor(ArrayRef<Operation *> carriedForChain,
                             scf::ForOp nextFor) {
  Operation *next = nextFor.getOperation();
  return llvm::any_of(carriedForChain, [next](Operation *carriedFor) {
    if (carriedFor == next)
      return false;
    return carriedFor->isProperAncestor(next) ||
           next->isProperAncestor(carriedFor);
  });
}

// Walks forward from the modulo result and collects the loads it addresses.
// Returns false as soon as the value reaches anything else, which is what makes
// dropping the wrap unobservable: no other consumer can see the widened index.
//
// The allowlist is restricted to ops that preserve dimension positions, so the
// axis recorded for the modulo result stays the axis it varies along in every
// load that it reaches.  tt.trans and tt.reshape are therefore rejected.
bool collectAddressedLoads(Value value, SmallPtrSetImpl<Value> &visited,
                           SmallVectorImpl<triton::LoadOp> &loads,
                           ArrayRef<Operation *> carriedForChain) {
  if (!visited.insert(value).second)
    return true;

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();

    if (auto load = dyn_cast<triton::LoadOp>(user)) {
      // Reject a value that is the `other` operand rather than the address:
      // rewriting it would change the loaded data itself.
      if (load.getPtr() != value)
        return false;
      loads.push_back(load);
      continue;
    }
    if (isa<triton::ExpandDimsOp, triton::BroadcastOp, triton::AddPtrOp,
            arith::MulIOp, arith::AddIOp>(user)) {
      for (Value result : user->getResults()) {
        if (!collectAddressedLoads(result, visited, loads, carriedForChain))
          return false;
      }
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      // An address carried around a loop stays an address only if the iter arg
      // and the loop result are both used as one.
      for (auto [index, initArg] : llvm::enumerate(forOp.getInitArgs())) {
        if (initArg != value)
          continue;
        if (crossesNestedCarriedFor(carriedForChain, forOp))
          return false;
        SmallVector<Operation *, 2> nextChain(carriedForChain);
        nextChain.push_back(forOp.getOperation());
        if (!collectAddressedLoads(forOp.getRegionIterArg(index), visited,
                                   loads, nextChain))
          return false;
      }
      continue;
    }
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp());
      if (!forOp)
        return false;
      unsigned index = use.getOperandNumber();
      if (index >= forOp.getNumResults())
        return false;
      if (crossesNestedCarriedFor(carriedForChain, forOp))
        return false;
      SmallVector<Operation *, 2> nextChain(carriedForChain);
      nextChain.push_back(forOp.getOperation());
      if (!collectAddressedLoads(forOp.getResult(index), visited, loads,
                                 nextChain))
        return false;
      continue;
    }

    return false;
  }
  return true;
}

// Reports the single axis that expand_dims inserted on the modulo result.
// Distinct axes mean the one index addresses two different dimensions, and no
// single mask orientation would be correct for both, so such a result is
// rejected rather than rewritten on a guessed axis.
std::optional<int64_t> getUniqueExpandDimsAxis(Value value) {
  std::optional<int64_t> axis;
  for (OpOperand &use : value.getUses()) {
    auto expand = dyn_cast<triton::ExpandDimsOp>(use.getOwner());
    if (!expand)
      continue;
    int64_t current = expand.getAxis();
    if (axis && *axis != current)
      return std::nullopt;
    axis = current;
  }
  return axis;
}

// Reports whether a boundary mask of `tileSize` lanes, expanded on `axis`, can
// be broadcast to every loaded shape.  Checking it up front means the mask ops
// this rule creates are valid by construction.
bool areLoadsMaskable(ArrayRef<triton::LoadOp> loads, int64_t axis,
                      int64_t tileSize) {
  if (loads.empty() || axis < 0 || axis > 1)
    return false;
  for (triton::LoadOp load : loads) {
    auto type = dyn_cast<RankedTensorType>(load.getResult().getType());
    if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
        type.getShape()[1 - axis] != tileSize)
      return false;
    // A load carrying `other` without a mask reads it nowhere, so injecting a
    // mask would suddenly make that operand observable.
    if (!load.getMask() && load.getOther())
      return false;
    if (!load.getBoundaryCheck().empty() || load.getIsVolatile())
      return false;
  }
  return true;
}

// Matches the divsi/muli/subi expansion of a modulo.  It is equivalent to
// `x % d` by the signed-division identity x == (x / d) * d + x % d.
bool matchExpandedModulo(arith::SubIOp subtract, Value &dividend,
                         Value &divisor,
                         SmallVectorImpl<Operation *> &patternOps) {
  auto multiply = subtract.getRhs().getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return false;

  arith::DivSIOp divide;
  Value candidateDivisor;
  if (auto lhs = multiply.getLhs().getDefiningOp<arith::DivSIOp>()) {
    divide = lhs;
    candidateDivisor = multiply.getRhs();
  } else if (auto rhs = multiply.getRhs().getDefiningOp<arith::DivSIOp>()) {
    divide = rhs;
    candidateDivisor = multiply.getLhs();
  } else {
    return false;
  }

  if (divide.getLhs() != subtract.getLhs() ||
      divide.getRhs() != candidateDivisor)
    return false;

  dividend = subtract.getLhs();
  divisor = candidateDivisor;
  patternOps.assign({subtract.getOperation(), multiply.getOperation(),
                     divide.getOperation()});
  return true;
}

std::optional<ModuloCandidate> analyzeModulo(Operation *op) {
  ModuloCandidate candidate;
  if (auto remainder = dyn_cast<arith::RemSIOp>(op)) {
    candidate.dividend = remainder.getLhs();
    candidate.divisor = remainder.getRhs();
    candidate.patternOps.push_back(op);
  } else if (auto subtract = dyn_cast<arith::SubIOp>(op)) {
    if (!matchExpandedModulo(subtract, candidate.dividend, candidate.divisor,
                             candidate.patternOps))
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  candidate.anchor = op;
  candidate.result = op->getResult(0);

  // A 1-D i32 index is the shape every tile offset has at this point in the
  // pipeline.  Anything else is left alone rather than reasoned about.
  auto type = dyn_cast<RankedTensorType>(candidate.result.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 1 ||
      !type.getElementType().isInteger(32) || type.getEncoding())
    return std::nullopt;

  candidate.tileSize = getMakeRangeExtent(candidate.dividend);
  if (candidate.tileSize <= 0 || candidate.tileSize != type.getShape()[0])
    return std::nullopt;

  candidate.boundScalar = getSplatScalar(candidate.divisor);
  if (!candidate.boundScalar)
    return std::nullopt;

  // A constant divisor belongs to TritonToStructured, whose visitOperandRem
  // keeps the wrap and re-expresses it as a strided access.  That is exactly
  // equivalent, so it is always preferable to discarding the wrap here.
  if (isCompileTimeConstant(candidate.boundScalar))
    return std::nullopt;

  if (!hasStoreMaskGuard(candidate.dividend, candidate.boundScalar))
    return std::nullopt;

  std::optional<int64_t> axis = getUniqueExpandDimsAxis(candidate.result);
  if (!axis)
    return std::nullopt;
  candidate.axis = *axis;

  SmallPtrSet<Value, 32> visited;
  SmallVector<Operation *, 2> carriedForChain;
  if (!collectAddressedLoads(candidate.result, visited, candidate.loads,
                             carriedForChain))
    return std::nullopt;
  if (!areLoadsMaskable(candidate.loads, candidate.axis, candidate.tileSize))
    return std::nullopt;

  return candidate;
}

bool matchesCandidate(const ModuloCandidate &candidate,
                      const ModuloCandidate &current) {
  return candidate.anchor == current.anchor &&
         candidate.dividend == current.dividend &&
         candidate.divisor == current.divisor &&
         candidate.boundScalar == current.boundScalar &&
         candidate.result == current.result &&
         candidate.tileSize == current.tileSize &&
         candidate.axis == current.axis &&
         candidate.patternOps.size() == current.patternOps.size() &&
         std::equal(candidate.patternOps.begin(), candidate.patternOps.end(),
                    current.patternOps.begin()) &&
         candidate.loads.size() == current.loads.size() &&
         std::equal(candidate.loads.begin(), candidate.loads.end(),
                    current.loads.begin());
}

void eraseCreatedOperations(IRRewriter &rewriter,
                            ArrayRef<Operation *> created) {
  for (Operation *operation : llvm::reverse(created))
    rewriter.eraseOp(operation);
}

bool recordVerifiedOperation(Operation *operation,
                             SmallVectorImpl<Operation *> &created) {
  if (!operation)
    return false;
  created.push_back(operation);
  return succeeded(mlir::verify(operation));
}

// The mask that a single load needs, plus the zero fill for a load that had no
// mask before.  Nothing is attached to the load until every load's operands
// have been built and verified.
struct LoadMask {
  triton::LoadOp load;
  Value mask;
  Value other;
};

LogicalResult applyCandidate(IRRewriter &rewriter,
                             const ModuloCandidate &candidate) {
  SmallVector<Operation *, 16> created;
  auto fail = [&]() {
    eraseCreatedOperations(rewriter, created);
    return failure();
  };

  Location loc = candidate.anchor->getLoc();
  rewriter.setInsertionPointAfter(candidate.anchor);
  auto boundary = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, candidate.dividend, candidate.divisor);
  if (!recordVerifiedOperation(boundary.getOperation(), created))
    return fail();

  SmallVector<LoadMask, 4> loadMasks;
  for (triton::LoadOp load : candidate.loads) {
    auto loadType = dyn_cast<RankedTensorType>(load.getResult().getType());
    if (!loadType)
      return fail();

    rewriter.setInsertionPoint(load);
    Location loadLoc = load.getLoc();
    auto expanded = rewriter.create<triton::ExpandDimsOp>(
        loadLoc, boundary.getResult(), static_cast<int32_t>(candidate.axis));
    if (!recordVerifiedOperation(expanded.getOperation(), created))
      return fail();

    auto maskType =
        RankedTensorType::get(loadType.getShape(), rewriter.getI1Type());
    auto broadcast = rewriter.create<triton::BroadcastOp>(loadLoc, maskType,
                                                          expanded.getResult());
    if (!recordVerifiedOperation(broadcast.getOperation(), created))
      return fail();

    LoadMask loadMask{load, broadcast.getResult(), nullptr};
    if (Value existing = load.getMask()) {
      auto combined = rewriter.create<arith::AndIOp>(loadLoc, existing,
                                                     broadcast.getResult());
      if (!recordVerifiedOperation(combined.getOperation(), created))
        return fail();
      loadMask.mask = combined.getResult();
    } else {
      Type elementType = loadType.getElementType();
      Attribute zero;
      if (isa<FloatType>(elementType))
        zero = rewriter.getFloatAttr(elementType, 0.0);
      else if (isa<IntegerType>(elementType))
        zero = rewriter.getIntegerAttr(elementType, 0);
      else
        return fail();
      auto fill = rewriter.create<arith::ConstantOp>(
          loadLoc, DenseElementsAttr::get(loadType, zero));
      if (!recordVerifiedOperation(fill.getOperation(), created))
        return fail();
      loadMask.other = fill.getResult();
    }
    loadMasks.push_back(loadMask);
  }
  if (loadMasks.empty())
    return fail();

  // Everything needed has been built and verified, so the observable rewrite
  // can now run without any step that could still fail.
  for (LoadMask &loadMask : loadMasks) {
    loadMask.load.getMaskMutable().assign(loadMask.mask);
    if (loadMask.other)
      loadMask.load.getOtherMutable().assign(loadMask.other);
  }

  rewriter.replaceAllUsesWith(candidate.result, candidate.dividend);
  for (Operation *operation : candidate.patternOps) {
    if (operation->use_empty())
      rewriter.eraseOp(operation);
  }
  return success();
}

class ConvertModuloToMaskPlan final : public RewritePlan {
public:
  ConvertModuloToMaskPlan(ModuloCandidate candidate, unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::ConvertModuloToMask;
  }

  // Every load that stops wrapping can become a contiguous transfer.
  unsigned getBenefit() const override {
    return static_cast<unsigned>(std::min<size_t>(
        candidate.loads.size(), std::numeric_limits<unsigned>::max()));
  }

  Operation *getAnchor() const override { return candidate.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (candidate.anchor->getParentOfType<triton::FuncOp>() !=
        context.getFunction())
      return failure();
    std::optional<ModuloCandidate> current = analyzeModulo(candidate.anchor);
    return current && matchesCandidate(candidate, *current) ? success()
                                                            : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    // Re-prove locally so that a stale plan can never mutate the IR.
    std::optional<ModuloCandidate> current = analyzeModulo(candidate.anchor);
    if (!current || !matchesCandidate(candidate, *current))
      return failure();
    return applyCandidate(rewriter, *current);
  }

private:
  ModuloCandidate candidate;
  unsigned epoch;
};

class ConvertModuloToMaskRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::ConvertModuloToMask;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    context.getFunction().walk([&](Operation *op) {
      if (!isa<arith::RemSIOp, arith::SubIOp>(op))
        return;
      if (std::optional<ModuloCandidate> candidate = analyzeModulo(op)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] matched graph optimization rule "
                   << static_cast<unsigned>(getId()) << " ("
                   << getGraphOptimizationRuleName(getId()) << ") at "
                   << op->getLoc() << ": " << op->getName()
                   << " tileSize=" << candidate->tileSize
                   << " maskedLoads=" << candidate->loads.size() << "\n");
        plans.push_back(std::make_unique<ConvertModuloToMaskPlan>(
            std::move(*candidate), context.getEpoch()));
      }
    });
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createConvertModuloToMaskRule() {
  return std::make_unique<ConvertModuloToMaskRule>();
}
