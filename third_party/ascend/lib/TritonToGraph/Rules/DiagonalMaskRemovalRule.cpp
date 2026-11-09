/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

// Replaces the O(N^2) "diagonal mask + reduce" vector shift with the O(N)
// cumulative-sum identity.
//
// A shift-by-one of a scan result is commonly expressed in the DSL as
//
//   mask    = (row == col + 1)
//   matrix  = tl.where(mask, tl.cumsum(x, axis=0)[None, :], 0)
//   shifted = tl.sum(matrix, axis=1)
//
// which materializes an N x N intermediate to read scan[i - 1].  Because an
// inclusive scan satisfies scan[i] - x[i] == scan[i - 1] (and the reverse scan
// satisfies scan[i] - x[i] == scan[i + 1]), the whole pattern collapses into a
// single elementwise subtraction.
//
// Only a unit shift can be rewritten.  For |shift| > 1 the pattern denotes a
// genuine data movement (scan[i - s] is scan[i] minus a sliding-window sum),
// and TTIR has no cheap way to express that, which is precisely why the DSL
// resorts to the diagonal-mask formulation in the first place.
//
// For floating point the rewrite is not bit-exact: scan[i] is computed as
// fl(scan[i - 1] + x[i]), so fl(scan[i] - x[i]) may differ from scan[i - 1] by
// about eps * |scan[i]|.  The boundary element, where nothing is selected, is
// still exact because x - x is exactly +0.  This trade is intentional and
// matches the DSL author's intent of avoiding the quadratic intermediate.

#include "TritonToGraph/GraphOptimizationRule.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

// The scan and the reduce must both combine with an add, and the rewrite has
// to materialize its inverse, so track which arithmetic family it belongs to.
enum class AddCombineKind { Float, Int };

std::optional<AddCombineKind> getAddCombineKind(Region &region) {
  if (region.empty())
    return std::nullopt;

  Block &body = region.front();
  if (body.getNumArguments() != 2)
    return std::nullopt;

  auto bodyOps = body.without_terminator();
  if (std::distance(bodyOps.begin(), bodyOps.end()) != 1)
    return std::nullopt;

  Operation &combine = *bodyOps.begin();
  if (combine.getNumResults() != 1 || combine.getNumOperands() != 2)
    return std::nullopt;

  Operation *terminator = body.getTerminator();
  if (terminator->getNumOperands() != 1 ||
      terminator->getOperand(0) != combine.getResult(0))
    return std::nullopt;

  // Reject a combine that does not consume both block arguments, such as
  // `a + a`, which is not an accumulation.
  Value lhs = combine.getOperand(0);
  Value rhs = combine.getOperand(1);
  Value first = body.getArgument(0);
  Value second = body.getArgument(1);
  if (!((lhs == first && rhs == second) || (lhs == second && rhs == first)))
    return std::nullopt;

  if (isa<arith::AddFOp>(combine))
    return AddCombineKind::Float;
  if (isa<arith::AddIOp>(combine))
    return AddCombineKind::Int;
  return std::nullopt;
}

// Peels producers that only reshape a uniform value.  A splat constant does
// not always reach this rule in splat form: reorder-broadcast runs after
// canonicalization in make_ttir and leaves constants behind a tt.broadcast
// that nothing folds away again.
Value peelUniformShapeOps(Value value) {
  while (Operation *definingOp = value.getDefiningOp()) {
    if (auto splat = dyn_cast<triton::SplatOp>(definingOp)) {
      value = splat.getSrc();
      continue;
    }
    if (auto broadcast = dyn_cast<triton::BroadcastOp>(definingOp)) {
      value = broadcast.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(definingOp)) {
      value = expand.getSrc();
      continue;
    }
    return value;
  }
  return value;
}

// Matches a compile-time integer held by a scalar constant or by a uniform
// constant tensor, however it was shaped.
bool matchConstantInt(Value value, int64_t &result) {
  Value uniform = peelUniformShapeOps(value);

  APInt scalar;
  if (matchPattern(uniform, m_ConstantInt(&scalar))) {
    result = scalar.getSExtValue();
    return true;
  }

  DenseElementsAttr elements;
  if (matchPattern(uniform, m_Constant(&elements)) && elements.isSplat() &&
      isa<IntegerType>(elements.getElementType())) {
    result = elements.getSplatValue<APInt>().getSExtValue();
    return true;
  }
  return false;
}

// The unselected lanes must contribute the reduce combine's identity, so that
// a row selecting nothing reduces to zero, which is exactly what the
// subtraction produces at the boundary element.
bool isAddIdentity(Value value, AddCombineKind kind) {
  if (kind == AddCombineKind::Int) {
    int64_t scalar = 0;
    return matchConstantInt(value, scalar) && scalar == 0;
  }

  Value uniform = peelUniformShapeOps(value);

  // Negative zero is rejected so the boundary element stays bit-identical to
  // the +0 produced by `x - x`.
  APFloat scalar(0.0);
  if (matchPattern(uniform, m_ConstantFloat(&scalar)))
    return scalar.isZero() && !scalar.isNegative();

  DenseElementsAttr elements;
  if (matchPattern(uniform, m_Constant(&elements)) && elements.isSplat() &&
      isa<FloatType>(elements.getElementType())) {
    APFloat splatValue = elements.getSplatValue<APFloat>();
    return splatValue.isZero() && !splatValue.isNegative();
  }
  return false;
}

// Describes a value that is elementwise equal to `coordinate(dim) + offset`,
// where `coordinate(dim)` runs along dimension `dim` of the value's own tensor
// type.  `length` is the extent of the originating tt.make_range, which is
// needed to prove that both sides of the diagonal span the same axis.
struct AxisIndex {
  int64_t dim = 0;
  int64_t offset = 0;
  int64_t length = 0;
};

// Peels expand_dims / broadcast / constant offsets off an index expression
// down to its tt.make_range.  Accepting the offset at any level, on either
// side of the add, keeps the match robust against canonicalization moving it
// across the reshapes.
std::optional<AxisIndex> matchAxisIndex(Value value) {
  // Reshaping ops, outermost first.
  SmallVector<Operation *> reshapes;
  int64_t offset = 0;
  Value current = value;
  triton::MakeRangeOp range;

  while (true) {
    Operation *definingOp = current.getDefiningOp();
    if (!definingOp)
      return std::nullopt;

    if (auto makeRange = dyn_cast<triton::MakeRangeOp>(definingOp)) {
      range = makeRange;
      break;
    }
    if (auto broadcast = dyn_cast<triton::BroadcastOp>(definingOp)) {
      // tt.broadcast only stretches unit dimensions, so dimension indices are
      // preserved and it needs no dimension bookkeeping.
      current = broadcast.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(definingOp)) {
      reshapes.push_back(definingOp);
      current = expand.getSrc();
      continue;
    }
    if (auto add = dyn_cast<arith::AddIOp>(definingOp)) {
      int64_t constant = 0;
      if (matchConstantInt(add.getRhs(), constant)) {
        offset += constant;
        current = add.getLhs();
        continue;
      }
      if (matchConstantInt(add.getLhs(), constant)) {
        offset += constant;
        current = add.getRhs();
        continue;
      }
      return std::nullopt;
    }
    if (auto sub = dyn_cast<arith::SubIOp>(definingOp)) {
      // Only `index - constant` keeps the coordinate's unit stride; the
      // mirrored `constant - index` would negate it.
      int64_t constant = 0;
      if (matchConstantInt(sub.getRhs(), constant)) {
        offset -= constant;
        current = sub.getLhs();
        continue;
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  // The range varies along dimension 0 of its own type; replay the inserted
  // dimensions from the innermost outwards to find where it ended up.
  int64_t dim = 0;
  for (Operation *reshape : llvm::reverse(reshapes)) {
    auto expand = cast<triton::ExpandDimsOp>(reshape);
    if (dim >= static_cast<int64_t>(expand.getAxis()))
      ++dim;
  }

  const int64_t start = static_cast<int64_t>(range.getStart());
  const int64_t length = static_cast<int64_t>(range.getEnd()) - start;
  if (length <= 0)
    return std::nullopt;

  // A non-zero range start shifts the diagonal just like an explicit offset
  // does, so it has to participate in the shift computation.
  return AxisIndex{dim, offset + start, length};
}

struct DiagonalCandidate {
  triton::ReduceOp reduce;
  arith::SelectOp select;
  triton::BroadcastOp broadcast;
  triton::ExpandDimsOp expand;
  triton::ScanOp scan;
  Value scanResult;
  Value scanInput;
  AddCombineKind kind = AddCombineKind::Float;
  // The rewrite reads scanResult[p + shift] at result position p.
  int64_t shift = 0;
  int64_t eliminatedElements = 0;
};

std::optional<DiagonalCandidate> analyzeDiagonal(triton::ReduceOp reduce) {
  if (reduce.getNumOperands() != 1 || reduce->getNumResults() != 1)
    return std::nullopt;

  std::optional<AddCombineKind> kind = getAddCombineKind(reduce.getCombineOp());
  if (!kind)
    return std::nullopt;

  auto select = reduce.getOperand(0).getDefiningOp<arith::SelectOp>();
  if (!select || !isAddIdentity(select.getFalseValue(), *kind))
    return std::nullopt;

  auto broadcast = select.getTrueValue().getDefiningOp<triton::BroadcastOp>();
  if (!broadcast)
    return std::nullopt;
  auto maskType = dyn_cast<RankedTensorType>(broadcast.getType());
  if (!maskType || maskType.getRank() < 2 || !maskType.hasStaticShape())
    return std::nullopt;

  auto expand = broadcast.getSrc().getDefiningOp<triton::ExpandDimsOp>();
  if (!expand)
    return std::nullopt;
  Value scanResult = expand.getSrc();
  auto scanResultType = dyn_cast<RankedTensorType>(scanResult.getType());
  if (!scanResultType || !scanResultType.hasStaticShape())
    return std::nullopt;

  // The replacement is `scan_result - scan_input`, so the reduction has to
  // produce exactly the scan result's type.  This subsumes the historical
  // square-matrix requirement and extends it to leading batch dimensions.
  if (reduce->getResult(0).getType() != scanResult.getType())
    return std::nullopt;

  const int64_t expandAxis = static_cast<int64_t>(expand.getAxis());
  const int64_t fastestDim = scanResultType.getRank() - 1;
  // The replicated dimension must be adjacent to the scan's own fastest
  // dimension.  Inserting it anywhere else would transpose the reduced result
  // against the scan result, and the subtraction would no longer be
  // equivalent even though both types still match.
  if (expandAxis != fastestDim && expandAxis != fastestDim + 1)
    return std::nullopt;

  // Where the scan result's fastest dimension ends up inside the mask.
  const int64_t reducedDim =
      expandAxis == fastestDim ? fastestDim + 1 : fastestDim;
  if (static_cast<int64_t>(reduce.getAxis()) != reducedDim)
    return std::nullopt;

  auto scan = scanResult.getDefiningOp<triton::ScanOp>();
  if (!scan || scan.getNumOperands() != 1 || scan->getNumResults() != 1)
    return std::nullopt;
  if (static_cast<int64_t>(scan.getAxis()) != fastestDim)
    return std::nullopt;
  std::optional<AddCombineKind> scanKind =
      getAddCombineKind(scan.getCombineOp());
  if (!scanKind || *scanKind != *kind)
    return std::nullopt;

  auto compare = select.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::eq)
    return std::nullopt;

  std::optional<AxisIndex> lhs = matchAxisIndex(compare.getLhs());
  std::optional<AxisIndex> rhs = matchAxisIndex(compare.getRhs());
  if (!lhs || !rhs)
    return std::nullopt;

  // One side indexes the reduced dimension; the other indexes the replicated
  // dimension and survives as the result's own coordinate.
  const AxisIndex *reduced = nullptr;
  const AxisIndex *kept = nullptr;
  if (lhs->dim == reducedDim && rhs->dim == expandAxis) {
    reduced = &*lhs;
    kept = &*rhs;
  } else if (rhs->dim == reducedDim && lhs->dim == expandAxis) {
    reduced = &*rhs;
    kept = &*lhs;
  } else {
    return std::nullopt;
  }

  // Both sides must span their whole axis, and the two axes must agree;
  // otherwise the element picked per row is not simply scanResult[p + shift].
  if (reduced->length != kept->length ||
      reduced->length != maskType.getDimSize(reduced->dim) ||
      kept->length != maskType.getDimSize(kept->dim))
    return std::nullopt;

  // The mask holds `kept + keptOffset == reduced + reducedOffset`, so the row
  // at coordinate p selects source index p + (keptOffset - reducedOffset).
  const int64_t shift = kept->offset - reduced->offset;
  if (shift != 1 && shift != -1)
    return std::nullopt;

  // shift == -1 reads scan[p - 1], which only the forward identity
  // scan[p] - x[p] == scan[p - 1] provides; shift == +1 reads scan[p + 1],
  // which is the reverse-scan identity.
  const bool needsReverseScan = shift == 1;
  if (scan.getReverse() != needsReverseScan)
    return std::nullopt;

  return DiagonalCandidate{reduce,
                           select,
                           broadcast,
                           expand,
                           scan,
                           scanResult,
                           scan.getOperand(0),
                           *kind,
                           shift,
                           maskType.getNumElements()};
}

bool matchesCandidate(const DiagonalCandidate &candidate,
                      const DiagonalCandidate &current) {
  return candidate.reduce == current.reduce &&
         candidate.select == current.select &&
         candidate.broadcast == current.broadcast &&
         candidate.expand == current.expand && candidate.scan == current.scan &&
         candidate.scanResult == current.scanResult &&
         candidate.scanInput == current.scanInput &&
         candidate.kind == current.kind && candidate.shift == current.shift;
}

// Erases the quadratic intermediate once it is unused.  The walk is restricted
// to the side-effect-free ops this pattern is built from, so it can never
// remove anything observable.
void eraseDeadPatternOp(IRRewriter &rewriter, Operation *op) {
  if (!op || !op->use_empty())
    return;
  if (!isa<arith::SelectOp, arith::CmpIOp, arith::AddIOp, arith::SubIOp,
           arith::ConstantOp, triton::BroadcastOp, triton::ExpandDimsOp,
           triton::SplatOp, triton::MakeRangeOp>(op))
    return;

  SmallVector<Operation *> producers;
  for (Value operand : op->getOperands()) {
    if (Operation *producer = operand.getDefiningOp())
      producers.push_back(producer);
  }

  rewriter.eraseOp(op);
  for (Operation *producer : producers)
    eraseDeadPatternOp(rewriter, producer);
}

class DiagonalMaskRemovalPlan final : public RewritePlan {
public:
  DiagonalMaskRemovalPlan(DiagonalCandidate candidate, unsigned epoch)
      : candidate(candidate), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::DiagonalMaskRemoval;
  }

  // Larger intermediates are worth collapsing first.
  unsigned getBenefit() const override {
    constexpr int64_t maxBenefit = std::numeric_limits<unsigned>::max();
    return static_cast<unsigned>(
        std::min<int64_t>(candidate.eliminatedElements, maxBenefit));
  }

  Operation *getAnchor() const override { return candidate.reduce; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (candidate.reduce->getParentOfType<triton::FuncOp>() !=
        context.getFunction())
      return failure();
    std::optional<DiagonalCandidate> current =
        analyzeDiagonal(candidate.reduce);
    return current && matchesCandidate(candidate, *current) ? success()
                                                            : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    // Re-prove locally so that a stale plan can never mutate the IR.
    std::optional<DiagonalCandidate> current =
        analyzeDiagonal(candidate.reduce);
    if (!current || !matchesCandidate(candidate, *current))
      return failure();

    Value reduceResult = candidate.reduce->getResult(0);
    if (reduceResult.getType() != candidate.scanResult.getType() ||
        candidate.scanInput.getType() != candidate.scanResult.getType())
      return failure();

    rewriter.setInsertionPoint(candidate.reduce);
    Location loc = candidate.reduce.getLoc();
    Value replacement =
        candidate.kind == AddCombineKind::Float
            ? rewriter
                  .create<arith::SubFOp>(loc, candidate.scanResult,
                                         candidate.scanInput)
                  .getResult()
            : rewriter
                  .create<arith::SubIOp>(loc, candidate.scanResult,
                                         candidate.scanInput)
                  .getResult();

    Operation *replacementOp = replacement.getDefiningOp();
    if (failed(mlir::verify(replacementOp))) {
      rewriter.eraseOp(replacementOp);
      return failure();
    }

    rewriter.replaceAllUsesWith(reduceResult, replacement);

    arith::SelectOp select = candidate.select;
    rewriter.eraseOp(candidate.reduce);
    eraseDeadPatternOp(rewriter, select);
    return success();
  }

private:
  DiagonalCandidate candidate;
  unsigned epoch;
};

class DiagonalMaskRemovalRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::DiagonalMaskRemoval;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    context.getFunction().walk([&](triton::ReduceOp reduce) {
      if (std::optional<DiagonalCandidate> candidate =
              analyzeDiagonal(reduce)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] matched graph optimization rule "
                   << static_cast<unsigned>(getId()) << " ("
                   << getGraphOptimizationRuleName(getId()) << ") at "
                   << reduce.getLoc() << "\n");
        plans.push_back(std::make_unique<DiagonalMaskRemovalPlan>(
            *candidate, context.getEpoch()));
      }
    });
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createDiagonalMaskRemovalRule() {
  return std::make_unique<DiagonalMaskRemovalRule>();
}
