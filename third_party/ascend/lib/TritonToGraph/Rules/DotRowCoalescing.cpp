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

#include "TritonToGraph/DotRowCoalescing.h"
#include "TritonToGraph/ProgramGridTransform.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

// Internal RowCoalescing pattern for a counted, independent row matmul loop.
// No kernel name, Python argument name, or public tuning option is inspected.
// A static x-grid selects an exact divisor. Without it, use the existing
// original-grid hidden arguments and mask the final partial group. Indirect
// scalar loads deliberately remain on the original path.

constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";
constexpr llvm::StringLiteral kCoalesceGridCeilDivAttr =
    "hacc.coalesce_grid_ceil_div";

static std::optional<int64_t> getConstantIntValue(Value v) {
  APInt value;
  if (matchPattern(v, m_ConstantInt(&value)))
    return value.getSExtValue();

  return std::nullopt;
}

struct IntegerBounds {
  int64_t lo;
  int64_t hi;
};

// Prove monotonicity together with a signed range. A syntactically increasing
// expression is not monotonic if an intermediate integer operation wraps.
static std::optional<IntegerBounds> inferMonotonicBounds(Value value, Value pid,
                                                         int64_t grid) {
  if (value == pid)
    return IntegerBounds{0, grid - 1};

  if (auto constant = getConstantIntValue(value))
    return IntegerBounds{*constant, *constant};

  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    return inferMonotonicBounds(ext.getIn(), pid, grid);

  Operation *op = value.getDefiningOp();
  if (!op || op->getNumOperands() != 2)
    return std::nullopt;

  auto lhs = inferMonotonicBounds(op->getOperand(0), pid, grid);
  auto rhs = inferMonotonicBounds(op->getOperand(1), pid, grid);
  if (!lhs || !rhs)
    return std::nullopt;

  IntegerBounds result;
  if (isa<arith::AddIOp>(op)) {
    if (__builtin_add_overflow(lhs->lo, rhs->lo, &result.lo) ||
        __builtin_add_overflow(lhs->hi, rhs->hi, &result.hi))
      return std::nullopt;
  } else if (isa<arith::MulIOp>(op)) {
    if (lhs->lo < 0 || rhs->lo < 0 ||
        __builtin_mul_overflow(lhs->lo, rhs->lo, &result.lo) ||
        __builtin_mul_overflow(lhs->hi, rhs->hi, &result.hi))
      return std::nullopt;
  } else if (isa<arith::DivSIOp>(op) && rhs->lo == rhs->hi && rhs->lo > 0) {
    result = {lhs->lo / rhs->lo, lhs->hi / rhs->lo};
  } else {
    return std::nullopt;
  }

  auto integer = dyn_cast<IntegerType>(value.getType());
  if (!integer || integer.getWidth() > 64 ||
      !APInt(64, result.lo, true).isSignedIntN(integer.getWidth()) ||
      !APInt(64, result.hi, true).isSignedIntN(integer.getWidth()))
    return std::nullopt;

  return result;
}

static bool dependsOnValue(Value v, Value pid) {
  if (v == pid)
    return true;

  Operation *op = v.getDefiningOp();
  return op && llvm::any_of(op->getOperands(),
                            [&](Value a) { return dependsOnValue(a, pid); });
}

struct RowAddressRange {
  int64_t stride = 0, lo = 0, hi = 0;
};

// Scalar offsets independent of x translate every lane by the same amount.
// Ignore that common translation, but retain the constant x coefficient and
// tensor lane span. Unknown products/casts fail closed.
static std::optional<RowAddressRange> analyzeRowAddress(Value v, Value pid) {
  if (v == pid)
    return RowAddressRange{1, 0, 0};

  if (!isa<RankedTensorType>(v.getType()) && !dependsOnValue(v, pid))
    return RowAddressRange{};

  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;

  if (auto range = dyn_cast<triton::MakeRangeOp>(op))
    return RowAddressRange{0, range.getStart(), range.getEnd() - 1};

  if (isa<arith::ExtSIOp, triton::SplatOp, triton::ExpandDimsOp,
          triton::BroadcastOp>(op))
    return analyzeRowAddress(op->getOperand(0), pid);

  if (isa<triton::AddPtrOp, arith::AddIOp>(op)) {
    auto a = analyzeRowAddress(op->getOperand(0), pid),
         b = analyzeRowAddress(op->getOperand(1), pid);
    if (!a || !b)
      return std::nullopt;

    RowAddressRange out;
    if (__builtin_add_overflow(a->stride, b->stride, &out.stride) ||
        __builtin_add_overflow(a->lo, b->lo, &out.lo) ||
        __builtin_add_overflow(a->hi, b->hi, &out.hi))
      return std::nullopt;

    return out;
  }

  if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    auto c = getConstantIntValue(mul.getRhs());
    Value other = mul.getLhs();
    if (!c) {
      c = getConstantIntValue(mul.getLhs());
      other = mul.getRhs();
    }

    if (!c || *c <= 0)
      return std::nullopt;

    auto a = analyzeRowAddress(other, pid);
    if (!a)
      return std::nullopt;

    RowAddressRange out;
    if (__builtin_mul_overflow(a->stride, *c, &out.stride) ||
        __builtin_mul_overflow(a->lo, *c, &out.lo) ||
        __builtin_mul_overflow(a->hi, *c, &out.hi))
      return std::nullopt;

    return out;
  }

  return std::nullopt;
}

static int64_t chooseRowCoalescingFactor(int64_t grid, int64_t maxDotRows,
                                         int64_t maxTensorBytes) {
  // Profitability policy, not a correctness requirement. Bound the widened M
  // dimension and the largest materialized value, including pointer tensors.
  // This is deliberately conservative, not a claim about peak live UB use or
  // an autotuned optimum. A known grid selects an exact factor; zero denotes
  // a runtime extent whose partial group is masked.
  constexpr int64_t targetDotRows = 128;
  constexpr int64_t maxLiftedValueBytes = 128 * 1024;

  for (int64_t factor : {32, 16, 8, 4, 2}) {
    if ((!grid || grid % factor == 0) && maxDotRows <= targetDotRows / factor &&
        maxTensorBytes <= maxLiftedValueBytes / factor) {
      return factor;
    }
  }

  return 1;
}

struct RowCandidate {
  triton::FuncOp function;
  Operation *anchor = nullptr;
  int64_t rowsPerProgram = 1;
  bool runtimeGrid = false;
};

static std::optional<RowCandidate> matchDotRows(triton::FuncOp fn) {
  ModuleOp module = fn->getParentOfType<ModuleOp>();
  auto grid = module->getAttrOfType<IntegerAttr>("hacc.grid_num_tiles");
  if ((grid && (grid.getInt() <= 0 || grid.getInt() > INT32_MAX)) ||
      !llvm::hasSingleElement(fn.getBody()) || !fn.isPublic() ||
      llvm::count_if(module.getOps<triton::FuncOp>(),
                     [](triton::FuncOp entry) { return entry.isPublic(); }) !=
          1 ||
      module->hasAttr(kCoalesceFactorAttr) ||
      module->hasAttr("hacc.program_grid_transforms"))
    return std::nullopt;

  LLVM_DEBUG(llvm::dbgs() << "dot-row: grid gate passed\n");

  triton::GetProgramIdOp pid;
  scf::ForOp loop;
  triton::StoreOp output;
  unsigned pids = 0, loops = 0, dots = 0, stores = 0;
  int64_t maxDotRows = 1;
  int64_t maxTensorBytes = 1;
  bool safe = true;
  DenseSet<Value> lifted;

  // Propagate row dependence while checking operations in definition order.
  fn.walk<WalkOrder::PreOrder>([&](Operation *op) {
    bool dependent = llvm::any_of(
        op->getOperands(), [&](Value value) { return lifted.contains(value); });
    if (dependent)
      lifted.insert(op->getResults().begin(), op->getResults().end());

    for (Type type : op->getResultTypes()) {
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        Type element = tensor.getElementType();
        int64_t bytes = isa<triton::PointerType>(element)
                            ? 8
                            : (element.getIntOrFloatBitWidth() + 7) / 8;
        int64_t size;
        if (__builtin_mul_overflow(tensor.getNumElements(), bytes, &size))
          safe = false;
        else
          maxTensorBytes = std::max(maxTensorBytes, size);
      }
    }

    if (auto p = dyn_cast<triton::GetProgramIdOp>(op)) {
      if (p.getAxisAsInt() == 0) {
        pid = p;
        ++pids;
        lifted.insert(p.getResult());
      }
      return;
    }

    if (auto f = dyn_cast<scf::ForOp>(op)) {
      loop = f;
      ++loops;
      // Loop states are per-row even when their initial values are uniform.
      lifted.insert(f.getRegionIterArgs().begin(), f.getRegionIterArgs().end());
      lifted.insert(f.getResults().begin(), f.getResults().end());
      if (f->getBlock() != &fn.getBody().front())
        safe = false;
      return;
    }

    if (auto d = dyn_cast<triton::DotOp>(op)) {
      ++dots;
      auto a = dyn_cast<RankedTensorType>(d.getA().getType());
      auto b = dyn_cast<RankedTensorType>(d.getB().getType());
      if (!a || !b || a.getRank() != 2 || b.getRank() != 2 ||
          !(a.getElementType().isF16() || a.getElementType().isBF16()) ||
          (dependent &&
           (!lifted.contains(d.getA()) || lifted.contains(d.getB()))))
        safe = false;
      else
        maxDotRows = std::max(maxDotRows, a.getShape()[0]);
      return;
    }

    if (auto l = dyn_cast<triton::LoadOp>(op)) {
      // A scalar metadata load is the varlen/indirect case, not this pattern.
      if (!isa<RankedTensorType>(l.getType()) || l.getIsVolatile() ||
          (dependent && op->getParentOfType<scf::ForOp>()))
        safe = false;
      return;
    }

    if (auto store = dyn_cast<triton::StoreOp>(op)) {
      output = store;
      ++stores;
      if (op->getBlock() != &fn.getBody().front())
        safe = false;
      return;
    }

    // Reduction regions are outside this pattern's supported rewrite scope.
    if (isa<triton::ReduceOp>(op)) {
      safe = false;
      return;
    }

    if (isa<triton::FuncOp, triton::ReturnOp, scf::YieldOp, triton::MakeRangeOp,
            triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
            triton::AddPtrOp, triton::TransOp, triton::ReshapeOp>(op))
      return;
    StringRef ns = op->getName().getDialectNamespace();
    // A partial group evaluates arithmetic for inactive rows as well.
    if (!grid && dependent && (ns == "arith" || ns == "math") &&
        !isSpeculatable(op))
      safe = false;
    if ((ns != "arith" && ns != "math") || op->getNumRegions())
      safe = false;
  });

  LLVM_DEBUG(llvm::dbgs() << "dot-row: safe=" << safe << " pids=" << pids
                          << " loops=" << loops << " dots=" << dots
                          << " stores=" << stores << "\n");
  if (!safe || pids != 1 || loops != 1 || dots == 0 || stores != 1 ||
      loop.getNumResults() == 0 ||
      getConstantIntValue(loop.getLowerBound()) != std::optional<int64_t>(0) ||
      (dependsOnValue(loop.getUpperBound(), pid.getResult()) &&
       !inferMonotonicBounds(loop.getUpperBound(), pid.getResult(),
                             grid ? grid.getInt() : INT32_MAX)))
    return std::nullopt;

  // Merging unequal trip counts speculates pure arithmetic for inactive rows.
  // Preserve the old accumulator, but do not speculate potentially trapping
  // operations such as integer division with an unproven nonzero divisor.
  if (dependsOnValue(loop.getUpperBound(), pid.getResult())) {
    for (Operation &op : loop.getBody()->without_terminator()) {
      StringRef ns = op.getName().getDialectNamespace();
      if ((ns == "arith" || ns == "math") && !isSpeculatable(&op))
        return std::nullopt;
    }
  }

  for (Value init : loop.getInitArgs())
    if (!isa<RankedTensorType>(init.getType()))
      return std::nullopt;

  auto address = analyzeRowAddress(output.getPtr(), pid.getResult());
  if (!address || address->stride <= 0 || address->hi < address->lo ||
      static_cast<uint64_t>(address->hi) - static_cast<uint64_t>(address->lo) >=
          static_cast<uint64_t>(address->stride))
    return std::nullopt;

  auto step = getConstantIntValue(loop.getStep());
  if (!step || *step <= 0)
    return std::nullopt;

  int64_t rows = chooseRowCoalescingFactor(grid ? grid.getInt() : 0, maxDotRows,
                                           maxTensorBytes);
  if (rows <= 1)
    return std::nullopt;

  return RowCandidate{fn, pid.getOperation(), rows, !grid};
}

// Lift only query-dependent values. Invariant K/V loads stay two-dimensional;
// a lifted lhs [BT,G,K] is reshaped to [BT*G,K] at each dot, not batched dots.
class DotRowBuilder {
  IRRewriter &rewriter;
  RowCandidate candidate;

  // Original SSA values mapped to the rebuilt computation.
  IRMapping valueMapping;
  // Scalar values evaluated at the last row to form the shared loop bound.
  IRMapping lastRowMapping;
  Value validRows;

public:
  DotRowBuilder(IRRewriter &rewriter, RowCandidate candidate)
      : rewriter(rewriter), candidate(candidate) {}

  bool run(triton::FuncOp fn) {
    Block *old = &fn.getBody().front();
    Value upperBound = (*old->getOps<scf::ForOp>().begin()).getUpperBound();
    auto *fresh = new Block();
    fn.getBody().push_back(fresh);

    for (BlockArgument arg : old->getArguments()) {
      Value newArg = fresh->addArgument(arg.getType(), arg.getLoc());
      valueMapping.map(arg, newArg);
      lastRowMapping.map(arg, newArg);
    }

    rewriter.setInsertionPointToStart(fresh);
    for (Operation &op : old->without_terminator()) {
      if (&op == candidate.anchor) {
        auto pid = cast<triton::GetProgramIdOp>(rewriter.clone(op));
        Value base = rewriter.create<arith::MulIOp>(
            op.getLoc(), pid.getResult(),
            rewriter.create<arith::ConstantIntOp>(
                op.getLoc(), candidate.rowsPerProgram, 32));
        Value last = rewriter.create<arith::AddIOp>(
            op.getLoc(), base,
            rewriter.create<arith::ConstantIntOp>(
                op.getLoc(), candidate.rowsPerProgram - 1, 32));
        if (candidate.runtimeGrid) {
          Value extent = fresh->getArgument(fresh->getNumArguments() - 2);
          Value lastValid = rewriter.create<arith::SubIOp>(
              op.getLoc(), extent,
              rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, 32));
          last = rewriter.create<arith::MinSIOp>(op.getLoc(), last, lastValid);
        }
        lastRowMapping.map(op.getResult(0), last);

        auto t = getLiftedType(op.getResult(0).getType());
        Value lanes = rewriter.create<triton::MakeRangeOp>(
            op.getLoc(), t, 0, candidate.rowsPerProgram);
        Value start = rewriter.create<triton::SplatOp>(op.getLoc(), t, base);
        valueMapping.map(
            op.getResult(0),
            rewriter.create<arith::AddIOp>(op.getLoc(), start, lanes)
                .getResult());
        if (candidate.runtimeGrid) {
          Value extent = fresh->getArgument(fresh->getNumArguments() - 2);
          Value limit =
              rewriter.create<triton::SplatOp>(op.getLoc(), t, extent);
          validRows = rewriter.create<arith::CmpIOp>(
              op.getLoc(), arith::CmpIPredicate::slt,
              valueMapping.lookup(op.getResult(0)), limit);
        }
        continue;
      }

      // Only clone the pure scalar bound slice for the last-row evaluation.
      if (op.getNumResults() == 1 &&
          !isa<RankedTensorType>(op.getResult(0).getType()) &&
          op.getName().getDialectNamespace() == "arith" &&
          dependsOnValue(upperBound, op.getResult(0)))
        rewriter.clone(op, lastRowMapping);

      if (!rewrite(&op))
        return false;

      for (Value v : op.getResults())
        if (!hasRowDimension(v) && !lastRowMapping.contains(v))
          lastRowMapping.map(v, getMappedValue(v));
    }

    rewriter.create<triton::ReturnOp>(fn.getLoc());
    old->dropAllDefinedValueUses();
    old->erase();
    return true;
  }

private:
  bool rewrite(Operation *op) {
    if (auto loop = dyn_cast<scf::ForOp>(op))
      return rewriteLoop(loop);

    bool dependent = llvm::any_of(op->getOperands(),
                                  [&](Value v) { return hasRowDimension(v); });
    if (!dependent) {
      rewriter.clone(*op, valueMapping);
      return true;
    }

    if (isa<triton::LoadOp>(op) && op->getParentOfType<scf::ForOp>())
      return false; // No speculative query-dependent memory accesses.
    if (auto dot = dyn_cast<triton::DotOp>(op)) {
      if (!hasRowDimension(dot.getA()) || hasRowDimension(dot.getB()))
        return false;

      auto result = cast<RankedTensorType>(dot.getType());
      SmallVector<Value> operands{
          flattenDotRows(ensureRowDimension(dot.getA())),
          getMappedValue(dot.getB()),
          flattenDotRows(ensureRowDimension(dot.getC()))};
      Type flatType = RankedTensorType::get(
          {candidate.rowsPerProgram * result.getShape()[0],
           result.getShape()[1]},
          result.getElementType());
      Operation *d =
          rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                          TypeRange{flatType}, op->getAttrs());
      valueMapping.map(
          dot.getResult(),
          reshape(d->getResult(0), getLiftedType(result).getShape()));
      return true;
    }

    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      valueMapping.map(splat.getResult(),
                       broadcastRowValue(ensureRowDimension(splat.getSrc()),
                                         getLiftedType(splat.getType())));
      return true;
    }

    if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
      valueMapping.map(
          expand.getResult(),
          rewriter
              .create<triton::ExpandDimsOp>(op->getLoc(),
                                            ensureRowDimension(expand.getSrc()),
                                            expand.getAxis() + 1)
              .getResult());
      return true;
    }

    if (auto view = dyn_cast<triton::ReshapeOp>(op)) {
      // Preserve each row's element order. Allowing arbitrary reordering of
      // the lifted tensor could mix values belonging to different
      // programs.
      valueMapping.map(view.getResult(),
                       reshape(ensureRowDimension(view.getSrc()),
                               getLiftedType(view.getType()).getShape()));
      return true;
    }

    if (auto trans = dyn_cast<triton::TransOp>(op)) {
      SmallVector<int32_t> order{0};
      for (int32_t axis : trans.getOrder())
        order.push_back(axis + 1);
      valueMapping.map(
          trans.getResult(),
          rewriter
              .create<triton::TransOp>(
                  op->getLoc(), ensureRowDimension(trans.getSrc()), order)
              .getResult());
      return true;
    }

    if (candidate.runtimeGrid) {
      if (auto load = dyn_cast<triton::LoadOp>(op)) {
        Value pointer = ensureRowDimension(load.getPtr());
        Value mask = getMemoryMask(pointer, load.getMask());
        auto type = getLiftedType(load.getType());
        Value other = load.getOther()
                          ? ensureRowDimension(load.getOther())
                          : rewriter.create<arith::ConstantOp>(
                                op->getLoc(), rewriter.getZeroAttr(type));
        auto replacement = rewriter.create<triton::LoadOp>(
            op->getLoc(), pointer, mask, other, load.getBoundaryCheck(),
            load.getPadding(), load.getCache(), load.getEvict(),
            load.getIsVolatile());
        valueMapping.map(load.getResult(), replacement.getResult());
        return true;
      }

      if (auto store = dyn_cast<triton::StoreOp>(op)) {
        Value pointer = ensureRowDimension(store.getPtr());
        rewriter.create<triton::StoreOp>(
            op->getLoc(), pointer, ensureRowDimension(store.getValue()),
            getMemoryMask(pointer, store.getMask()), store.getBoundaryCheck(),
            store.getCache(), store.getEvict());
        return true;
      }
    }

    SmallVector<Value> operands;
    for (Value v : op->getOperands())
      operands.push_back(ensureRowDimension(v));

    SmallVector<Type> types;
    for (Type t : op->getResultTypes())
      types.push_back(getLiftedType(t));

    Operation *cloned =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                        types, op->getAttrs());
    valueMapping.map(op->getResults(), cloned->getResults());
    return true;
  }

  Value getMemoryMask(Value pointer, Value originalMask) {
    auto type = cast<RankedTensorType>(pointer.getType());
    Value mask = broadcastRowValue(
        validRows,
        RankedTensorType::get(type.getShape(), rewriter.getI1Type()));
    if (originalMask)
      mask = rewriter.create<arith::AndIOp>(pointer.getLoc(), mask,
                                            ensureRowDimension(originalMask));
    return mask;
  }

  bool rewriteLoop(scf::ForOp old) {
    SmallVector<Value> init;
    for (Value arg : old.getInitArgs())
      init.push_back(ensureRowDimension(arg));

    // The monotonic upper-bound proof means the last row supplies the maximum.
    auto loop = rewriter.create<scf::ForOp>(
        old.getLoc(), getMappedValue(old.getLowerBound()),
        lastRowMapping.lookup(old.getUpperBound()),
        getMappedValue(old.getStep()), init);
    loop->setAttrs(old->getAttrs());

    valueMapping.map(old.getInductionVar(), loop.getInductionVar());
    valueMapping.map(old.getRegionIterArgs(), loop.getRegionIterArgs());

    rewriter.setInsertionPointToStart(loop.getBody());
    for (Operation &op : old.getBody()->without_terminator())
      if (!rewrite(&op))
        return false;

    auto yield = cast<scf::YieldOp>(old.getBody()->getTerminator());
    Value iv = rewriter.create<triton::SplatOp>(
        old.getLoc(), getLiftedType(old.getInductionVar().getType()),
        loop.getInductionVar());
    Value active = rewriter.create<arith::CmpIOp>(
        old.getLoc(), arith::CmpIPredicate::slt, iv,
        ensureRowDimension(old.getUpperBound()));

    SmallVector<Value> output;
    for (auto [v, prev] :
         llvm::zip(yield.getOperands(), loop.getRegionIterArgs())) {
      auto t = cast<RankedTensorType>(prev.getType());
      Value mask = broadcastRowValue(
          active, RankedTensorType::get(t.getShape(), rewriter.getI1Type()));
      // Retain exactly the previous accumulator for rows whose original loop
      // did not execute this iteration (also preserves NaN/Inf behavior).
      output.push_back(rewriter.create<arith::SelectOp>(
          old.getLoc(), mask, ensureRowDimension(v), prev));
    }

    if (!loop.getBody()->empty() && isa<scf::YieldOp>(loop.getBody()->back()))
      rewriter.eraseOp(loop.getBody()->getTerminator());
    rewriter.setInsertionPointToEnd(loop.getBody());
    rewriter.create<scf::YieldOp>(old.getLoc(), output);

    valueMapping.map(old.getResults(), loop.getResults());

    rewriter.setInsertionPointAfter(loop);
    return true;
  }

  RankedTensorType getLiftedType(Type old) {
    SmallVector<int64_t> shape{candidate.rowsPerProgram};
    Type elem = old;
    if (auto tensor = dyn_cast<RankedTensorType>(old)) {
      shape.append(tensor.getShape().begin(), tensor.getShape().end());
      elem = tensor.getElementType();
    }
    return RankedTensorType::get(shape, elem);
  }

  Value getMappedValue(Value old) { return valueMapping.lookup(old); }

  // Every lifted value gains a leading row dimension, so its type already
  // records this state; no separate boolean needs to be maintained.
  bool hasRowDimension(Value old) {
    return getMappedValue(old).getType() != old.getType();
  }

  Value ensureRowDimension(Value old) {
    Value value = getMappedValue(old);
    if (hasRowDimension(old))
      return value;

    if (!isa<RankedTensorType>(old.getType()))
      return rewriter.create<triton::SplatOp>(
          old.getLoc(), getLiftedType(old.getType()), value);

    Value expanded =
        rewriter.create<triton::ExpandDimsOp>(old.getLoc(), value, 0);
    return rewriter.create<triton::BroadcastOp>(
        old.getLoc(), getLiftedType(old.getType()), expanded);
  }

  Value reshape(Value v, ArrayRef<int64_t> shape) {
    auto t = cast<RankedTensorType>(v.getType());
    return rewriter.create<triton::ReshapeOp>(
        v.getLoc(), RankedTensorType::get(shape, t.getElementType()), v, false);
  }

  Value flattenDotRows(Value v) {
    auto t = cast<RankedTensorType>(v.getType());
    return reshape(v, {t.getShape()[0] * t.getShape()[1], t.getShape()[2]});
  }

  Value broadcastRowValue(Value v, RankedTensorType target) {
    while (cast<RankedTensorType>(v.getType()).getRank() < target.getRank())
      v = rewriter.create<triton::ExpandDimsOp>(
          v.getLoc(), v, cast<RankedTensorType>(v.getType()).getRank());
    return rewriter.create<triton::BroadcastOp>(v.getLoc(), target, v);
  }
};

class DotRowPlan final : public RewritePlan {
  RowCandidate candidate;
  unsigned epoch;
  OwningOpRef<ModuleOp> prepared;

  DotRowPlan(RowCandidate candidate, unsigned epoch,
             OwningOpRef<ModuleOp> prepared)
      : candidate(candidate), epoch(epoch), prepared(std::move(prepared)) {}

public:
  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::RowCoalescing;
  }

  unsigned getBenefit() const override { return 1; }

  Operation *getAnchor() const override { return candidate.function; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();

    auto current = matchDotRows(candidate.function);
    return success(current && current->anchor == candidate.anchor &&
                   current->rowsPerProgram == candidate.rowsPerProgram &&
                   current->runtimeGrid == candidate.runtimeGrid);
  }

  static std::unique_ptr<DotRowPlan> create(RowCandidate candidate,
                                            unsigned epoch) {
    // Prepare and verify privately before offering a plan to the scheduler.
    // An unsupported rewrite must leave the input and its diagnostics
    // untouched.
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    OwningOpRef<ModuleOp> sandbox(
        ModuleOp::create(candidate.function.getLoc()));
    sandbox->getOperation()->setAttrs(module->getAttrs());

    IRMapping mapping;
    sandbox->getBody()->push_back(candidate.function->clone(mapping));

    RowCandidate copy = candidate;
    copy.function = cast<triton::FuncOp>(&sandbox->getBody()->front());
    copy.anchor =
        mapping.lookup(candidate.anchor->getResult(0)).getDefiningOp();

    IRRewriter rw(module.getContext());
    ScopedDiagnosticHandler diagnostics(
        module.getContext(), [](Diagnostic &diag) {
          LLVM_DEBUG(llvm::dbgs()
                     << "dot-row: skipped candidate: " << diag << "\n");
          return success();
        });
    if (copy.runtimeGrid &&
        failed(addProgramGridHiddenExtentArguments(copy.function)))
      return nullptr;

    DotRowBuilder builder(rw, copy);
    if (!builder.run(copy.function) || failed(verify(*sandbox)))
      return nullptr;

    if (copy.runtimeGrid) {
      ProgramGridTransformContract contract;
      contract.transforms.push_back({0, 0, copy.rowsPerProgram, false, false});
      if (failed(setProgramGridTransformContract(*sandbox, contract)))
        return nullptr;
    }

    return std::unique_ptr<DotRowPlan>(
        new DotRowPlan(candidate, epoch, std::move(sandbox)));
  }

  LogicalResult apply(IRRewriter &rw) override {
    // Commit the verified body and, for runtime grids, its hidden arguments.
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    auto replacement = cast<triton::FuncOp>(&prepared->getBody()->front());
    if (candidate.runtimeGrid) {
      if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                      replacement)))
        return failure();
      module->setAttr(
          kProgramGridTransformsAttr,
          prepared->getOperation()->getAttr(kProgramGridTransformsAttr));
      return success();
    }

    candidate.function.getBody().takeBody(replacement.getBody());
    module->setAttr(kCoalesceFactorAttr,
                    rw.getI32IntegerAttr(candidate.rowsPerProgram));
    module->setAttr(kCoalesceAxisAttr, rw.getI32IntegerAttr(0));
    module->setAttr(kCoalesceGridCeilDivAttr, rw.getI32IntegerAttr(1));
    return success();
  }
};

} // namespace

std::unique_ptr<RewritePlan>
cfg::createDotRowCoalescingPlan(triton::FuncOp function, unsigned epoch) {
  // Dot row coalescing is supported only on Ascend 950.
  auto module = function->getParentOfType<ModuleOp>();
  if (!module || !hacc::utils::isAscend950(module))
    return nullptr;

  auto candidate = matchDotRows(function);
  if (!candidate)
    return nullptr;

  return DotRowPlan::create(*candidate, epoch);
}
