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

#include "TritonToGraph/LegacyMemoryAccess/RowCoalescing.h"

#include "TritonToGraph/GraphOptimization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <functional>
#include <optional>

#define DEBUG_TYPE "graph-optimize"

namespace RowCoalescing {

using namespace mlir;
using namespace triton;

namespace {

constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";
constexpr llvm::StringLiteral kCoalesceGridCeilDivAttr =
    "hacc.coalesce_grid_ceil_div";

// Default to the original FBGEMM dense-token point for very small row tiles.
constexpr int64_t kDefaultRowsPerProgram = 8;
constexpr int64_t kMaxBaseElementsPerLift = 1024;

struct RowSeed {
  triton::GetProgramIdOp pid;
  int32_t axis = 0;
  Value validCount;
  arith::CmpIOp entryGuard;
  Block *workBlock = nullptr;
};

static int64_t inferRowsPerProgram(int64_t maxBaseElements) {
  if (maxBaseElements > kMaxBaseElementsPerLift)
    return 1;
  if (maxBaseElements >= 1024)
    return 2;
  if (maxBaseElements > 16)
    return 4;
  return kDefaultRowsPerProgram;
}

static bool readsAxisNumPrograms(ModuleOp moduleOp, int32_t axis) {
  bool reads = false;
  moduleOp.walk([&](triton::GetNumProgramsOp np) {
    if (np.getAxisAsInt() == axis)
      reads = true;
  });
  return reads;
}

static bool isScalarIntegerLike(Value value) {
  Type type = value.getType();
  if (isa<IndexType>(type))
    return true;
  auto intTy = dyn_cast<IntegerType>(type);
  return intTy && intTy.getWidth() > 1;
}

static bool isAutomaticOverflowAssert(triton::AssertOp assertOp) {
  if (!assertOp || !assertOp->hasAttr("tt.auto_overflow_assert"))
    return false;
  auto message = dyn_cast<StringAttr>(assertOp.getMessageAttr());
  return message &&
         message.getValue().contains("overflow detected for operation");
}

// Match the canonical rowwise guard:
//
//   %pid = tt.get_program_id x
//   %n   = tt.load %valid_count_ptr
//   %p   = arith.cmpi sge, %pid, %n
//   cf.cond_br %p, ^return_block, ^work_block
//
// This is intentionally a seed matcher only. It proves that pid is a row id
// guarded by a runtime row count and identifies the work block that should be
// lifted. The actual IR rewrite is kept separate.
static std::optional<RowSeed> matchRowSeed(ModuleOp moduleOp) {
  if (moduleOp->hasAttr(kCoalesceFactorAttr))
    return std::nullopt;

  SmallVector<triton::GetProgramIdOp> pids;
  moduleOp.walk([&](triton::GetProgramIdOp pid) { pids.push_back(pid); });
  if (pids.size() != 1)
    return std::nullopt;

  triton::GetProgramIdOp pid = pids.front();
  int32_t axis = pid.getAxisAsInt();
  if (readsAxisNumPrograms(moduleOp, axis))
    return std::nullopt;

  for (Operation *user : pid.getResult().getUsers()) {
    auto cmp = dyn_cast<arith::CmpIOp>(user);
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::sge ||
        cmp.getLhs() != pid.getResult())
      continue;

    Block *cmpBlock = cmp->getBlock();
    if (!cmpBlock || !cmpBlock->mightHaveTerminator())
      continue;
    auto cond = dyn_cast_or_null<cf::CondBranchOp>(cmpBlock->getTerminator());
    if (!cond || cond.getCondition() != cmp.getResult())
      continue;

    Block *trueBlock = cond.getTrueDest();
    Block *falseBlock = cond.getFalseDest();
    if (!trueBlock || !falseBlock)
      continue;
    if (!trueBlock->mightHaveTerminator())
      continue;
    if (!isa<triton::ReturnOp>(trueBlock->getTerminator()))
      continue;
    if (!isScalarIntegerLike(cmp.getRhs()))
      continue;

    return RowSeed{pid, axis, cmp.getRhs(), cmp, falseBlock};
  }

  return std::nullopt;
}

static bool isRowLiftable(Operation *op) {
  if (isa<triton::ReturnOp, cf::BranchOp, cf::CondBranchOp>(op))
    return false;
  if (auto assertOp = dyn_cast<triton::AssertOp>(op))
    return isAutomaticOverflowAssert(assertOp);
  if (auto *dialect = op->getDialect()) {
    StringRef ns = dialect->getNamespace();
    if (ns == arith::ArithDialect::getDialectNamespace() ||
        ns == math::MathDialect::getDialectNamespace())
      return true;
  }
  return isa<triton::SplatOp, triton::AddPtrOp, triton::BroadcastOp,
             triton::ExpandDimsOp, triton::LoadOp, triton::StoreOp,
             triton::MakeRangeOp, triton::ScanOp, triton::ReduceOp,
             triton::ClampFOp, triton::FpToFpOp, scf::ForOp>(op);
}

static bool collectRowRegion(const RowSeed &seed,
                             SmallVectorImpl<Operation *> &ordered) {
  if (!seed.workBlock)
    return false;

  bool hasStore = false;
  for (Operation &op : seed.workBlock->without_terminator()) {
    bool safe = true;
    op.walk([&](Operation *nested) {
      if (nested == &op)
        return;
      if (isa<scf::YieldOp, triton::ReduceReturnOp, triton::ScanReturnOp>(
              nested))
        return;
      if (!isRowLiftable(nested))
        safe = false;
    });
    if (!safe || !isRowLiftable(&op))
      return false;
    if (isa<triton::StoreOp>(op))
      hasStore = true;
    ordered.push_back(&op);
  }

  return hasStore;
}

static int64_t getMaxStaticTensorElements(ArrayRef<Operation *> ordered) {
  int64_t maxElements = 1;
  auto update = [&](Type type) {
    auto rt = dyn_cast<RankedTensorType>(type);
    if (!rt)
      return;
    if (!rt.hasStaticShape()) {
      maxElements = kMaxBaseElementsPerLift + 1;
      return;
    }
    maxElements = std::max<int64_t>(maxElements, rt.getNumElements());
  };

  for (Operation *op : ordered) {
    for (Value operand : op->getOperands())
      update(operand.getType());
    for (Value result : op->getResults())
      update(result.getType());
  }
  return maxElements;
}

static bool rewriteMatchedRow(ModuleOp moduleOp, const RowSeed &seed,
                              ArrayRef<Operation *> ordered, IRRewriter &rw,
                              int64_t H) {
  if (H <= 1)
    return false;
  triton::GetProgramIdOp pid = seed.pid;
  Value pidVal = pid.getResult();
  Location loc = pid.getLoc();
  if (!seed.entryGuard || !seed.workBlock)
    return false;

  auto liftTy = [&](Type t) -> RankedTensorType {
    if (auto rt = dyn_cast<RankedTensorType>(t)) {
      SmallVector<int64_t> shape;
      shape.push_back(H);
      shape.append(rt.getShape().begin(), rt.getShape().end());
      return RankedTensorType::get(shape, rt.getElementType());
    }
    return RankedTensorType::get({H}, t);
  };

  auto makeZero = [&](Location zloc, Type ty) -> Value {
    OpBuilder::InsertionGuard guard(rw);
    auto rt = dyn_cast<RankedTensorType>(ty);
    Type elemTy = rt ? rt.getElementType() : ty;
    if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
      TypedAttr zero = IntegerAttr::get(intTy, 0);
      if (rt)
        return rw.create<arith::ConstantOp>(zloc, ty,
                                            DenseElementsAttr::get(rt, zero));
      return rw.create<arith::ConstantOp>(zloc, ty, zero);
    }
    if (auto fpTy = dyn_cast<FloatType>(elemTy)) {
      TypedAttr zero = FloatAttr::get(fpTy, 0.0);
      if (rt)
        return rw.create<arith::ConstantOp>(zloc, ty,
                                            DenseElementsAttr::get(rt, zero));
      return rw.create<arith::ConstantOp>(zloc, ty, zero);
    }
    return Value();
  };

  rw.setInsertionPoint(seed.entryGuard);
  Value cH = rw.create<arith::ConstantIntOp>(loc, H, 32);
  Value pidH = rw.create<arith::MulIOp>(loc, pidVal, cH);
  auto hI32Ty = RankedTensorType::get({H}, rw.getI32Type());
  Value lane = rw.create<triton::MakeRangeOp>(loc, hI32Ty, 0, H);
  Value pidHSplat = rw.create<triton::SplatOp>(loc, hI32Ty, pidH);
  Value rows = rw.create<arith::AddIOp>(loc, pidHSplat, lane);
  Value validSplat = rw.create<triton::SplatOp>(loc, hI32Ty, seed.validCount);
  Value rowMask = rw.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, rows,
                                           validSplat);

  DenseMap<Value, Value> vmap;
  vmap[pidVal] = rows;
  vmap[seed.validCount] = validSplat;

  auto maskForType = [&](Location mloc, Type type) -> Value {
    auto rt = dyn_cast<RankedTensorType>(type);
    if (!rt)
      return Value();
    if (rt.getRank() == 1)
      return rowMask;
    Value cur = rowMask;
    for (int64_t rank = 1; rank < rt.getRank(); ++rank) {
      cur = rw.create<triton::ExpandDimsOp>(
          mloc, cur, cast<RankedTensorType>(cur.getType()).getRank());
    }
    auto maskTy = RankedTensorType::get(rt.getShape(), rw.getI1Type());
    return rw.create<triton::BroadcastOp>(mloc, maskTy, cur);
  };

  std::function<Value(Value, DenseMap<Value, Value> *)> lift =
      [&](Value v, DenseMap<Value, Value> *localMap) -> Value {
    if (localMap) {
      auto lit = localMap->find(v);
      if (lit != localMap->end())
        return lit->second;
    }
    auto it = vmap.find(v);
    if (it != vmap.end())
      return it->second;
    if (!isa<RankedTensorType>(v.getType()))
      return v;
    Value expanded = rw.create<triton::ExpandDimsOp>(v.getLoc(), v, 0);
    Value broadcast = rw.create<triton::BroadcastOp>(
        v.getLoc(), liftTy(v.getType()), expanded);
    if (localMap)
      (*localMap)[v] = broadcast;
    else
      vmap[v] = broadcast;
    return broadcast;
  };

  auto liftOperand = [&](Value v, DenseMap<Value, Value> *localMap) -> Value {
    Value lv = lift(v, localMap);
    if (!lv)
      return Value();
    if (!isa<RankedTensorType>(lv.getType()))
      return rw.create<triton::SplatOp>(lv.getLoc(), liftTy(v.getType()), lv);
    return lv;
  };

  auto copyAttrs = [&](Operation *from, Operation *to) {
    for (NamedAttribute attr : from->getAttrs())
      if (!to->hasAttr(attr.getName()))
        to->setAttr(attr.getName(), attr.getValue());
  };

  auto liftConstant = [&](arith::ConstantOp cst, Location opLoc,
                          DenseMap<Value, Value> *localMap) -> bool {
    Operation *nu = rw.clone(*cst.getOperation());
    Value result = nu->getResult(0);
    if (isa<RankedTensorType>(cst.getType())) {
      Value expanded = rw.create<triton::ExpandDimsOp>(opLoc, result, 0);
      result = rw.create<triton::BroadcastOp>(opLoc, liftTy(cst.getType()),
                                              expanded);
    }
    (*localMap)[cst.getResult()] = result;
    return true;
  };

  std::function<bool(Operation *, DenseMap<Value, Value> *, bool)> rebuildOp;
  rebuildOp = [&](Operation *op, DenseMap<Value, Value> *localMap,
                  bool resetInsertionPoint) -> bool {
    if (resetInsertionPoint)
      rw.setInsertionPoint(op);
    Location opLoc = op->getLoc();

    if (auto cst = dyn_cast<arith::ConstantOp>(op))
      return liftConstant(cst, opLoc, localMap);

    if (auto range = dyn_cast<triton::MakeRangeOp>(op)) {
      Value oneD = rw.create<triton::MakeRangeOp>(
          opLoc, range.getType(), range.getStart(), range.getEnd());
      Value expanded = rw.create<triton::ExpandDimsOp>(opLoc, oneD, 0);
      (*localMap)[range.getResult()] = rw.create<triton::BroadcastOp>(
          opLoc, liftTy(range.getType()), expanded);
      return true;
    }

    if (auto sp = dyn_cast<triton::SplatOp>(op)) {
      Value src = lift(sp.getSrc(), localMap);
      if (!src)
        return false;
      Value result;
      if (!isa<RankedTensorType>(src.getType())) {
        result = rw.create<triton::SplatOp>(opLoc, liftTy(sp.getType()), src);
      } else {
        Value cur = src;
        int64_t addDims = cast<RankedTensorType>(sp.getType()).getRank();
        for (int64_t i = 0; i < addDims; ++i) {
          cur = rw.create<triton::ExpandDimsOp>(
              opLoc, cur, cast<RankedTensorType>(cur.getType()).getRank());
        }
        result =
            rw.create<triton::BroadcastOp>(opLoc, liftTy(sp.getType()), cur);
      }
      (*localMap)[sp.getResult()] = result;
      return true;
    }

    if (auto ed = dyn_cast<triton::ExpandDimsOp>(op)) {
      Value src = liftOperand(ed.getSrc(), localMap);
      if (!src)
        return false;
      (*localMap)[ed.getResult()] =
          rw.create<triton::ExpandDimsOp>(opLoc, src, ed.getAxis() + 1);
      return true;
    }

    if (auto bc = dyn_cast<triton::BroadcastOp>(op)) {
      Value src = liftOperand(bc.getSrc(), localMap);
      if (!src)
        return false;
      (*localMap)[bc.getResult()] =
          rw.create<triton::BroadcastOp>(opLoc, liftTy(bc.getType()), src);
      return true;
    }

    if (auto red = dyn_cast<triton::ReduceOp>(op)) {
      SmallVector<Value> srcs;
      for (Value src : red.getSrcs()) {
        Value lifted = liftOperand(src, localMap);
        if (!lifted)
          return false;
        srcs.push_back(lifted);
      }
      auto nu = rw.create<triton::ReduceOp>(opLoc, srcs, red.getAxis() + 1);
      rw.cloneRegionBefore(red.getCombineOp(), nu.getCombineOp(),
                           nu.getCombineOp().end());
      copyAttrs(red, nu);
      for (auto [oldR, newR] : llvm::zip(red.getResults(), nu.getResults()))
        (*localMap)[oldR] = newR;
      return true;
    }

    if (auto scan = dyn_cast<triton::ScanOp>(op)) {
      SmallVector<Value> srcs;
      for (Value src : scan.getSrcs()) {
        Value lifted = liftOperand(src, localMap);
        if (!lifted)
          return false;
        srcs.push_back(lifted);
      }
      auto nu = rw.create<triton::ScanOp>(opLoc, srcs, scan.getAxis() + 1,
                                          scan.getReverse());
      rw.cloneRegionBefore(scan.getCombineOp(), nu.getCombineOp(),
                           nu.getCombineOp().end());
      copyAttrs(scan, nu);
      for (auto [oldR, newR] : llvm::zip(scan.getResults(), nu.getResults()))
        (*localMap)[oldR] = newR;
      return true;
    }

    if (auto ld = dyn_cast<triton::LoadOp>(op)) {
      Value ptr = liftOperand(ld.getPtr(), localMap);
      if (!ptr)
        return false;
      Value mask = ld.getMask() ? liftOperand(ld.getMask(), localMap) : Value();
      Value rowMaskForLoad =
          maskForType(opLoc, cast<ShapedType>(ptr.getType()));
      if (rowMaskForLoad) {
        mask = mask ? rw.create<arith::AndIOp>(opLoc, mask, rowMaskForLoad)
                    : rowMaskForLoad;
      }
      Value other = ld.getOther()
                        ? liftOperand(ld.getOther(), localMap)
                        : makeZero(opLoc, liftTy(ld.getResult().getType()));
      auto nu = rw.create<triton::LoadOp>(
          opLoc, ptr, mask, other, ld.getBoundaryCheck(), ld.getPadding(),
          ld.getCache(), ld.getEvict(), ld.getIsVolatile());
      copyAttrs(ld, nu);
      (*localMap)[ld.getResult()] = nu.getResult();
      return true;
    }

    if (auto st = dyn_cast<triton::StoreOp>(op)) {
      Value ptr = liftOperand(st.getPtr(), localMap);
      Value val = liftOperand(st.getValue(), localMap);
      if (!ptr || !val)
        return false;
      Value mask = st.getMask() ? liftOperand(st.getMask(), localMap) : Value();
      Value rowMaskForStore =
          maskForType(opLoc, cast<ShapedType>(ptr.getType()));
      if (rowMaskForStore) {
        mask = mask ? rw.create<arith::AndIOp>(opLoc, mask, rowMaskForStore)
                    : rowMaskForStore;
      }
      rw.create<triton::StoreOp>(opLoc, ptr, val, mask, st.getBoundaryCheck(),
                                 st.getCache(), st.getEvict());
      return true;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      OpBuilder::InsertionGuard guard(rw);
      Value lowerBound = lift(forOp.getLowerBound(), localMap);
      Value upperBound = lift(forOp.getUpperBound(), localMap);
      Value step = lift(forOp.getStep(), localMap);
      if (!lowerBound || !upperBound || !step ||
          isa<RankedTensorType>(lowerBound.getType()) ||
          isa<RankedTensorType>(upperBound.getType()) ||
          isa<RankedTensorType>(step.getType()))
        return false;
      SmallVector<Value> initArgs;
      for (Value init : forOp.getInitArgs()) {
        Value lifted = liftOperand(init, localMap);
        if (!lifted)
          return false;
        initArgs.push_back(lifted);
      }
      auto newFor =
          rw.create<scf::ForOp>(opLoc, lowerBound, upperBound, step, initArgs);
      copyAttrs(forOp, newFor);

      DenseMap<Value, Value> loopMap = *localMap;
      loopMap[forOp.getInductionVar()] = newFor.getInductionVar();
      for (auto [oldArg, newArg] :
           llvm::zip(forOp.getRegionIterArgs(), newFor.getRegionIterArgs())) {
        loopMap[oldArg] = newArg;
      }

      Block *oldBody = forOp.getBody();
      Block *newBody = newFor.getBody();
      if (!oldBody || !newBody || !oldBody->mightHaveTerminator())
        return false;
      if (!newBody->mightHaveTerminator()) {
        SmallVector<Value> passthrough;
        for (Value arg : newFor.getRegionIterArgs())
          passthrough.push_back(arg);
        rw.setInsertionPointToEnd(newBody);
        rw.create<scf::YieldOp>(opLoc, passthrough);
      }
      if (!newBody->mightHaveTerminator())
        return false;
      Operation *newTerm = newBody->getTerminator();
      rw.setInsertionPoint(newTerm);
      for (Operation &bodyOp : oldBody->without_terminator()) {
        if (!rebuildOp(&bodyOp, &loopMap, false))
          return false;
      }
      auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
      SmallVector<Value> yieldVals;
      for (Value y : oldYield.getResults()) {
        auto mapped = loopMap.find(y);
        if (mapped == loopMap.end() || !mapped->second)
          return false;
        yieldVals.push_back(mapped->second);
      }
      rw.setInsertionPoint(newTerm);
      rw.create<scf::YieldOp>(oldYield.getLoc(), yieldVals);
      rw.eraseOp(newTerm);

      for (auto [oldR, newR] :
           llvm::zip(forOp.getResults(), newFor.getResults()))
        (*localMap)[oldR] = newR;
      rw.setInsertionPointAfter(newFor);
      return true;
    }

    SmallVector<Value> operands;
    bool hasTensorOperand = false;
    for (Value operand : op->getOperands()) {
      Value lifted = lift(operand, localMap);
      if (!lifted)
        return false;
      hasTensorOperand |= isa<RankedTensorType>(lifted.getType());
      operands.push_back(lifted);
    }
    if (hasTensorOperand) {
      for (size_t idx = 0; idx < operands.size(); ++idx) {
        if (isa<RankedTensorType>(operands[idx].getType()))
          continue;
        Value operand = op->getOperand(idx);
        operands[idx] = rw.create<triton::SplatOp>(
            operands[idx].getLoc(), liftTy(operand.getType()), operands[idx]);
      }
    }
    SmallVector<Type> resultTypes;
    for (Type resultTy : op->getResultTypes()) {
      if (hasTensorOperand)
        resultTypes.push_back(liftTy(resultTy));
      else
        resultTypes.push_back(resultTy);
    }
    Operation *nu = rw.create(opLoc, op->getName().getIdentifier(), operands,
                              resultTypes, op->getAttrs());
    for (auto [oldR, newR] : llvm::zip(op->getResults(), nu->getResults()))
      (*localMap)[oldR] = newR;
    return true;
  };

  DenseMap<Value, Value> topMap;
  for (Operation *op : ordered) {
    if (!rebuildOp(op, &topMap, true))
      return false;
  }

  Block *entryBlock = seed.entryGuard->getBlock();
  if (!entryBlock || !entryBlock->mightHaveTerminator())
    return false;
  auto cond = dyn_cast<cf::CondBranchOp>(entryBlock->getTerminator());
  if (!cond)
    return false;
  rw.setInsertionPoint(cond);
  rw.create<cf::BranchOp>(cond.getLoc(), seed.workBlock);
  rw.eraseOp(cond);

  std::function<void(Operation *)> eraseNestedOps = [&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : llvm::make_early_inc_range(region)) {
        while (!block.empty()) {
          Operation &nested = block.back();
          eraseNestedOps(&nested);
          nested.dropAllUses();
          rw.eraseOp(&nested);
        }
      }
    }
  };
  for (auto it = ordered.rbegin(); it != ordered.rend(); ++it) {
    Operation *op = *it;
    eraseNestedOps(op);
    op->dropAllUses();
    rw.eraseOp(op);
  }
  if (seed.entryGuard->use_empty())
    rw.eraseOp(seed.entryGuard);

  (void)moduleOp;
  return true;
}

static void rewriteModule(ModuleOp moduleOp, IRRewriter &rw) {
  auto seed = matchRowSeed(moduleOp);
  if (!seed)
    return;

  SmallVector<Operation *> ordered;
  if (!collectRowRegion(*seed, ordered))
    return;
  int64_t maxBaseElements = getMaxStaticTensorElements(ordered);
  int64_t rowsPerProgram = inferRowsPerProgram(maxBaseElements);
  if (rowsPerProgram <= 1)
    return;

  if (!rewriteMatchedRow(moduleOp, *seed, ordered, rw, rowsPerProgram))
    return;

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] matched graph optimization rule "
                          << static_cast<unsigned>(
                                 cfg::GraphOptimizationRuleId::RowCoalescing)
                          << " ("
                          << cfg::getGraphOptimizationRuleName(
                                 cfg::GraphOptimizationRuleId::RowCoalescing)
                          << ") at " << seed->pid.getLoc()
                          << ": axis=" << seed->axis
                          << " rowsPerProgram=" << rowsPerProgram << "\n");

  auto i32Ty = IntegerType::get(moduleOp.getContext(), 32);
  moduleOp->setAttr(kCoalesceFactorAttr,
                    IntegerAttr::get(i32Ty, rowsPerProgram));
  moduleOp->setAttr(kCoalesceAxisAttr, IntegerAttr::get(i32Ty, seed->axis));
  moduleOp->setAttr(kCoalesceGridCeilDivAttr, IntegerAttr::get(i32Ty, 1));
}

} // namespace

void rewriteRowCoalesce(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());
  rewriteModule(moduleOp, rw);
}

} // namespace RowCoalescing
