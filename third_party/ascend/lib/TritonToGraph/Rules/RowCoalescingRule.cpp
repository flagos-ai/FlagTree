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
#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/LegacyMemoryAccess/RowCoalescing.h"
#include "TritonToGraph/ProgramAxisDependenceAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <memory>
#include <optional>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";
constexpr llvm::StringLiteral kCoalesceGridCeilDivAttr =
    "hacc.coalesce_grid_ceil_div";
constexpr llvm::StringLiteral kIndependentAxisTensorizeMarkerAttr =
    "hacc.independent_axis_tensorize";
constexpr llvm::StringLiteral kPersistentTaskStripMiningMarkerAttr =
    "hacc.persistent_task_strip_mining";

constexpr int64_t kDefaultRowsPerProgram = 8;
constexpr int64_t kMaxBaseElementsPerLift = 1024;

struct RowSeed {
  triton::GetProgramIdOp pid;
  int32_t axis = 0;
  Value validCount;
  arith::CmpIOp entryGuard;
  Block *workBlock = nullptr;
};

struct RowCandidate {
  triton::FuncOp function;
  Operation *anchor = nullptr;
  int32_t axis = 0;
  int64_t rowsPerProgram = 1;
};

bool isPublicEntry(triton::FuncOp function) {
  auto visibility = function->getAttrOfType<StringAttr>("sym_visibility");
  return !visibility || visibility.getValue() == "public";
}

bool isOnlyPublicEntry(ModuleOp module, triton::FuncOp function) {
  unsigned publicEntries = 0;
  for (triton::FuncOp candidate : module.getOps<triton::FuncOp>()) {
    if (!isPublicEntry(candidate))
      continue;
    ++publicEntries;
    if (candidate != function)
      return false;
  }
  return publicEntries == 1;
}

bool hasDirectCall(triton::FuncOp function) {
  bool hasCall = false;
  function.walk([&](Operation *operation) {
    if (isa<CallOpInterface>(operation))
      hasCall = true;
  });
  return hasCall;
}

bool readsAxisNumPrograms(
    triton::FuncOp function, int32_t axis,
    const ProgramAxisDependenceAnalysis *programAxisAnalysis = nullptr) {
  if (programAxisAnalysis)
    return programAxisAnalysis->get(axis).readsNumPrograms;
  bool reads = false;
  function.walk([&](triton::GetNumProgramsOp np) {
    if (np.getAxisAsInt() == axis)
      reads = true;
  });
  return reads;
}

bool isScalarIntegerLike(Value value) {
  Type type = value.getType();
  if (isa<IndexType>(type))
    return true;
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && integerType.getWidth() > 1;
}

bool isAutomaticOverflowAssert(triton::AssertOp assertOp) {
  if (!assertOp || !assertOp->hasAttr("tt.auto_overflow_assert"))
    return false;
  auto message = dyn_cast<StringAttr>(assertOp.getMessageAttr());
  return message &&
         message.getValue().contains("overflow detected for operation");
}

bool isInWorkRegion(Operation *operation, Block *workBlock) {
  for (Operation *current = operation; current;
       current = current->getParentOp()) {
    if (current->getBlock() == workBlock)
      return true;
  }
  return false;
}

bool isRowLiftable(Operation *operation) {
  if (isa<triton::ReturnOp, cf::BranchOp, cf::CondBranchOp>(operation))
    return false;
  if (auto assertOp = dyn_cast<triton::AssertOp>(operation))
    return isAutomaticOverflowAssert(assertOp);
  if (Dialect *dialect = operation->getDialect()) {
    StringRef dialectNamespace = dialect->getNamespace();
    if (dialectNamespace == arith::ArithDialect::getDialectNamespace() ||
        dialectNamespace == math::MathDialect::getDialectNamespace())
      return true;
  }
  return isa<triton::SplatOp, triton::AddPtrOp, triton::BroadcastOp,
             triton::ExpandDimsOp, triton::LoadOp, triton::StoreOp,
             triton::MakeRangeOp, triton::ScanOp, triton::ReduceOp,
             triton::ClampFOp, triton::FpToFpOp, scf::ForOp>(operation);
}

bool collectRowRegion(const RowSeed &seed,
                      SmallVectorImpl<Operation *> &ordered) {
  if (!seed.workBlock || !seed.workBlock->mightHaveTerminator() ||
      !isa<triton::ReturnOp>(seed.workBlock->getTerminator()))
    return false;

  bool hasStore = false;
  for (Operation &operation : seed.workBlock->without_terminator()) {
    bool safe = true;
    operation.walk([&](Operation *nested) {
      if (nested == &operation)
        return;
      if (isa<scf::YieldOp, triton::ReduceReturnOp, triton::ScanReturnOp>(
              nested))
        return;
      if (!isRowLiftable(nested))
        safe = false;
    });
    if (!safe || !isRowLiftable(&operation))
      return false;
    hasStore |= isa<triton::StoreOp>(operation);
    ordered.push_back(&operation);
  }
  return hasStore;
}

int64_t getMaxStaticTensorElements(ArrayRef<Operation *> ordered) {
  int64_t maxElements = 1;
  auto update = [&](Type type) {
    auto rankedType = dyn_cast<RankedTensorType>(type);
    if (!rankedType)
      return;
    if (!rankedType.hasStaticShape()) {
      maxElements = kMaxBaseElementsPerLift + 1;
      return;
    }
    maxElements = std::max(maxElements, rankedType.getNumElements());
  };
  for (Operation *operation : ordered) {
    for (Value operand : operation->getOperands())
      update(operand.getType());
    for (Value result : operation->getResults())
      update(result.getType());
  }
  return maxElements;
}

int64_t inferRowsPerProgram(int64_t maxBaseElements) {
  if (maxBaseElements > kMaxBaseElementsPerLift)
    return 1;
  if (maxBaseElements >= 1024)
    return 2;
  if (maxBaseElements > 16)
    return 4;
  return kDefaultRowsPerProgram;
}

bool hasPidDerivedValueOutsideWork(RowSeed &seed) {
  DenseSet<Value> visited;
  SmallVector<Value> worklist{seed.pid.getResult()};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      // The canonical guard intentionally consumes the scalar PID before the
      // work block.  Every other PID-derived calculation must be rebuilt in
      // the lifted region; otherwise legacy Row would splat it as uniform.
      if (user == seed.entryGuard.getOperation())
        continue;
      if (!isInWorkRegion(user, seed.workBlock))
        return true;
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
  return false;
}

bool hasEscapingWorkResult(ArrayRef<Operation *> ordered, Block *workBlock) {
  for (Operation *operation : ordered) {
    for (Value result : operation->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!isInWorkRegion(user, workBlock))
          return true;
      }
    }
  }
  return false;
}

std::optional<RowSeed> matchRowSeed(
    triton::FuncOp function,
    const ProgramAxisDependenceAnalysis *programAxisAnalysis = nullptr) {
  SmallVector<triton::GetProgramIdOp> pids;
  function.walk([&](triton::GetProgramIdOp pid) { pids.push_back(pid); });
  if (pids.size() != 1)
    return std::nullopt;

  triton::GetProgramIdOp pid = pids.front();
  const int32_t axis = pid.getAxisAsInt();
  if (readsAxisNumPrograms(function, axis, programAxisAnalysis))
    return std::nullopt;

  for (Operation *user : pid.getResult().getUsers()) {
    auto guard = dyn_cast<arith::CmpIOp>(user);
    if (!guard || guard.getPredicate() != arith::CmpIPredicate::sge ||
        guard.getLhs() != pid.getResult() || !guard.getResult().hasOneUse())
      continue;

    Block *entryBlock = guard->getBlock();
    if (!entryBlock || entryBlock != pid->getBlock() ||
        !entryBlock->mightHaveTerminator())
      continue;
    auto branch =
        dyn_cast_or_null<cf::CondBranchOp>(entryBlock->getTerminator());
    if (!branch || branch.getCondition() != guard.getResult())
      continue;

    Block *returnBlock = branch.getTrueDest();
    Block *workBlock = branch.getFalseDest();
    if (!returnBlock || !workBlock || returnBlock->getNumArguments() != 0 ||
        workBlock->getNumArguments() != 0 ||
        !returnBlock->mightHaveTerminator() ||
        !isa<triton::ReturnOp>(returnBlock->getTerminator()) ||
        returnBlock->getTerminator()->getNumOperands() != 0 ||
        !isScalarIntegerLike(guard.getRhs()))
      continue;
    auto predecessors = workBlock->getPredecessors();
    if (!llvm::hasSingleElement(predecessors) ||
        *predecessors.begin() != entryBlock)
      continue;

    RowSeed seed{pid, axis, guard.getRhs(), guard, workBlock};
    if (hasPidDerivedValueOutsideWork(seed))
      continue;
    return seed;
  }
  return std::nullopt;
}

std::optional<RowCandidate>
analyzeRow(triton::FuncOp function,
           const ProgramAxisDependenceAnalysis *programAxisAnalysis = nullptr) {
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module || !isPublicEntry(function) ||
      !isOnlyPublicEntry(module, function) || function->getNumRegions() != 1 ||
      function->getRegion(0).empty() || hasDirectCall(function) ||
      module->hasAttr(kIndependentAxisTensorizeMarkerAttr) ||
      module->hasAttr(kPersistentTaskStripMiningMarkerAttr) ||
      module->hasAttr(kCoalesceFactorAttr) ||
      module->hasAttr(kCoalesceAxisAttr) ||
      module->hasAttr(kCoalesceGridCeilDivAttr))
    return std::nullopt;

  std::optional<RowSeed> seed = matchRowSeed(function, programAxisAnalysis);
  if (!seed)
    return std::nullopt;

  SmallVector<Operation *> ordered;
  if (!collectRowRegion(*seed, ordered) ||
      hasEscapingWorkResult(ordered, seed->workBlock))
    return std::nullopt;

  const int64_t rowsPerProgram =
      inferRowsPerProgram(getMaxStaticTensorElements(ordered));
  if (rowsPerProgram <= 1)
    return std::nullopt;
  return RowCandidate{function, seed->entryGuard.getOperation(), seed->axis,
                      rowsPerProgram};
}

bool matchesCandidate(const RowCandidate &candidate,
                      const RowCandidate &current) {
  return candidate.function == current.function &&
         candidate.anchor == current.anchor && candidate.axis == current.axis &&
         candidate.rowsPerProgram == current.rowsPerProgram;
}

class RowCoalescingPlan final : public RewritePlan {
public:
  RowCoalescingPlan(RowCandidate candidate, unsigned epoch)
      : candidate(candidate), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::RowCoalescing;
  }

  unsigned getBenefit() const override { return 1; }
  Operation *getAnchor() const override { return candidate.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    std::optional<RowCandidate> current = analyzeRow(context.getFunction());
    return current && matchesCandidate(candidate, *current) ? success()
                                                            : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    return applyWithResult(rewriter) == RewritePlanApplyResult::Applied
               ? success()
               : failure();
  }

  RewritePlanApplyResult applyWithResult(IRRewriter &rewriter) override {
    (void)rewriter;
    std::optional<RowCandidate> current = analyzeRow(candidate.function);
    if (!current || !matchesCandidate(candidate, *current))
      return RewritePlanApplyResult::NotApplicable;

    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    if (!module)
      return RewritePlanApplyResult::Failed;

    // The old Row implementation can still encounter a late unsupported
    // shape while materializing.  Run it on a detached one-function module;
    // the original function and its launch attrs remain untouched until the
    // cloned IR verifies successfully.
    ModuleOp sandbox = ModuleOp::create(candidate.function.getLoc());
    sandbox.getBody()->push_back(candidate.function->clone());
    auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
    if (!clonedFunction)
      return RewritePlanApplyResult::Failed;

    RowCoalescing::rewriteRowCoalesce(sandbox);
    auto factor = sandbox->getAttrOfType<IntegerAttr>(kCoalesceFactorAttr);
    auto axis = sandbox->getAttrOfType<IntegerAttr>(kCoalesceAxisAttr);
    auto ceilDiv =
        sandbox->getAttrOfType<IntegerAttr>(kCoalesceGridCeilDivAttr);
    if (!factor || !axis || !ceilDiv ||
        factor.getInt() != candidate.rowsPerProgram ||
        axis.getInt() != candidate.axis || ceilDiv.getInt() != 1 ||
        failed(mlir::verify(sandbox.getOperation())))
      return RewritePlanApplyResult::NotApplicable;

    // takeBody() is non-failing.  Commit the function IR first, then publish
    // the complete launcher contract as one final, non-failing step.
    candidate.function->getRegion(0).takeBody(clonedFunction->getRegion(0));
    auto i32Type = IntegerType::get(module.getContext(), 32);
    module->setAttr(kCoalesceFactorAttr,
                    IntegerAttr::get(i32Type, candidate.rowsPerProgram));
    module->setAttr(kCoalesceAxisAttr,
                    IntegerAttr::get(i32Type, candidate.axis));
    module->setAttr(kCoalesceGridCeilDivAttr, IntegerAttr::get(i32Type, 1));
    return RewritePlanApplyResult::Applied;
  }

private:
  RowCandidate candidate;
  unsigned epoch;
};

class RowCoalescingRule final : public GraphOptimizationRule {
  bool enableLegacyPattern;

public:
  explicit RowCoalescingRule(bool enableLegacyPattern)
      : enableLegacyPattern(enableLegacyPattern) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::RowCoalescing;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::ProgramAxisDependence;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    // The legacy pattern is enabled only for simt_only. Dot row coalescing
    // belongs to the non-pure-SIMT path and must not override that pattern.
    if (!enableLegacyPattern) {
      if (auto plan = createDotRowCoalescingPlan(context.getFunction(),
                                                 context.getEpoch()))
        plans.push_back(std::move(plan));
      return success();
    }

    if (std::optional<RowCandidate> candidate =
            analyzeRow(context.getFunction(),
                       &context.getProgramAxisDependenceAnalysis())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] matched graph optimization rule "
                 << static_cast<unsigned>(getId()) << " ("
                 << getGraphOptimizationRuleName(getId()) << ") in @"
                 << candidate->function.getName()
                 << ": axis=" << candidate->axis
                 << " rowsPerProgram=" << candidate->rowsPerProgram << "\n");
      plans.push_back(
          std::make_unique<RowCoalescingPlan>(*candidate, context.getEpoch()));
    }
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule>
cfg::createRowCoalescingRule(bool enableLegacy) {
  return std::make_unique<RowCoalescingRule>(enableLegacy);
}
