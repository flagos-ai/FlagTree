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

#include "TritonToGraph/GraphOptimizationContext.h"
#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/Passes.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"
#include "Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

namespace mlir {
namespace triton {
namespace cfg {
#define GEN_PASS_DEF_GRAPHOPTIMIZE
#include "ascend/include/TritonToGraph/Passes.h.inc"
} // namespace cfg
} // namespace triton
} // namespace mlir

namespace mlir {
namespace triton {
namespace cfg {
namespace {

constexpr std::array<GraphOptimizationRulePhase, 11> kRulePhases = {
    GraphOptimizationRulePhase::DiagonalMaskRemoval,
    GraphOptimizationRulePhase::ConvertModuloToMask,
    GraphOptimizationRulePhase::ProgramMapping,
    GraphOptimizationRulePhase::PersistentTaskMapping,
    GraphOptimizationRulePhase::LoadStoreTranspose,
    GraphOptimizationRulePhase::TransposePointwiseReorder,
    GraphOptimizationRulePhase::ResidentLoadForwarding,
    GraphOptimizationRulePhase::IntermediatePrecisionBoundaryElision,
    GraphOptimizationRulePhase::StoreCoveragePlanning,
    GraphOptimizationRulePhase::StoreCoalescing,
    GraphOptimizationRulePhase::ContiguousBlockAccessFormation,
};

using ProgramOrderMap = llvm::DenseMap<Operation *, unsigned>;

ProgramOrderMap buildProgramOrderMap(triton::FuncOp function) {
  ProgramOrderMap programOrder;
  unsigned nextOrder = 0;
  function.walk(
      [&](Operation *operation) { programOrder[operation] = nextOrder++; });
  return programOrder;
}

unsigned getProgramOrder(const RewritePlan &plan,
                         const ProgramOrderMap &programOrder) {
  Operation *anchor = plan.getAnchor();
  if (!anchor)
    return std::numeric_limits<unsigned>::max();

  auto it = programOrder.find(anchor);
  if (it == programOrder.end())
    return std::numeric_limits<unsigned>::max();
  return it->second;
}

bool isRuleEnabled(GraphOptimizationRuleMask ruleMask,
                   GraphOptimizationRuleId ruleId) {
  return (ruleMask & getGraphOptimizationRuleMask(ruleId)) != 0;
}

constexpr bool requiresProgramMappingCleanup(GraphOptimizationRuleId ruleId) {
  return ruleId == GraphOptimizationRuleId::IndependentAxisTensorize ||
         ruleId == GraphOptimizationRuleId::PersistentTaskStripMining;
}

constexpr bool isPlanHigherPriority(unsigned lhsBenefit, unsigned lhsOrder,
                                    GraphOptimizationRuleId lhsRuleId,
                                    unsigned rhsBenefit, unsigned rhsOrder,
                                    GraphOptimizationRuleId rhsRuleId) {
  if (lhsBenefit != rhsBenefit)
    return lhsBenefit > rhsBenefit;
  if (lhsOrder != rhsOrder)
    return lhsOrder < rhsOrder;
  return getGraphOptimizationRuleMask(lhsRuleId) <
         getGraphOptimizationRuleMask(rhsRuleId);
}

class GraphOptimizePass final
    : public impl::GraphOptimizeBase<GraphOptimizePass> {
public:
  GraphOptimizePass() = default;

  explicit GraphOptimizePass(const GraphOptimizationOptions &options) {
    this->ruleMask = options.enabledRuleMask;
    this->maxRewritesPerFunction = options.maxRewritesPerFunction;
    this->ubCapacityBytes = options.ubCapacityBytes;
    this->mappingUBCapacityBytes = options.mappingUBCapacityBytes;
    this->storeCoalescingUBBudgetBytes = options.storeCoalescingUBBudgetBytes;
    this->ubSafetyPercent = options.ubSafetyPercent;
    this->reservedUBBytes = options.reservedUBBytes;
    this->compileMode = options.compileMode;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;

private:
  LogicalResult getStableOptions(GraphOptimizationOptions &options);
  LogicalResult runStructuralCleanup();
};

LogicalResult GraphOptimizePass::runStructuralCleanup() {
  PassManager cleanup(&getContext(), getOperation().getOperationName());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  cleanup.addPass(createLoopInvariantCodeMotionPass());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  return runPipeline(cleanup, getOperation());
}

LogicalResult
GraphOptimizePass::getStableOptions(GraphOptimizationOptions &options) {
  const uint64_t cliRuleMask = this->ruleMask;
  const uint64_t cliMaxRewrites = this->maxRewritesPerFunction;
  const uint64_t cliUBCapacityBytes = this->ubCapacityBytes;
  const uint64_t cliMappingUBCapacityBytes = this->mappingUBCapacityBytes;
  const uint64_t cliStoreCoalescingUBBudgetBytes =
      this->storeCoalescingUBBudgetBytes;
  const uint64_t cliUBSafetyPercent = this->ubSafetyPercent;
  const uint64_t cliReservedUBBytes = this->reservedUBBytes;

  if (cliRuleMask > std::numeric_limits<GraphOptimizationRuleMask>::max() ||
      !isValidGraphOptimizationRuleMask(
          static_cast<GraphOptimizationRuleMask>(cliRuleMask))) {
    getOperation().emitError()
        << "graph-optimize rule-mask contains unknown or out-of-range bits: "
        << cliRuleMask;
    return failure();
  }

  if (cliMaxRewrites > std::numeric_limits<unsigned>::max()) {
    getOperation().emitError()
        << "graph-optimize max-rewrites-per-function is out of range: "
        << cliMaxRewrites;
    return failure();
  }

  if (cliUBCapacityBytes > std::numeric_limits<unsigned>::max()) {
    getOperation().emitError()
        << "graph-optimize ub-capacity-bytes is out of range: "
        << cliUBCapacityBytes;
    return failure();
  }

  if (cliMappingUBCapacityBytes > std::numeric_limits<unsigned>::max() ||
      cliStoreCoalescingUBBudgetBytes > std::numeric_limits<unsigned>::max()) {
    getOperation().emitError()
        << "graph-optimize explicit UB budget is out of range: mapping="
        << cliMappingUBCapacityBytes
        << " store-coalescing=" << cliStoreCoalescingUBBudgetBytes;
    return failure();
  }

  const uint64_t effectiveMappingUBCapacity = cliMappingUBCapacityBytes != 0
                                                  ? cliMappingUBCapacityBytes
                                                  : cliUBCapacityBytes;
  const uint64_t effectiveStoreCoalescingUBBudget =
      cliStoreCoalescingUBBudgetBytes != 0 ? cliStoreCoalescingUBBudgetBytes
                                           : cliUBCapacityBytes;

  if (cliUBSafetyPercent == 0 || cliUBSafetyPercent > 100 ||
      cliReservedUBBytes > std::numeric_limits<unsigned>::max() ||
      cliReservedUBBytes > effectiveMappingUBCapacity) {
    getOperation().emitError()
        << "graph-optimize resource options are invalid: ub-safety-percent="
        << cliUBSafetyPercent << " reserved-ub-bytes=" << cliReservedUBBytes;
    return failure();
  }

  const auto compileMode = triton::ascend::parseCompileMode(this->compileMode);
  if (!compileMode) {
    getOperation().emitError()
        << "graph-optimize compile-mode is invalid: " << this->compileMode;
    return failure();
  }

  options.enabledRuleMask = static_cast<GraphOptimizationRuleMask>(cliRuleMask);
  options.maxRewritesPerFunction = static_cast<unsigned>(cliMaxRewrites);
  options.ubCapacityBytes =
      static_cast<unsigned>(effectiveStoreCoalescingUBBudget);
  options.mappingUBCapacityBytes =
      static_cast<unsigned>(effectiveMappingUBCapacity);
  options.storeCoalescingUBBudgetBytes =
      static_cast<unsigned>(effectiveStoreCoalescingUBBudget);
  options.ubSafetyPercent = static_cast<unsigned>(cliUBSafetyPercent);
  options.reservedUBBytes = static_cast<unsigned>(cliReservedUBBytes);
  options.compileMode = this->compileMode;
  options.independentAxisTensorize.enabledForCompileMode =
      *compileMode != triton::ascend::CompileMode::SimtOnly;
  options.independentAxisTensorize.iatAndPtsmEnabled =
      isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::IndependentAxisTensorize) &&
      isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::PersistentTaskStripMining);
  options.persistentTaskStripMining.enabledForCompileMode =
      *compileMode != triton::ascend::CompileMode::SimtOnly;
  options.storeCoalescing.enabledForCompileMode =
      *compileMode != triton::ascend::CompileMode::SimtOnly;
  return success();
}

void GraphOptimizePass::runOnOperation() {
  GraphOptimizationOptions options;
  if (failed(getStableOptions(options))) {
    signalPassFailure();
    return;
  }

  ModuleOp module = getOperation();
  SmallVector<std::unique_ptr<GraphOptimizationRule>> ownedRules;
  populateBuiltinGraphOptimizationRules(options, ownedRules);

  SmallVector<GraphOptimizationRule *> enabledRules;
  llvm::DenseSet<GraphOptimizationRuleMask> registeredRuleMasks;
  for (const std::unique_ptr<GraphOptimizationRule> &rule : ownedRules) {
    if (!rule) {
      getOperation().emitError()
          << "graph-optimize builtin rule factory returned null";
      signalPassFailure();
      return;
    }

    const GraphOptimizationRuleMask ruleMask =
        getGraphOptimizationRuleMask(rule->getId());
    if (ruleMask == 0 || !isValidGraphOptimizationRuleMask(ruleMask) ||
        !isRuleEnabled(options.enabledRuleMask, rule->getId()) ||
        !registeredRuleMasks.insert(ruleMask).second) {
      getOperation().emitError()
          << "graph-optimize registered an invalid, disabled, or duplicate "
             "rule ID: "
          << ruleMask;
      signalPassFailure();
      return;
    }
    enabledRules.push_back(rule.get());
  }

  for (triton::FuncOp function : module.getOps<triton::FuncOp>()) {
    const ResourceSnapshot resources = ResourceSnapshot::fromExplicit(
        options.mappingUBCapacityBytes, options.ubSafetyPercent,
        options.reservedUBBytes);
    GraphOptimizationContext context(function, resources);
    unsigned rewriteCount = 0;

    for (GraphOptimizationRulePhase phase : kRulePhases) {
      while (rewriteCount < options.maxRewritesPerFunction) {
        ProgramOrderMap programOrder = buildProgramOrderMap(function);
        SmallVector<std::unique_ptr<RewritePlan>> plans;

        for (GraphOptimizationRule *rule : enabledRules) {
          if (getGraphOptimizationRulePhase(rule->getId()) != phase)
            continue;

          if (failed(context.ensure(rule->getAnalysisRequirements()))) {
            function.emitError() << "graph-optimize failed to build analyses";
            signalPassFailure();
            return;
          }

          if (failed(rule->findCandidates(context, plans))) {
            function.emitError()
                << "graph-optimize failed to discover rewrite candidates";
            signalPassFailure();
            return;
          }
        }

        plans.erase(
            std::remove_if(plans.begin(), plans.end(),
                           [phase](const std::unique_ptr<RewritePlan> &plan) {
                             return !plan || getGraphOptimizationRulePhase(
                                                 plan->getRuleId()) != phase;
                           }),
            plans.end());
        if (plans.empty())
          break;

        std::stable_sort(
            plans.begin(), plans.end(),
            [&programOrder](const std::unique_ptr<RewritePlan> &lhs,
                            const std::unique_ptr<RewritePlan> &rhs) {
              const unsigned lhsOrder = getProgramOrder(*lhs, programOrder);
              const unsigned rhsOrder = getProgramOrder(*rhs, programOrder);
              return isPlanHigherPriority(lhs->getBenefit(), lhsOrder,
                                          lhs->getRuleId(), rhs->getBenefit(),
                                          rhsOrder, rhs->getRuleId());
            });

        std::unique_ptr<RewritePlan> selectedPlan;
        for (std::unique_ptr<RewritePlan> &plan : plans) {
          if (plan->getCreationEpoch() != context.getEpoch())
            continue;
          if (failed(plan->revalidate(context))) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "[" DEBUG_TYPE "] dropped stale graph optimization rule "
                << getGraphOptimizationRuleMask(plan->getRuleId()) << " ("
                << getGraphOptimizationRuleName(plan->getRuleId()) << ")\n");
            continue;
          }

          selectedPlan = std::move(plan);
          break;
        }

        if (!selectedPlan) {
          plans.clear();
          break;
        }

        const GraphOptimizationRuleId appliedRuleId = selectedPlan->getRuleId();
        IRRewriter rewriter(&getContext());
        if (failed(selectedPlan->apply(rewriter))) {
          selectedPlan.reset();
          plans.clear();
          function.emitError() << "graph-optimize failed to apply rewrite";
          signalPassFailure();
          return;
        }
        if (requiresProgramMappingCleanup(appliedRuleId) &&
            failed(runStructuralCleanup())) {
          selectedPlan.reset();
          plans.clear();
          function.emitError()
              << "graph-optimize failed structural rewrite cleanup";
          signalPassFailure();
          return;
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] applied graph optimization rule "
                   << getGraphOptimizationRuleMask(appliedRuleId) << " ("
                   << getGraphOptimizationRuleName(appliedRuleId) << ")\n");

        // Plans can retain pointers into analysis results, so destroy all of
        // them before invalidating the context for the next IR epoch.
        selectedPlan.reset();
        plans.clear();
        context.invalidate();
        ++rewriteCount;
      }

      if (rewriteCount == options.maxRewritesPerFunction)
        break;
    }

    // RowCoalescing has the same function-local candidate/rewrite interface
    // as the native graph rules, but it has distinct launch semantics.  Its
    // historical pass ran once and was not subject to the generic rewrite
    // budget, so run it once after LoadStoreTranspose,
    // TransposePointwiseReorder, and StoreCoalescing even when that budget
    // has already been exhausted.
    if (!isRuleEnabled(options.enabledRuleMask,
                       GraphOptimizationRuleId::RowCoalescing))
      continue;
    if (module->hasAttr(kProgramGridTransformsAttr))
      continue;

    GraphOptimizationRule *rowRule = nullptr;
    for (GraphOptimizationRule *rule : enabledRules) {
      if (rule->getId() == GraphOptimizationRuleId::RowCoalescing) {
        rowRule = rule;
        break;
      }
    }
    if (!rowRule)
      continue;

    if (failed(context.ensure(rowRule->getAnalysisRequirements()))) {
      function.emitError() << "graph-optimize failed to build Row analyses";
      signalPassFailure();
      return;
    }

    SmallVector<std::unique_ptr<RewritePlan>> rowPlans;
    if (failed(rowRule->findCandidates(context, rowPlans))) {
      function.emitError() << "graph-optimize failed to discover Row candidate";
      signalPassFailure();
      return;
    }
    rowPlans.erase(
        std::remove_if(rowPlans.begin(), rowPlans.end(),
                       [](const std::unique_ptr<RewritePlan> &plan) {
                         return !plan ||
                                plan->getRuleId() !=
                                    GraphOptimizationRuleId::RowCoalescing;
                       }),
        rowPlans.end());
    if (rowPlans.empty())
      continue;

    ProgramOrderMap programOrder = buildProgramOrderMap(function);
    std::stable_sort(
        rowPlans.begin(), rowPlans.end(),
        [&programOrder](const std::unique_ptr<RewritePlan> &lhs,
                        const std::unique_ptr<RewritePlan> &rhs) {
          const unsigned lhsOrder = getProgramOrder(*lhs, programOrder);
          const unsigned rhsOrder = getProgramOrder(*rhs, programOrder);
          return isPlanHigherPriority(lhs->getBenefit(), lhsOrder,
                                      lhs->getRuleId(), rhs->getBenefit(),
                                      rhsOrder, rhs->getRuleId());
        });

    std::unique_ptr<RewritePlan> selectedRowPlan;
    for (std::unique_ptr<RewritePlan> &plan : rowPlans) {
      if (plan->getCreationEpoch() != context.getEpoch())
        continue;
      if (failed(plan->revalidate(context)))
        continue;
      selectedRowPlan = std::move(plan);
      break;
    }
    if (!selectedRowPlan)
      continue;

    IRRewriter rewriter(&getContext());
    switch (selectedRowPlan->applyWithResult(rewriter)) {
    case RewritePlanApplyResult::Applied:
      break;
    case RewritePlanApplyResult::NotApplicable:
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE << "] declined graph optimization rule "
                 << static_cast<unsigned>(
                        GraphOptimizationRuleId::RowCoalescing)
                 << " ("
                 << getGraphOptimizationRuleName(
                        GraphOptimizationRuleId::RowCoalescing)
                 << ") after detached materialization\n");
      selectedRowPlan.reset();
      rowPlans.clear();
      continue;
    case RewritePlanApplyResult::Failed:
      selectedRowPlan.reset();
      rowPlans.clear();
      function.emitError() << "graph-optimize failed to apply Row rewrite";
      signalPassFailure();
      return;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] applied graph optimization rule "
               << getGraphOptimizationRuleMask(
                      GraphOptimizationRuleId::RowCoalescing)
               << " ("
               << getGraphOptimizationRuleName(
                      GraphOptimizationRuleId::RowCoalescing)
               << ")\n");

    selectedRowPlan.reset();
    rowPlans.clear();
    context.invalidate();
  }
}

} // namespace

void populateBuiltinGraphOptimizationRules(
    const GraphOptimizationOptions &options,
    SmallVectorImpl<std::unique_ptr<GraphOptimizationRule>> &rules) {
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::DiagonalMaskRemoval)) {
    rules.push_back(createDiagonalMaskRemovalRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::ConvertModuloToMask)) {
    rules.push_back(createConvertModuloToMaskRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::IndependentAxisTensorize)) {
    rules.push_back(
        createIndependentAxisTensorizeRule(options.independentAxisTensorize));
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::PersistentTaskStripMining)) {
    rules.push_back(
        createPersistentTaskStripMiningRule(options.persistentTaskStripMining));
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::LoadStoreTranspose)) {
    rules.push_back(createLoadStoreTransposeRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::TransposePointwiseReorder)) {
    rules.push_back(createTransposePointwiseReorderRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::StoreCoalescing)) {
    rules.push_back(createStoreCoalescingRule(
        options.storeCoalescingUBBudgetBytes, options.storeCoalescing));
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::ResidentLoadForwarding)) {
    rules.push_back(
        createResidentLoadForwardingRule(options.residentLoadForwarding));
  }
  if (isRuleEnabled(
          options.enabledRuleMask,
          GraphOptimizationRuleId::IntermediatePrecisionBoundaryElision)) {
    rules.push_back(createIntermediatePrecisionBoundaryElisionRule(
        options.intermediatePrecisionBoundaryElision));
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::StoreCoveragePlanning)) {
    rules.push_back(
        createStoreCoveragePlanningRule(options.storeCoveragePlanning));
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::ContiguousBlockAccessFormation)) {
    rules.push_back(createContiguousBlockAccessFormationRule(
        options.contiguousBlockAccessFormation));
  }
  const auto compileMode =
      triton::ascend::parseCompileMode(options.compileMode);
  if (compileMode && isRuleEnabled(options.enabledRuleMask,
                                   GraphOptimizationRuleId::RowCoalescing)) {
    rules.push_back(createRowCoalescingRule(
        *compileMode == triton::ascend::CompileMode::SimtOnly));
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createGraphOptimizePass(GraphOptimizationOptions options) {
  return std::make_unique<GraphOptimizePass>(options);
}

} // namespace cfg
} // namespace triton
} // namespace mlir
