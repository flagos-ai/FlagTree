//===- AdvancedPipeliner.cpp - Advanced Multi-Level Pipelining Pass ------===//
//
// This file implements the main orchestrator for advanced pipelining
// optimization, coordinating buffer analysis, opportunity detection,
// circular buffer transformation, and synchronization insertion.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/AdvancedPipeliner.h"
#include "triton/Dialect/TritonGPU/Transforms/BufferAccessAnalysis.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineOpportunityDetector.h"
#include "triton/Dialect/TritonGPU/Transforms/CircularBufferTransform.h"
#include "triton/Dialect/TritonGPU/Transforms/SynchronizationInsertion.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "triton/Dialect/TritonGPU/Transforms/TMASupport.h"
#include "triton/Dialect/TritonGPU/Transforms/MultiBufferFusion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "advanced-pipeliner"

namespace mlir {
namespace triton {
namespace gpu {

// Define the pass implementation - need both DECL (for Options type) and DEF
#define GEN_PASS_DECL_TRITONGPUADVANCEDPIPELINER
#define GEN_PASS_DEF_TRITONGPUADVANCEDPIPELINER
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// RegisterPrefetcher - Loop-Carried Register Double-Buffering
//===----------------------------------------------------------------------===//
// This class implements true loop-carried prefetching for Shared→Register loads.
// The transformation changes:
//   scf.for %iv = ... {
//     %a = local_load %smem_a
//     %c = dot %a, ...
//   }
// Into:
//   %a_pre = local_load %smem_a[0]           // prologue
//   scf.for %iv = ... iter_args(%a_buf = %a_pre) {
//     %c = dot %a_buf, ...                   // use prefetched
//     %a_next = local_load %smem_a[next]     // prefetch next
//     yield %a_next
//   }
//===----------------------------------------------------------------------===//

class RegisterPrefetcher {
public:
  RegisterPrefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    debugEnabled = std::getenv("FLAGTREE_DEBUG_PIPELINE") != nullptr;
  }

  // Find LocalLoadOp that feeds DotOp and can be prefetched
  LogicalResult initialize() {
    Block *loop = forOp.getBody();

    // Find all DotOps in the loop
    SmallVector<triton::DotOp> dotsInFor;
    for (Operation &op : *loop) {
      if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
        dotsInFor.push_back(dotOp);
      }
    }

    if (dotsInFor.empty()) {
      if (debugEnabled) {
        llvm::errs() << "[RegisterPrefetcher] No DotOp found in loop\n";
      }
      return failure();
    }

    // For each DotOp, find LocalLoadOp that produces its operands
    for (triton::DotOp dot : dotsInFor) {
      Value aOperand = dot.getA();
      Value bOperand = dot.getB();

      // Trace back through any ConvertLayoutOp to find LocalLoadOp
      auto findLocalLoad = [&](Value v) -> triton::gpu::LocalLoadOp {
        Operation *defOp = v.getDefiningOp();
        while (defOp) {
          if (auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(defOp)) {
            return localLoad;
          }
          if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(defOp)) {
            defOp = cvt.getSrc().getDefiningOp();
            continue;
          }
          break;
        }
        return nullptr;
      };

      triton::gpu::LocalLoadOp aLocalLoad = findLocalLoad(aOperand);
      triton::gpu::LocalLoadOp bLocalLoad = findLocalLoad(bOperand);

      // Check if LocalLoadOp is inside the loop and has valid source
      auto isValidForPrefetch = [&](triton::gpu::LocalLoadOp localLoad) -> bool {
        if (!localLoad)
          return false;
        // Must be in this loop
        if (localLoad->getParentOfType<scf::ForOp>() != forOp)
          return false;
        // Source must be a MemDescType (shared memory)
        Value src = localLoad.getSrc();
        if (!mlir::isa<triton::MemDescType>(src.getType()))
          return false;

        // For prologue to work, the source must be:
        // 1. A block argument (loop iter_arg) - can use init value
        // 2. Defined outside the loop - can clone
        if (auto blockArg = mlir::dyn_cast<BlockArgument>(src)) {
          if (blockArg.getOwner() == forOp.getBody() && blockArg.getArgNumber() > 0) {
            // Block arg of this loop (not IV) - OK, can use init value
            return true;
          }
        }

        if (auto defOp = src.getDefiningOp()) {
          if (defOp->getParentOfType<scf::ForOp>() != forOp) {
            // Defined outside this loop - OK, can clone
            return true;
          }
        }

        // Source depends on loop-internal values - cannot prefetch
        return false;
      };

      // Helper to get yield value for a block arg
      auto getYieldValueForBlockArg = [&](Value blockArgVal) -> Value {
        if (auto blockArg = mlir::dyn_cast<BlockArgument>(blockArgVal)) {
          if (blockArg.getOwner() == forOp.getBody()) {
            unsigned argNum = blockArg.getArgNumber();
            if (argNum > 0) {  // Skip induction variable
              unsigned yieldIdx = argNum - 1;  // -1 because of IV
              if (yieldIdx < yieldOp.getNumOperands()) {
                return yieldOp.getOperand(yieldIdx);
              }
            }
          }
        }
        return Value();
      };

      if (isValidForPrefetch(aLocalLoad)) {
        dots.insert(dot);
        dot2aLocalLoad[dot] = aLocalLoad;

        // Store yield value for the source
        Value src = aLocalLoad.getSrc();
        if (Value yieldVal = getYieldValueForBlockArg(src)) {
          src2yieldValue[src] = yieldVal;
        }

        if (debugEnabled) {
          llvm::errs() << "[RegisterPrefetcher] Found prefetchable A operand for DotOp\n";
        }
      }

      if (isValidForPrefetch(bLocalLoad)) {
        dots.insert(dot);
        dot2bLocalLoad[dot] = bLocalLoad;

        // Store yield value for the source
        Value src = bLocalLoad.getSrc();
        if (Value yieldVal = getYieldValueForBlockArg(src)) {
          src2yieldValue[src] = yieldVal;
        }

        if (debugEnabled) {
          llvm::errs() << "[RegisterPrefetcher] Found prefetchable B operand for DotOp\n";
        }
      }
    }

    if (dots.empty()) {
      if (debugEnabled) {
        llvm::errs() << "[RegisterPrefetcher] No prefetchable loads found\n";
      }
      return failure();
    }

    if (debugEnabled) {
      llvm::errs() << "[RegisterPrefetcher] Found " << dots.size()
                   << " DotOps with prefetchable operands\n";
    }
    return success();
  }

  // Generate prologue: prefetch first iteration before loop
  // Returns false if prologue generation fails
  bool emitPrologue() {
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();

    // Helper to generate prologue for a single LocalLoadOp
    auto generatePrologueForLoad = [&](triton::gpu::LocalLoadOp localLoad,
                                        const char *name) -> std::optional<Value> {
      Value src = localLoad.getSrc();

      // Check if source is a block argument (loop iter_arg)
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(src)) {
        if (blockArg.getOwner() == forOp.getBody()) {
          unsigned argNum = blockArg.getArgNumber();
          if (argNum > 0) {  // Skip induction variable
            Value initVal = forOp.getInitArgs()[argNum - 1];  // -1 because of IV

            // Create new LocalLoadOp with init value as source
            Value prefetched = builder.create<triton::gpu::LocalLoadOp>(
                loc, localLoad.getType(), initVal);

            if (debugEnabled) {
              llvm::errs() << "[RegisterPrefetcher] Generated prologue for " << name
                           << " operand (from iter_arg init)\n";
            }
            return prefetched;
          }
        }
      }

      // Check if source is defined outside the loop
      if (auto defOp = src.getDefiningOp()) {
        if (defOp->getParentOfType<scf::ForOp>() != forOp) {
          // Source defined outside loop - safe to clone
          IRMapping mapping;
          Operation *cloned = builder.clone(*localLoad.getOperation(), mapping);
          Value prefetched = cloned->getResult(0);

          if (debugEnabled) {
            llvm::errs() << "[RegisterPrefetcher] Generated prologue for " << name
                         << " operand (source outside loop)\n";
          }
          return prefetched;
        }
      }

      // Cannot generate prologue
      if (debugEnabled) {
        llvm::errs() << "[RegisterPrefetcher] Cannot generate prologue for " << name
                     << " operand - source depends on loop values\n";
      }
      return std::nullopt;
    };

    for (triton::DotOp dot : dots) {
      // Process A operand
      if (auto aLocalLoad = dot2aLocalLoad.lookup(dot)) {
        auto result = generatePrologueForLoad(aLocalLoad, "A");
        if (!result.has_value()) {
          return false;
        }
        operand2headPrefetch[dot.getA()] = result.value();
      }

      // Process B operand
      if (auto bLocalLoad = dot2bLocalLoad.lookup(dot)) {
        auto result = generatePrologueForLoad(bLocalLoad, "B");
        if (!result.has_value()) {
          return false;
        }
        operand2headPrefetch[dot.getB()] = result.value();
      }
    }

    return true;
  }

  // Create new ForOp with prefetched values as iter_args
  scf::ForOp createNewForOp() {
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();

    // Collect new loop init args: original + prefetched values
    SmallVector<Value> loopArgs;
    for (auto v : forOp.getInitArgs()) {
      loopArgs.push_back(v);
    }

    // Add prefetched values as new init args
    SmallVector<Value> prefetchedInits;
    for (triton::DotOp dot : dots) {
      if (Value aPrefetch = operand2headPrefetch.lookup(dot.getA())) {
        loopArgs.push_back(aPrefetch);
        prefetchedInits.push_back(aPrefetch);
      }
      if (Value bPrefetch = operand2headPrefetch.lookup(dot.getB())) {
        loopArgs.push_back(bPrefetch);
        prefetchedInits.push_back(bPrefetch);
      }
    }

    if (prefetchedInits.empty()) {
      if (debugEnabled) {
        llvm::errs() << "[RegisterPrefetcher] No prefetched values to add to loop\n";
      }
      return nullptr;
    }

    // Create new ForOp with additional iter_args
    auto newForOp = builder.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        loopArgs);

    // Build mapping from old block args to new block args
    builder.setInsertionPointToStart(newForOp.getBody());
    IRMapping mapping;

    // Map induction variable
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    // Map original iter_args
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
      mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
    }

    // Map prefetched values to their iter_args in new loop
    unsigned prefetchArgIdx = forOp.getRegionIterArgs().size();
    for (triton::DotOp dot : dots) {
      if (operand2headPrefetch.lookup(dot.getA())) {
        Value newIterArg = newForOp.getRegionIterArgs()[prefetchArgIdx++];
        prefetchIterArgMapping[dot.getA()] = newIterArg;
      }
      if (operand2headPrefetch.lookup(dot.getB())) {
        Value newIterArg = newForOp.getRegionIterArgs()[prefetchArgIdx++];
        prefetchIterArgMapping[dot.getB()] = newIterArg;
      }
    }

    // First, set up mappings for LocalLoadOps we're replacing
    for (triton::DotOp dot : dots) {
      if (auto aLocalLoad = dot2aLocalLoad.lookup(dot)) {
        mapping.map(aLocalLoad.getResult(), prefetchIterArgMapping[dot.getA()]);
      }
      if (auto bLocalLoad = dot2bLocalLoad.lookup(dot)) {
        mapping.map(bLocalLoad.getResult(), prefetchIterArgMapping[dot.getB()]);
      }
    }

    // Collect LocalLoadOps to skip
    DenseSet<Operation *> opsToSkip;
    for (triton::DotOp dot : dots) {
      if (auto aLocalLoad = dot2aLocalLoad.lookup(dot)) {
        opsToSkip.insert(aLocalLoad.getOperation());
      }
      if (auto bLocalLoad = dot2bLocalLoad.lookup(dot)) {
        opsToSkip.insert(bLocalLoad.getOperation());
      }
    }

    // Clone loop body operations (except LocalLoadOps we're replacing)
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (opsToSkip.contains(&op)) {
        continue;
      }
      builder.clone(op, mapping);
    }

    // Generate prefetch for next iteration and collect yield values
    SmallVector<Value> yieldValues;

    // Original yield values (mapped)
    for (Value v : yieldOp.getOperands()) {
      yieldValues.push_back(mapping.lookupOrDefault(v));
    }

    // Prefetch for next iteration - use the YIELDED buffer (which is the next iteration's buffer)
    for (triton::DotOp dot : dots) {
      if (auto aLocalLoad = dot2aLocalLoad.lookup(dot)) {
        // Get the yield value for this source and map it
        Value origSrc = aLocalLoad.getSrc();
        Value yieldVal = src2yieldValue.lookup(origSrc);
        if (!yieldVal) {
          // Fallback to mapped current source if no yield value
          yieldVal = origSrc;
        }
        Value mappedYieldVal = mapping.lookupOrDefault(yieldVal);

        // Create LocalLoadOp with mapped yield value (next iteration's buffer)
        Value prefetchNext = builder.create<triton::gpu::LocalLoadOp>(
            loc, aLocalLoad.getType(), mappedYieldVal);
        yieldValues.push_back(prefetchNext);

        if (debugEnabled) {
          llvm::errs() << "[RegisterPrefetcher] Generated next-iteration prefetch for A\n";
        }
      }
      if (auto bLocalLoad = dot2bLocalLoad.lookup(dot)) {
        Value origSrc = bLocalLoad.getSrc();
        Value yieldVal = src2yieldValue.lookup(origSrc);
        if (!yieldVal) {
          yieldVal = origSrc;
        }
        Value mappedYieldVal = mapping.lookupOrDefault(yieldVal);

        Value prefetchNext = builder.create<triton::gpu::LocalLoadOp>(
            loc, bLocalLoad.getType(), mappedYieldVal);
        yieldValues.push_back(prefetchNext);

        if (debugEnabled) {
          llvm::errs() << "[RegisterPrefetcher] Generated next-iteration prefetch for B\n";
        }
      }
    }

    // Create yield with all values
    builder.create<scf::YieldOp>(loc, yieldValues);

    if (debugEnabled) {
      llvm::errs() << "[RegisterPrefetcher] Created new ForOp with "
                   << prefetchedInits.size() << " prefetched iter_args\n";
    }

    return newForOp;
  }

  // Get the DotOps that are being transformed
  const SetVector<triton::DotOp> &getDots() const { return dots; }

private:
  scf::ForOp forOp;
  scf::YieldOp yieldOp;
  bool debugEnabled = false;

  SetVector<triton::DotOp> dots;
  DenseMap<Value, triton::gpu::LocalLoadOp> dot2aLocalLoad;
  DenseMap<Value, triton::gpu::LocalLoadOp> dot2bLocalLoad;
  DenseMap<Value, Value> operand2headPrefetch;  // Original operand → prefetched value
  DenseMap<Value, Value> prefetchIterArgMapping;  // Original operand → iter_arg in new loop
  DenseMap<Value, Value> src2yieldValue;  // LocalLoadOp source → corresponding yield value
};

//===----------------------------------------------------------------------===//
// AdvancedPipelinerPass Implementation
//===----------------------------------------------------------------------===//

struct AdvancedPipelinerPass
    : public impl::TritonGPUAdvancedPipelinerBase<AdvancedPipelinerPass> {

  using impl::TritonGPUAdvancedPipelinerBase<AdvancedPipelinerPass>::TritonGPUAdvancedPipelinerBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Check if debug output is enabled via environment variable
    bool debugEnabled = std::getenv("FLAGTREE_DEBUG_PIPELINE") != nullptr;

    if (debugEnabled) {
      llvm::errs() << "[AdvancedPipeliner] Running on module\n";
    }
    LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] Running on module\n");

    // Process each function in the module - use triton::FuncOp (tt.func), not func::FuncOp
    for (triton::FuncOp function : module.getOps<triton::FuncOp>()) {
      if (debugEnabled) {
        llvm::errs() << "[AdvancedPipeliner] Processing function: " << function.getName() << "\n";
      }
      LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] Processing function: " << function.getName() << "\n");

      // Skip if no pipelining is enabled
      if (globalToSharedStages <= 1 && sharedToRegisterStages <= 1) {
        if (debugEnabled) {
          llvm::errs() << "[AdvancedPipeliner] Pipelining disabled (stages <= 1), skipping\n";
        }
        LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] Pipelining disabled (stages <= 1), skipping\n");
        continue;
      }

      if (debugEnabled) {
        llvm::errs() << "[AdvancedPipeliner] globalStages=" << globalToSharedStages
                     << " registerStages=" << sharedToRegisterStages << "\n";
      }
      LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] globalStages=" << globalToSharedStages
                              << " registerStages=" << sharedToRegisterStages << "\n");

      // Step 1: Run buffer access analysis
      BufferAccessAnalysis accessAnalysis;
      accessAnalysis.analyze(function);

      // Step 2: Detect pipeline opportunities
      if (debugEnabled) {
        llvm::errs() << "[AdvancedPipeliner] Running opportunity detection...\n";
      }
      PipelineOpportunityDetector detector(accessAnalysis);
      auto opportunities = detector.detect(function);

      if (opportunities.empty()) {
        if (debugEnabled) {
          llvm::errs() << "[AdvancedPipeliner] No pipeline opportunities found\n";
        }
        LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] No pipeline opportunities found\n");
        continue;
      }

      if (debugEnabled) {
        llvm::errs() << "[AdvancedPipeliner] Found " << opportunities.size()
                     << " pipeline opportunities\n";
      }
      LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] Found " << opportunities.size()
                              << " pipeline opportunities\n");

      // Step 3: Sort by dependency order (predecessors first)
      sortOpportunitiesByDependency(opportunities);

      // Step 3b: Apply multi-buffer fusion if enabled
      OpBuilder builder(&getContext());
      if (enableMultiBufferFusion) {
        MultiBufferFusion fusion(builder);
        auto groups = fusion.findFusionGroups(opportunities);
        for (auto &group : groups) {
          auto fusionInfo = fusion.apply(group, nextPipelineId++);
          LLVM_DEBUG(llvm::dbgs() << "Applied multi-buffer fusion: "
                                  << group.buffers.size() << " buffers\n");
        }
      }

      // Step 4: Apply transformations
      for (auto &opp : opportunities) {
        // Apply speedup threshold filter
        if (opp.expectedSpeedup < minSpeedup) {
          LLVM_DEBUG(llvm::dbgs() << "Skipping opportunity with speedup "
                                  << opp.expectedSpeedup << " < " << minSpeedup << "\n");
          continue;
        }

        applyPipelineTransformation(opp, accessAnalysis);
      }

      // Step 5: Cleanup
      cleanupUnusedAllocations(function);

      LLVM_DEBUG(llvm::dbgs() << "Advanced pipeliner completed for function: "
                              << function.getName() << "\n");
    }
  }

private:
  unsigned nextPipelineId = 0;
  DenseMap<Value, CircularBufferInfo> circularBuffers;
  DenseMap<unsigned, PipelineInfo> pipelines;
  DenseSet<Operation *> transformedLoopPtrs;  // Track loop Operation* that have been transformed

  void applyPipelineTransformation(PipelineOpportunity &opp,
                                    BufferAccessAnalysis &analysis);
  void sortOpportunitiesByDependency(SmallVector<PipelineOpportunity> &opportunities);
  void generatePrologue(const PipelineOpportunity &opp,
                        CircularBufferInfo &circularInfo,
                        BufferAccessInfo *info);
  void generateEpilogue(const PipelineOpportunity &opp,
                        CircularBufferInfo &circularInfo);
  void cleanupUnusedAllocations(triton::FuncOp function);
  bool verifyIRIntegrity(triton::FuncOp function);

  // Apply loop-carried register prefetching for S2R optimization
  bool applyRegisterPrefetching(scf::ForOp forOp);
};

void AdvancedPipelinerPass::applyPipelineTransformation(
    PipelineOpportunity &opp, BufferAccessAnalysis &analysis) {

  bool debugEnabled = std::getenv("FLAGTREE_DEBUG_PIPELINE") != nullptr;

  // Get the loop pointer early - do NOT dereference if it's been transformed
  Operation *loopPtr = opp.loop ? opp.loop.getOperation() : nullptr;

  // Check if this loop was already transformed by a previous opportunity
  // This check uses pointer comparison only - no dereferencing
  if (loopPtr && transformedLoopPtrs.contains(loopPtr)) {
    if (debugEnabled) {
      llvm::errs() << "[AdvancedPipeliner] Loop already transformed, skipping opportunity\n";
    }
    return;
  }

  OpBuilder builder(&getContext());

  // Get buffer access info from analysis, or create one from the opportunity
  BufferAccessInfo *info = analysis.getAccessInfo(opp.buffer);
  BufferAccessInfo localInfo;

  if (!info) {
    // At TTGIR stage, tt.load doesn't have allocation attribute, so BufferAccessAnalysis
    // won't track it. Create a local BufferAccessInfo from the opportunity.
    bool debugEnabled = std::getenv("FLAGTREE_DEBUG_PIPELINE") != nullptr;
    if (debugEnabled) {
      llvm::errs() << "[AdvancedPipeliner] No analysis info, creating from opportunity\n";
    }

    // For Global→Shared pipeline, find the LoadOp consumers in the loop
    if (opp.level == PipelineLevel::GlobalToShared && opp.loop) {
      opp.loop.getBody()->walk([&](triton::LoadOp loadOp) {
        if (loadOp->getParentOfType<scf::ForOp>() == opp.loop) {
          localInfo.consumers.push_back(loadOp.getOperation());
        }
      });

      if (localInfo.consumers.empty()) {
        if (debugEnabled) {
          llvm::errs() << "[AdvancedPipeliner] No LoadOp consumers found in loop\n";
        }
        return;
      }

      localInfo.scope = MemoryScope::Global;
      localInfo.loopContext = opp.loop;
      localInfo.producer = nullptr;  // Global memory has no explicit producer
      info = &localInfo;

      if (debugEnabled) {
        llvm::errs() << "[AdvancedPipeliner] Created G2S info with "
                     << localInfo.consumers.size() << " consumers\n";
      }
    }
    // For Shared→Register pipeline, find the LocalLoadOp consumers in the loop
    else if (opp.level == PipelineLevel::SharedToRegister && opp.loop) {
      opp.loop.getBody()->walk([&](triton::gpu::LocalLoadOp localLoadOp) {
        if (localLoadOp->getParentOfType<scf::ForOp>() == opp.loop) {
          localInfo.consumers.push_back(localLoadOp.getOperation());
        }
      });

      if (localInfo.consumers.empty()) {
        if (debugEnabled) {
          llvm::errs() << "[AdvancedPipeliner] No LocalLoadOp consumers found in loop\n";
        }
        return;
      }

      localInfo.scope = MemoryScope::Shared;
      localInfo.loopContext = opp.loop;
      localInfo.producer = nullptr;  // Producer is async copy (handled separately)
      info = &localInfo;

      if (debugEnabled) {
        llvm::errs() << "[AdvancedPipeliner] Created S2R info with "
                     << localInfo.consumers.size() << " consumers\n";
      }
    } else {
      if (debugEnabled) {
        llvm::errs() << "[AdvancedPipeliner] Cannot create info for unknown pipeline level\n";
      }
      return;
    }
  }

  // Allocate pipeline ID
  unsigned pipelineId = nextPipelineId++;

  if (debugEnabled) {
    llvm::errs() << "[AdvancedPipeliner] Applying transformation: level="
                 << static_cast<int>(opp.level) << " stages=" << opp.numStages << "\n";
  }
  LLVM_DEBUG(llvm::dbgs() << "Applying pipeline transformation: buffer="
                          << opp.buffer << " pipeline_id=" << pipelineId
                          << " num_stages=" << opp.numStages
                          << " level=" << static_cast<int>(opp.level) << "\n");

  // Step 1: Transform allocation to circular buffer
  CircularBufferTransform circularTransform(builder);
  CircularBufferInfo circularInfo;

  if (opp.level == PipelineLevel::SharedToRegister) {
    // Check if aggressive register prefetching is enabled
    // By default, disabled because shared memory latency is already low
    // and the iter_args overhead often hurts more than it helps
    bool enableAggressiveRegPrefetch = std::getenv("FLAGTREE_AGGRESSIVE_S2R") != nullptr;

    // For S2R, try loop-carried register prefetching first
    // This creates a structural transformation that adds iter_args to the loop
    if (enableAggressiveRegPrefetch && opp.numStages >= 2 && loopPtr) {
      if (debugEnabled) {
        llvm::errs() << "[AdvancedPipeliner] S2R: Attempting loop-carried register prefetching\n";
      }

      // Apply register prefetching transformation
      if (applyRegisterPrefetching(opp.loop)) {
        // Mark this loop as transformed (store the pointer for future comparisons)
        transformedLoopPtrs.insert(loopPtr);

        if (debugEnabled) {
          llvm::errs() << "[AdvancedPipeliner] S2R: Register prefetching applied successfully!\n";
        }
        // Transformation succeeded - the loop has been replaced
        // Skip the rest of the transformation for this opportunity
        return;
      }

      if (debugEnabled) {
        llvm::errs() << "[AdvancedPipeliner] S2R: Register prefetching not applicable, "
                     << "falling back to instruction reordering\n";
      }
    } else if (opp.numStages >= 2 && loopPtr && debugEnabled) {
      llvm::errs() << "[AdvancedPipeliner] S2R: Register prefetching disabled (set FLAGTREE_AGGRESSIVE_S2R=1 to enable)\n";
    }

    // Fallback: use existing buffer without allocation transformation
    circularInfo.originalBuffer = opp.buffer;
    circularInfo.circularBuffer = opp.buffer;
    circularInfo.numStages = opp.numStages;
    circularInfo.loop = opp.loop;
    circularInfo.pipelineId = pipelineId;
    circularInfo.useAsyncCopy = false;
    circularInfo.useSwizzle = false;
    circularInfo.stride = 0;

    if (debugEnabled) {
      llvm::errs() << "[AdvancedPipeliner] S2R: Using existing buffer (no new allocation)\n";
    }
  } else {
    circularInfo = circularTransform.transformAllocation(opp, pipelineId);
  }

  circularBuffers[opp.buffer] = circularInfo;

  // Step 2: Transform stores based on pipeline level
  if (info->producer && opp.level != PipelineLevel::SharedToRegister) {
    // For S2R, skip store transformation (already handled by Triton's pipeline)
    circularTransform.transformStore(info->producer, circularInfo);
  }

  // Step 3: Transform loads based on pipeline level
  for (auto *consumer : info->consumers) {
    if (opp.level == PipelineLevel::SharedToRegister) {
      // For Shared→Register, apply register double-buffering optimization
      // This overlaps shared memory loads with tensor core compute
      if (auto localLoadOp = dyn_cast<triton::gpu::LocalLoadOp>(consumer)) {
        if (debugEnabled) {
          llvm::errs() << "[AdvancedPipeliner] S2R: Processing LocalLoadOp\n";
        }

        // Find the DotOp that consumes this load (may be through convert ops)
        triton::DotOp dotOp = nullptr;
        SmallVector<Operation *, 4> users(localLoadOp->getUsers().begin(),
                                          localLoadOp->getUsers().end());
        while (!users.empty() && !dotOp) {
          Operation *user = users.pop_back_val();
          if (auto dot = dyn_cast<triton::DotOp>(user)) {
            dotOp = dot;
          } else if (isa<triton::gpu::ConvertLayoutOp>(user)) {
            // Follow through convert ops
            for (auto nextUser : user->getUsers()) {
              users.push_back(nextUser);
            }
          }
        }

        if (debugEnabled) {
          llvm::errs() << "[AdvancedPipeliner] S2R: Found dotOp=" << (dotOp ? "yes" : "no")
                       << " numStages=" << opp.numStages << "\n";
        }

        if (dotOp && opp.numStages >= 2) {
          // Apply register double-buffering:
          // 1. Move LocalLoadOp to beginning of loop body
          // 2. This allows load to execute while previous dot is computing

          auto loopBody = opp.loop.getBody();

          // Find a safe position to move the load (after IV computation)
          Operation *insertPoint = nullptr;
          for (auto &op : *loopBody) {
            // Skip block arguments and IV-related ops
            if (isa<arith::ConstantOp, arith::IndexCastOp>(&op)) {
              insertPoint = op.getNextNode();
              continue;
            }
            // Find first "real" operation
            if (!insertPoint) {
              insertPoint = &op;
            }
            break;
          }

          // Check dependencies - can we safely move this load?
          bool canMove = true;
          SmallVector<Operation *> dependentOps;

          for (Value operand : localLoadOp->getOperands()) {
            if (auto defOp = operand.getDefiningOp()) {
              if (defOp->getBlock() == loopBody) {
                // Check if defOp is before the current localLoadOp position
                if (!defOp->isBeforeInBlock(localLoadOp)) {
                  // Operand defined after localLoadOp - need to also move this
                  dependentOps.push_back(defOp);
                }
              }
            }
          }

          // Only move if we have no complex dependencies
          if (canMove && dependentOps.empty() && insertPoint &&
              localLoadOp.getOperation() != insertPoint) {
            // Move load to execute earlier
            localLoadOp->moveBefore(insertPoint);

            if (debugEnabled) {
              llvm::errs() << "[AdvancedPipeliner] S2R: Applied register double-buffering - "
                           << "moved LocalLoadOp earlier for compute overlap\n";
            }
            LLVM_DEBUG(llvm::dbgs() << "S2R: Applied register double-buffering\n");
          } else if (!dependentOps.empty()) {
            // Try to move dependent ops too
            if (debugEnabled) {
              llvm::errs() << "[AdvancedPipeliner] S2R: LocalLoadOp has "
                           << dependentOps.size() << " dependent ops, skipping move\n";
            }
          }
        } else if (dotOp) {
          // Fallback: just move the load earlier if possible
          auto loopBody = opp.loop.getBody();
          Operation *firstOp = &loopBody->front();
          if (localLoadOp.getOperation() != firstOp) {
            // Check if we can move to beginning
            bool canMove = true;
            for (Value operand : localLoadOp->getOperands()) {
              if (auto defOp = operand.getDefiningOp()) {
                if (defOp->getBlock() == loopBody) {
                  canMove = false;
                  break;
                }
              }
            }
            if (canMove) {
              localLoadOp->moveBefore(firstOp);
              if (debugEnabled) {
                llvm::errs() << "[AdvancedPipeliner] S2R: Moved LocalLoadOp to loop start\n";
              }
            }
          }
        }
      }
    } else {
      // For Global→Shared, transform LoadOp with async copy
      if (auto loadOp = dyn_cast<triton::LoadOp>(consumer)) {
        if (opp.useAsyncCopy && circularInfo.numStages > 1) {
          // Compute insert/extract indices for circular buffer
          // insertIdx = (iter + numStages - 1) % numStages  (producer writes ahead)
          // extractIdx = iter % numStages  (consumer reads current)
          Location loc = loadOp.getLoc();
          builder.setInsertionPoint(loadOp);

          // Get loop induction variable
          Value iv = circularInfo.loop.getInductionVar();
          Value lb = circularInfo.loop.getLowerBound();
          Value step = circularInfo.loop.getStep();

          // Compute iteration: iter = (iv - lb) / step
          Value diff = builder.create<arith::SubIOp>(loc, iv, lb);
          Value iter = builder.create<arith::DivSIOp>(loc, diff, step);

          // Ensure we have i32 type for index computation
          Type i32Type = builder.getI32Type();
          Value iter32;
          if (iter.getType().isIndex()) {
            iter32 = builder.create<arith::IndexCastOp>(loc, i32Type, iter);
          } else if (iter.getType() != i32Type) {
            iter32 = builder.create<arith::TruncIOp>(loc, i32Type, iter);
          } else {
            iter32 = iter;  // Already i32
          }

          Value numStages32 = builder.create<arith::ConstantIntOp>(loc, circularInfo.numStages, 32);
          Value one32 = builder.create<arith::ConstantIntOp>(loc, 1, 32);

          // insertIdx = (iter + numStages - 1) % numStages
          Value insertSum = builder.create<arith::AddIOp>(loc, iter32, numStages32);
          insertSum = builder.create<arith::SubIOp>(loc, insertSum, one32);
          Value insertIdx = builder.create<arith::RemSIOp>(loc, insertSum, numStages32);

          // extractIdx = iter % numStages
          Value extractIdx = builder.create<arith::RemSIOp>(loc, iter32, numStages32);

          // Transform using async copy
          circularTransform.transformGlobalLoad(loadOp, circularInfo, insertIdx, extractIdx);
          LLVM_DEBUG(llvm::dbgs() << "Transformed LoadOp with async copy for Global→Shared pipeline\n");
        } else {
          // Fallback: use simple load transformation (no async)
          circularTransform.transformLoad(loadOp, circularInfo);
        }
      }
    }
  }

  // Step 4: Insert synchronization
  SynchronizationInsertion syncInsertion(builder);
  syncInsertion.registerPipeline(pipelineId, circularInfo, opp);
  syncInsertion.insertSynchronization(opp, circularInfo, info);

  // Step 5: Apply warp specialization if beneficial
  WarpSpecialization warpSpec(builder);
  if (enableWarpSpecialization && warpSpec.isProfitable(opp, circularInfo)) {
    auto warpInfo = warpSpec.apply(opp, circularInfo, pipelineId);
    LLVM_DEBUG(llvm::dbgs() << "Applied warp specialization: "
                            << warpInfo.config.numProducerWarps << " producers, "
                            << warpInfo.config.numConsumerWarps << " consumers\n");
  }

  // Step 5b: Apply TMA optimization if available and beneficial (Hopper+)
  TMASupport tmaSupport(builder);
  if (enableAsyncCopy && tmaSupport.isProfitable(opp, circularInfo)) {
    auto tmaInfo = tmaSupport.apply(opp, circularInfo, pipelineId);
    if (!tmaInfo.descriptors.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Applied TMA transformation: "
                              << tmaInfo.descriptors.size() << " transfers\n");
    }
  }

  // Step 6: Generate prologue (warm-up loop)
  generatePrologue(opp, circularInfo, info);

  // Step 7: Generate epilogue (drain pipeline)
  generateEpilogue(opp, circularInfo);

  // Register pipeline info
  PipelineInfo pipelineInfo;
  pipelineInfo.pipelineId = pipelineId;
  pipelineInfo.buffers.push_back(circularInfo.circularBuffer);
  pipelineInfo.loop = circularInfo.loop;
  pipelineInfo.numStages = circularInfo.numStages;
  pipelineInfo.scope = (opp.level == PipelineLevel::SharedToRegister) ? "register" : "shared";
  pipelineInfo.canFuseSync = false;
  pipelines[pipelineId] = pipelineInfo;

  LLVM_DEBUG(llvm::dbgs() << "Pipeline transformation applied successfully\n");
}

void AdvancedPipelinerPass::sortOpportunitiesByDependency(
    SmallVector<PipelineOpportunity> &opportunities) {

  // Build dependency graph
  DenseMap<Value, SmallVector<Value>> dependencies;

  for (auto &opp : opportunities) {
    if (opp.predecessorBuffer) {
      dependencies[opp.buffer].push_back(opp.predecessorBuffer);
    }
  }

  // Topological sort (simplified - assumes no cycles)
  SmallVector<PipelineOpportunity> sorted;
  DenseSet<Value> processed;

  // Process opportunities without dependencies first
  for (auto &opp : opportunities) {
    if (!dependencies.count(opp.buffer) ||
        dependencies[opp.buffer].empty()) {
      sorted.push_back(opp);
      processed.insert(opp.buffer);
    }
  }

  // Process remaining opportunities
  bool changed = true;
  while (changed && sorted.size() < opportunities.size()) {
    changed = false;

    for (auto &opp : opportunities) {
      if (processed.count(opp.buffer)) {
        continue;
      }

      // Check if all dependencies are processed
      bool allDepsProcessed = true;
      if (dependencies.count(opp.buffer)) {
        for (auto dep : dependencies[opp.buffer]) {
          if (!processed.count(dep)) {
            allDepsProcessed = false;
            break;
          }
        }
      }

      if (allDepsProcessed) {
        sorted.push_back(opp);
        processed.insert(opp.buffer);
        changed = true;
      }
    }
  }

  // Replace with sorted list
  opportunities = std::move(sorted);

  LLVM_DEBUG(llvm::dbgs() << "Sorted opportunities by dependency\n");
}

void AdvancedPipelinerPass::generatePrologue(
    const PipelineOpportunity &opp, CircularBufferInfo &circularInfo,
    BufferAccessInfo *info) {

  // TODO: Prologue generation has type mismatch issues between index and i32 loop bounds.
  // Skip prologue for now - rely on Triton's built-in pipeline to handle prologue/epilogue.
  // The main transformation (async copy) will still provide performance benefits.
  LLVM_DEBUG(llvm::dbgs() << "Skipping prologue generation (not yet implemented for mixed types)\n");
  return;

  if (!circularInfo.loop || circularInfo.numStages <= 1) {
    return;
  }

  OpBuilder builder(&getContext());
  builder.setInsertionPoint(circularInfo.loop);
  Location loc = circularInfo.loop->getLoc();

  // Prologue warms up pipeline by pre-loading (numStages - 1) iterations
  unsigned prologueIters = circularInfo.numStages - 1;

  // Collect producer operations to clone
  SmallVector<Operation *> producerOps;
  if (info && info->producer) {
    // Collect the producer and its dependent operations within the loop body
    Operation *producerOp = info->producer;

    // Walk backwards to find all operations that produce values used by the producer
    SmallVector<Operation *> workList;
    DenseSet<Operation *> visited;
    workList.push_back(producerOp);

    while (!workList.empty()) {
      Operation *op = workList.pop_back_val();
      if (visited.count(op))
        continue;
      visited.insert(op);

      // Only include operations within the loop body
      if (op->getParentOp() != circularInfo.loop.getBody()->getParentOp())
        continue;

      producerOps.push_back(op);

      // Add defining operations of operands
      for (Value operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          if (!visited.count(defOp)) {
            workList.push_back(defOp);
          }
        }
      }
    }

    // Reverse to maintain topological order (definitions before uses)
    std::reverse(producerOps.begin(), producerOps.end());
  }

  // Create prologue loop: for (i = 0; i < numStages-1; i++)
  Value lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value upperBound =
      builder.create<arith::ConstantIndexOp>(loc, prologueIters);
  Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

  // Get original loop bounds to compute actual iteration index
  Value origLowerBound = circularInfo.loop.getLowerBound();
  Value origStep = circularInfo.loop.getStep();

  auto prologueLoop = builder.create<scf::ForOp>(
      loc, lowerBound, upperBound, step, ValueRange{},
      [&](OpBuilder &b, Location innerLoc, Value iv, ValueRange iterArgs) {
        // Create IRMapping to substitute loop induction variable
        IRMapping mapping;

        // Map prologue iv to actual loop iteration:
        // actual_iv = orig_lower + iv * orig_step
        // Handle type mismatch: prologue iv is index, orig bounds might be i32
        Type origStepType = origStep.getType();
        Value ivCasted = iv;
        if (ivCasted.getType() != origStepType) {
          // Cast prologue iv to the type of the original loop
          if (origStepType.isIndex()) {
            ivCasted = b.create<arith::IndexCastOp>(innerLoc, origStepType, iv);
          } else {
            // Cast index to integer type
            ivCasted = b.create<arith::IndexCastOp>(innerLoc, origStepType, iv);
          }
        }
        Value actualIv = b.create<arith::MulIOp>(innerLoc, ivCasted, origStep);
        actualIv = b.create<arith::AddIOp>(innerLoc, origLowerBound, actualIv);

        // Map original loop's induction variable to computed actual_iv
        mapping.map(circularInfo.loop.getInductionVar(), actualIv);

        // Map original buffer to circular buffer
        if (circularInfo.originalBuffer && circularInfo.circularBuffer) {
          mapping.map(circularInfo.originalBuffer, circularInfo.circularBuffer);
        }

        // Clone producer operations with mapping
        for (Operation *op : producerOps) {
          // Skip if it's a terminator or yield
          if (op->hasTrait<OpTrait::IsTerminator>())
            continue;

          b.clone(*op, mapping);
        }

        b.create<scf::YieldOp>(innerLoc);
      });

  LLVM_DEBUG(llvm::dbgs() << "Generated prologue with " << prologueIters
                          << " iterations, cloned " << producerOps.size()
                          << " producer operations\n");
}

void AdvancedPipelinerPass::generateEpilogue(
    const PipelineOpportunity &opp, CircularBufferInfo &circularInfo) {

  // Epilogue is handled by pipeline flush in synchronization
  // No additional code generation needed

  LLVM_DEBUG(llvm::dbgs() << "Epilogue handled by pipeline flush\n");
}

void AdvancedPipelinerPass::cleanupUnusedAllocations(triton::FuncOp function) {
  // Remove operations marked for deletion
  function.walk([&](Operation *op) {
    if (op->hasAttr("to_delete")) {
      op->erase();
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Cleaned up unused allocations\n");
}

bool AdvancedPipelinerPass::applyRegisterPrefetching(scf::ForOp forOp) {
  bool debugEnabled = std::getenv("FLAGTREE_DEBUG_PIPELINE") != nullptr;

  if (debugEnabled) {
    llvm::errs() << "[AdvancedPipeliner] Attempting register prefetching transformation\n";
  }

  RegisterPrefetcher prefetcher(forOp);

  // Initialize: find LocalLoadOps that feed DotOps
  if (prefetcher.initialize().failed()) {
    if (debugEnabled) {
      llvm::errs() << "[AdvancedPipeliner] RegisterPrefetcher initialization failed\n";
    }
    return false;
  }

  // Generate prologue: prefetch first iteration before loop
  if (!prefetcher.emitPrologue()) {
    if (debugEnabled) {
      llvm::errs() << "[AdvancedPipeliner] RegisterPrefetcher prologue generation failed\n";
    }
    return false;
  }

  // Create new ForOp with prefetched values as iter_args
  scf::ForOp newForOp = prefetcher.createNewForOp();
  if (!newForOp) {
    if (debugEnabled) {
      llvm::errs() << "[AdvancedPipeliner] Failed to create new ForOp\n";
    }
    return false;
  }

  // Replace the original loop with the new one
  // Only replace the results that the original loop produced
  unsigned numOrigResults = forOp->getNumResults();
  for (unsigned i = 0; i < numOrigResults; ++i) {
    forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
  }

  // Erase the old loop
  forOp->erase();

  if (debugEnabled) {
    llvm::errs() << "[AdvancedPipeliner] Successfully applied register prefetching!\n";
  }

  LLVM_DEBUG(llvm::dbgs() << "Applied loop-carried register prefetching\n");
  return true;
}

bool AdvancedPipelinerPass::verifyIRIntegrity(triton::FuncOp function) {
  // Basic verification: check that all uses are defined
  bool valid = true;

  function.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (!operand.getDefiningOp() && !mlir::isa<BlockArgument>(operand)) {
        LLVM_DEBUG(llvm::dbgs() << "ERROR: Undefined operand in " << *op
                                << "\n");
        valid = false;
      }
    }
  });

  // Verify synchronization pairing
  // Collect all synchronization operations
  SmallVector<func::CallOp> acquireOps;
  SmallVector<func::CallOp> commitOps;
  SmallVector<func::CallOp> waitOps;
  SmallVector<func::CallOp> releaseOps;

  function.walk([&](func::CallOp callOp) {
    StringRef callee = callOp.getCallee();
    if (callee == "triton_gpu.pipeline_producer_acquire") {
      acquireOps.push_back(callOp);
    } else if (callee == "triton_gpu.pipeline_producer_commit") {
      commitOps.push_back(callOp);
    } else if (callee == "triton_gpu.pipeline_consumer_wait") {
      waitOps.push_back(callOp);
    } else if (callee == "triton_gpu.pipeline_consumer_release") {
      releaseOps.push_back(callOp);
    }
  });

  // Verify acquire/commit pairing
  if (acquireOps.size() != commitOps.size()) {
    LLVM_DEBUG(llvm::dbgs() << "ERROR: Mismatched acquire/commit count: "
                            << acquireOps.size() << " acquires vs "
                            << commitOps.size() << " commits\n");
    valid = false;
  }

  // Verify wait/release pairing
  if (waitOps.size() != releaseOps.size()) {
    LLVM_DEBUG(llvm::dbgs() << "ERROR: Mismatched wait/release count: "
                            << waitOps.size() << " waits vs "
                            << releaseOps.size() << " releases\n");
    valid = false;
  }

  // Verify proper nesting of barriers within each pipeline
  for (const auto &pipelinePair : pipelines) {
    const PipelineInfo &pipelineInfo = pipelinePair.second;
    scf::ForOp loop = pipelineInfo.loop;
    if (!loop) {
      continue;
    }

    // Verify barriers are within the loop body
    bool hasProducerBarriers = false;
    bool hasConsumerBarriers = false;

    loop.getBody()->walk([&](func::CallOp callOp) {
      StringRef callee = callOp.getCallee();
      if (callee == "triton_gpu.pipeline_producer_acquire" ||
          callee == "triton_gpu.pipeline_producer_commit") {
        hasProducerBarriers = true;
      } else if (callee == "triton_gpu.pipeline_consumer_wait" ||
                 callee == "triton_gpu.pipeline_consumer_release") {
        hasConsumerBarriers = true;
      }
    });

    // Verify dominance: acquire should dominate commit within the same block
    for (auto acquireOp : acquireOps) {
      Block *acquireBlock = acquireOp->getBlock();
      bool foundMatchingCommit = false;

      for (auto commitOp : commitOps) {
        if (commitOp->getBlock() == acquireBlock) {
          // Check that acquire comes before commit in the same block
          if (acquireOp->isBeforeInBlock(commitOp)) {
            foundMatchingCommit = true;
            break;
          }
        }
      }

      if (!foundMatchingCommit && !acquireOps.empty()) {
        // Acquire in different blocks is allowed for nested control flow
        bool hasCommitInNestedRegion = false;
        for (auto commitOp : commitOps) {
          if (acquireOp->getParentRegion()->isAncestor(
                  commitOp->getParentRegion())) {
            hasCommitInNestedRegion = true;
            break;
          }
        }
        if (!hasCommitInNestedRegion) {
          LLVM_DEBUG(llvm::dbgs()
                     << "WARNING: Acquire without matching commit in scope\n");
        }
      }
    }

    // Verify dominance: wait should dominate release
    for (auto waitOp : waitOps) {
      Block *waitBlock = waitOp->getBlock();
      bool foundMatchingRelease = false;

      for (auto releaseOp : releaseOps) {
        if (releaseOp->getBlock() == waitBlock) {
          if (waitOp->isBeforeInBlock(releaseOp)) {
            foundMatchingRelease = true;
            break;
          }
        }
      }

      if (!foundMatchingRelease && !waitOps.empty()) {
        bool hasReleaseInNestedRegion = false;
        for (auto releaseOp : releaseOps) {
          if (waitOp->getParentRegion()->isAncestor(
                  releaseOp->getParentRegion())) {
            hasReleaseInNestedRegion = true;
            break;
          }
        }
        if (!hasReleaseInNestedRegion) {
          LLVM_DEBUG(llvm::dbgs()
                     << "WARNING: Wait without matching release in scope\n");
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Pipeline " << pipelineInfo.pipelineId
                            << ": producer_barriers=" << hasProducerBarriers
                            << ", consumer_barriers=" << hasConsumerBarriers
                            << "\n");
  }

  if (valid) {
    LLVM_DEBUG(llvm::dbgs() << "IR integrity verified\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "IR integrity check FAILED\n");
  }

  return valid;
}

} // anonymous namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
