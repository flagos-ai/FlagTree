//===- PipelineOpportunityDetector.cpp - Detect Pipelining Opportunities -===//
//
// This file implements detection of profitable pipelining opportunities
// in Triton GPU kernels at the TTGIR stage.
//
// FIXED: Now detects scf::ForOp + triton::LoadOp patterns which exist at TTGIR.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/PipelineOpportunityDetector.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cmath>

#define DEBUG_TYPE "pipeline-opportunity-detector"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// PipelineOpportunityDetector Implementation - FIXED for TTGIR stage
//===----------------------------------------------------------------------===//

SmallVector<PipelineOpportunity>
PipelineOpportunityDetector::detect(triton::FuncOp function) {
  SmallVector<PipelineOpportunity> opportunities;

  bool debugEnabled = std::getenv("FLAGTREE_DEBUG_PIPELINE") != nullptr;

  LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] Detecting opportunities in function: "
                          << function.getName() << "\n");

  function.walk([&](scf::ForOp forOp) {
    LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] Found ForOp\n");

    // ==== Phase 1: Detect Global→Shared opportunities (triton::LoadOp) ====
    SmallVector<triton::LoadOp> globalLoadsInLoop;
    forOp.getBody()->walk([&](triton::LoadOp loadOp) {
      if (loadOp->getParentOfType<scf::ForOp>() == forOp) {
        globalLoadsInLoop.push_back(loadOp);
      }
    });

    // ==== Phase 2: Detect Shared→Register opportunities (LocalLoadOp) ====
    // This runs AFTER Triton's pipeline has converted LoadOp→AsyncCopyGlobalToLocalOp+LocalLoadOp
    SmallVector<triton::gpu::LocalLoadOp> localLoadsInLoop;
    forOp.getBody()->walk([&](triton::gpu::LocalLoadOp localLoadOp) {
      if (localLoadOp->getParentOfType<scf::ForOp>() == forOp) {
        localLoadsInLoop.push_back(localLoadOp);
      }
    });

    if (debugEnabled) {
      llvm::errs() << "[AdvancedPipeliner] Loop has " << globalLoadsInLoop.size()
                   << " global loads, " << localLoadsInLoop.size() << " local loads\n";
    }

    if (globalLoadsInLoop.empty() && localLoadsInLoop.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] No loads found in loop\n");
      return;
    }

    // Check if loads feed into compute operations (DotOp)
    bool hasDotConsumer = false;
    forOp.getBody()->walk([&](triton::DotOp dotOp) {
      hasDotConsumer = true;
    });

    if (!hasDotConsumer) {
      LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] No DotOp consumer, skipping\n");
      return;
    }

    // Get loop extent
    auto loopExtent = getLoopExtent(forOp);
    if (!loopExtent || *loopExtent < 3) {
      LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] Loop extent too small\n");
      return;
    }

    // Create Global→Shared opportunities for triton::LoadOp
    for (auto loadOp : globalLoadsInLoop) {
      Value ptr = loadOp.getPtr();

      BufferAccessInfo info;
      info.scope = MemoryScope::Global;
      info.loopContext = forOp;
      info.producer = nullptr;
      info.consumers.push_back(loadOp.getOperation());

      if (auto tensorType = dyn_cast<RankedTensorType>(loadOp.getType())) {
        int64_t elements = 1;
        for (int64_t dim : tensorType.getShape()) {
          elements *= dim;
        }
        info.elementCount = elements;
        info.elementType = tensorType.getElementType();
      }

      PipelineOpportunity opp;
      opp.buffer = ptr;
      opp.loop = forOp;
      opp.level = PipelineLevel::GlobalToShared;
      opp.numStages = estimateNumStages(forOp, &info);
      opp.useAsyncCopy = true;
      opp.useSwizzle = true;
      opp.expectedSpeedup = estimateSpeedup(opp, &info);

      if (opp.expectedSpeedup > 1.05) {
        opportunities.push_back(opp);
        if (debugEnabled) {
          llvm::errs() << "[AdvancedPipeliner] Found G2S opportunity: stages="
                       << opp.numStages << " speedup=" << opp.expectedSpeedup << "x\n";
        }
      }
    }

    // Create Shared→Register opportunities for LocalLoadOp
    for (auto localLoadOp : localLoadsInLoop) {
      Value src = localLoadOp.getSrc();

      BufferAccessInfo info;
      info.scope = MemoryScope::Shared;  // Source is shared memory
      info.loopContext = forOp;
      info.producer = nullptr;  // Producer is the async copy
      info.consumers.push_back(localLoadOp.getOperation());

      if (auto tensorType = dyn_cast<RankedTensorType>(localLoadOp.getType())) {
        int64_t elements = 1;
        for (int64_t dim : tensorType.getShape()) {
          elements *= dim;
        }
        info.elementCount = elements;
        info.elementType = tensorType.getElementType();
      }

      PipelineOpportunity opp;
      opp.buffer = src;  // Shared memory buffer
      opp.loop = forOp;
      opp.level = PipelineLevel::SharedToRegister;
      opp.numStages = 2;  // Double-buffering for S2R
      opp.useAsyncCopy = false;  // No async copy for S2R
      opp.useSwizzle = false;    // Swizzle already applied at allocation
      opp.expectedSpeedup = 1.1; // ~10% speedup from register prefetching

      opportunities.push_back(opp);
      if (debugEnabled) {
        llvm::errs() << "[AdvancedPipeliner] Found S2R opportunity: stages="
                     << opp.numStages << " speedup=" << opp.expectedSpeedup << "x\n";
      }
    }
  });

  // Sort by expected speedup (highest first)
  std::sort(opportunities.begin(), opportunities.end(),
            [](const PipelineOpportunity &a, const PipelineOpportunity &b) {
              return a.expectedSpeedup > b.expectedSpeedup;
            });

  LLVM_DEBUG(llvm::dbgs() << "[AdvancedPipeliner] Total opportunities: "
                          << opportunities.size() << "\n");

  return opportunities;
}

bool PipelineOpportunityDetector::isPipelinable(Value buffer,
                                                 BufferAccessInfo *info) {
  if (!info || !info->loopContext) {
    return false;
  }

  auto loopExtent = getLoopExtent(info->loopContext);
  if (!loopExtent || *loopExtent < 3) {
    return false;
  }

  return true;
}

PipelineLevel
PipelineOpportunityDetector::determinePipelineLevel(BufferAccessInfo *info) {
  if (!info) {
    return PipelineLevel::GlobalToShared;
  }

  if (info->scope == MemoryScope::Shared) {
    return PipelineLevel::SharedToRegister;
  }

  return PipelineLevel::GlobalToShared;
}

unsigned
PipelineOpportunityDetector::estimateNumStages(scf::ForOp loop,
                                                BufferAccessInfo *info) {
  auto extent = getLoopExtent(loop);
  if (!extent) {
    return 3;  // Default
  }

  // Estimate based on loop extent and memory latency
  // More stages for longer loops, but cap at 5
  unsigned stages = 3;

  if (*extent >= 64) {
    stages = 4;
  }
  if (*extent >= 128) {
    stages = 5;
  }

  // Don't exceed loop extent
  stages = std::min(stages, static_cast<unsigned>(*extent));

  return stages;
}

double PipelineOpportunityDetector::estimateSpeedup(
    PipelineOpportunity &opp, BufferAccessInfo *info) {

  // Simplified speedup model for TTGIR stage
  // Base speedup from pipelining
  double baseSpeedup = 1.0;

  if (opp.numStages >= 2) {
    baseSpeedup = 1.1;  // 10% base improvement
  }
  if (opp.numStages >= 3) {
    baseSpeedup = 1.2;  // 20% improvement
  }
  if (opp.numStages >= 4) {
    baseSpeedup = 1.25; // 25% improvement
  }

  // Additional benefit from async copy
  if (opp.useAsyncCopy) {
    baseSpeedup *= 1.05;  // 5% additional
  }

  // Additional benefit from swizzle
  if (opp.useSwizzle) {
    baseSpeedup *= 1.02;  // 2% additional
  }

  return baseSpeedup;
}

// Legacy method - redirects to new implementation
double PipelineOpportunityDetector::estimateSpeedup(PipelineOpportunity &opp) {
  return estimateSpeedup(opp, nullptr);
}

bool PipelineOpportunityDetector::shouldUseAsyncCopy(BufferAccessInfo *info) {
  if (!info) return true;
  return info->scope == MemoryScope::Global;
}

bool PipelineOpportunityDetector::shouldUseSwizzle(BufferAccessInfo *info) {
  if (!info) return true;

  // Enable swizzle for shared memory buffers
  if (info->scope == MemoryScope::Shared) {
    return true;
  }

  // Enable for large buffers
  if (info->elementCount >= 1024) {
    return true;
  }

  return false;
}

std::optional<int64_t>
PipelineOpportunityDetector::getLoopExtent(scf::ForOp loop) {
  if (!loop) {
    return std::nullopt;
  }

  auto lowerBound = loop.getLowerBound();
  auto upperBound = loop.getUpperBound();
  auto step = loop.getStep();

  // Try to extract constant bounds
  auto getConstantValue = [](Value v) -> std::optional<int64_t> {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        return intAttr.getInt();
      }
    }
    return std::nullopt;
  };

  auto lb = getConstantValue(lowerBound);
  auto ub = getConstantValue(upperBound);
  auto s = getConstantValue(step);

  if (lb && ub && s && *s > 0) {
    return (*ub - *lb + *s - 1) / *s;
  }

  // If bounds are not constant, estimate based on typical GEMM sizes
  // K dimension is usually >= 32
  return 32;
}

double PipelineOpportunityDetector::estimateMemoryLatency(
    MemoryScope scope, int64_t elementCount) {
  constexpr double clockFrequency = 1.4e9;
  int64_t bytesTransferred = elementCount * 2;  // fp16

  switch (scope) {
  case MemoryScope::Global: {
    constexpr double bandwidth = 1000e9;  // 1 TB/s
    double transferTime = bytesTransferred / bandwidth * clockFrequency;
    return 500.0 + transferTime;
  }
  case MemoryScope::Shared:
    return 25.0;
  case MemoryScope::Register:
    return 1.0;
  default:
    return 100.0;
  }
}

double PipelineOpportunityDetector::estimateComputeTime(
    scf::ForOp loop, BufferAccessInfo *info) {
  int64_t totalOps = 0;

  loop.getBody()->walk([&](Operation *op) {
    if (isa<arith::MulFOp, arith::AddFOp, arith::SubFOp>(op)) {
      totalOps += 1;
    } else if (isa<triton::DotOp>(op)) {
      totalOps += 100;
    }
  });

  return std::max(totalOps / 4.0, 10.0);
}

double PipelineOpportunityDetector::estimateRegisterPressure(
    PipelineOpportunity &opp) {
  // Conservative estimate
  int64_t estimatedRegs = 64 + opp.numStages * 16;

  if (estimatedRegs > 128) {
    return 128.0 / estimatedRegs;
  }

  return 1.0;
}
