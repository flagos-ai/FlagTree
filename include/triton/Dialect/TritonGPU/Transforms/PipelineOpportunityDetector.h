#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINEOPPORTUNITYDETECTOR_H
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINEOPPORTUNITYDETECTOR_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/BufferAccessAnalysis.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace gpu {

/// Pipeline hierarchy level
enum class PipelineLevel {
  GlobalToShared,    /// Global memory → Shared memory
  SharedToRegister,  /// Shared memory → Registers
  GlobalToRegister   /// Global memory → Registers (direct)
};

/// Represents a detected pipelining opportunity
struct PipelineOpportunity {
  /// Loop to pipeline
  scf::ForOp loop;

  /// Buffer to pipeline
  Value buffer;

  /// Memory hierarchy level
  PipelineLevel level;

  /// Recommended number of pipeline stages
  unsigned numStages;

  /// Expected performance speedup (multiplicative factor)
  double expectedSpeedup;

  /// Predecessor buffer (if chained pipeline)
  Value predecessorBuffer;

  /// Whether to use async copy intrinsics
  bool useAsyncCopy;

  /// Whether to use swizzled indexing
  bool useSwizzle;

  PipelineOpportunity()
      : loop(nullptr), buffer(nullptr), level(PipelineLevel::GlobalToShared),
        numStages(1), expectedSpeedup(1.0), predecessorBuffer(nullptr),
        useAsyncCopy(false), useSwizzle(false) {}
};

/// Detector for finding profitable pipelining opportunities
class PipelineOpportunityDetector {
public:
  explicit PipelineOpportunityDetector(BufferAccessAnalysis &analysis)
      : analysis(analysis) {}

  /// Detect all pipeline opportunities in a function
  SmallVector<PipelineOpportunity> detect(triton::FuncOp function);

private:
  /// Reference to buffer access analysis
  BufferAccessAnalysis &analysis;

  /// Check if a buffer access pattern is suitable for pipelining
  bool isPipelinable(Value buffer, BufferAccessInfo *info);

  /// Determine the pipeline level based on memory scopes
  PipelineLevel determinePipelineLevel(BufferAccessInfo *info);

  /// Estimate optimal number of pipeline stages
  unsigned estimateNumStages(scf::ForOp loop, BufferAccessInfo *info);

  /// Estimate expected performance improvement
  double estimateSpeedup(PipelineOpportunity &opp);
  double estimateSpeedup(PipelineOpportunity &opp, BufferAccessInfo *info);

  /// Determine if async copy should be used
  bool shouldUseAsyncCopy(BufferAccessInfo *info);

  /// Determine if swizzling should be used
  bool shouldUseSwizzle(BufferAccessInfo *info);

  /// Helper: Get loop extent (trip count) if constant
  std::optional<int64_t> getLoopExtent(scf::ForOp loop);

  /// Helper: Estimate memory latency in cycles
  double estimateMemoryLatency(MemoryScope scope, int64_t elementCount);

  /// Helper: Estimate compute time per iteration in cycles
  double estimateComputeTime(scf::ForOp loop, BufferAccessInfo *info);

  /// Helper: Estimate register pressure impact
  double estimateRegisterPressure(PipelineOpportunity &opp);
};

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINEOPPORTUNITYDETECTOR_H
