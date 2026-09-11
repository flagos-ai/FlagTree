#ifndef TRITON_METAX_GLUON_DOT_ANALYSIS_H
#define TRITON_METAX_GLUON_DOT_ANALYSIS_H

#include "Gluon/Analysis/GluonMemDescAliasAnalysis.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <array>
#include <optional>

namespace mlir::triton::gpu::metax::gluon {

struct DotOperandMemoryFacts {
  bool hasOrderSources = false;
  std::optional<unsigned> contiguousDimension;
  unsigned maxContiguousElements = 0;

  bool hasProvenPhysicalOrder() const {
    return contiguousDimension.has_value();
  }
};

struct DotOperandPathFacts {
  bool passesThroughBsmPermutation = false;
  bool operator==(const DotOperandPathFacts &other) const {
    return passesThroughBsmPermutation ==
           other.passesThroughBsmPermutation;
  }
};

struct DotPipelineLocalLoadFacts {
  triton::gpu::LocalLoadOp localLoad;
  MemDescAliasPath sharedView;
};

struct DotPipelineOperandFacts {
  unsigned logicalKDimension = 0;
  DotOperandMemoryFacts memory;
  DotOperandPathFacts path;
  SmallVector<Operation *, 1> bsmPermutations;
  SmallVector<Value, 2> terminals;
  SmallVector<DotPipelineLocalLoadFacts, 2> localLoads;
  bool hasDotProducer = false;

  std::optional<bool> isKContiguous() const {
    if (!memory.hasProvenPhysicalOrder())
      return std::nullopt;
    return *memory.contiguousDimension == logicalKDimension;
  }
  bool requiresBsmPermutation() const {
    return path.passesThroughBsmPermutation;
  }
};

/// All machine-independent facts for one actual tt.dot. Target encodings and
/// candidate domains are intentionally absent.
struct DotPipelineFacts {
  triton::DotOp dot;
  std::array<DotPipelineOperandFacts, 2> operands;
};

/// Collect each dot's region-aware value path and logical shared alias path in
/// one read-only pass.
FailureOr<SmallVector<DotPipelineFacts, 4>>
collectDotPipelineFacts(ModuleOp module,
                        const GluonMemDescAliasAnalysis &aliases);

} // namespace mlir::triton::gpu::metax::gluon

#endif
