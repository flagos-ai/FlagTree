#ifndef TRITON_METAX_GLUON_C500_LAYOUT_RULES_H
#define TRITON_METAX_GLUON_C500_LAYOUT_RULES_H

#include "Gluon/Analysis/GluonDotAnalysis.h"
#include "Gluon/GluonLayoutCandidate.h"

#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::gpu::metax::gluon {

/// A dot is tunable when its C/D accumulator contract is Auto and its operand
/// paths do not use the fixed C500 BSM permutation ABI. A/B may be fixed; the
/// selected requirement then materializes as a local conversion boundary.
bool isTunableC500Dot(const DotPipelineFacts &facts);

/// Populate only the target-owned physical-order facts needed by C500 dot
/// construction. Logical value paths and shared alias paths are collected by
/// the machine-independent dot analysis.
LogicalResult inferC500DotMemoryFacts(
    ModuleOp module, triton::ModuleAxisInfoAnalysis &axisInfo,
    SmallVectorImpl<DotPipelineFacts> &pipelines);

/// Generate a finite domain of complete, individually verified C500 dot
/// contracts. Entry zero is the deterministic fallback.
FailureOr<SmallVector<DotLayoutPlan, 8>>
inferC500DotLayoutDomain(const DotPipelineFacts &facts,
                         int computeCapability);

/// Derive the concrete shared-memory view required by one local_load feeding a
/// selected dot operand. This query does not mutate IR.
FailureOr<Attribute>
inferC500DotSharedEncoding(Attribute dotOperandEncoding,
                           triton::gpu::LocalLoadOp localLoad,
                           const DotOperandPathFacts &path);

/// Compare accumulator ownership while intentionally ignoring load-policy
/// fields such as isATrans/isBTrans. Those fields are re-derived per dot from
/// the proven A/B physical order.
bool haveSameC500AccumulatorProfile(Attribute lhs, Attribute rhs);

} // namespace mlir::triton::gpu::metax::gluon

#endif // TRITON_METAX_GLUON_C500_LAYOUT_RULES_H
