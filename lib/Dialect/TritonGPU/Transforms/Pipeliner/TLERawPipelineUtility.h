// Copyright 2025- FlagOS Contributors
// SPDX-License-Identifier: MIT

#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINER_TLERAWPIPELINEUTILITY_H
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINER_TLERAWPIPELINEUTILITY_H

#ifdef __TLE__

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Operation;
class RewriterBase;
class Value;

namespace triton {
class CoarseSchedule;

namespace gpu {

// Returns true only for an explicitly requested TLE-Raw producer.  The hint is
// deliberately not inferred: without hint="pipeline" the native pipeline is
// completely inert for tle.dsl_region.
bool isTLERawPipelineOp(Operation *op);

// Validate every explicitly requested TLE-Raw pipeline before lowerLoops starts
// mutating the module.  This keeps lowerLoops' original infallible interface;
// failures are reported by the owning pipeline pass before loop lowering.
LogicalResult validateTLERawPipelineOps(ModuleOp moduleOp);

// Materialize shared-memory rings for the TLE-Raw producers in `forOp`.  Loop
// staging and prologue/epilogue expansion remain owned by Triton's native
// software pipeliner.
FailureOr<scf::ForOp> lowerTLERawPipelineOps(scf::ForOp forOp,
                                             CoarseSchedule &schedule);

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // __TLE__
#endif
