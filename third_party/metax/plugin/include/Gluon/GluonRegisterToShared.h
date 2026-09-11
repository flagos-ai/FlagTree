#ifndef TRITON_METAX_GLUON_REGISTER_TO_SHARED_H
#define TRITON_METAX_GLUON_REGISTER_TO_SHARED_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton::gpu::metax::gluon {

/// Legalize only the concrete C500 register/shared boundaries left after
/// propagation and generic layout cleanup.
LogicalResult legalizeLocalDotStaging(ModuleOp module);

} // namespace mlir::triton::gpu::metax::gluon

#endif // TRITON_METAX_GLUON_REGISTER_TO_SHARED_H
