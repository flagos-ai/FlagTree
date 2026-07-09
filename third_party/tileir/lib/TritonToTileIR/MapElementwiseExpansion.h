//===- MapElementwiseExpansion.h - Pre-processing for map_elementwise -----===//
//
// Declares pre-processing utilities for tt.map_elementwise:
//   - ifConvertMapElementwiseRegions: scf.if → arith.select at scalar level
//   - expandMapElementwiseOps: tensor-level expansion for all map_elementwise
//   ops
//
// These are pure IR rewrites that run before dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_TILEIR_MAP_ELEMENTWISE_EXPANSION_H
#define TRITON_TO_TILEIR_MAP_ELEMENTWISE_EXPANSION_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace triton {

/// Pre-process all map_elementwise regions: convert scf.if → arith.select
/// at scalar level within the region body.
LogicalResult ifConvertMapElementwiseRegions(Operation *rootOp);

/// Expand all map_elementwise ops into tensor-level arith/math ops via
/// tensor lifting. Each scalar op in the region is lifted to its tensor
/// equivalent, and the map_elementwise op is erased.
LogicalResult expandMapElementwiseOps(Operation *rootOp);

} // namespace triton
} // namespace mlir

#endif // TRITON_TO_TILEIR_MAP_ELEMENTWISE_EXPANSION_H
