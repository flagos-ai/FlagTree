// Copyright 2026 FlagOS Contributors

#ifndef TRITON_TLE_ANALYSIS_GATHER_LAYOUT_H
#define TRITON_TLE_ANALYSIS_GATHER_LAYOUT_H

#include "mlir/IR/Matchers.h"
#include "triton/Analysis/Utility.h"

namespace mlir::triton::tle {

/// A uniform, valid gather index selects one source coordinate per output.
/// With matching warp-local row ownership, constant folding selects the source
/// register and one lane shuffle suffices. Repacking the whole source to a
/// dynamic-index layout cannot reduce that communication and may add Shared
/// round-trips at a fixed-layout producer such as WGMMA.
inline bool hasUniformWarpLocalGather(triton::GatherOp op) {
  Attribute value;
  if (!matchPattern(op.getIndices(), m_Constant(&value)))
    return false;
  auto indices = dyn_cast<DenseIntElementsAttr>(value);
  if (!indices || !indices.isSplat())
    return false;
  auto index = indices.getSplatValue<APInt>();
  if (index.isNegative() ||
      index.uge(op.getSrc().getType().getDimSize(op.getAxis())))
    return false;
  return GatherLoweringHelper(op).isWarpLocal();
}

} // namespace mlir::triton::tle

#endif // TRITON_TLE_ANALYSIS_GATHER_LAYOUT_H
