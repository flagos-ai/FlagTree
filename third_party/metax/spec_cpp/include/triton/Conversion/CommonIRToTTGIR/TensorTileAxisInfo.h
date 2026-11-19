#ifndef TRITON_METAX_TENSOR_TILE_AXIS_INFO_H
#define TRITON_METAX_TENSOR_TILE_AXIS_INFO_H

namespace mlir::triton {
class AxisInfoVisitorList;

namespace metax {

// Extend an existing Triton AxisInfo analysis with logical tensor tile rules.
void addTensorTileAxisInfoVisitors(AxisInfoVisitorList &visitors);

} // namespace metax
} // namespace mlir::triton

#endif
