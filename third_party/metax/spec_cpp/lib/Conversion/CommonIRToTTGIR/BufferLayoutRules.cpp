#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;

bool TTGIRBufferLayoutAdapter::isAllocation(Value value) const {
  return bool(value.getDefiningOp<ttg::LocalAllocOp>());
}

Value TTGIRBufferLayoutAdapter::getViewSource(Value value) const {
  Operation *op = value.getDefiningOp();
  if (isa_and_nonnull<ttg::MemDescIndexOp, ttg::MemDescTransOp>(op))
    return op->getOperand(0);
  return {};
}

namespace {
Attribute projectTranspose(Attribute encoding, ArrayRef<int64_t> shape,
                           ArrayRef<int32_t> order) {
  Attribute result;
  auto *interface =
      encoding.getDialect()
          .getRegisteredInterface<triton::DialectInferLayoutInterface>();
  if (!interface || failed(interface->inferTransOpEncoding(
                        encoding, shape, order, result, /*loc=*/{})))
    return {};
  return result;
}

// A pipeline slot adds/drops only the leading allocation dimension.
Attribute projectIndex(Attribute encoding, bool toRoot) {
  auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(encoding);
  if (!shared || ttg::getNumCTAs(shared) != 1)
    return {};
  SmallVector<unsigned> order;
  if (toRoot) {
    for (unsigned dim : shared.getOrder())
      order.push_back(dim + 1);
    order.push_back(0);
  } else {
    if (shared.getOrder().back() != 0)
      return {};
    for (unsigned dim : shared.getOrder().drop_back())
      order.push_back(dim - 1);
  }
  return ttg::SwizzledSharedEncodingAttr::get(
      encoding.getContext(), shared.getVec(), shared.getPerPhase(),
      shared.getMaxPhase(), order,
      ttg::CTAEncodingAttr::getDefault(encoding.getContext(), order.size()));
}
} // namespace

Attribute TTGIRBufferLayoutAdapter::toSourceEncoding(Value view,
                                                     Attribute encoding) const {
  if (view.getDefiningOp<ttg::MemDescIndexOp>())
    return projectIndex(encoding, true);
  if (auto trans = view.getDefiningOp<ttg::MemDescTransOp>())
    return projectTranspose(encoding, trans.getType().getShape(),
                            triton::inversePermutation(trans.getOrder()));
  return {};
}

Attribute TTGIRBufferLayoutAdapter::toViewEncoding(Value view,
                                                   Attribute encoding) const {
  if (view.getDefiningOp<ttg::MemDescIndexOp>())
    return projectIndex(encoding, false);
  if (auto trans = view.getDefiningOp<ttg::MemDescTransOp>())
    return projectTranspose(encoding, trans.getSrc().getType().getShape(),
                            trans.getOrder());
  return {};
}

Type TTGIRBufferLayoutAdapter::withEncoding(Value value,
                                            Attribute encoding) const {
  auto type = dyn_cast<ttg::MemDescType>(value.getType());
  if (!type || !isa<ttg::SharedEncodingTrait>(encoding))
    return {};
  return ttg::MemDescType::get(type.getShape(), type.getElementType(), encoding,
                               type.getMemorySpace(), type.getMutableMemory(),
                               type.getAllocShape());
}

} // namespace mlir::triton::metax
