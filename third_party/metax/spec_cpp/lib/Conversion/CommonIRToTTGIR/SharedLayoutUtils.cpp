// MIT License
// Copyright (c) 2026 The FlagOS Contributors
#include "triton/Conversion/CommonIRToTTGIR/SharedLayoutUtils.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;

Value getSharedViewSource(Value value) {
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

Attribute getSharedSourceEncoding(Value view, Attribute encoding) {
  if (view.getDefiningOp<ttg::MemDescIndexOp>())
    return projectIndex(encoding, true);
  if (auto trans = view.getDefiningOp<ttg::MemDescTransOp>())
    return projectTranspose(encoding, trans.getType().getShape(),
                            triton::inversePermutation(trans.getOrder()));
  return {};
}

Attribute getSharedViewEncoding(Value view, Attribute encoding) {
  if (view.getDefiningOp<ttg::MemDescIndexOp>())
    return projectIndex(encoding, false);
  if (auto trans = view.getDefiningOp<ttg::MemDescTransOp>())
    return projectTranspose(encoding, trans.getSrc().getType().getShape(),
                            trans.getOrder());
  return {};
}

Type SharedLayoutUpdates::getType(Value value) const {
  Type type = types.lookup(value);
  return type ? type : value.getType();
}

void SharedLayoutUpdates::apply() const {
  for (auto [value, type] : types)
    value.setType(type);
}

Value getSharedAllocation(Value value) {
  while (value && !value.getDefiningOp<ttg::LocalAllocOp>())
    value = getSharedViewSource(value);
  return value;
}

LogicalResult collectSharedLayoutUpdates(Value allocation, Attribute encoding,
                                         SharedLayoutUpdates &updates) {
  if (!allocation.getDefiningOp<ttg::LocalAllocOp>())
    return failure();
  SmallVector<std::pair<Value, Attribute>> worklist{{allocation, encoding}};
  for (unsigned i = 0; i < worklist.size(); ++i) {
    auto [value, layout] = worklist[i];
    auto type = dyn_cast<ttg::MemDescType>(value.getType());
    if (!type || !isa<ttg::SharedEncodingTrait>(layout))
      return failure();
    auto replacement = ttg::MemDescType::get(
        type.getShape(), type.getElementType(), layout, type.getMemorySpace(),
        type.getMutableMemory(), type.getAllocShape());
    auto [it, inserted] = updates.types.try_emplace(value, replacement);
    if (!inserted) {
      if (it->second != replacement)
        return failure();
      continue;
    }
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (user->getNumResults() == 1 &&
          getSharedViewSource(user->getResult(0)) == value) {
        Value view = user->getResult(0);
        Attribute projected = getSharedViewEncoding(view, layout);
        if (!projected)
          return failure();
        worklist.emplace_back(view, projected);
      } else {
        updates.accesses.push_back(&use);
      }
    }
  }
  return success();
}

} // namespace mlir::triton::metax
