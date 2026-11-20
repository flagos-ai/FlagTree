#ifndef TRITON_METAX_COMMONIR_LAYOUT_CONVERSION_UTILS_H
#define TRITON_METAX_COMMONIR_LAYOUT_CONVERSION_UTILS_H

#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::metax {

inline Value getConversionSource(Value value) {
  if (auto op = value.getDefiningOp<gpu::ConvertLayoutOp>()) {
    // A fixed producer may serve different dot ports through conversions.
    // Do not merge those ports into one shared-layout requirement.
    if (auto explicitEncoding = tile::getExplicitValueEncoding(op.getSrc()))
      if (explicitEncoding != op.getType().getEncoding())
        return {};
    return op.getSrc();
  }
  return {};
}

// Follow a conversion's input or the incoming values of a loop-carried tensor.
inline bool appendTensorSources(Value value,
                                llvm::SmallSetVector<Value, 4> &sources) {
  if (Value input = getConversionSource(value)) {
    sources.insert(input);
    return true;
  }
  if (auto loop = value.getDefiningOp<scf::ForOp>())
    value = loop.getTiedLoopRegionIterArg(cast<OpResult>(value));
  auto arg = dyn_cast<BlockArgument>(value);
  auto loop = arg ? dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())
                  : scf::ForOp{};
  if (!loop || arg.getArgNumber() == 0)
    return false;
  unsigned index = arg.getArgNumber() - 1;
  sources.insert(loop.getInitArgs()[index]);
  sources.insert(loop.getYieldedValues()[index]);
  return true;
}

// Load layout that can be used through the store's slice and transpose chain.
Attribute getTileStoreSourcePreference(gpu::LocalStoreOp store);

inline Value convertLayout(Value value, RankedTensorType type,
                           Operation *before) {
  if (value.getType() == type)
    return value;
  OpBuilder builder(before);
  return gpu::ConvertLayoutOp::create(builder, value.getLoc(), type, value);
}

} // namespace mlir::triton::metax

#endif
