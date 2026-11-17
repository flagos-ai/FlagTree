#ifndef TRITON_METAX_COMMONIR_TENSOR_LAYOUT_RULES_H
#define TRITON_METAX_COMMONIR_TENSOR_LAYOUT_RULES_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::triton::metax {

// TTGIR operation adapters. Tensor tiles reuse the lowering's direct-slice
// check.
class TTGIRLayoutAdapter {
public:
  bool canChangeEncoding(Value value, Attribute encoding) const;
  Value getConversionSource(Value value) const;
  Attribute inferOperandEncoding(Operation *op, Attribute encoding) const;
  Value cloneWithLayout(Operation *op, RankedTensorType type,
                        ValueRange operands) const;
  Value convertLayout(Value value, RankedTensorType type,
                      Operation *before) const;
};

// Initial value, region argument, result and yielded value for one loop slot.
SmallVector<Value> getLoopSlotValues(scf::ForOp loop, unsigned index);

// Collect values tied by layout conversions and scf.for slots.
SmallVector<Value>
collectTensorLayoutComponent(Value seed, const TTGIRLayoutAdapter &adapter);

} // namespace mlir::triton::metax
#endif
