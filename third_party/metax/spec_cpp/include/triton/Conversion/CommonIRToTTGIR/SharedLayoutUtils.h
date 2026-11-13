#ifndef TRITON_METAX_COMMONIR_SHARED_LAYOUT_UTILS_H
#define TRITON_METAX_COMMONIR_SHARED_LAYOUT_UTILS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::triton::metax {

// Stage allocation/view types until all affected accesses have been checked.
struct SharedLayoutUpdates {
  llvm::MapVector<Value, Type> types;
  SmallVector<OpOperand *> accesses;

  Type getType(Value value) const;
  void apply() const;
};

Value getSharedViewSource(Value view);
Value getSharedAllocation(Value value);
Attribute getSharedSourceEncoding(Value view, Attribute encoding);
Attribute getSharedViewEncoding(Value view, Attribute encoding);

// Does not mutate IR. Discard the updates if collection or validation fails.
LogicalResult collectSharedLayoutUpdates(Value allocation, Attribute encoding,
                                         SharedLayoutUpdates &updates);

} // namespace mlir::triton::metax
#endif
