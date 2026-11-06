#ifndef TRITON_METAX_COMMONIR_LAYOUT_RULES_H
#define TRITON_METAX_COMMONIR_LAYOUT_RULES_H

#include "mlir-ext/Dialect/CommonIR/Transforms/BufferLayoutPropagation.h"
#include "mlir-ext/Dialect/CommonIR/Transforms/TensorLayoutPropagation.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::metax {

using tile::collectTensorLayoutComponent;
using tile::planTensorLayouts;
using tile::TensorLayoutAdapter;
using tile::TensorLayoutRequirements;

class TTGIRBufferLayoutAdapter : public tile::BufferLayoutAdapter {
public:
  bool isAllocation(Value value) const override;
  Value getViewSource(Value value) const override;
  Attribute toSourceEncoding(Value view, Attribute encoding) const override;
  Attribute toViewEncoding(Value view, Attribute encoding) const override;
  Type withEncoding(Value value, Attribute encoding) const override;
};

// TTGIR operation adapters. Tensor tiles reuse the lowering's direct-slice check.
class TTGIRLayoutAdapter : public TensorLayoutAdapter {
public:
  bool canChangeEncoding(Value value, Attribute encoding) const override;
  Value getConversionSource(Value value) const override;
  Attribute inferOperandEncoding(Operation *op,
                                 Attribute encoding) const override;
  Value cloneWithLayout(Operation *op, RankedTensorType type,
                        ValueRange operands) const override;
  Value convertLayout(Value value, RankedTensorType type,
                      Operation *before) const override;
};

// Writer preferences guide shared order; they are not applied to tensors here.
void collectCopyWriterPreferences(ModuleOp module,
                                  TensorLayoutRequirements &preferences);
// Copy requirements account for the selected shared layout before propagation.
void collectLegalCopyLayoutRequirements(ModuleOp module,
                                        TensorLayoutRequirements &requirements);
void collectLoadStoreLayoutRequirements(ModuleOp module,
                                        TensorLayoutRequirements &requirements);
void applyLayoutRequirements(ModuleOp module,
                             const TensorLayoutRequirements &requirements,
                             const TensorLayoutAdapter &adapter);

} // namespace mlir::triton::metax
#endif
