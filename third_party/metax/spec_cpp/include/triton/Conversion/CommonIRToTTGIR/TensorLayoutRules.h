#ifndef TRITON_METAX_COMMONIR_TENSOR_LAYOUT_RULES_H
#define TRITON_METAX_COMMONIR_TENSOR_LAYOUT_RULES_H

#include "mlir-ext/Dialect/CommonIR/Transforms/TensorLayoutPropagation.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::metax {

using tile::collectTensorLayoutComponent;
using tile::TensorLayoutAdapter;
using tile::TensorLayoutRequirements;

// TTGIR operation adapters. Tensor tiles reuse the lowering's direct-slice
// check.
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

void applyLayoutRequirements(ModuleOp module,
                             const TensorLayoutRequirements &requirements,
                             const TensorLayoutAdapter &adapter);

} // namespace mlir::triton::metax
#endif
