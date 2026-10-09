#ifndef TRITON_METAX_COMMONIR_LAYOUT_RULES_H
#define TRITON_METAX_COMMONIR_LAYOUT_RULES_H

#include "mlir-ext/Dialect/CommonIR/Transforms/BufferLayoutPropagation.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Conversion/CommonIRToTTGIR/DotLayoutPlanner.h"
#include "triton/Conversion/CommonIRToTTGIR/TensorLayoutPropagation.h"

namespace mlir::triton::metax {

class TTGIRBufferLayoutAdapter : public tile::BufferLayoutAdapter {
public:
  bool isAllocation(Value value) const override;
  Value getViewSource(Value value) const override;
  Attribute toSourceEncoding(Value view, Attribute encoding) const override;
  Attribute toViewEncoding(Value view, Attribute encoding) const override;
  Type withEncoding(Value value, Attribute encoding) const override;
};

// TTGIR operation adapters. MMA and hardware legality checks belong to the
// requirement collectors and the tile-specific override in the applying pass.
class TTGIRLayoutAdapter : public TensorLayoutAdapter {
public:
  Value getConversionSource(Value value) const override;
  Attribute inferOperandEncoding(Operation *op,
                                 Attribute encoding) const override;
  Value cloneWithLayout(Operation *op, RankedTensorType type,
                        ValueRange operands) const override;
  Value convertLayout(Value value, RankedTensorType type,
                      Operation *before) const override;
};

void collectCopyLayoutRequirements(ModuleOp module,
                                   TensorLayoutRequirements &requirements);
void collectLoadStoreLayoutRequirements(ModuleOp module,
                                        TensorLayoutRequirements &requirements);
void applyLayoutRequirements(ModuleOp module,
                             const TensorLayoutRequirements &requirements,
                             const TensorLayoutAdapter &adapter);

// All collectors are read-only. Requirements are committed once, after every
// rule has accepted the candidate.
LogicalResult
collectBsmLayoutRequirements(ModuleOp module, ArrayRef<DotLayoutChoice> choices,
                             TensorLayoutRequirements &requirements);
void collectLocalLoadLayoutRequirements(ArrayRef<DotLayoutChoice> choices,
                                        TensorLayoutRequirements &requirements,
                                        const tile::BufferLayoutPlan &buffers);
FailureOr<tile::BufferLayoutPlan>
planSharedLayouts(ModuleOp module, ArrayRef<DotLayoutChoice> choices,
                  TensorLayoutRequirements &requirements);

} // namespace mlir::triton::metax
#endif
