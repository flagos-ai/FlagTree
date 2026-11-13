#include "triton/Conversion/CommonIRToTTGIR/TensorLayoutRules.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Conversion/CommonIRToTTGIR/CommonIRLowerTensorTiles.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;

bool TTGIRLayoutAdapter::canChangeEncoding(Value value,
                                           Attribute encoding) const {
  auto fixed = tile::getExplicitValueEncoding(value);
  return !fixed || fixed == encoding;
}

Value TTGIRLayoutAdapter::getConversionSource(Value value) const {
  if (auto op = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
    // A fixed producer may serve different dot ports through conversions.
    // Do not merge those ports into one shared-layout requirement.
    if (auto explicitEncoding = tile::getExplicitValueEncoding(op.getSrc()))
      if (explicitEncoding != op.getType().getEncoding())
        return {};
    return op.getSrc();
  }
  return {};
}

Attribute TTGIRLayoutAdapter::inferOperandEncoding(Operation *op,
                                                   Attribute encoding) const {
  if (auto constant = dyn_cast<arith::ConstantOp>(op))
    return isa<DenseElementsAttr>(constant.getValue()) ? encoding : Attribute{};
  if (isa<triton::MakeRangeOp, triton::SplatOp, triton::BroadcastOp>(op))
    return encoding;
  if (isa<tile::ExtractTileOp, tile::InsertTileOp>(op))
    return canLowerTensorTileInEncoding(op, encoding) ? encoding : Attribute{};
  // Rematerialize only address/mask arithmetic and simple tensor views.
  // Loads and other operations with effects remain conversion boundaries.
  bool arithmetic = op->getName().getDialectNamespace() == "arith" &&
                    op->hasTrait<OpTrait::Elementwise>();
  if (!isMemoryEffectFree(op) ||
      !(arithmetic ||
        isa<triton::AddPtrOp, triton::ExpandDimsOp, triton::TransOp>(op)))
    return {};
  return mlir::inferSrcEncoding(op, encoding);
}

Value TTGIRLayoutAdapter::cloneWithLayout(Operation *op, RankedTensorType type,
                                          ValueRange operands) const {
  OpBuilder builder(op);
  IRMapping mapping;
  for (auto [before, after] : llvm::zip(op->getOperands(), operands))
    mapping.map(before, after);
  Operation *copy = builder.clone(*op, mapping);
  copy->getResult(0).setType(type);
  if (auto constant = dyn_cast<arith::ConstantOp>(copy)) {
    auto value = cast<DenseElementsAttr>(constant.getValue());
    constant.setValueAttr(value.reshape(type));
  }
  return copy->getResult(0);
}

Value TTGIRLayoutAdapter::convertLayout(Value value, RankedTensorType type,
                                        Operation *before) const {
  if (value.getType() == type)
    return value;
  OpBuilder builder(before);
  return ttg::ConvertLayoutOp::create(builder, value.getLoc(), type, value);
}

namespace {
void eraseDeadProducers(ModuleOp module) {
  llvm::SetVector<Operation *> worklist;
  module.walk([&](Operation *op) {
    if (op->getNumRegions() == 0)
      worklist.insert(op);
  });
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isOpTriviallyDead(op))
      continue;
    for (Value operand : op->getOperands())
      if (Operation *producer = operand.getDefiningOp())
        if (producer->getNumRegions() == 0)
          worklist.insert(producer);
    op->erase();
  }
}

} // namespace

void applyLayoutRequirements(ModuleOp module,
                             const TensorLayoutRequirements &requirements,
                             const TensorLayoutAdapter &adapter) {
  tile::applyTensorLayoutRequirements(requirements, adapter);
  eraseDeadProducers(module);
}

} // namespace mlir::triton::metax
