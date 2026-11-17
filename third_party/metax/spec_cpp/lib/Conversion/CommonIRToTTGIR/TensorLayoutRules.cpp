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

SmallVector<Value> getLoopSlotValues(scf::ForOp loop, unsigned index) {
  return {
      loop.getInitArgs()[index], loop.getRegionIterArgs()[index],
      loop.getResult(index),
      cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(index)};
}

SmallVector<Value>
collectTensorLayoutComponent(Value seed, const TTGIRLayoutAdapter &adapter) {
  llvm::SetVector<Value> values;
  values.insert(seed);
  auto addSlot = [&](scf::ForOp loop, unsigned index) {
    for (Value value : getLoopSlotValues(loop, index))
      values.insert(value);
  };
  for (unsigned i = 0; i < values.size(); ++i) {
    Value value = values[i];
    if (Value source = adapter.getConversionSource(value))
      values.insert(source);
    else if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()))
        if (arg.getArgNumber() > 0)
          addSlot(loop, arg.getArgNumber() - 1);
    } else if (auto loop = value.getDefiningOp<scf::ForOp>()) {
      addSlot(loop, cast<OpResult>(value).getResultNumber());
    }
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto loop = dyn_cast<scf::ForOp>(user)) {
        if (use.getOperandNumber() >= loop.getNumControlOperands())
          addSlot(loop, use.getOperandNumber() - loop.getNumControlOperands());
      } else if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        if (auto loop = dyn_cast<scf::ForOp>(yield->getParentOp()))
          addSlot(loop, use.getOperandNumber());
      } else if (user->getNumResults() == 1 &&
                 adapter.getConversionSource(user->getResult(0)) == value) {
        values.insert(user->getResult(0));
      }
    }
  }
  return values.takeVector();
}

} // namespace mlir::triton::metax
