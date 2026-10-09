#include "mlir-ext/Dialect/CommonIR/Analysis/AxisInfoExt.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;

Value TTGIRLayoutAdapter::getConversionSource(Value value) const {
  if (auto op = value.getDefiningOp<ttg::ConvertLayoutOp>())
    return op.getSrc();
  return {};
}

Attribute TTGIRLayoutAdapter::inferOperandEncoding(Operation *op,
                                                   Attribute encoding) const {
  if (auto constant = dyn_cast<arith::ConstantOp>(op))
    return isa<DenseElementsAttr>(constant.getValue()) ? encoding : Attribute{};
  if (isa<triton::MakeRangeOp, triton::SplatOp, triton::BroadcastOp>(op))
    return encoding;
  if (isa<tile::ExtractTileOp, tile::InsertTileOp>(op))
    return isa<ttg::BlockedEncodingAttr>(encoding) ? encoding : Attribute{};
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
bool comesFromLogicalTile(Value value) {
  // Coalesce may already have placed a conversion immediately after the tile.
  while (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>())
    value = convert.getSrc();
  return isa_and_nonnull<tile::ExtractTileOp, tile::InsertTileOp>(
      value.getDefiningOp());
}

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

void collectCopyLayoutRequirements(ModuleOp module,
                                   TensorLayoutRequirements &requirements) {
  SmallVector<ttg::AsyncCopyGlobalToLocalOp> copies;
  module.walk([&](ttg::AsyncCopyGlobalToLocalOp copy) {
    if (comesFromLogicalTile(copy.getSrc()))
      copies.push_back(copy);
  });
  if (copies.empty())
    return;
  ModuleAxisInfoAnalysis axisInfo(module, tile::addCommonIRAxisInfoVisitors);
  for (auto copy : copies) {
    auto type = cast<RankedTensorType>(copy.getSrc().getType());
    auto encoding = ttg::buildCoalescedEncoding(
        module.getContext(), axisInfo, copy, ttg::lookupNumWarps(copy),
        ttg::TritonGPUDialect::getThreadsPerWarp(module),
        ttg::getCTALayout(type.getEncoding()), ttg::getShapePerCTA(type));
    for (OpOperand &operand : copy->getOpOperands())
      if (isa<RankedTensorType>(operand.get().getType()))
        requirements.uses.push_back({&operand, encoding});
  }
}

void collectLoadStoreLayoutRequirements(
    ModuleOp module, TensorLayoutRequirements &requirements) {
  TTGIRLayoutAdapter adapter;
  module.walk([&](ttg::LocalStoreOp store) {
    Value source = store.getSrc();
    SmallVector<Operation *> views;
    bool hasTile = false;
    // Follow only logical views. Loads stay effect boundaries: use their
    // existing coalesced encoding without cloning or moving the memory read.
    while (true) {
      if (Value input = adapter.getConversionSource(source)) {
        source = input;
        continue;
      }
      Operation *op = source.getDefiningOp();
      if (!isa_and_nonnull<tile::ExtractTileOp, triton::TransOp>(op))
        break;
      hasTile |= isa<tile::ExtractTileOp>(op);
      views.push_back(op);
      source = op->getOperand(0);
    }
    if (!hasTile)
      return;

    Attribute encoding;
    // Include prefetched loads carried across loop iterations. Conflicting
    // load encodings do not provide one unambiguous transport requirement.
    for (Value value : collectTensorLayoutComponent(source, adapter)) {
      auto load = value.getDefiningOp<triton::LoadOp>();
      if (!load)
        continue;
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || !isa<ttg::BlockedEncodingAttr>(type.getEncoding()))
        return;
      if (encoding && encoding != type.getEncoding())
        return;
      encoding = type.getEncoding();
    }
    if (!encoding)
      return;
    for (Operation *view : llvm::reverse(views)) {
      if (isa<triton::TransOp>(view))
        encoding = mlir::inferDstEncoding(view, encoding);
      if (!encoding)
        return;
    }
    requirements.uses.push_back({&store->getOpOperand(0), encoding});
  });
}

void applyLayoutRequirements(ModuleOp module,
                             const TensorLayoutRequirements &requirements,
                             const TensorLayoutAdapter &adapter) {
  auto plan = planTensorLayouts(requirements, adapter);
  plan->apply();
  eraseDeadProducers(module);
}

} // namespace mlir::triton::metax
