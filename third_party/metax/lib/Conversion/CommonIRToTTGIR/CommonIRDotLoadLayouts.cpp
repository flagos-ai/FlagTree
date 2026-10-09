// MIT License
// Copyright (c) 2026 The FlagOS Contributors
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;

namespace {
bool canLoadDirectly(ttg::LocalLoadOp load, RankedTensorType target,
                     const tile::BufferLayoutPlan &buffers) {
  auto dot = dyn_cast<ttg::DotOperandEncodingAttr>(target.getEncoding());
  auto mma =
      dot ? dyn_cast<ttg::MACAMmaEncodingAttr>(dot.getParent()) : nullptr;
  auto shared = dyn_cast<ttg::MemDescType>(buffers.getType(load.getSrc()));
  if (!mma || mma.getVersionMajor() != 2 || mma.getIsATrans() ||
      mma.getIsBTrans() || load.getMmaMode() != -1 || load.getToken() ||
      target.getRank() != 2 || !shared ||
      target.getShape() != shared.getShape() ||
      target.getElementType() != shared.getElementType() ||
      (!target.getElementType().isF16() && !target.getElementType().isBF16()) ||
      !isa<ttg::SwizzledSharedEncodingAttr>(shared.getEncoding()))
    return false;

  auto conversion =
      ttg::toLinearLayout(target).invertAndCompose(ttg::toLinearLayout(shared));
  return conversion.isTrivialOver(
      {StringAttr::get(load.getContext(), "block")});
}

} // namespace

void collectLocalLoadLayoutRequirements(ArrayRef<DotLayoutChoice> choices,
                                        TensorLayoutRequirements &requirements,
                                        const tile::BufferLayoutPlan &buffers) {
  DenseMap<OpOperand *, Attribute> ports;
  for (auto [dot, plan] : choices) {
    ports[&dot->getOpOperand(0)] = plan.operandA;
    ports[&dot->getOpOperand(1)] = plan.operandB;
  }
  TTGIRLayoutAdapter adapter;
  DenseSet<Value> visited;
  for (auto [dot, plan] : choices) {
    for (unsigned index : {0u, 1u}) {
      Value seed = dot->getOperand(index);
      if (visited.contains(seed))
        continue;
      Attribute encoding = ports.lookup(&dot->getOpOperand(index));
      if (!isa_and_nonnull<ttg::DotOperandEncodingAttr>(encoding))
        continue;
      auto target =
          cast<RankedTensorType>(seed.getType()).cloneWithEncoding(encoding);
      auto values = collectTensorLayoutComponent(seed, adapter);
      DenseSet<Value> members(values.begin(), values.end());
      bool valid = true, hasLoad = false;
      // This rule stays conservative: unknown uses or fixed producers keep
      // the existing layouts. The propagation helper traverses
      // loops/conversions.
      for (Value value : values) {
        auto type = dyn_cast<RankedTensorType>(value.getType());
        if (!type || type.getShape() != target.getShape() ||
            type.getElementType() != target.getElementType()) {
          valid = false;
          break;
        }
        if (auto load = value.getDefiningOp<ttg::LocalLoadOp>()) {
          valid &= canLoadDirectly(load, target, buffers);
          hasLoad = true;
        } else if (auto arg = dyn_cast<BlockArgument>(value)) {
          valid &= isa<scf::ForOp>(arg.getOwner()->getParentOp()) &&
                   arg.getArgNumber() > 0;
        } else {
          valid &= bool(adapter.getConversionSource(value)) ||
                   bool(value.getDefiningOp<scf::ForOp>());
        }
        for (OpOperand &use : value.getUses()) {
          Operation *user = use.getOwner();
          if (auto loop = dyn_cast<scf::ForOp>(user)) {
            valid &= use.getOperandNumber() >= loop.getNumControlOperands();
          } else if (auto yield = dyn_cast<scf::YieldOp>(user)) {
            valid &= isa<scf::ForOp>(yield->getParentOp());
          } else if (user->getNumResults() == 1 &&
                     adapter.getConversionSource(user->getResult(0)) == value) {
            valid &= members.contains(user->getResult(0));
          } else {
            valid &= ports.lookup(&use) == encoding;
          }
        }
      }
      visited.insert(values.begin(), values.end());
      if (!valid || !hasLoad)
        continue;
      for (Value value : values) {
        requirements.assignments[value] = encoding;
        for (OpOperand &use : value.getUses())
          requirements.uses.push_back({&use, encoding, false});
      }
    }
  }
}

} // namespace mlir::triton::metax
