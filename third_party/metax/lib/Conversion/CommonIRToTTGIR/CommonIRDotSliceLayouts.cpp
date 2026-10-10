// MIT License
// Copyright (c) 2026 The FlagOS Contributors
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Conversion/CommonIRToTTGIR/TensorOps.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::metax {
namespace {
namespace ttg = mlir::triton::gpu;
using Key = std::pair<Value, Attribute>;

// Discover layouts without changing the IR. Control flow and memory operations
// remain boundaries; existing propagation handles their inputs when applying.
Attribute forwardEncoding(Operation *op, Attribute encoding) {
  if (op->getNumResults() != 1 || !isMemoryEffectFree(op))
    return {};
  if (isa<tile::ExtractTileOp>(op))
    return canLowerTensorTileInEncoding(op, encoding) ? encoding : Attribute{};
  if (isa<ttg::ConvertLayoutOp>(op))
    return encoding;
  auto ns = op->getName().getDialectNamespace();
  bool elementwise = (ns == "arith" || ns == "math") &&
                     op->hasTrait<OpTrait::Elementwise>();
  if (!elementwise &&
      !isa<triton::ReduceOp, triton::ExpandDimsOp, triton::BroadcastOp,
           triton::TransOp>(op))
    return {};
  return mlir::inferDstEncoding(op, encoding);
}
} // namespace

void collectDotSliceLayoutRequirements(ArrayRef<DotLayoutChoice> choices,
                                      TensorLayoutRequirements &requirements) {
  llvm::SetVector<Key> reachable;
  DenseMap<OpOperand *, Attribute> ports;
  for (auto [dot, plan] : choices) {
    reachable.insert({dot.getResult(), plan.mma});
    ports[&dot->getOpOperand(0)] = plan.operandA;
    ports[&dot->getOpOperand(1)] = plan.operandB;
  }
  for (unsigned i = 0; i < reachable.size(); ++i) {
    auto [value, encoding] = reachable[i];
    for (Operation *user : value.getUsers())
      if (Attribute result = forwardEncoding(user, encoding))
        reachable.insert({user->getResult(0), result});
  }

  // A use selects a compatible register carrier. Keep dot operand conversion
  // explicit: LLVM can reorder per-thread values without a shared allocation.
  for (auto [dot, plan] : choices) {
    for (unsigned index : {0u, 1u}) {
      OpOperand &port = dot->getOpOperand(index);
      auto type = cast<RankedTensorType>(port.get().getType());
      auto target = type.cloneWithEncoding(ports.lookup(&port));
      for (auto [value, encoding] : reachable) {
        if (value != port.get() || !isa<ttg::MACAMmaEncodingAttr>(encoding) ||
            !cvtReordersRegisters(type.cloneWithEncoding(encoding), target))
          continue;

        llvm::SetVector<Key> path;
        path.insert({value, encoding});
        bool valid = true, hasTile = false;
        for (unsigned i = 0; i < path.size() && valid; ++i) {
          auto [current, layout] = path[i];
          Operation *op = current.getDefiningOp();
          if (auto assigned = requirements.assignments.lookup(current))
            if (assigned != layout) {
              valid = false;
              break;
            }
          if (isa<triton::DotOp>(op))
            continue;
          hasTile |= isa<tile::ExtractTileOp>(op);
          Attribute input = isa<tile::ExtractTileOp, ttg::ConvertLayoutOp>(op)
                                ? layout
                                : mlir::inferSrcEncoding(op, layout);
          if (!input) {
            valid = false;
            break;
          }
          for (Value operand : op->getOperands())
            if (reachable.contains({operand, input}))
              path.insert({operand, input});
        }
        if (!valid || !hasTile)
          continue;

        for (auto [current, layout] : path) {
          Operation *op = current.getDefiningOp();
          if (isa<triton::DotOp, ttg::ConvertLayoutOp>(op))
            continue;
          requirements.assignments[current] = layout;
          Attribute input = isa<tile::ExtractTileOp>(op)
                                ? layout
                                : mlir::inferSrcEncoding(op, layout);
          for (OpOperand &operand : op->getOpOperands())
            if (isa<RankedTensorType>(operand.get().getType()))
              requirements.uses.push_back({&operand, input});
        }
        // The original hardware-port requirement already requests dot A/B.
        break;
      }
    }
  }
}
} // namespace mlir::triton::metax
