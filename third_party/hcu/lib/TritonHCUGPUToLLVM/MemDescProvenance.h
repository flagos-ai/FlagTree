#ifndef TRITON_HCU_MEMDESC_PROVENANCE_H
#define TRITON_HCU_MEMDESC_PROVENANCE_H
#include "WarpSpecializeUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <optional>

namespace mlir::triton::HCU::shared_view {
// Read-only SSA forwarding facts. No equal-type or allocation may-alias result
// establishes identity. Region ownership and lifetime proofs belong to callers.
inline std::optional<SmallVector<Value>> incoming(Value value) {
  if (Value capture = resolveWarpSpecializeCapture(value); capture != value)
    return SmallVector<Value>{capture};
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return std::nullopt;
  Block *block = arg.getOwner();
  SmallVector<Value> result;
  for (Block *pred : block->getPredecessors()) {
    auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
    if (!branch)
      return std::nullopt;
    bool edge = false;
    for (unsigned i = 0; i < branch->getNumSuccessors(); ++i) {
      if (branch->getSuccessor(i) != block)
        continue;
      auto operands = branch.getSuccessorOperands(i);
      unsigned produced = operands.getProducedOperandCount();
      unsigned number = arg.getArgNumber();
      if (number < produced ||
          number - produced >= operands.getForwardedOperands().size())
        return std::nullopt;
      Value next = operands.getForwardedOperands()[number - produced];
      if (next.getType() != value.getType())
        return std::nullopt;
      result.push_back(next);
      edge = true;
    }
    if (!edge)
      return std::nullopt;
  }
  return result.empty() ? std::nullopt : std::optional(std::move(result));
}

inline Value identical(Value value) {
  SmallVector<Value, 8> seen;
  for (unsigned depth = 0; depth < 64; ++depth) {
    if (llvm::is_contained(seen, value))
      return {};
    seen.push_back(value);
    auto edges = incoming(value);
    if (!edges)
      return isa<BlockArgument>(value) ? Value{} : value;
    if (!llvm::all_of(*edges,
                      [&](Value next) { return next == edges->front(); }))
      return {};
    value = edges->front();
  }
  return {};
}

// Bounds for a logical slice are reusable facts, not physical disjointness.
inline bool boundedSlice(ArrayRef<int64_t> source, ArrayRef<int64_t> shape,
                         ArrayRef<int32_t> offsets) {
  if (source.size() != shape.size() || source.size() != offsets.size())
    return false;
  for (auto [extent, size, offset] : llvm::zip(source, shape, offsets))
    if (extent <= 0 || size <= 0 || offset < 0 || size > extent ||
        int64_t(offset) > extent - size)
      return false;
  return true;
}
} // namespace mlir::triton::HCU::shared_view
#endif
