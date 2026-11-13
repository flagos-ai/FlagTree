#ifndef TRITON_HCU_WARP_SPECIALIZE_UTILS_H
#define TRITON_HCU_WARP_SPECIALIZE_UTILS_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::triton::HCU {
// Follow only the structural, type-preserving captures of a partition entry.
// CF loop arguments are deliberately not treated as captures or ABI arguments.
inline Value resolveWarpSpecializeCapture(Value value) {
  for (unsigned depth = 0; depth != 64; ++depth) {
    auto arg = dyn_cast<BlockArgument>(value);
    if (!arg) return value;
    Block *block = arg.getOwner();
    auto parts = dyn_cast<gpu::WarpSpecializePartitionsOp>(block->getParentOp());
    if (!parts || block != &block->getParent()->front()) return value;
    auto captures = parts.getParentOp().getExplicitCaptures();
    unsigned n = arg.getArgNumber();
    if (n >= captures.size() || captures[n].getType() != arg.getType())
      return value;
    value = captures[n];
  }
  return value; // A remaining non-ABI argument still fails the caller's proof.
}

// Thread IDs are relative to the nearest specialization region. Hoisting the
// query into the enclosing function would discard that partition's start ID.
inline Region *warpSpecializeScope(Operation *op) {
  Region *region = op->getParentRegion();
  while (region && !isa<gpu::WarpSpecializeOp,
                        gpu::WarpSpecializePartitionsOp, triton::FuncOp,
                        LLVM::LLVMFuncOp>(region->getParentOp()))
    region = region->getParentOp()->getParentRegion();
  return region;
}
} // namespace mlir::triton::HCU
#endif
