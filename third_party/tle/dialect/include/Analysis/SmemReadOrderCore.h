/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_TLE_ANALYSIS_SMEM_READ_ORDER_CORE_H_
#define TRITON_TLE_ANALYSIS_SMEM_READ_ORDER_CORE_H_

// The part of the keep-smem-read-order analysis that needs no TLE dialect: the
// allocation a shared memory descriptor is a view of, the window of it a view
// covers, and the test whether an effect may write them. A build without the
// TLE dialect (so without its generated headers) uses only this part, as
// tritongpu-reorder-instructions does there. SmemReadOrder.h adds what needs
// the TLE ops. Header-only, so that the core TritonGPU transforms need no link
// dependency on the TLE libraries.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::triton::tle {

// Address space of a tt.ptr of global memory; the pointers of
// tle.local_pointers are in the shared address space (3), and its verifier
// accepts no other.
inline constexpr int kSmemReadOrderGlobalAddressSpace = 1;

// The capture operand of the warp_specialize that a partition argument stands
// for, or null for any other block argument. A partition argument is the
// capture of the same position, so the two are the same descriptor.
inline Value getWarpSpecializeCapture(BlockArgument arg) {
  Operation *parent = arg.getOwner()->getParentOp();
  if (!parent || !isa<triton::gpu::WarpSpecializePartitionsOp>(parent))
    return Value();
  auto ws = dyn_cast<triton::gpu::WarpSpecializeOp>(parent->getParentOp());
  if (!ws || arg.getArgNumber() >= ws.getExplicitCaptures().size())
    return Value();
  return ws.getExplicitCaptures()[arg.getArgNumber()];
}

// The local_alloc a shared memory descriptor is a view of, or null when the
// descriptor cannot be traced to one: a block argument other than a partition
// argument, or a view whose first operand is not its source descriptor. The
// shared memory alias analysis reads the source of a view from its first
// operand too.
inline triton::gpu::LocalAllocOp findViewedAlloc(Value memDesc) {
  while (true) {
    if (auto arg = dyn_cast<BlockArgument>(memDesc)) {
      memDesc = getWarpSpecializeCapture(arg);
      if (!memDesc)
        return nullptr;
      continue;
    }
    Operation *def = memDesc.getDefiningOp();
    if (!def)
      return nullptr;
    if (auto alloc = dyn_cast<triton::gpu::LocalAllocOp>(def))
      return alloc;
    if (!def->hasTrait<OpTrait::MemDescViewTrait>() ||
        def->getNumOperands() == 0 ||
        !isa<triton::gpu::MemDescType>(def->getOperand(0).getType()))
      return nullptr;
    memDesc = def->getOperand(0);
  }
}

// Whether an effect may write the shared memory of `alloc` (null when the
// buffer read is not known). A write through a descriptor of another local
// allocation, or of another memory space, is of a different buffer. A write
// through a pointer is of shared memory in the shared address space and of
// global memory in the global one; a pointer of any other address space, and
// a write without a value, may be of anything.
inline bool mayWriteSharedAlloc(const MemoryEffects::EffectInstance &effect,
                                triton::gpu::LocalAllocOp alloc) {
  if (isa<MemoryEffects::Read, MemoryEffects::Allocate, MemoryEffects::Free>(
          effect.getEffect()))
    return false;
  Value value = effect.getValue();
  if (!value)
    return true;
  Type type = value.getType();
  if (auto memDescTy = dyn_cast<triton::gpu::MemDescType>(type)) {
    if (!isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
            memDescTy.getMemorySpace()))
      return false;
    auto written = findViewedAlloc(value);
    return !alloc || !written || written == alloc;
  }
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    type = tensorTy.getElementType();
  if (auto ptrTy = dyn_cast<triton::PointerType>(type))
    return ptrTy.getAddressSpace() != kSmemReadOrderGlobalAddressSpace;
  return true;
}

// A rectangular window of a shared memory descriptor in the coordinates of its
// local_alloc: the memdesc_subslice offsets and the shape of the view. The
// window has the rank of the allocation; a memdesc_index takes one slot of the
// leading dimension, which becomes a window of extent one at a stage that is
// not known, so that dimension never separates two windows (the stage may be
// the same). A view that is not a chain of subslices and indexes over the
// allocation (an alias, a reshape, a transpose) has no window; claiming one
// would be a guess.
struct SmemWindow {
  llvm::SmallVector<int64_t> offsets;
  llvm::SmallVector<int64_t> shape;
  // Dimensions that a memdesc_index fixed at a stage not known statically.
  llvm::SmallVector<bool> stageDim;
};

inline std::optional<SmemWindow> getSmemWindow(Value memDesc) {
  auto leafTy = dyn_cast<triton::gpu::MemDescType>(memDesc.getType());
  if (!leafTy)
    return std::nullopt;
  // The shape is built from the leaf, one memdesc_index at a time, so that a
  // dimension an index dropped is put back with its whole extent.
  SmemWindow window;
  window.shape.assign(leafTy.getShape().begin(), leafTy.getShape().end());
  window.offsets.assign(window.shape.size(), 0);
  window.stageDim.assign(window.shape.size(), false);
  while (true) {
    if (auto arg = dyn_cast<BlockArgument>(memDesc)) {
      memDesc = getWarpSpecializeCapture(arg);
      if (!memDesc)
        return std::nullopt;
      continue;
    }
    Operation *def = memDesc.getDefiningOp();
    if (!def)
      return std::nullopt;
    if (auto subslice = dyn_cast<triton::gpu::MemDescSubsliceOp>(def)) {
      ArrayRef<int32_t> offsets = subslice.getOffsets();
      if (offsets.size() != window.offsets.size())
        return std::nullopt;
      for (auto [dim, offset] : llvm::enumerate(offsets)) {
        if (offset < 0)
          return std::nullopt;
        window.offsets[dim] += offset;
      }
      memDesc = subslice.getSrc();
      continue;
    }
    if (auto index = dyn_cast<triton::gpu::MemDescIndexOp>(def)) {
      // The view lost the leading dimension: put it back as a dimension whose
      // window is the whole extent of the source, at an unknown stage.
      auto srcTy = cast<triton::gpu::MemDescType>(index.getSrc().getType());
      if (srcTy.getRank() != static_cast<int64_t>(window.shape.size()) + 1)
        return std::nullopt;
      window.shape.insert(window.shape.begin(), srcTy.getShape().front());
      window.offsets.insert(window.offsets.begin(), 0);
      window.stageDim.insert(window.stageDim.begin(), true);
      memDesc = index.getSrc();
      continue;
    }
    // The allocation itself, of the rank the window has by now.
    auto rootTy = dyn_cast<triton::gpu::MemDescType>(memDesc.getType());
    if (!isa<triton::gpu::LocalAllocOp>(def) || !rootTy ||
        rootTy.getRank() != static_cast<int64_t>(window.shape.size()))
      return std::nullopt;
    return window;
  }
}

// Whether two windows of one allocation cannot overlap: some dimension in
// which their ranges are apart. A dimension that a memdesc_index fixed at an
// unknown stage never separates them.
inline bool areSmemWindowsDisjoint(const SmemWindow &lhs,
                                   const SmemWindow &rhs) {
  if (lhs.shape.size() != rhs.shape.size())
    return false;
  for (size_t dim = 0; dim < lhs.shape.size(); ++dim) {
    if (lhs.stageDim[dim] || rhs.stageDim[dim])
      continue;
    if (lhs.offsets[dim] + lhs.shape[dim] <= rhs.offsets[dim] ||
        rhs.offsets[dim] + rhs.shape[dim] <= lhs.offsets[dim])
      return true;
  }
  return false;
}

// Narrows the window of a descriptor to the static subview of it that a load
// reads. `offsets` and `shape` are in the coordinates of that descriptor, which
// are the trailing dimensions of the window: a memdesc_index put the stage
// dimensions in front. A subview of another rank leaves the window as it was.
inline void narrowSmemWindow(SmemWindow &window, ArrayRef<int32_t> offsets,
                             ArrayRef<int64_t> shape) {
  size_t rank = offsets.size();
  if (rank != shape.size() || rank > window.shape.size())
    return;
  size_t lead = window.shape.size() - rank;
  for (size_t dim = 0; dim < rank; ++dim) {
    window.offsets[lead + dim] += offsets[dim];
    window.shape[lead + dim] = shape[dim];
  }
}

// What a load of mutable shared memory reads: the allocation its descriptor is
// a view of (null when it cannot be traced to one) and the part of it the load
// covers (none: take it to be the whole allocation).
struct SmemRead {
  triton::gpu::LocalAllocOp alloc;
  std::optional<SmemWindow> window;
};

inline SmemRead getSmemRead(Value src) {
  return {findViewedAlloc(src), getSmemWindow(src)};
}

// Whether an op may write the shared memory a load reads, judged by its memory
// effects alone: one that declares none, or none it can name, counts as a
// write.
inline bool mayWriteReadBufferByEffects(Operation *op, const SmemRead &read) {
  auto effects = getEffectsRecursively(op);
  if (!effects)
    return true;
  return llvm::any_of(*effects, [&](const auto &effect) {
    return mayWriteSharedAlloc(effect, read.alloc);
  });
}

// The write test of a build without the TLE dialect: memory effects alone. It
// has the interface of SmemReadOrderAnalysis (SmemReadOrder.h), which also
// knows the TLE ops, so that a pass picks one of the two by build.
class EffectsSmemReadOrderAnalysis {
public:
  explicit EffectsSmemReadOrderAnalysis(ModuleOp) {}

  bool mayWriteReadBuffer(Operation *op, const SmemRead &read) {
    return mayWriteReadBufferByEffects(op, read);
  }
};

} // namespace mlir::triton::tle

#endif // TRITON_TLE_ANALYSIS_SMEM_READ_ORDER_CORE_H_
