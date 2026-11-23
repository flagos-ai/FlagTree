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

#ifndef TRITON_TLE_ANALYSIS_SMEM_READ_ORDER_H_
#define TRITON_TLE_ANALYSIS_SMEM_READ_ORDER_H_

// The write test behind keep-smem-read-order, shared by
// tritongpu-reorder-instructions and triton-tle-optimize-local-pointer-loads:
// a load of mutable shared memory moves only past ops that cannot write the
// buffer it reads. Header-only, so that the core TritonGPU transforms need no
// link dependency on the TLE libraries. It is deliberately more conservative
// than TleMemoryEffectAnalysis: a descriptor that cannot be traced to its
// local_alloc may be any buffer, instead of being a distinct root. A block
// argument of a warp_specialize partition is traced through the capture it
// stands for.

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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

// Whether `release` may let a writer overwrite shared memory that a load of
// `readSrc` (the descriptor of its local_pointers, within `readAlloc`) reads.
// The release hands a slot of each of its fields back to the writer, so it
// counts as a write of every field, except of one that is a static window of
// the allocation apart from the window the load reads. A window of the read
// descriptor is not known when it is not a chain of subslices, and then the
// whole of the read buffer is read.
inline bool mayReleaseReadBuffer(tle::PipeReaderReleaseOp release,
                                 Value readSrc,
                                 triton::gpu::LocalAllocOp readAlloc) {
  if (!readAlloc)
    return true;
  std::optional<SmemWindow> readWindow = getSmemWindow(readSrc);
  for (Value field : release.getFields()) {
    triton::gpu::LocalAllocOp fieldAlloc = findViewedAlloc(field);
    if (!fieldAlloc)
      return true;
    if (fieldAlloc != readAlloc)
      continue;
    std::optional<SmemWindow> fieldWindow = getSmemWindow(field);
    if (!readWindow || !fieldWindow ||
        !areSmemWindowsDisjoint(*fieldWindow, *readWindow))
      return true;
  }
  return false;
}

// The descriptor a store through a shared-address-space pointer writes, when
// the pointer is that of a tle.local_pointers (seen through layout
// conversions): the store then writes within that descriptor, whatever its
// indices are. Null for any other pointer, which may point anywhere in shared
// memory.
inline Value getLocalPointersStoreTarget(triton::StoreOp store) {
  Value ptr = store.getPtr();
  while (auto cvt = ptr.getDefiningOp<triton::gpu::ConvertLayoutOp>())
    ptr = cvt.getSrc();
  if (auto localPointers = ptr.getDefiningOp<tle::LocalPointersOp>())
    return localPointers.getSrc();
  return Value();
}

// Whether an op between a load of mutable shared memory and the place it would
// be re-issued may change what the load reads. `op` is the op itself, not one
// of its effects, because a pipe op declares no memory effects of its own:
//  - reader_wait, writer_acquire and writer_commit move no data. The writer's
//    data goes through stores, which are tested as writes on their own.
//  - reader_release gives the slot back, after which a writer may overwrite it:
//    mayReleaseReadBuffer.
//  - any other pipe op (create, close) is not known to be harmless.
// Every other op is tested through its effects; one that declares none is a
// write.
inline bool mayWriteReadBuffer(Operation *op, Value readSrc,
                               triton::gpu::LocalAllocOp readAlloc) {
  if (isa<tle::PipeReaderWaitOp, tle::PipeWriterAcquireOp,
          tle::PipeWriterCommitOp>(op))
    return false;
  if (auto release = dyn_cast<tle::PipeReaderReleaseOp>(op))
    return mayReleaseReadBuffer(release, readSrc, readAlloc);
  // A store through the pointers of a local_pointers writes that descriptor's
  // buffer, which is another one than the load reads when the two are known
  // and differ. The write is judged as a write through the descriptor.
  if (auto store = dyn_cast<triton::StoreOp>(op))
    if (Value target = getLocalPointersStoreTarget(store))
      if (auto memDescTy = dyn_cast<triton::gpu::MemDescType>(target.getType()))
        if (isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
                memDescTy.getMemorySpace())) {
          triton::gpu::LocalAllocOp written = findViewedAlloc(target);
          return !readAlloc || !written || written == readAlloc;
        }
  auto effects = getEffectsRecursively(op);
  if (!effects)
    return true;
  return llvm::any_of(*effects, [&](const auto &effect) {
    return mayWriteSharedAlloc(effect, readAlloc);
  });
}

} // namespace mlir::triton::tle

#endif // TRITON_TLE_ANALYSIS_SMEM_READ_ORDER_H_
