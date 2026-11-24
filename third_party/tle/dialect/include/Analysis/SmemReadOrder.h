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
// buffer it reads. This is the part that knows the TLE ops; what needs no TLE
// dialect is in SmemReadOrderCore.h, which a build without the dialect uses
// alone. Header-only, so that the core TritonGPU transforms need no link
// dependency on the TLE libraries. It is deliberately more conservative than
// TleMemoryEffectAnalysis: a descriptor that cannot be traced to its
// local_alloc may be any buffer, instead of being a distinct root.

#include "tle/dialect/include/Analysis/SmemReadOrderCore.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::triton::tle {

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
// be re-issued may change what the load reads. The ops are judged by their
// memory effects, except the pipe ops, which declare none:
//  - reader_wait, writer_acquire and writer_commit move no data. The writer's
//    data goes through stores, which are tested as writes on their own.
//  - reader_release gives the slot back to the writer of its pipe, which may
//    then overwrite what it writes (see mayReleaseReadBuffer).
//  - any other pipe op (create, close) is not known to be harmless.
// The analysis caches what it learns about the writers of the pipes of the
// module it is made for, so one object serves one pass run, over a module whose
// pipe ops and writer regions do not move.
class SmemReadOrderAnalysis {
public:
  explicit SmemReadOrderAnalysis(ModuleOp module) : module(module) {}

  bool mayWriteReadBuffer(Operation *op, const SmemRead &read) {
    if (isa<tle::PipeReaderWaitOp, tle::PipeWriterAcquireOp,
            tle::PipeWriterCommitOp>(op))
      return false;
    if (auto release = dyn_cast<tle::PipeReaderReleaseOp>(op))
      return mayReleaseReadBuffer(release, read);
    // A store through the pointers of a local_pointers writes that
    // descriptor's buffer, which is another one than the load reads when the
    // two are known and differ. The write is judged as a write through the
    // descriptor.
    if (auto store = dyn_cast<triton::StoreOp>(op))
      if (Value target = getLocalPointersStoreTarget(store))
        if (auto memDescTy =
                dyn_cast<triton::gpu::MemDescType>(target.getType()))
          if (isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
                  memDescTy.getMemorySpace())) {
            triton::gpu::LocalAllocOp written = findViewedAlloc(target);
            return !read.alloc || !written || written == read.alloc;
          }
    return mayWriteReadBufferByEffects(op, read);
  }

private:
  // One write of the writer of a pipe: the allocation it writes (null: not
  // known, it may be any) and the window of it (none: the whole allocation).
  struct PipeWrite {
    triton::gpu::LocalAllocOp alloc;
    std::optional<SmemWindow> window;
  };
  // What the writer of a pipe writes between its writer_acquire and its
  // writer_commit, everywhere in the module. Not known when a writer region
  // holds an op whose effects are not known, or has no commit in its block.
  struct PipeWrites {
    bool known = false;
    llvm::SmallVector<PipeWrite> writes;
  };

  // Whether a reader_release may let the writer of its pipe overwrite what the
  // load reads. The writer may then write what it writes between its
  // writer_acquire and writer_commit and nothing else, whatever fields the pipe
  // names: another pipe's writer is held back by that pipe's own release. So
  // the release is a write of the read buffer only where that writer writes it,
  // or, when that is not known, wherever the field covers it. This tells apart
  // two pipes that name the whole buffer but whose writers fill its halves.
  bool mayReleaseReadBuffer(tle::PipeReaderReleaseOp release,
                            const SmemRead &read) {
    if (!read.alloc)
      return true;
    for (Value field : release.getFields()) {
      triton::gpu::LocalAllocOp fieldAlloc = findViewedAlloc(field);
      if (!fieldAlloc)
        return true;
      if (fieldAlloc != read.alloc)
        continue;
      std::optional<SmemWindow> fieldWindow = getSmemWindow(field);
      if (fieldWindow && read.window &&
          areSmemWindowsDisjoint(*fieldWindow, *read.window))
        continue;
      if (!pipeWritesAreApartFrom(release, read))
        return true;
    }
    return false;
  }

  // Whether the writer of the pipe of `release` is known to write nothing the
  // load reads.
  bool pipeWritesAreApartFrom(tle::PipeReaderReleaseOp release,
                              const SmemRead &read) {
    if (!read.window)
      return false;
    auto name = release.getPipeName();
    // An unnamed pipe cannot be told from another one.
    if (!name || name->empty())
      return false;
    const PipeWrites &pipe = getPipeWrites(*name);
    if (!pipe.known)
      return false;
    for (const PipeWrite &write : pipe.writes) {
      if (!write.alloc)
        return false;
      if (write.alloc != read.alloc)
        continue;
      if (!write.window || !areSmemWindowsDisjoint(*write.window, *read.window))
        return false;
    }
    return true;
  }

  const PipeWrites &getPipeWrites(StringRef name) {
    auto it = pipeWrites.find(name);
    if (it != pipeWrites.end())
      return it->second;
    return pipeWrites.try_emplace(name, collectPipeWrites(name)).first->second;
  }

  PipeWrites collectPipeWrites(StringRef name) {
    PipeWrites pipe;
    bool sawAcquire = false;
    bool known = true;
    module.walk([&](tle::PipeWriterAcquireOp acquire) {
      if (acquire.getPipeName() != name)
        return WalkResult::advance();
      sawAcquire = true;
      // The writer region: from the acquire to the commit of the same pipe in
      // the same block.
      Operation *op = acquire->getNextNode();
      for (; op; op = op->getNextNode()) {
        auto commit = dyn_cast<tle::PipeWriterCommitOp>(op);
        if (commit && commit.getPipeName() == name)
          break;
        if (!addWritesOf(op, pipe.writes)) {
          known = false;
          return WalkResult::interrupt();
        }
      }
      if (!op) {
        known = false;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    pipe.known = known && sawAcquire;
    if (!pipe.known)
      pipe.writes.clear();
    return pipe;
  }

  // Appends the shared memory `op` writes. False when that is not known: the
  // op declares no effects, or writes through a pointer that is not that of a
  // local_pointers.
  static bool addWritesOf(Operation *op, llvm::SmallVector<PipeWrite> &writes) {
    auto add = [&](Value memDesc) {
      writes.push_back({findViewedAlloc(memDesc), getSmemWindow(memDesc)});
    };
    if (auto store = dyn_cast<triton::StoreOp>(op))
      if (Value target = getLocalPointersStoreTarget(store)) {
        add(target);
        return true;
      }
    auto effects = getEffectsRecursively(op);
    if (!effects)
      return false;
    for (const MemoryEffects::EffectInstance &effect : *effects) {
      if (isa<MemoryEffects::Read, MemoryEffects::Allocate,
              MemoryEffects::Free>(effect.getEffect()))
        continue;
      Value value = effect.getValue();
      if (!value)
        return false;
      Type type = value.getType();
      if (auto memDescTy = dyn_cast<triton::gpu::MemDescType>(type)) {
        if (isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
                memDescTy.getMemorySpace()))
          add(value);
        continue;
      }
      if (auto tensorTy = dyn_cast<RankedTensorType>(type))
        type = tensorTy.getElementType();
      auto ptrTy = dyn_cast<triton::PointerType>(type);
      if (!ptrTy || ptrTy.getAddressSpace() != kSmemReadOrderGlobalAddressSpace)
        return false;
    }
    return true;
  }

  ModuleOp module;
  llvm::StringMap<PipeWrites> pipeWrites;
};

} // namespace mlir::triton::tle

#endif // TRITON_TLE_ANALYSIS_SMEM_READ_ORDER_H_
