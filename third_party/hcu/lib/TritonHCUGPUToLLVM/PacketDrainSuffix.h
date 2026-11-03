// Accept only a complete acyclic, memory-free/global-store suffix after the
// already modeled unique drain. No shared event or queue state can escape the
// drain, and every additional block must belong to this unchanged suffix.
#pragma once
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
namespace flagtree::hcu {
template <typename IsNoop>
static bool pureGlobalStoreSuffix(llvm::Function &kernel,
                                  llvm::BasicBlock *pre,
                                  llvm::BasicBlock *body,
                                  llvm::BasicBlock *drain,
                                  IsNoop isNoop) {
  using namespace llvm;
  SmallPtrSet<BasicBlock *, 32> seen;
  SmallVector<BasicBlock *> blocks;
  for (BasicBlock *b : successors(drain)) blocks.push_back(b);
  for (unsigned n = 0; n < blocks.size(); ++n) {
    BasicBlock *b = blocks[n];
    if (b == pre || b == body || b == drain) return false;
    if (!seen.insert(b).second) continue;
    if (!isa<BranchInst, ReturnInst>(b->getTerminator())) return false;
    for (Instruction &i : *b) {
      if (auto *st = dyn_cast<StoreInst>(&i)) {
        if (st->getPointerAddressSpace() != 1 || st->isVolatile() || st->isAtomic()) return false;
        continue;
      }
      if (auto *call = dyn_cast<CallBase>(&i))
        if (auto *f = call->getCalledFunction()) {
          StringRef name = f->getName();
          if (name.starts_with("llvm.amdgcn.raw.ptr.buffer.store.") ||
              name.starts_with("llvm.amdgcn.struct.ptr.buffer.store.")) continue;
        }
      if (!isNoop(&i)) return false;
    }
    for (BasicBlock *next : successors(b)) blocks.push_back(next);
  }
  if (seen.size() + 3 != kernel.size()) return false;
  // Kahn's algorithm rejects even an otherwise pure suffix with a cycle.
  DenseMap<BasicBlock *, unsigned> incoming;
  SmallVector<BasicBlock *> ready;
  for (BasicBlock *b : seen) incoming[b] = 0;
  for (BasicBlock *b : seen)
    for (BasicBlock *next : successors(b)) ++incoming[next];
  for (auto item : incoming) if (!item.second) ready.push_back(item.first);
  unsigned count = 0;
  while (!ready.empty()) {
    BasicBlock *b = ready.pop_back_val(); ++count;
    for (BasicBlock *next : successors(b)) if (!--incoming[next]) ready.push_back(next);
  }
  return count == seen.size();
}
}
