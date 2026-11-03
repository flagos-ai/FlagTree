#ifndef TRITON_HCU_TERMINAL_READONLY_PUBLICATION_H
#define TRITON_HCU_TERMINAL_READONLY_PUBLICATION_H

#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <utility>

namespace mlir::triton::HCU {
namespace terminal_readonly_publication {
namespace ttg = ::mlir::triton::gpu;
namespace hcug = ::mlir::triton::hcugpu;

// These have register-only lowering. Pure alone is NOT a sufficient proof:
// reduce/scan and inline asm may allocate/write LDS or synchronize.
// A convert_layout needs an independent no-shared-memory lowering proof.
inline bool isRegisterOnly(Operation *op) {
  if (op->getNumRegions() || op->getNumSuccessors() || !isMemoryEffectFree(op))
    return false;
  for (Type type : op->getOperandTypes())
    if (isa<ttg::MemDescType>(type))
      return false;
  for (Type type : op->getResultTypes())
    if (isa<ttg::MemDescType>(type))
      return false;
  if (op->getName().getDialectNamespace() == "arith")
    return true;
  if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
          triton::AddPtrOp, triton::MakeRangeOp>(op))
    return true;
  if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    auto src = dyn_cast<RankedTensorType>(convert.getSrc().getType());
    auto dst = dyn_cast<RankedTensorType>(convert.getType());
    if (!src || !dst || src.getShape() != dst.getShape() ||
        src.getElementType() != dst.getElementType())
      return false;
    // Use the same classification as allocation/lowering. Register reorders
    // and supported within-wave shuffles allocate no LDS and publish no data
    // to another warp. Any shuffle-result completion remains an SSA dependency
    // for ordinary lowering; no DS-only waits are deleted here. Cross-warp
    // conversions (or unproved decompositions) keep the original publication.
    return !cvtNeedsSharedMemory(src, dst);
  }
  if (auto dot = dyn_cast<triton::DotOp>(op)) {
    auto a = dyn_cast<RankedTensorType>(dot.getA().getType());
    auto b = dyn_cast<RankedTensorType>(dot.getB().getType());
    auto result = dyn_cast<RankedTensorType>(dot.getResult().getType());
    if (!a || !b || !result)
      return false;
    auto encoding = dyn_cast_or_null<ttg::AMDMfmaEncodingAttr>(result.getEncoding());
    auto aEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(a.getEncoding());
    auto bEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(b.getEncoding());
    return encoding && aEncoding && bEncoding &&
           aEncoding.getParent() == encoding && bEncoding.getParent() == encoding;
  }
  return false;
}

inline bool isPlainGlobalStore(Operation *op) {
  Value ptr;
  if (auto store = dyn_cast<hcug::BufferStoreOp>(op))
    ptr = store.getPtr();
  else if (auto store = dyn_cast<triton::StoreOp>(op))
    ptr = store.getPtr();
  else
    return false;
  Type type = ptr.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  auto pointer = dyn_cast<triton::PointerType>(type);
  return pointer && pointer.getAddressSpace() == 1 && !op->getNumRegions();
}

struct Pair {
  hcug::AsyncWaitOp wait;
  ttg::LocalBarrierOp barrier;
};

// Prove the WHOLE suffix before deleting anything. Same-block lexical order
// gives dominance; tt.return (not cf.br/cond_br) makes this a terminal exit.
// Other function blocks/loops may precede it but cannot follow this suffix.
inline bool proveSuffix(hcug::AsyncWaitOp anchor, ttg::LocalBarrierOp first,
                        llvm::SmallVectorImpl<Pair> &redundant) {
  Block *block = anchor->getBlock();
  if (anchor.getNumInst() != 0 || anchor->getNextNode() != first.getOperation() ||
      !isa<triton::ReturnOp>(block->getTerminator()))
    return false;
  bool afterDealloc = false;
  bool outputStoresStarted = false;
  for (Operation *op = first->getNextNode(); op;) {
    if (isa<triton::ReturnOp>(op))
      return op == block->getTerminator() && op->getNumOperands() == 0 &&
             !redundant.empty();
    if (op->getNumRegions() || op->getNumSuccessors())
      return false; // No SCF, warp-specialized region, call/branch speculation.
    if (auto wait = dyn_cast<hcug::AsyncWaitOp>(op)) {
      auto barrier = dyn_cast_or_null<ttg::LocalBarrierOp>(op->getNextNode());
      if (wait.getNumInst() != 0 || !wait.getResult().use_empty() ||
          !barrier || afterDealloc || outputStoresStarted)
        return false;
      redundant.push_back({wait, barrier});
      op = barrier->getNextNode();
      continue;
    }
    if (auto read = dyn_cast<ttg::LocalLoadOp>(op)) {
      if (afterDealloc || outputStoresStarted ||
          !isa<ttg::SharedMemorySpaceAttr>(read.getSrc().getType().getMemorySpace()))
        return false;
      // Reads remain in place. Their SSA data dependencies still force DS
      // completion at the original consumers; we never delete DS-only waits.
    } else if (isa<ttg::LocalDeallocOp>(op)) {
      // No later shared read/view/alloc/store is accepted. Dealloc itself has
      // no hardware write; there can be no future allocation/slot reuse before
      // this terminal return. Multiple final deallocs are harmless.
      if (outputStoresStarted)
        return false;
      afterDealloc = true;
    } else if (isPlainGlobalStore(op)) {
      // No global effects are allowed BETWEEN the deleted publication pairs.
      // Once stores begin only pure register work/more output stores/return
      // are accepted; local reads and waits/barriers can no longer appear.
      outputStoresStarted = true;
    } else if (!isRegisterOnly(op)) {
      return false;
    }
    op = op->getNextNode();
  }
  return false;
}
} // namespace terminal_readonly_publication

// Opt-in late HCU-IR cleanup AFTER unchanged Membar hazard analysis and BEFORE
// addWaitCntHCU. The full anchor wait0 + original DS0/CTA barrier is preserved.
inline unsigned coalesceTerminalReadonlyPublication(ModuleOp mod,
                                                    llvm::StringRef arch) {
  using namespace terminal_readonly_publication;
  auto enabled = mod->getAttrOfType<BoolAttr>(
      "ttg.hcu.terminal_readonly_publication");
  auto ctas = mod->getAttrOfType<IntegerAttr>("ttg.num-ctas");
  if (!enabled || !enabled.getValue() || arch != "gfx936" || !ctas ||
      ctas.getInt() != 1)
    return 0;
  unsigned removedPairs = 0;
  mod.walk([&](triton::FuncOp func) {
    if (!triton::isKernel(func) || func.getFunctionType().getNumResults())
      return; // A private callee may return into a caller that reuses LDS.
    for (Block &block : func.getBody()) {
      llvm::SmallVector<Pair> erase;
      // Skip unsuccessful anchors conservatively. A later full publication may
      // start an independently proved terminal suffix (no crossed writes).
      for (Operation &op : block) {
        auto wait = dyn_cast<hcug::AsyncWaitOp>(&op);
        auto barrier = dyn_cast_or_null<ttg::LocalBarrierOp>(op.getNextNode());
        if (!wait || !barrier)
          continue;
        llvm::SmallVector<Pair> candidate;
        if (proveSuffix(wait, barrier, candidate)) {
          erase = std::move(candidate);
          break;
        }
      }
      // All proof decisions completed before IR mutation; erase no live SSA.
      for (Pair pair : erase) {
        pair.barrier.erase();
        pair.wait.erase();
        ++removedPairs;
      }
    }
  });
  mod->setAttr("ttg.hcu.terminal_readonly_publication_pairs",
               IntegerAttr::get(IntegerType::get(mod.getContext(), 32), removedPairs));
  return removedPairs;
}
} // namespace mlir::triton::HCU
#endif
