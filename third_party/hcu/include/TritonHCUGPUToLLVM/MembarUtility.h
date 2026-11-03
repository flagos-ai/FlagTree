#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include <optional>
#include "llvm/ADT/SmallVector.h"
#include <utility>

namespace mlir { class ModuleAllocation; }

namespace mlir::triton::HCU {

// Ephemeral witness from the SAME v0-prefix2 verifier used by preflight.
// No IR certificate, global state, or proof survives a lowering invocation.
struct LegacyV0Prefix2Queue {
  Value root, initialView, futureView, currentDescriptor, currentToken;
  Value counter, nextIndex;
  Block *entry = nullptr, *header = nullptr, *body = nullptr, *exit = nullptr;
  Operation *initialCopy = nullptr, *futureCopy = nullptr;
  Operation *initialCommit = nullptr, *futureCommit = nullptr;
  Operation *mainRead = nullptr, *exitRead = nullptr;
  Operation *mainWait = nullptr, *exitWait = nullptr;
};
std::optional<LegacyV0Prefix2Queue>
getVerifiedLegacyV0Prefix2Queue(ModuleOp mod);

// Private exact-pair WAR proof for the existing opaque-B + ordinary async-A
// two-slot queue. Does not change IR, publication, waits, or alias annotations.
llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareLegacyV0Prefix2WAR(ModuleOp mod, ModuleAllocation &allocation);

// Private default-off LP1 diagnostic. Reuse the full two-slot native-B proof
// on this same CF, require its existing head publication barrier, and return
// only future MatrixLoadToLocal -> current body LocalLoad (distinct slots).
// No IR mutation, certificate, cross-iteration WAR or generic alias change.
llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareLegacyV0LocalPrefetchRAW(ModuleOp mod);

// Private N4 synchronous-A exact WAR witness. Reuses the current B proof's
// CF anchor, independently checks A queue/aliases/values and real disjoint frames.
// Quiet unsupported modules return no pairs; no IR or old B-path change.
llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareLegacyV0LocalPrefetchAWAR(ModuleOp mod, ModuleAllocation &allocation,
    ArrayRef<std::pair<Operation *, Operation *>> bPairs);

// Current paired indexed A/B packets: exact same-iteration WAR pairs.
// Call after head wait barriers exist and immediately before membar.
llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareLegacyV0IndexedABWAR(ModuleOp mod, ModuleAllocation &allocation);

// Private post-membar schedule experiment. Takes only the ephemeral pairs
// already proved above; unsupported requests fail without moving anything.
LogicalResult scheduleLegacyV0LocalPrefetchReadFirst(ModuleOp mod,
    ArrayRef<std::pair<Operation *, Operation *>> pairs);

// Ignore only async-copy RAW dependencies covered by AsyncWait; retain WAR
// dependencies because the next copy may overwrite a tile still being read.
// The pipeline does not provide an unconditional two-buffer guarantee.
bool membarFilter(Operation *op1, Operation *op2);

// Private gfx936 diagnostic. Prove a narrowly shaped synchronous two-slot CF
// loop immediately before membar, synchronize its unique preheader, and return
// only its exact LocalLoad -> LocalStore WAR pairs. No global state or IR
// certificates survive this invocation. Unsupported CFGs return no pairs.
llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareTwoSlotPrefetchWAR(ModuleOp mod);

// Private gfx936 read-first diagnostic. Prove the final two-root async CF
// queue's slot phases and mandatory publication barrier without mutating IR.
// Return only exact main-body LocalLoad -> BufferLoadToLocal WAR pairs;
// unsupported CFGs/aliases return no pairs. Call immediately before membar.
llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareAsyncSlotPrefetchWAR(ModuleOp mod);
} // namespace mlir::triton::HCU

#endif
