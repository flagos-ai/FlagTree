#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include <optional>
#include "llvm/ADT/SmallVector.h"
#include <utility>

namespace mlir { class ModuleAllocation; }

namespace mlir::triton::HCU {

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
