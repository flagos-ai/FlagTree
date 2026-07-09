#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_WAITCNTHCUUTILITY_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_WAITCNTHCUUTILITY_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::AMD {

// HCU Note: Starting with ZD and newer chips, s_barrier no longer implies
// s_waitcnt vmcnt(0). For patterns that require this ordering guarantee,
// we detect them explicitly and insert s_waitcnt vmcnt(0) in IR.
void addWaitCntHCU(ModuleOp mod);

} // namespace mlir::triton::AMD

#endif
