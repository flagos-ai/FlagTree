#ifndef TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_TRITONILUVATARGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_TRITONILUVATARGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/Operation.h"

namespace mlir::triton::ILUVATAR {

// Filter function used in the Iluvatar backend to drop unnecessary barriers
// during Membar analysis. Filters applied by this function:
//
// 1) Do not create a barrier between an SME AsyncCopyGlobalToLocal (a copy with
//    an SME blocked encoding and explicit inputStride, lowered through the SME
//    G2S engine) and a LocalLoad that is synced via AsyncWait. The pipeliner
//    double-buffers shared memory: it reads slot i while async-copying into
//    slot i+1. Membar cannot prove the rotating subview indices from the same
//    shared allocation do not alias, so it would insert heavy CTA barriers both
//    before the next async copy (spurious WAR) and around the local_load (RAW),
//    serializing the prefetch against the dot and defeating pipelining. The
//    async_wait's waitcnt already synchronizes the data and the ping-pong slots
//    are disjoint, so the barrier is redundant.
//
// Both gates are important: block-pointer rewriting can attach inputStride to a
// regular copy before SME eligibility is known. A regular cp.async stages data
// through shared memory whose cross-warp visibility still needs the heavy
// barrier, so it must NOT be filtered.
bool membarFilter(Operation *op1, Operation *op2);

} // namespace mlir::triton::ILUVATAR

#endif
