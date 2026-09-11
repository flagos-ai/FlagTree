#ifndef TRITON_METAX_GLUON_KERNEL_REPLAY_H
#define TRITON_METAX_GLUON_KERNEL_REPLAY_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace mlir::triton {
class FuncOp;
}

namespace mlir::triton::gpu::metax::gluon {

/// Conservative replay contract. `replayable` is true only when every
/// externally visible memory effect is traced to a kernel tensor argument and
/// the reported no-alias preconditions are sufficient to redirect all writes
/// to independent runtime scratch buffers. Atomic and read/write arguments are
/// explicit because their scratch state must be restored before every replay.
/// This is not a general race-freedom proof.
struct KernelReplayEffectContract {
  bool replayable = false;
  SmallVector<unsigned, 4> tensorArgs;
  SmallVector<unsigned, 4> readArgs;
  SmallVector<unsigned, 4> writtenArgs;
  SmallVector<unsigned, 4> writeOnlyArgs;
  SmallVector<unsigned, 4> atomicArgs;
  SmallVector<std::pair<unsigned, unsigned>, 4> requiredNoAlias;
};

/// Derive external buffer effects from MemoryEffectOpInterface and pointer SSA
/// backward slices. Missing effect or pointer provenance yields
/// replayable=false. Argument indices are flattened lowered `tt.func` ABI
/// ordinals: constexpr parameters are absent. A JIT caller whose bound-argument
/// list retains constexpr values must translate through the compiler signature
/// instead of interpreting these as Python parameter ordinals.
KernelReplayEffectContract analyzeKernelReplayEffectContract(triton::FuncOp func);

/// Serialize the compiler/runtime replay contract as deterministic compact
/// JSON suitable for an internal module string attribute.
std::string serializeKernelReplayEffectContract(
    const KernelReplayEffectContract &contract);

} // namespace mlir::triton::gpu::metax::gluon

#endif // TRITON_METAX_GLUON_KERNEL_REPLAY_H
