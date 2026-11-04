#ifndef ANALYSIS_TKO_OP_SEMANTICS_H
#define ANALYSIS_TKO_OP_SEMANTICS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace triton {
namespace tileir {
namespace tko_op_semantics {

/// Coarse effect class for token-ordered TKO operations.
enum class EffectKind { None, Read, Write, Atomic };

/// Return the memory effect carried by a TKO memory operation.
EffectKind getEffectKind(Operation *op);

/// True iff `op` is a TKO read/write/atomic memory operation.
bool isMemOp(Operation *op);

/// True iff `op` writes memory. Atomics are treated as writes by token
/// generation because they need both WAW and WAR ordering.
bool isWriteLike(Operation *op);

/// True iff `op` reads memory and does not write memory.
bool isReadOnly(Operation *op);

/// True iff `op` has acquire-fence semantics for token generation.
///
/// Current mapping:
///   - gdc_wait_tko is treated as acquire because later same-kernel memory ops
///     must execute after the predecessor kernel has completed.
///   - gpu.barrier is also acquire because it is a total local fence.
///
/// Acquire fences publish their result token to later tokenized memory ops, but
/// do not consume prior local memory state.
bool isAcquireFence(Operation *op);

/// True iff `op` has standard release-fence semantics for token generation.
/// Release fences consume prior local memory state, but do not publish their
/// result token to later tokenized memory ops.
///
/// Current mapping:
///   - gpu.barrier is release because it is a total local fence.
///   - There is no standalone release-only TKO op modeled here today.
bool isReleaseFence(Operation *op);

/// True iff `op` signals dependent kernels after producer memory is ready.
///
/// gdc_launch_dependents_tko is intentionally not modeled as a full release
/// fence here: AutoGenMemoryToken orders it after prior stores/atomics and
/// acquire-side fences, but not after read-only local state. This preserves the
/// PDL overlap pattern where pre-signal reads in the producer should not make
/// the dependent launch wait for unrelated read-token accumulation.
bool isDependentLaunchSignal(Operation *op);

/// True iff `op` is a total local ordering point. Today this is only
/// gpu.barrier, which AutoGenMemoryToken erases after transferring its ordering
/// semantics into the token graph.
bool isTotalFence(Operation *op);

/// Return true iff a TKO memory/fence op already has a non-null token operand.
bool hasUserToken(Operation *op);

/// Return the access operand that determines the allocation root for a TKO
/// memory op: pointer tile, view, or tensor-memory handle. Returns null for
/// non-memory ops.
Value getAccessValue(Operation *op);

/// Return the single access value forwarded by an addressing-only op. Alias
/// analysis uses this to walk through view construction, pointer offsetting,
/// and shape-only tile rewrites.
Value getPassthroughAccessValue(Operation *op);

} // namespace tko_op_semantics
} // namespace tileir
} // namespace triton
} // namespace mlir

#endif // ANALYSIS_TKO_OP_SEMANTICS_H
