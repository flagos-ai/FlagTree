#ifndef TRITON_TLE_ANALYSIS_PIPE_EFFECT_ANALYSIS_H_
#define TRITON_TLE_ANALYSIS_PIPE_EFFECT_ANALYSIS_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

#include <optional>

namespace mlir::triton::tle {

bool sameIndexValue(Value lhs, Value rhs);

std::optional<int> getPointerAddressSpace(Value value);
bool isSharedPointer(Value value);
bool isProvenGlobalPointer(Value value);
bool isNonSharedPointer(Value value);

Value stripConvertLayouts(Value value);

struct LocalStoreTarget {
  Value memdesc;
  Type valueType;
};

std::optional<LocalStoreTarget> getLocalStoreTarget(Operation *op);
std::optional<LocalStoreTarget> getAsyncCopyTarget(Operation *op);

struct CompletedAsyncCopyState {
  llvm::DenseSet<Value> completedTokens;
  bool allPriorAsyncCopiesComplete = false;
};

bool recordsCompletedAsyncCopies(triton::gpu::AsyncWaitOp wait);
void recordCompletedAsyncWait(triton::gpu::AsyncWaitOp wait,
                              CompletedAsyncCopyState &state);
void propagateCompletedAsyncCommitGroup(triton::gpu::AsyncCommitGroupOp commit,
                                        CompletedAsyncCopyState &state);
bool isAsyncCopyComplete(triton::gpu::AsyncCopyGlobalToLocalOp copy,
                         const CompletedAsyncCopyState &state);

bool isCtaInvariantSpecialRegisterRead(Operation *op);
bool canInterleaveBeforePipeMetadataOp(Operation *op);

} // namespace mlir::triton::tle

#endif // TRITON_TLE_ANALYSIS_PIPE_EFFECT_ANALYSIS_H_
