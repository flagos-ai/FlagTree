/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
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
