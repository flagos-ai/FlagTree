/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_OUTER_SCOPE_PASS_H
#define TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_OUTER_SCOPE_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace triton {

// Data structures

/// Buffer alloc pair: {allocOp, markOp}
struct BufferAllocPair {
    Operation *allocOp = nullptr;
    Operation *markOp = nullptr;
};

/// Transfer operation chain for sender or receiver side
struct TransferOpChain {
    Operation *waitOp = nullptr;
    Operation *transferOp = nullptr;  // fixpipe / hir.copy / memory_space_cast / convert_layout
    Operation *toTensorOp = nullptr;  // bufferization.to_tensor (memory_space_cast scenario only)
    Operation *setOp = nullptr;
};

/// Buffer alloc info: {sender, receiver}
struct BufferAllocInfo {
    BufferAllocPair sender;
    BufferAllocPair receiver;
};

/// Extra sync info for sync ops whose parent does NOT have main_loop attribute
struct ExtraSyncInfo {
    Operation *setOp = nullptr;
    Operation *waitOp = nullptr;
};

/// Transfer chain info: {senderChain, receiverChain}
struct TransferChainInfo {
    TransferOpChain sender;
    TransferOpChain receiver;
};

/// Complete transfer group information
struct TransferGroupInfo {
    int tid = -1;
    int originalFlag = -1;
    int outputFlag = -1;
    bool isCtoV = false;  // true=C→V, false=V→C

    BufferAllocPair senderBuf;
    BufferAllocPair receiverBuf;

    TransferOpChain senderChain;
    TransferOpChain receiverChain;

    // Input/output buffer values
    Value senderInputBuffer;
    Value senderOutputBuffer;
    Value receiverInputBuffer;
    Value receiverOutputBuffer;

    // TCB ID shared across all 4 buffers in the same transfer group
    int tcbId = -1;

    // Extra sync positions for output flag synchronization
    Operation *extraSyncSetOp = nullptr;
    Operation *extraSyncWaitOp = nullptr;
};

/// AddMultiBufferOuterScopePass for adding outer (CV inter-core) multi-buffer optimization
class AddMultiBufferOuterScopePass
    : public PassWrapper<AddMultiBufferOuterScopePass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddMultiBufferOuterScopePass)

    // Constructor
    AddMultiBufferOuterScopePass() = default;

    // Pass argument
    StringRef getArgument() const override { return "add_multi_buffer_outer_scope"; }

    // Run the pass
    void runOnOperation() override;

    // Get dependent dialects
    void getDependentDialects(DialectRegistry &registry) const override;
};

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferOuterScopePass();

void registerAddMultiBufferOuterScopePasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_OUTER_SCOPE_PASS_H
