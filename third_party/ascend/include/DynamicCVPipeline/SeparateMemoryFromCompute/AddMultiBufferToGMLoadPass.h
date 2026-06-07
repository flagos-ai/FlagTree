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

#ifndef TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_PASS_H
#define TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_PASS_H

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace triton {

// Rewrites GM load operations inside scf.for loops into a multi-buffer
// producer/consumer structure to overlap memory transfers with compute.
class AddMultiBufferToGMLoadPass : public PassWrapper<AddMultiBufferToGMLoadPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddMultiBufferToGMLoadPass)

    StringRef getArgument() const override { return "gm-load-multi-buffer"; }
    void runOnOperation() override;

private:
    // Step 1: Collect marked ops and group by enclosing forOp
    void collectAndGroupMarkedOps();

    // Step 2: Sort contexts inner-first
    void sortContextsInnerFirst();

    // Step 3: Transform each for loop with multi-buffer logic
    LogicalResult applyMultiBufferToGMLoadLoops();

    // Step 4: Cleanup transformed IR
    void cleanupTransformedIR();

    // Shared state between steps
    llvm::SmallVector<gmload::MarkedLoad> markedOps_;
    llvm::SmallVector<gmload::ForBufferCtx, 0> contexts_;
    llvm::DenseSet<Operation *> allCtxForOps_;
};

// Create the pass
std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferToGMLoadPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_PASS_H
