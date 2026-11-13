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

#ifndef TRITON_ADAPTER_BLOCK_ID_OPT_PASSES_H
#define TRITON_ADAPTER_BLOCK_ID_OPT_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUBUsageOptPass();
std::unique_ptr<OperationPass<ModuleOp>> createMergeSameSourceAxisPass();
void registerMergeSameSourceAxisPass();
std::unique_ptr<OperationPass<ModuleOp>> createUnifyAllocBlockPass();
void registerUnifyAllocBlockPass();
std::unique_ptr<OperationPass<ModuleOp>> createMergeVectorIfBlockPass();
void registerMergeVectorIfBlockPass();
std::unique_ptr<OperationPass<ModuleOp>> createUnifyStoreBlockPass();
void registerUnifyStoreBlockPass();
std::unique_ptr<OperationPass<ModuleOp>> createFixpipeOptPass();
std::unique_ptr<OperationPass<ModuleOp>> createMergeSmallBlockPass();
void registerMergeSmallBlockPass();
std::unique_ptr<OperationPass<ModuleOp>> createSinkI1ProducersIntoUsersPass();
std::unique_ptr<OperationPass<ModuleOp>> createBroadcastUBOptPass();
std::unique_ptr<OperationPass<ModuleOp>> createMoveLoadIntoUserPass();
std::unique_ptr<OperationPass<ModuleOp>> createPosMaskPatternPass();
std::unique_ptr<OperationPass<ModuleOp>> createRelocateMemrefDeclPass();
std::unique_ptr<OperationPass<ModuleOp>> createMergeComputeBlockPass();
std::unique_ptr<OperationPass<ModuleOp>>
createMergeInputInitSharedCubeBlockPass();

std::unique_ptr<OperationPass<ModuleOp>> createSplitIfByBlockIdPass();
std::unique_ptr<OperationPass<ModuleOp>> createExpSubfPatternPass();
std::unique_ptr<OperationPass<ModuleOp>> createExpLoadPatternPass();
void registerSplitIfByBlockIdPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_BLOCK_ID_OPT_PASSES_H
