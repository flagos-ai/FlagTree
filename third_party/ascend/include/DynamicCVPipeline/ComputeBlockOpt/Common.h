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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_OPT_COMMON_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_OPT_COMMON_H

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace CVPipeline {

/**
 * @brief Detect if unifying a list of operations to target block_id would
 * create a cycle
 *
 * This helper temporarily assigns every op in @p opsToUnify to @p
 * targetBlockId, walks the SSA + memory dependency edges, and reports whether
 * the resulting block-level dependency graph would contain a cycle. The
 * temporary block_id assignments are always rolled back before returning, so
 * the function leaves
 * @p bm in its original state regardless of the result.
 *
 * Shared by the ComputeBlockOpt passes (e.g. UnifyAllocBlockPass and
 * MergeVectorIfBlockPass) that merge operations into a common block_id.
 *
 * @param opsToUnify Block-level operations to add to the safe set (okSet)
 * @param memGraph Memory dependence graph for RAW/WAW/WAR dependency analysis
 * @param targetBlockId Target block_id after unification
 * @param bm Block-id manager used to query/temporarily mutate block ids
 * @return bool Returns true if unification would create a cycle, false
 * otherwise
 */
bool willCreateCycle(llvm::ArrayRef<Operation *> opsToUnify,
                     const MemoryDependenceGraph &memGraph, int targetBlockId,
                     ComputeBlockIdManager &bm);

/**
 * @brief Clone scalar-producing ops shared between a pattern and other blocks
 *
 * Walks the pattern's scalar-producing ops in reverse topological order. If
 * such an op's result is used by an op that is outside both the pattern and
 * the target's original block (matchedOps[0]), clones the op and redirects
 * those external uses to the clone, keeping the original for pattern-internal
 * use. This prevents the cross-block dependency cycle that would otherwise
 * appear after unifying the pattern ops into matchedOps[0]'s block_id.
 *
 * @param bmOriginal "Original" block_id view (before any fusion happens in the
 *                   caller); used to decide whether a user is in a different
 *                   block than the target (matchedOps[0]).
 * @param matchedOps The op set of one pattern (target op first).
 */
void cloneScalarOpsForCrossBlockUses(ComputeBlockIdManager &bmOriginal,
                                     SetVector<Operation *> &matchedOps,
                                     int targetBlockId);

/**
 * @brief Check if a value originates from global memory (GM) and collect
 * viewOps
 *
 * Traces back through nested view-like operations (subview, reinterpret_cast,
 * etc.) to determine if the source is a function argument (block argument in
 * the entry block). Only view-like operations in the same block as the input
 * viewOp are collected.
 *
 * @param viewValue The value to check
 * @param matchedOps SetVector to collect all same-block view-like operations in
 * the trace
 * @return bool Returns true if the source is from global memory (function
 * argument), false otherwise
 */
bool collectViewOpsAndCheckGlobalMemory(Value viewValue,
                                        SetVector<Operation *> &matchedOps);

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_OPT_COMMON_H
