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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UTILS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UTILS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

namespace mlir {

// Collect all nested ops within an operation's regions
LogicalResult collectAllNestedOps(Operation *op,
                                  llvm::DenseSet<Operation *> &regionOps);

// Group body ops of a main-loop op (scf.for or scf.while carrying
// ssbuffer.main_loop) by block_id.
LogicalResult
collectOpsByBlockId(Operation *op,
                    llvm::DenseMap<int, SmallVector<Operation *>> &blockOps);

// Topological sort of operations based on operand dependencies
LogicalResult topologicalSort(llvm::DenseSet<Operation *> &ops,
                              llvm::DenseMap<Operation *, int> *opOrder,
                              SmallVectorImpl<Operation *> &sorted);

LogicalResult topologicalSort(SmallVector<Operation *> &ops);

// Get block_ids in order of appearance in the main-loop body (forOp body or
// whileOp after-region body). Returns empty if `op` is neither.
SmallVector<int> getBlockIdsInOrder(Operation *op);

// Count unique ssbuffer.if values inside a main-loop op (scf.for or scf.while
// carrying ssbuffer.main_loop), walking all nested ops. Returns 0 if none.
int countUniqueIfBlockIds(Operation *loopOp);

// Get block_id of immediate child of main-loop (scf.for/scf.while carrying
// ssbuffer.main_loop) that contains op. For scf.while, "body" means the
// after-region block.
std::optional<int> getLoopDirectChildBlockId(Operation *op);

// Find the tcb group id that contains value v
int findTcbGroupId(
    Value v,
    llvm::DenseMap<int, SmallVector<Value>> &tightlyCoupledBufferGroups);

// Set isCube/isVector based on the scope's tcore_type attribute
// Returns failure if scopeOp does not have tcore_type attribute
LogicalResult getScopeType(Operation *scopeOp, bool &isCube, bool &isVector);

// Check if op is scf.if whose body only contains hivm.hir.sync_block_wait,
// hivm.hir.sync_block_set and hivm.fixpipe ops (excluding terminators). Returns
// false if op is not scf.if or contains any other op.
bool isIfOpWithOnlySyncOps(Operation *op);

// Migrate ops from oldBlock to newBlock; replaceAllUsesWith on oldBlock's
// args to newBlock's args (same index). Used for both branches of scf.while
// (before/after) and for scf.for body replacement.
void migrateBody(Block *oldBlock, Block *newBlock);

// Migrate both before and after regions of a scf.while op. Does not touch
// terminators — the caller is expected to build a new scf.condition and
// scf.yield in the new regions.
void migrateWhileBodies(scf::WhileOp oldWhileOp, scf::WhileOp newWhileOp);

// Build new scf.yield at end of `newBlock`: copies oldBlock's yield operands,
// appends `extraYieldValues`, creates new scf::YieldOp, erases old yield.
LogicalResult buildNewYieldOp(Block *oldBlock, Block *newBlock,
                              Operation *newOp,
                              ArrayRef<Value> extraYieldValues);

// Replace all uses of `oldOp`'s results with `newOp`'s matching results.
// No-op when `oldOp` is result-less.
void replaceOpResultUses(Operation *oldOp, Operation *newOp);

// Build new scf.condition in `newWhileOp`'s before region. Condition preserved
// from `whileOp`; forwarded values = new before-block args (incl. extras).
void buildNewWhileCondition(scf::WhileOp whileOp, scf::WhileOp newWhileOp);

// Creates a new scf.for with `extraInitArgs` appended to the original init
// args. Returns `oldForOp` unchanged when `extraInitArgs` is empty.
scf::ForOp createNewForOpWithExtras(scf::ForOp oldForOp,
                                    ArrayRef<Value> extraInitArgs);

// Creates a new scf.while with `extraInitArgs` appended to the original inits
// and empty before/after blocks. Returns `oldWhileOp` unchanged when empty.
scf::WhileOp createNewWhileOpWithExtras(scf::WhileOp oldWhileOp,
                                        ArrayRef<Value> extraInitArgs);

// Dispatches createNewForOpWithExtras / createNewWhileOpWithExtras by op type.
// Returns nullptr if `oldOp` is neither scf.for nor scf.while.
Operation *createMainLoopOpWithExtras(Operation *oldOp,
                                      ArrayRef<Value> extraInitArgs);

// Prints whileBlockArgMap (whileOp -> block_id -> (new_arg_idx -> old_arg_idx))
// to the debug stream, gated by LLVM_DEBUG. `header` is logged once before the
// iteration.
void dumpWhileBlockArgMap(const triton::WhileBlockArgMap &map,
                          llvm::StringRef header);

} // namespace mlir
#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UTILS_H
