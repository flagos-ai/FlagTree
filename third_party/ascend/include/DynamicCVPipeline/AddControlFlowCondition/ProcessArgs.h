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

#ifndef TRITON_ASCEND_SSBUF_PROCESS_ARGS_FOR_CONTROL_FLOW_H
#define TRITON_ASCEND_SSBUF_PROCESS_ARGS_FOR_CONTROL_FLOW_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"

namespace mlir {
namespace triton {

struct ControlFlowConditionInfo;

// For each shared iter_arg tracks: which block_ids use it, who is owner (first
// block_id in order), and what new iter_arg index each non-owner block uses
struct SharedArgInfo {
  int argIndex;
  Value iterArg;
  int ownerBlockId;
  int newArgIndex;
  int nonOwnerBlockId;

  SharedArgInfo(int arg, int owner, int newIdx, int nonOwner)
      : argIndex(arg), iterArg(Value()), ownerBlockId(owner),
        newArgIndex(newIdx), nonOwnerBlockId(nonOwner) {}
};

// Per-whileOp state for cloning cond-used iter_arg update chains into the new
// scf.while's after body. Populated by planWhileIterArgDescriptors; consumed by
// cloneWhileBlockChains, buildNewWhileYield, recordWhileBlockArgMap.
struct WhileIterArgClonePlan {
  // Per-origIdx metadata, keyed on the original iter_arg index.
  llvm::DenseMap<unsigned, Operation *> compOp;
  llvm::DenseMap<unsigned, llvm::DenseSet<Operation *>> chainOps;
  llvm::DenseMap<unsigned, unsigned> posInClonedVec;
  // (blockId, newArgIdx, origIdx) triples in planning order, one per (blockId,
  // cond-used iter_arg) pair
  SmallVector<std::tuple<int, unsigned, unsigned>> newArgDescriptors;
  // Output: cloned compOp results per blockId, indexed by posInClonedVec
  llvm::DenseMap<int, SmallVector<Value>> clonedPerBlock;
};

class ProcessArgsPass
    : public PassWrapper<ProcessArgsPass, OperationPass<ModuleOp>> {
public:
  ProcessArgsPass() = default;

  void runOnOperation() override;

  LogicalResult processSharedIterArgs(ModuleOp module);

  // Snapshots whileOp iter_args; clones cond-used update chain per block (same
  // ssbuffer.block_id run); records (new_arg_idx, old_arg_idx) in
  // ControlFlowConditionInfo.
  LogicalResult updateIndependentCondsInWhileBlocks(ModuleOp module);

  // Per-whileOp driver for updateIndependentCondsInWhileBlocks.
  LogicalResult processWhileIterArgsInWhileOp(scf::WhileOp whileOp,
                                              ControlFlowConditionInfo *info);

  // Per-op driver for shared-iter_args processing.
  LogicalResult processSharedIterArgsInLoop(Operation *op,
                                            ControlFlowConditionInfo *info);

  // Completes the scf.while path: migrate before/after bodies, rebuild
  // yield/condition, transfer maps.
  LogicalResult processSharedArgsInWhileOp(
      scf::WhileOp whileOp, scf::WhileOp newWhileOp,
      SmallVector<SharedArgInfo> &sharedArgsInfo,
      const llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
      const llvm::DenseMap<int, llvm::DenseSet<Operation *>>
          &sharedArgToChainOps,
      ControlFlowConditionInfo *info);

  void setConditionInfo(ControlFlowConditionInfo *info_) { info = info_; }

  llvm::StringRef getArgument() const override { return "process-args"; }

  ControlFlowConditionInfo *info = nullptr;

  // Original iter_args of every scf.while op with main_loop attr, captured at
  // start of ProcessArgs. Used to identify iter_args referenced by
  // scf.condition.
  llvm::DenseMap<scf::WhileOp, SmallVector<unsigned>>
      originalWhileIterArgIndices;

  // Local copy of whileBlockArgMap; also mirrored to info->whileBlockArgMap
  // when info is set, so the mapping is observable when --process-args runs
  // standalone (info may be null).
  WhileBlockArgMap localWhileBlockArgMap;
};

std::unique_ptr<OperationPass<ModuleOp>> createProcessArgsPass();

} // namespace triton
} // namespace mlir
#endif // TRITON_ASCEND_SSBUF_PROCESS_ARGS_FOR_CONTROL_FLOW_H
