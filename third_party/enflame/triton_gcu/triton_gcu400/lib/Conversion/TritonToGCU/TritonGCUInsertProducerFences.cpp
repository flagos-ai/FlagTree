/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/GCUWS/IR/Dialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONGCUINSERTPRODUCERFENCESPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgcuws = mlir::triton::gcuws;
namespace ttgcu = mlir::triton::gcu;

using namespace mlir;

namespace {

// Check if an operation is a non-DTE store that writes to shared memory.
// DTE ops (copy_global_to_local, gather_global_to_local) are excluded —
// they notify the pipeline directly via hardware.
// For region ops (e.g., elementwise_fusion_region), walk inside to find
// any non-DTE store.
static bool hasNonDTEStore(Operation *op) {
  // DTE ops — safe, no fence needed.
  if (isa<ttgcu::CopyGlobalToLocalOp, ttgcu::GatherGlobalToLocalOp,
          ttgcu::StoreOp>(op))
    return false;

  // Direct non-DTE stores.
  if (isa<ttg::LocalStoreOp>(op))
    return true;
  if (isa<tt::StoreOp>(op))
    return true;
  if (isa<ttgcu::MaskedStoreOp>(op))
    return true;

  // Region ops (e.g., elementwise_fusion_region) — walk inside.
  if (op->getNumRegions() > 0) {
    bool found = false;
    op->walk([&](Operation *inner) {
      if (isa<ttg::LocalStoreOp, tt::StoreOp, ttgcu::MaskedStoreOp>(inner))
        found = true;
    });
    if (found)
      return true;
  }

  return false;
}

// Determine whether a fence is needed before `producer_commit`.
//
// Walk backward from `producer_commit` to the nearest `producer_acquire`
// in the same block. Check every op between them:
//   - If all ops are DTE (copy_global_to_local / gather_global_to_local)
//     or pure (arith, memdesc_index, etc.) → no fence needed.
//   - If any op is a non-DTE store (local_store, tt.store, maskedstore,
//     or elementwise_fusion_region containing such) → fence required.
//
// This approach is per-pipe: each producer_commit only checks the ops
// between it and its corresponding producer_acquire, so multiple pipes
// in the same partition are handled independently.
static bool needsFence(ttgcuws::ProducerCommitOp commitOp) {
  Block *block = commitOp->getBlock();

  for (auto it = block->rbegin(); it != block->rend(); ++it) {
    Operation &op = *it;

    // Skip the commit op itself.
    if (&op == commitOp.getOperation())
      continue;

    // Found the acquire — all ops between acquire and commit are checked.
    if (isa<ttgcuws::ProducerAcquireOp>(op))
      return false;

    // Check if this op is a non-DTE store.
    if (hasNonDTEStore(&op))
      return true;

    // Pure ops and DTE ops are skipped — continue backward walk.
  }

  // No producer_acquire found in this block — conservatively insert fence.
  return true;
}

struct TritonGCUInsertProducerFencesPass
    : public mlir::impl::TritonGCUInsertProducerFencesPassBase<
          TritonGCUInsertProducerFencesPass> {
  using Base::Base;

  void runOnOperation() override {
    auto mod = getOperation();

    // For TLE WS, the producer may use either DTE (copy_global_to_local)
    // or non-DTE (local_store / maskedstore via pointer) transport.
    // After LocalMemOptimize, stores that can be fused into DTE have already
    // been converted. We only need a fence when non-DTE stores remain
    // between producer_acquire and producer_commit.
    //
    // DTE-based producers notify the pipeline directly via hardware, so no
    // fence is needed. Non-DTE producers require an explicit mfence(Local)
    // to ensure store visibility to the consumer.
    mod.walk([&](ttg::WarpSpecializeOp wsOp) {
      if (!wsOp->hasAttr("tle.warp_specialize"))
        return;
      wsOp.walk([&](ttgcuws::ProducerCommitOp commitOp) {
        if (!needsFence(commitOp))
          return;
        OpBuilder builder(commitOp);
        builder.create<mlir::gcu::MFenceOp>(commitOp.getLoc(),
                                            mlir::gcu::MFenceType::Local);
      });
    });
  }
};

} // namespace
