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

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_ANNOTATEDOTACCREUSEPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
#define DEBUG_TYPE "annotate-dot-acc-reuse"

namespace {

// Pre-conversion analysis pass that identifies tt.dot ops eligible for
// in-place accumulator buffer reuse (D = A * B + C where D and C share
// the same storage).
//
// This pass annotates qualifying dot ops with "acc_reuse_candidate" so that
// the downstream ConvertTritonToGCU pass can skip allocating a separate
// output buffer.
//
// Structural conditions checked here (Triton IR level):
//   1. 2-D matmul (rank == 2).
//   2. The accumulator (C) is a loop-carried block argument of an scf.for.
//   3. The accumulator tensor has exactly one use (this dot op).
//   4. The dot result is yielded back at the same iter-arg position.
//   5. The scf.for is NOT nested inside another scf.for.
//
// Type-compatibility conditions (accMemRefType == resultMemRefType, init-arg
// is an AllocOp) are deferred to conversion time since they require the
// TypeConverter.
struct AnnotateDotAccReusePass
    : public impl::AnnotateDotAccReusePassBase<AnnotateDotAccReusePass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation().walk([](triton::DotOp dotOp) {
      if (dotOp.getType().getRank() != 2)
        return;

      Value origAcc = dotOp.getC();
      if (!origAcc.hasOneUse())
        return;

      auto blockArg = dyn_cast<BlockArgument>(origAcc);
      if (!blockArg)
        return;

      auto *parentOp = blockArg.getOwner()->getParentOp();
      auto forOp = dyn_cast<scf::ForOp>(parentOp);
      if (!forOp)
        return;

      unsigned argIdx = blockArg.getArgNumber();
      if (argIdx == 0)
        return;
      unsigned iterArgIdx = argIdx - 1;

      auto *terminator = forOp.getBody()->getTerminator();
      auto yieldOp = dyn_cast<scf::YieldOp>(terminator);
      if (!yieldOp || iterArgIdx >= yieldOp.getNumOperands())
        return;
      if (yieldOp.getOperand(iterArgIdx) != dotOp.getResult())
        return;

      if (forOp->getParentOfType<scf::ForOp>())
        return;

      dotOp->setAttr("acc_reuse_candidate", UnitAttr::get(dotOp.getContext()));
      LLVM_DEBUG(llvm::dbgs()
                 << "AnnotateDotAccReuse: marked dot op as reuse candidate\n");
    });
  }
};

} // namespace
