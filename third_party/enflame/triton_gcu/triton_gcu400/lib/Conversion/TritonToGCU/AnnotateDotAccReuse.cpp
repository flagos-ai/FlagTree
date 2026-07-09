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
#include "Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_ANNOTATEDOTACCREUSEPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
#define DEBUG_TYPE "annotate-dot-acc-reuse"

namespace {

// Check whether a Value is a splat-zero constant, looking through
// single-region wrapper ops (e.g. triton_gcu.elementwise_fusion_region).
static bool isSplatZeroConstant(Value v) {
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (!constOp->hasOneUse())
      return false;
    auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!denseAttr || !denseAttr.isSplat())
      return false;
    auto splatAttr = denseAttr.getSplatValue<Attribute>();
    if (auto fAttr = dyn_cast<FloatAttr>(splatAttr))
      return fAttr.getValue().isZero();
    if (auto iAttr = dyn_cast<IntegerAttr>(splatAttr))
      return iAttr.getValue().isZero();
    return false;
  }
  // Look through single-region wrapper ops whose terminator yields
  // the inner value (e.g. triton_gcu.elementwise_fusion_region).
  Operation *defOp = v.getDefiningOp();
  if (!defOp || defOp->getNumRegions() != 1)
    return false;
  Region &region = defOp->getRegion(0);
  if (!region.hasOneBlock())
    return false;
  Operation *terminator = region.front().getTerminator();
  auto resultIdx = cast<OpResult>(v).getResultNumber();
  if (resultIdx >= terminator->getNumOperands())
    return false;
  return isSplatZeroConstant(terminator->getOperand(resultIdx));
}

static bool canReuseOaccCache(triton::DotOp op) {
  ModuleOp mod = op->getParentOfType<ModuleOp>();
  int32_t numWarps = triton::gcu::getNumWarps(mod);

  int64_t maxM = OACC_MAX_NUM; // should get from max register size
  if (numWarps > 4) {
    maxM /= (numWarps / 4);
  }

  auto type = dyn_cast<RankedTensorType>(op.getType());
  auto originShape = type.getShape();
  auto numElems = triton::gcu::getElemsPerThread(type);
  int64_t mSize = numElems[0];
  int64_t nSize = numElems[1];
  return mSize <= maxM &&
         (nSize == OACC_F32_LENGTH ||
          (nSize < OACC_F32_LENGTH && originShape[1] == nSize));
}

// Determine the acc_store mode by tracing the scf.for result's use chain.
//   - "global":     result → triton_gcu.store (same type)
//   - "cvt_global": result → truncf/trunci → triton_gcu.store
//   - "local":      result → maxtrix_store → ...
//   - "cvt_local":  result → truncf/trunci → maxtrix_store → ...
static StringRef determineAccStoreMode(Value forResult) {
  Value val = forResult;
  bool hasCvt = false;
  while (val.hasOneUse()) {
    Operation *user = *val.getUsers().begin();
    if (isa<triton::gcu::StoreOp>(user))
      return hasCvt ? kAccStoreCvtGlobal : kAccStoreGlobal;
    if (isa<arith::TruncFOp, arith::TruncIOp>(user)) {
      hasCvt = true;
      val = user->getResult(0);
      continue;
    }
    break;
  }
  return kAccStoreLocal;
  // return hasCvt ? kAccStoreCvtLocal : kAccStoreLocal;
}

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
//   5. The accumulator init value must be a splat-zero constant (the
//      downstream conversion re-initializes via memset to zero).  The
//      constant may be wrapped in fusion regions.
//   6. Only single or two-level loop nesting is supported; three or more
//      levels are rejected.
//   7. If the scf.for is nested inside another scf.for, the accumulator
//      init arg must NOT be carried through the outer loop (i.e., it must
//      not be an outer for block arg), and the inner for's result must NOT
//      be yielded by the outer for.  The downstream conversion inserts a
//      re-initialization of the accumulator buffer at each outer iteration.
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

      // If the init arg for this iter position is the same SSA value as
      // another iter arg, two dots would try to reuse the same buffer.
      // This causes aliasing and a crash in fixAccBufferLifetime.
      auto initArgs = forOp.getInitArgs();
      Value initArg = initArgs[iterArgIdx];

      // Only support zero-initialized accumulators -- fixAccBufferLifetime
      // re-initializes via memset to zero, so a non-zero init would be lost.
      if (!isSplatZeroConstant(initArg))
        return;

      if (auto outerFor = forOp->getParentOfType<scf::ForOp>()) {
        // Reject if nested three or more levels deep -- only two-level
        // nesting is supported for accumulator reuse.
        if (outerFor->getParentOfType<scf::ForOp>())
          return;
        // Reject if init comes from outer for's block arg (carried through
        // the outer loop -- the buffer would not be re-initialized).
        if (auto outerBlockArg = dyn_cast<BlockArgument>(initArg))
          if (outerBlockArg.getOwner()->getParentOp() ==
              outerFor.getOperation())
            return;
        // Reject if outer for yields the inner for's accumulator result
        // (double accumulation across both loops).
        if (auto outerYield =
                dyn_cast<scf::YieldOp>(outerFor.getBody()->getTerminator())) {
          Value innerResult = forOp->getResult(iterArgIdx);
          for (auto yieldedVal : outerYield.getOperands())
            if (yieldedVal == innerResult)
              return;
        }
      }
      for (unsigned i = 0, e = initArgs.size(); i < e; ++i) {
        if (i != iterArgIdx && forOp.getInitArgs()[i] == initArg)
          return;
      }

      auto ctx = dotOp.getContext();
      bool isOaccCache = canReuseOaccCache(dotOp);
      StringRef accReuseMode = isOaccCache ? kAccReuseOacc : kAccReuseLocal;
      dotOp->setAttr(kAccReuseCandidate, StringAttr::get(ctx, accReuseMode));

      LLVM_DEBUG(llvm::dbgs()
                 << "AnnotateDotAccReuse: marked dot op as reuse candidate"
                 << ", acc_reuse_candidate=" << accReuseMode << "\n");

      if (isOaccCache) {
        Value forResult = forOp->getResult(iterArgIdx);
        StringRef accStoreMode = determineAccStoreMode(forResult);
        dotOp->setAttr(kAccStore, StringAttr::get(ctx, accStoreMode));
        LLVM_DEBUG(llvm::dbgs() << "AnnotateDotAccReuse: acc oacc store"
                                << ", acc_store=" << accStoreMode << "\n");
      }
    });
  }
};

} // namespace
