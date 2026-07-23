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

#include "Constants.h"
#include "Conversion/TritonToGCU/ReduceScanCommon.h"
#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Utility.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_ANNOTATEDOTFUSIONPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
#define DEBUG_TYPE "annotate-dot-fusion"

namespace {

/*
// TODO(support)
// Detect the "pre-fusion" accumulator-reuse pattern and, when found, annotate
// the ops so the downstream conversion keeps the accumulator in OACC across
// the loop:
//
//   %acc = <loop iter-arg>
//   %fused = triton_gcu.elementwise_fusion_region(%acc, ...) { ... }
//   %dot   = tt.dot %a, %b, %fused
//   scf.yield ... %dot ...        // yielded back to the same iter-arg
//
// Here the fusion reads the accumulator, the dot uses the fused value as its
// bias, and the dot result is carried back to the accumulator iter-arg. The
// whole accumulator can therefore live in a single OACC buffer: allocate it
// outside the loop, let the fusion read/write it in place, and let the dot
// use it as bias and output. Only the initial matrix_load (before the loop)
// and the final matrix_store (after the loop) touch local memory.
//
// On a match this sets on the dot:
//   acc_reuse_candidate = "acc_reuse_oacc"
// and on the fusion op:
//   acc_reuse_inplace_operand = <index of the iter-arg operand>
// The dot keeps its acc_load="local" / acc_store="local" tags: "local" load
// drives the one-time init matrix_load in the reuse path, and "local" store
// drives the one-time matrix_store after the loop.
static bool tryAnnotatePreFusionReuse(triton::DotOp dotOp) {
  auto accLoadAttr = dotOp->getAttrOfType<StringAttr>(kAccLoad);
  if (!accLoadAttr || accLoadAttr.getValue() != kAccLoadLocal)
    return false;

  Value acc = dotOp.getC();
  auto fusionOp = acc.getDefiningOp<triton::gcu::ElementwiseFusionRegionOp>();
  if (!fusionOp)
    return false;

  // The fusion output must feed only this dot.
  if (!acc.hasOneUse())
    return false;

  // OACC reuse assumes a single-result elementwise fusion (the accumulator).
  if (fusionOp->getNumResults() != 1)
    return false;

  // Find the fusion operand that is the accumulator loop iter-arg such that the
  // dot result is yielded back to that same iter-arg.
  for (auto operand : llvm::enumerate(fusionOp->getOperands())) {
    auto blockArg = dyn_cast<BlockArgument>(operand.value());
    if (!blockArg)
      continue;

    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      continue;

    unsigned argNum = blockArg.getArgNumber();
    if (argNum == 0)
      continue; // induction variable
    unsigned iterArgIdx = argNum - 1;

    auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yieldOp || iterArgIdx >= yieldOp.getNumOperands())
      continue;
    if (yieldOp.getOperand(iterArgIdx) != dotOp.getResult())
      continue;

    // The accumulator iter-arg must be consumed only by the fusion op so the
    // in-place rewrite does not clobber another live reader.
    if (!blockArg.hasOneUse())
      continue;

    auto *ctx = dotOp.getContext();
    dotOp->setAttr(kAccReuseCandidate, StringAttr::get(ctx, kAccReuseOacc));
    fusionOp->setAttr(
        kAccReuseInplaceOperand,
        IntegerAttr::get(IntegerType::get(ctx, 64), operand.index()));
    fusionOp->setAttr(kAccReuseInplaceResult,
                      IntegerAttr::get(IntegerType::get(ctx, 64), 0));

    LLVM_DEBUG(llvm::dbgs()
               << "AnnotateDotFusion: pre-fusion oacc reuse, inplace operand="
               << operand.index() << "\n");
    return true;
  }
  return false;
}
*/

struct AnnotateDotFusionPass
    : public impl::AnnotateDotFusionPassBase<AnnotateDotFusionPass> {
  using Base::Base;

  void runOnOperation() override {
    // return;
    auto module = getOperation();
    module.walk([&](triton::DotOp dotOp) {
      auto *ctx = dotOp.getContext();
      if (dotOp.getType().getRank() != 2)
        return;

      if (dotOp->hasAttr(kAccReuseCandidate))
        return;

      if (!dotOp->hasAttr(kAccLoad))
        return;

      auto accStoreAttr = dotOp->getAttrOfType<StringAttr>(kAccStore);
      if (!accStoreAttr || accStoreAttr.getValue() != kAccStoreLocal)
        return;

      // Pre-fusion accumulator reuse (elementwise on acc → dot → yield).
      // if (tryAnnotatePreFusionReuse(dotOp))
      //   return;

      // Post-fusion accumulator (dot → elementwise on acc → yield).
      Value result = dotOp.getResult();
      if (!result.hasOneUse())
        return;

      // GEMM size
      auto tensorType = dyn_cast<RankedTensorType>(result.getType());
      auto numElems = triton::gcu::getElemsPerThread(tensorType);
      int64_t mPerThread = numElems.front();
      int64_t nPerThread = numElems.back();

      Type elemType = tensorType.getElementType();
      int64_t elemBytes = elemType.getIntOrFloatBitWidth() / 8;
      int64_t oaccM = OACC_MAX_NUM;
      int64_t oaccN = kOaccSizeInBytes / elemBytes;

      bool canFuseElementwise = false;
      Operation *user = *result.getUsers().begin();
      if (auto elemwFusionOp =
              dyn_cast<triton::gcu::ElementwiseFusionRegionOp>(user)) {
        canFuseElementwise = mPerThread <= oaccM && nPerThread == oaccN;
        if (canFuseElementwise) {
          int64_t inplaceOperandIdx = -1;
          int64_t inplaceResultIdx = -1;
          for (auto [index, value] :
               llvm::enumerate(elemwFusionOp->getOperands())) {
            if (value == result) {
              Value curValue = elemwFusionOp.getBody()->getArgument(index);
              Value nextValue = nullptr;
              while (curValue) {
                for (Operation *curUser : curValue.getUsers()) {
                  if (auto yieldOp = dyn_cast<triton::gcu::YieldOp>(curUser)) {
                    for (auto &yieldOperand : yieldOp->getOpOperands()) {
                      if (yieldOperand.get() == curValue) {
                        inplaceOperandIdx = index;
                        inplaceResultIdx = yieldOperand.getOperandNumber();
                        break;
                      }
                    }
                    if (inplaceOperandIdx >= 0 && inplaceResultIdx >= 0)
                      break;
                  } else if (curUser->getNumResults() == 1) {
                    Value curRes = curUser->getResult(0);
                    if (auto curTensorType =
                            dyn_cast<RankedTensorType>(curRes.getType())) {
                      if (curTensorType.getElementType() == elemType)
                        nextValue = curRes;
                    }
                  }
                }
                curValue = nextValue;
                nextValue = nullptr;
              }
              break;
            }
          }
          if (inplaceOperandIdx >= 0 && inplaceResultIdx >= 0) {
            auto i64Ty = IntegerType::get(dotOp.getContext(), 64);
            elemwFusionOp->setAttr(kAccReuseInplaceOperand,
                                   IntegerAttr::get(i64Ty, inplaceOperandIdx));
            elemwFusionOp->setAttr(kAccReuseInplaceResult,
                                   IntegerAttr::get(i64Ty, inplaceResultIdx));

            // Determine acc_store mode
            Value val = elemwFusionOp.getResult(inplaceResultIdx);
            StringRef accStoreMode = kAccStoreLocal;
            if (val.hasOneUse()) {
              Operation *user = *val.getUsers().begin();
              if (auto storeOp = dyn_cast<triton::gcu::StoreOp>(user)) {
                storeOp->setAttr(kMaxtrixStore, UnitAttr::get(ctx));
                accStoreMode = kAccStoreGlobal;
              }
            }
            elemwFusionOp->setAttr(kAccStore,
                                   StringAttr::get(ctx, accStoreMode));
          }
          // } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(user)) {
          //   if (nPerThread == oaccN) {
          //     triton::gcu::CombineOpDesc desc(reduceOp);
          //     auto kind = desc.getCombiningKind();
          //     if (kind) {
          //       using CK = vector::CombiningKind;
          //       switch (*kind) {
          //       case CK::MAXNUMF:
          //       case CK::MAXIMUMF:
          //       case CK::MAXSI:
          //       case CK::MAXUI:
          //       case CK::MINNUMF:
          //       case CK::MINIMUMF:
          //       case CK::MINSI:
          //       case CK::MINUI:
          //       case CK::ADD:
          //         canFuseElementwise = true;
          //       default:
          //         break;
          //       }
          //     }
          //   }
        }
        if (canFuseElementwise) {
          dotOp->setAttr(kAccStore,
                         StringAttr::get(dotOp.getContext(), kAccStoreNone));
        }
      }
    });
  }
};

} // namespace
