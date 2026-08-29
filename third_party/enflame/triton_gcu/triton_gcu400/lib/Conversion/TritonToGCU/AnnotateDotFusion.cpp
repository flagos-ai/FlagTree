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
#include <string>

#include "Constants.h"
#include "Conversion/TritonToGCU/ReduceScanCommon.h"
#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

// Check whether an extern_elementwise op is a mac/mas/imas or mix-precision op
static bool isExtOpWithMultiOperandsSymbol(StringRef symbol) {
  static const std::string multiOperandSymbolList[] = {
      "__gcu_mac", "__gcu_mas",  "__gcu_imas", "__gcu_wadd",
      "__gcu_add", "__gcu_wmul", "__gcu_mul"};
  return llvm::any_of(multiOperandSymbolList,
                      [&symbol](StringRef symbolPrefix) {
                        return symbol.starts_with(symbolPrefix);
                      });
}

static bool isExtOpWithMultiOperandsSymbol(Operation *op) {
  if (auto extOp = dyn_cast<triton::ExternElementwiseOp>(op)) {
    if (isExtOpWithMultiOperandsSymbol(extOp.getSymbol())) {
      return true;
    }
  }
  return false;
}

// Trace a value back through ops until a block argument or
// broadcast/ExternElementwiseOp is reached. For broadcast, the result itself
// carries the fully-expanded tensor dimensions, so return it directly instead
// of tracing through. Returns the traced value, or nullptr if not traceable
// (e.g., constant).
static Value traceSourceValue(Value val) {
  while (true) {
    if (auto blockArg = dyn_cast<BlockArgument>(val))
      return blockArg;

    auto *defOp = val.getDefiningOp();
    if (!defOp || defOp->getNumOperands() == 0)
      return nullptr;

    // if (isa<arith::ConstantOp>(defOp))
    //  return nullptr;

    // Stop at broadcast: the result type has the full shape we care about.
    if (isa<triton::BroadcastOp>(defOp))
      return val;

    // Stop at ExternElementwiseOp: recurse through its operands.
    if (isExtOpWithMultiOperandsSymbol(defOp))
      return val;

    // Follow the first operand through pass-through / elementwise ops.
    val = defOp->getOperand(0);
  }
}

// Recursively collect distinct source tensor values from an extern_elementwise
// op's operands. When an operand traces to another extern_elementwise op, we
// recurse into its operands instead, flattening the entire source chain.
static void collectSourceValues(Value val, SetVector<Value> &sources) {
  auto tracedVal = traceSourceValue(val);
  if (!tracedVal)
    return;

  // If the traced value is itself produced by another extern_elementwise op,
  // recurse into its operands to flatten the chain.
  if (auto *defOp = tracedVal.getDefiningOp()) {
    if (isExtOpWithMultiOperandsSymbol(defOp)) {
      for (auto operand : defOp->getOperands()) {
        collectSourceValues(operand, sources);
      }
      if (defOp->getNumOperands() == 3)
        return;
    }
  }

  // Otherwise, check if it's a valid rank-2+ tensor and record it.
  auto tensorType = dyn_cast<RankedTensorType>(tracedVal.getType());
  if (tensorType && tensorType.getRank() > 1)
    sources.insert(tracedVal);
}

// Check whether an extern_elementwise op is a mac/mas/imas (multiply-accumulate
// variant) whose distinct source tensor count is low enough for hardware OACC.
// Sources are collected recursively through nested extern_elementwise ops.
static bool exceedOaccLimit(triton::ExternElementwiseOp extOp) {
  if (!isExtOpWithMultiOperandsSymbol(extOp.getSymbol()))
    return false;

  SetVector<Value> sources;
  collectSourceValues(extOp.getResult(), sources);

  unsigned numTensorValues = sources.size();
  if (numTensorValues == 0)
    return false;

  auto tensorType = dyn_cast<RankedTensorType>(extOp.getType());
  auto rank = tensorType.getRank();
  auto numElems = triton::gcu::getElemsPerThread(tensorType);
  return numElems[rank - 2] > OACC_MAX_NUM / numTensorValues;
}

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
      SmallVector<Operation *> users(result.user_begin(), result.user_end());
      llvm::sort(users, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });

      unsigned numUsers = std::distance(result.user_begin(), result.user_end());
      SmallVector<bool> canReuseOacc(numUsers, false);
      if (numUsers > 0 &&
          isa<triton::gcu::ElementwiseFusionRegionOp>(users.back()))
        canReuseOacc[numUsers - 1] = true;

      // GEMM size
      auto tensorType = dyn_cast<RankedTensorType>(result.getType());
      auto numElems = triton::gcu::getElemsPerThread(tensorType);
      int64_t mPerThread = numElems.front();
      int64_t nPerThread = numElems.back();

      Type elemType = tensorType.getElementType();
      int64_t elemBytes = triton::gcu::getBpe(elemType);
      int64_t oaccM = OACC_MAX_NUM;
      int64_t oaccN = kOaccSizeInBytes / elemBytes;

      bool canFuseElementwise = false;
      bool hasExceededOacc = false;
      for (auto [userIdx, user] : llvm::enumerate(users)) {
        if (auto elemwFusionOp =
                dyn_cast<triton::gcu::ElementwiseFusionRegionOp>(user)) {
          canFuseElementwise = mPerThread <= oaccM && nPerThread == oaccN;

          // mac/mas/imas with all-vector (rank>1) inputs cannot reuse OACC
          // because the extra operands would need separate OACC space.
          elemwFusionOp.getBody()->walk([&](triton::ExternElementwiseOp extOp) {
            if (exceedOaccLimit(extOp)) {
              hasExceededOacc = true;
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
          if (hasExceededOacc)
            return;
          // hasExceededOacc = false;

          if (canFuseElementwise && canReuseOacc[userIdx]) {
            int64_t inplaceOperandIdx = -1;
            int64_t inplaceResultIdx = -1;
            Operation *truncOp = nullptr;
            for (auto [index, value] :
                 llvm::enumerate(elemwFusionOp->getOperands())) {
              if (value == result) {
                Value curValue = elemwFusionOp.getBody()->getArgument(index);
                Value nextValue = nullptr;
                while (curValue) {
                  for (Operation *curUser : curValue.getUsers()) {
                    if (isa<arith::TruncFOp>(curUser)) {
                      truncOp = curUser;
                      curValue = curUser->getResult(0);
                      if (curValue.hasOneUse())
                        curUser = *curValue.getUsers().begin();
                    }
                    if (auto yieldOp =
                            dyn_cast<triton::gcu::YieldOp>(curUser)) {
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
              // Determine acc_store mode
              Value val = elemwFusionOp.getResult(inplaceResultIdx);
              StringRef accStoreMode =
                  truncOp ? kAccStoreCvtLocal : kAccStoreLocal;
              if (val.hasOneUse()) {
                Operation *user = *val.getUsers().begin();
                if (auto storeOp = dyn_cast<triton::gcu::StoreOp>(user)) {
                  storeOp->setAttr(kMaxtrixStore, UnitAttr::get(ctx));
                  accStoreMode = truncOp ? kAccStoreCvtGlobal : kAccStoreGlobal;
                }
              }
              // Currently cannot support kAccStoreCvtLocal, because
              // ElementwiseFusionOp maybe output OACC register
              if (accStoreMode == kAccStoreCvtLocal) {
                continue;
              } else if (accStoreMode == kAccStoreCvtGlobal) {
                auto newType = truncOp->getOperand(0).getType();
                truncOp->getResult(0).replaceAllUsesWith(
                    truncOp->getOperand(0));
                truncOp->erase();
                elemwFusionOp.getResult(inplaceResultIdx).setType(newType);
              }

              auto i64Ty = IntegerType::get(dotOp.getContext(), 64);
              elemwFusionOp->setAttr(
                  kAccReuseInplaceOperand,
                  IntegerAttr::get(i64Ty, inplaceOperandIdx));
              elemwFusionOp->setAttr(kAccReuseInplaceResult,
                                     IntegerAttr::get(i64Ty, inplaceResultIdx));
              elemwFusionOp->setAttr(kAccStore,
                                     StringAttr::get(ctx, accStoreMode));
            }
          }
        } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(user)) {
          auto canFuseToDot = [&](triton::ReduceOp op) {
            auto axis = op.getAxis();
            if (axis != 1) {
              return false;
            }
            auto combineDesc = triton::gcu::CombineOpDesc(op);
            if (nPerThread == oaccN) {
              return combineDesc.supportsFastLaneReduction();
            } else if (nPerThread == oaccN / 2 || nPerThread == oaccN / 4 ||
                       nPerThread == oaccN / 8) {
              return combineDesc.supportsStagedFastLaneReduction();
            } else {
              return false;
            }
          };
          canFuseElementwise = canFuseToDot(reduceOp);
        } else {
          canFuseElementwise = false;
        }
        if (!canFuseElementwise)
          break;
      }
      if (canFuseElementwise) {
        dotOp->setAttr(kAccStore,
                       StringAttr::get(dotOp.getContext(), kAccStoreNone));
      }
    });

    /*
    // Elementwise fusion + store
    module.walk([&](triton::gcu::ElementwiseFusionRegionOp fusionOp) {
      auto *ctx = fusionOp.getContext();

      if (llvm::any_of(
              fusionOp.getBody()->getOps<triton::ExternElementwiseOp>(),
              [](triton::ExternElementwiseOp extOp) {
                return exceedOaccLimit(extOp);
              })) {
        return WalkResult::advance();
      }

      if (fusionOp->hasAttr(kAccStore))
        return WalkResult::advance();

      auto yieldOp = cast<triton::gcu::YieldOp>(
          fusionOp.getRegion().back().getTerminator());
      for (unsigned i = 0; i < fusionOp.getNumResults(); ++i) {
        auto result = fusionOp.getResult(i);
        auto tensorType = dyn_cast<RankedTensorType>(result.getType());
        if (!tensorType || tensorType.getRank() != 2)
          return WalkResult::advance();

        auto numElems = triton::gcu::getElemsPerThread(tensorType);
        Type elemType = tensorType.getElementType();
        if (result.hasOneUse() &&
            isa<triton::gcu::StoreOp>(*result.user_begin())) {
          auto truncOp = dyn_cast_or_null<arith::TruncFOp>(
              yieldOp.getOperand(i).getDefiningOp());
          auto hasTruncOp = truncOp && truncOp->getResult(0).hasOneUse();
          if (hasTruncOp) {
            elemType =
            cast<RankedTensorType>(truncOp->getOperand(0).getType())
                           .getElementType();
          }
          int64_t elemBytes = triton::gcu::getBpe(elemType);
          int64_t oaccN = kOaccSizeInBytes / elemBytes;
          if (elemBytes != 4 || numElems.back() != oaccN)
            return WalkResult::advance();

          if (hasTruncOp) {
            auto newType = truncOp->getOperand(0).getType();
            truncOp->getResult(0).replaceAllUsesWith(truncOp->getOperand(0));
            truncOp->erase();
            fusionOp.getResult(i).setType(newType);
          }
          StringRef accStoreMode =
              truncOp ? kAccStoreCvtGlobal : kAccStoreGlobal;
          fusionOp->setAttr(kDirectStore,
                            IntegerAttr::get(IntegerType::get(ctx, 64), i));
          fusionOp->setAttr(kAccStore, StringAttr::get(ctx, accStoreMode));
          break;
        }
      }
      return WalkResult::advance();
    });
    */
  }
};

} // namespace
