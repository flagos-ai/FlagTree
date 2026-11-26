/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
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

#include "FusionOpUtils.h"
#include "Conversion/TritonToGCU/Utility.h"

namespace mlir {
namespace triton {
namespace gcu {

// Upper bound for a byte-per-lane histogram mask kept in a private
// per-subthread stack buffer.  The count loop runs on a single subthread and
// reads the whole mask linearly, so a single owner can count it in place; the
// 192-OACC-line budget (96 KiB) keeps that private buffer from outgrowing the
// register-resident footprint the `register` hint already allows.  Larger
// masks are staged through L1 instead.  The budget lives here, next to the
// histogram alloca analysis that needs it, and is reached by the histogram
// lowering through `fitsPrivateHistogramMask` so there is a single limit.
constexpr int64_t kPrivateHistogramMaskBytes =
    static_cast<int64_t>(kOaccSizeInBytes) * OACC_MAX_NUM;

bool fitsPrivateHistogramMask(RankedTensorType maskType) {
  if (!maskType)
    return false;
  if (getTotalElemsPerThread(maskType) != maskType.getNumElements())
    return false;
  int64_t maskBytes =
      maskType.getNumElements() * ((maskType.getElementTypeBitWidth() + 7) / 8);
  return maskBytes <= kPrivateHistogramMaskBytes;
}

Value getAllocaInputValue(Value val, ValueRemapFn remapFn) {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return nullptr;
    return getAllocaInputValue(forOp.getInitArgs()[blockArg.getArgNumber() -
                                                   forOp.getNumInductionVars()],
                               remapFn);
  }
  auto defOp = val.getDefiningOp();
  if (isa<memref::AllocaOp>(defOp)) {
    return val;
  } else if (isa<memref::ReinterpretCastOp>(defOp)) {
    return getAllocaInputValue(defOp->getOperand(0), remapFn);
  } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    unsigned resultIdx = cast<OpResult>(val).getResultNumber();
    auto initArg = forOp.getInitArgs()[resultIdx];
    auto remappedValue = remapFn(initArg);
    return getAllocaInputValue(remappedValue ? remappedValue : initArg,
                               remapFn);
  }
  return nullptr;
}

bool canUseAllocaForValue(Value val, ValueRemapFn remapFn) {
  if (!val)
    return false;
  SetVector<Value> worklist;
  worklist.insert(val);
  for (unsigned idx = 0; idx < worklist.size(); ++idx) {
    Value cur = worklist[idx];
    for (auto *user : cur.getUsers()) {
      if (isa<triton::gcu::ElementwiseFusionRegionOp>(user))
        continue;
      if (auto reduceOp = dyn_cast<triton::ReduceOp>(user)) {
        if (reduceOp->getNumOperands() != 1) {
          return false;
        }
        auto axis = reduceOp.getAxis();
        auto tensorType = reduceOp.getInputTypes()[0];
        if (tensorType.getRank() - 1 != axis ||
            triton::gcu::getElemsPerThread(tensorType)[axis] *
                    triton::gcu::getBpe(tensorType.getElementType()) <
                kOaccSizeInBytes) {
          return false;
        }
        continue;
      }
      if (auto gatherOp = dyn_cast<triton::GatherOp>(user)) {
        if (gatherOp.getSrc() != cur)
          continue;
        return false;
      }
      if (auto histOp = dyn_cast<triton::gcu::HistogramOp>(user)) {
        // 1. Only the mask may live in private memory;
        // 2. The key buffer is counted in place from L1;
        // 3. A private copy is only complete and bounded when a single
        //    subthread owns every lane and the byte-per-lane buffer fits the
        //    OACC budget; otherwise the mask has to be packed through L1.
        auto maskType = dyn_cast<RankedTensorType>(cur.getType());
        if (histOp.getMask() != cur || !fitsPrivateHistogramMask(maskType))
          return false;
        continue;
      }
      if (triton::gcu::isZeroCopyOp(user)) {
        worklist.insert(user->getResult(0));
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        for (auto [initArgIdx, initArg] :
             llvm::enumerate(forOp.getInitArgs())) {
          if (initArg == cur) {
            worklist.insert(forOp.getRegionIterArg(initArgIdx));
            worklist.insert(forOp.getResult(initArgIdx));
          }
        }
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
        if (!forOp)
          return false;
        auto operands = yieldOp.getOperands();
        auto iter = llvm::find(operands, cur);
        assert(iter != operands.end());
        auto initOperand = forOp.getInitArgs()[iter - operands.begin()];
        auto remappedValue = remapFn(initOperand);
        if (!getAllocaInputValue(remappedValue, remapFn))
          return false;
        continue;
      }
      return false;
    }
  }
  return true;
}

static bool
hasVboolIncompatibleOperand(triton::gcu::ElementwiseFusionRegionOp op) {
  return llvm::any_of(op.getOperands(), [&](Value operand) {
    auto type = dyn_cast<TensorType>(operand.getType());
    if (!type || !type.getElementType().isInteger(1))
      return false;
    Operation *producer = operand.getDefiningOp();
    while (isa<triton::ExpandDimsOp>(producer) &&
           !triton::gcu::isExpensiveView(cast<triton::ExpandDimsOp>(producer)))
      producer = producer->getOperand(0).getDefiningOp();
    return !isa<triton::gcu::ElementwiseFusionRegionOp>(producer);
  });
}

static bool hasVboolIncompatibleOp(triton::gcu::ElementwiseFusionRegionOp op) {
  return llvm::any_of(*op.getBody(), [&](Operation &innerOp) {
    auto resultTypes = innerOp.getResultTypes();
    if (!llvm::any_of(resultTypes, [&](Type resultType) {
          return cast<TensorType>(resultType).getElementType().isInteger(1);
        }))
      return false;
    return !isa<arith::AndIOp>(&innerOp) && !isa<arith::OrIOp>(&innerOp) &&
           !isa<arith::XOrIOp>(&innerOp) && !isa<arith::CmpIOp>(&innerOp) &&
           !isa<arith::CmpFOp>(&innerOp) && !isa<arith::ConstantOp>(&innerOp) &&
           !isa<arith::SelectOp>(&innerOp) && !isa<triton::SplatOp>(&innerOp) &&
           !isa<triton::BroadcastOp>(&innerOp);
  });
}

bool canUseAllocaForVboolValue(Value val) {
  SetVector<Value> worklist;
  worklist.insert(val);
  for (unsigned idx = 0; idx < worklist.size(); ++idx) {
    Value cur = worklist[idx];
    for (auto *user : cur.getUsers()) {
      if (auto fusionOp =
              dyn_cast<triton::gcu::ElementwiseFusionRegionOp>(user)) {
        if (hasVboolIncompatibleOperand(fusionOp) ||
            hasVboolIncompatibleOp(fusionOp))
          return false;
        continue;
      }
      if (isZeroCopyOp(user)) {
        worklist.insert(user->getResult(0));
        continue;
      }
      return false;
    }
  }
  return true;
}

} // namespace gcu
} // namespace triton
} // namespace mlir
