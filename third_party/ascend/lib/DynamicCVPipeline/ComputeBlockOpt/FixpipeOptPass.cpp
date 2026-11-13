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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <utility>

#include "ComputeBlockOpt/SplitIfByBlockId/Common.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "triton/Analysis/Utility.h"

#define DEBUG_TYPE "fixpipe-opt"
#define LOG_DEBUG(msg)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg << "\n")

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {

class FixpipeOptPass
    : public PassWrapper<FixpipeOptPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixpipeOptPass)

  FixpipeOptPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "fixpipe-opt"; }
  void getDependentDialects(DialectRegistry &registry) const override;

  llvm::StringRef getDescription() const final {
    return "Optimize matmul-cast-store pattern for fixpipe by setting "
           "core_type to CUBE";
  }

private:
  bool matchFixpipePattern(linalg::MatmulOp matmulOp,
                           SetVector<Operation *> &toMergeWithMatmul,
                           CVPipeline::ComputeBlockIdManager &bm,
                           int &targetBlockId);
  bool isFixpipeCastPattern(Operation *truncOp,
                            SetVector<Operation *> &matchedOps);
  bool isFixpipeMulPattern(Operation *mulOp,
                           SetVector<Operation *> &matchedOps);
  bool isStoreToGM(Operation *materializeOp,
                   SetVector<Operation *> &matchedOps);
  bool applyFixpipeOpt(SetVector<Operation *> &matchedOps, int targetBlockId,
                       const CVPipeline::MemoryDependenceGraph &memGraph,
                       CVPipeline::ComputeBlockIdManager &bm);
  bool isSubviewFromGlobalMemory(ViewLikeOpInterface viewOp,
                                 SetVector<Operation *> &matchedOps);
  bool isValidMul(Operation *op, Value matmulValues,
                  SetVector<Operation *> &matchedOps);
};

void transSource(Value value, SetVector<Operation *> &matchedOps,
                 Block *block) {
  if (!CVPipeline::isScalarLike(value)) {
    return;
  }

  if (auto defOp = value.getDefiningOp()) {
    // We need to find one point, value changes from tensor to scalar.
    // So use specific op condition to stop searching.
    if (isa<tensor::ExtractOp>(defOp)) {
      return;
    }
    if (llvm::find(matchedOps, defOp) != matchedOps.end()) {
      return;
    }
    if (defOp->getBlock() != block) {
      return;
    }
    matchedOps.insert(defOp);
    defOp->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        transSource(operand, matchedOps, block);
      }
    });
  }
}

bool hasQuantScaleCompileHint(Operation *op,
                              SetVector<Operation *> &matchedOps) {
  return any_of(op->getUsers(), [&](Operation *userOp) {
    auto markOp = dyn_cast<annotation::MarkOp>(userOp);
    if (!markOp) {
      return false;
    }
    matchedOps.insert(markOp);
    return markOp->hasAttr(CVPipeline::kInlinableQuantScaleAttr);
  });
}

bool FixpipeOptPass::isValidMul(Operation *op, Value matmulValue,
                                SetVector<Operation *> &matchedOps) {
  // Just filter: arith.mulf/muli(scalar)
  if (!isa<arith::MulFOp>(op) && !isa<arith::MulIOp>(op)) {
    return false;
  }
  if (!hasQuantScaleCompileHint(op, matchedOps)) {
    return false;
  }
  auto quantScalarValue =
      op->getOperand(0) == matmulValue ? op->getOperand(1) : op->getOperand(0);
  if (CVPipeline::isScalarLike(quantScalarValue)) {
    transSource(quantScalarValue, matchedOps, op->getBlock());
    return true;
  }

  // From one fill op with constant value or args....
  if (auto defOp = quantScalarValue.getDefiningOp()) {
    if (auto fillOp = dyn_cast<linalg::FillOp>(defOp)) {
      auto operands = fillOp->getOperands();
      if (!operands.empty()) {
        Value fillValue = operands[0];
        if (CVPipeline::isScalarLike(fillValue)) {
          if (llvm::find(matchedOps, fillOp) == matchedOps.end() &&
              fillOp.getBlock() == op->getBlock()) {
            matchedOps.insert(fillOp);
          }
          transSource(fillValue, matchedOps, op->getBlock());
          return true;
        } else {
          LOG_DEBUG(
              "Fill operand is not scalar-like, NOT match. fill=" << *fillOp);
          return false;
        }
      }
    }
  }
  return false;
}

bool FixpipeOptPass::isStoreToGM(Operation *storeOp,
                                 SetVector<Operation *> &matchedOps) {
  Value viewValue = nullptr;
  if (auto materializeOp =
          dyn_cast<bufferization::MaterializeInDestinationOp>(storeOp)) {
    viewValue = materializeOp.getDest();
  } else if (auto hivmStore = dyn_cast<hivm::StoreOp>(storeOp)) {
    viewValue = hivmStore.getDst();
  } else {
    LOG_DEBUG("Cannot find store op, NOT match");
    return false;
  }

  matchedOps.insert(storeOp);
  if (!CVPipeline::collectViewOpsAndCheckGlobalMemory(viewValue, matchedOps)) {
    LOG_DEBUG("Subview is not from global memory (GM), NOT match.");
    return false;
  }
  return true;
}

/** Fixpipe supports cast, the pattern should be like below:
    linalg.matmul
        ↓
    arith.truncf(f32->bf16, f32->f16, i32->i8)
        ↓
    tensor.extract_slice
        ↓
    bufferization.materialize_in_destination memref.subview(gm)
    After optimization, all these ops will be in same block with matmul and set
   core_type to CUBE.
 */
bool FixpipeOptPass::isFixpipeCastPattern(Operation *truncOp,
                                          SetVector<Operation *> &matchedOps) {
  Value truncResult = truncOp->getResult(0);
  if (!truncResult.hasOneUse()) {
    LOG_DEBUG("Trunc not only one user, NOT match.");
    return false;
  }
  auto maybeExtract = *truncResult.getUsers().begin();
  tensor::ExtractSliceOp extractSliceOp = nullptr;
  if (auto extract = dyn_cast<tensor::ExtractSliceOp>(maybeExtract)) {
    extractSliceOp = extract;
  } else if (auto consumerMatmul = dyn_cast<linalg::MatmulOp>(maybeExtract)) {
    // matmul -> trunc -> matmul pattern
    for (Value input : consumerMatmul.getDpsInputs()) {
      if (input == truncResult) {
        matchedOps.insert(truncOp);
        return true;
      }
    }
    LOG_DEBUG("Trunc result is not a DPS input of consumer matmul, NOT match.");
    return false;
  } else {
    LOG_DEBUG("Cannot find extract slice op or matmul, NOT match");
    return false;
  }

  Value extractResult = extractSliceOp.getResult();
  if (!extractResult.hasOneUse()) {
    LOG_DEBUG("Extract Slice not only one user, NOT match.");
    return false;
  }
  auto maybeMaterialize = *extractResult.getUsers().begin();
  bufferization::MaterializeInDestinationOp materializeOp = nullptr;

  if (auto materialize = dyn_cast<bufferization::MaterializeInDestinationOp>(
          maybeMaterialize)) {
    materializeOp = materialize;
  } else {
    LOG_DEBUG("Cannot find materialize op, NOT match");
    return false;
  }

  matchedOps.insert(truncOp);
  matchedOps.insert(extractSliceOp);
  if (!isStoreToGM(materializeOp, matchedOps)) {
    LOG_DEBUG("Not store to GM pattern, NOT match.");
    return false;
  }
  return true;
}

std::optional<Operation *> getOneUserExceptMarkOp(Operation *op) {
  int count = 0;
  Operation *onlyUser = nullptr;
  for (auto user : op->getUsers()) {
    if (!isa<annotation::MarkOp>(user)) {
      count += 1;
      onlyUser = user;
    }
  }
  if (count != 1) {
    return std::nullopt;
  } else {
    return onlyUser;
  }
}

/** Fixpipe supports scaling, the pattern should be like below:
    linalg.matmul
        ↓
    arith.mulf/muli (mul one scalar-like value for quantization)
        ↓
    tensor.extract_slice (optional)
        ↓
    bufferization.materialize_in_destination memref.subview(gm)
    After optimization, all these ops will be in same block with matmul and set
   core_type to CUBE.
 */
bool FixpipeOptPass::isFixpipeMulPattern(Operation *mulOp,
                                         SetVector<Operation *> &matchedOps) {
  Value mulResult = mulOp->getResult(0);
  auto maybeExtract = getOneUserExceptMarkOp(mulOp).value_or(nullptr);
  if (!maybeExtract) {
    LOG_DEBUG("Mul not only one user, NOT match.");
    return false;
  }
  tensor::ExtractSliceOp extractSliceOp = nullptr;
  if (auto extract = dyn_cast<tensor::ExtractSliceOp>(maybeExtract)) {
    extractSliceOp = extract;
    matchedOps.insert(extractSliceOp);
  }

  if (extractSliceOp && !getOneUserExceptMarkOp(extractSliceOp).has_value()) {
    LOG_DEBUG("Extract Slice not only one user, NOT match.");
    return false;
  }

  auto storeOp = extractSliceOp ? getOneUserExceptMarkOp(extractSliceOp).value()
                                : maybeExtract;
  matchedOps.insert(mulOp);
  if (!isStoreToGM(storeOp, matchedOps)) {
    LOG_DEBUG("Not store to GM pattern, NOT match.");
    return false;
  }
  return true;
}

/**
 * Resolve the first value that escapes a chain of nested `scf.for` loops
 * from a value defined inside the loop body.
 *
 * When a loop-carried value is updated by exactly one operation per
 * iteration (the incoming block argument has a single use, which is the
 * defining op of `nowV`, and `nowV` in turn is yielded as that same
 * loop-carried operand), this function walks outward to the result of the
 * outermost such loop.
 *
 * Example:
 *   scf.for (%a = %init) {
 *     %a_i = some_op(%a)            // %a has a single use -> some_op
 *     scf.yield %a_i                // %a_i has a single use -> yield
 *   }
 *   %b = user(%a_n, %k)            // %a_n is the loop result
 *
 * Given `nowV = %a_i` and `outerInValue = %a` (the block argument), this
 * returns `%a_n` — the first value visible to users outside the loop.
 *
 * \param nowV          Result of the op defined inside the loop body.
 * \param outerInValue  The loop-carried block argument (or, at the outermost
 *                      level, the loop init value) that feeds the op defining
 *                      `nowV`.
 * \return              The value that escapes the outermost qualifying loop;
 *                      if any precondition fails, `nowV` is returned unchanged.
 */
static Value getFirstResultAfterLoop(Value nowV, Value outerInValue) {

  if (outerInValue.getDefiningOp()) {
    // we find the outer loop,
    return nowV;
  }

  auto op = nowV.getDefiningOp();
  auto parentOp = op->getParentOp();
  auto nextSearchValue = nowV;

  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    auto blockArg = dyn_cast_if_present<BlockArgument>(outerInValue);
    if (!blockArg || blockArg.getOwner() != forOp.getBody() ||
        !blockArg.hasOneUse()) {
      return nowV;
    }
    auto argUser = blockArg.getUses().begin()->getOwner();
    if (argUser != op) {
      return nowV;
    }

    if (!nowV.hasOneUse()) {
      return nowV;
    }
    auto nowVUser = nowV.getUses().begin()->getOwner();
    int argIdx =
        CVPipeline::getLoopCarriedArgIndex(blockArg, blockArg.getOwner());
    auto yieldOp = dyn_cast<scf::YieldOp>(nowVUser);
    if (!yieldOp || yieldOp->getBlock() != forOp.getBody() ||
        nowV.getUses().begin()->getOperandNumber() != argIdx) {
      return nowV;
    }

    outerInValue = forOp.getInitArgs()[argIdx];
    nextSearchValue = forOp->getResult(argIdx);
  } else {
    LOG_DEBUG("WARN: not limited in one for");
    return nowV;
  }

  return getFirstResultAfterLoop(nextSearchValue, outerInValue);
}

/** Match fixpipe optimization patterns starting from a matmul operation.
 Pattern 1 (Cast Pattern):
   linalg.matmul -> arith.truncf/i -> tensor.extract_slice ->
   bufferization.materialize_in_destination(memref.subview(gm))

 Pattern 2 (Quantization Pattern):
   linalg.matmul -> arith.mulf -> tensor.extract_slice ->
   bufferization.materialize_in_destination(memref.subview(gm))

  NOTE: This function can assure target block id is not -1.
 */
bool FixpipeOptPass::matchFixpipePattern(
    linalg::MatmulOp matmulOp, SetVector<Operation *> &toMergeWithMatmul,
    CVPipeline::ComputeBlockIdManager &bm, int &targetBlockId) {
  LOG_DEBUG("Check matmul op: " << *matmulOp);
  Value matmulResult = matmulOp.getResult(0);
  if (!matmulResult.hasOneUse()) {
    LOG_DEBUG("Matmul not only one user, NOT match.");
    return false;
  }
  Value outerOutValue =
      getFirstResultAfterLoop(matmulResult, *matmulOp.getDpsInits().begin());

  auto outerOutOp = outerOutValue.getDefiningOp();
  LOG_DEBUG("outerOutOp = " << *outerOutOp);
  if (bm.getBlockIdByOp(outerOutOp) == -1) {
    if (llvm::failed(bm.markOpBlockId(outerOutOp))) {
      LOG_DEBUG("Matmul have no blockID, NOT match.");
      return false;
    }
  }
  targetBlockId = bm.getBlockIdByOp(outerOutOp);

  if (!outerOutValue.hasOneUse()) {
    LOG_DEBUG("Matmul(outerOutValue) not only one user, NOT match.");
    return false;
  }
  if (isa<linalg::MatmulOp>(outerOutOp)) {
    toMergeWithMatmul.insert(outerOutOp);
  }

  auto matmulUser = *outerOutValue.getUsers().begin();

  if (CVPipeline::getFixpipePreQuantMode(matmulUser).has_value()) {
    if (isFixpipeCastPattern(matmulUser, toMergeWithMatmul)) {
      return true;
    }
  } else if (isValidMul(matmulUser, matmulResult, toMergeWithMatmul)) {
    if (isFixpipeMulPattern(matmulUser, toMergeWithMatmul)) {
      return true;
    }
  } else {
    LOG_DEBUG("Cannot find valid consumer op (trunc or mulf), NOT match.");
    return false;
  }

  return false;
}

bool FixpipeOptPass::applyFixpipeOpt(
    SetVector<Operation *> &matchedOps, int targetBlockId,
    const CVPipeline::MemoryDependenceGraph &memGraph,
    CVPipeline::ComputeBlockIdManager &bm) {
  // If there are no cycle in Compute Block level, we apply:
  // 1. Change block_id to the matmul's block id
  // 2. Change core_type to CUBE.
  Operation *matmulOp = matchedOps[0];
  for (auto op : matchedOps) {
    if (isa<linalg::MatmulOp>(op)) {
      matmulOp = op;
      break;
    }
  }
  auto block = matmulOp->getBlock();

  if (CVPipeline::willCreateCycle(matchedOps.getArrayRef(), memGraph,
                                  targetBlockId, bm)) {
    return false;
  }
  for (Operation *op : matchedOps) {
    if (isa<scf::SCFDialect>(op->getDialect())) {
      op->walk([&](Operation *nestedOp) {
        // Never fold a sync into a compute block: it must keep its own unique
        // block id so the fence between before/after ops survives.
        if (CVPipeline::isSyncOp(nestedOp)) {
          return WalkResult::advance();
        }
        bm.updateBlockId(nestedOp, targetBlockId);
        nestedOp->setAttr(CVPipeline::kCoreType,
                          StringAttr::get(op->getContext(), "CUBE"));
        return WalkResult::advance();
      });
    } else {
      bm.updateBlockId(op, targetBlockId);
      op->setAttr(CVPipeline::kCoreType,
                  StringAttr::get(op->getContext(), "CUBE"));
    }
  }
  return true;
}

void FixpipeOptPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<hivm::HIVMDialect>();
}

void FixpipeOptPass::runOnOperation() {
  LOG_DEBUG("== FixpipeOpt Pass Start ==\n");
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LOG_DEBUG(module);
  auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memDepGraph(module, aliasAnalysis);
  LOG_DEBUG(module);

  SmallVector<std::pair<SetVector<Operation *>, int>> allMatchedPatterns;

  CVPipeline::ComputeBlockIdManager bm(module);
  module.walk([&](linalg::MatmulOp matmulOp) {
    SetVector<Operation *> matchedOps;
    int targetBlockId = -1;
    if (matchFixpipePattern(matmulOp, matchedOps, bm, targetBlockId)) {
      allMatchedPatterns.push_back({matchedOps, targetBlockId});
    }
  });
  LOG_DEBUG("== Found " << allMatchedPatterns.size()
                        << " fixpipe patterns ==\n");

  /** Inorder to avoid cycle, clone scalar-like ops.
      A-> B -> C
      ↘      ↗
          D
      Now we want to fuse A/B/C, so clone A' for D to avoid cycle.
  */
  for (auto &[matchedOps, targetBlockId] : allMatchedPatterns) {
    if (matchedOps.empty()) {
      continue;
    }
    CVPipeline::SplitIf::ScalarClosure closure{matchedOps.front()->getBlock(),
                                               matchedOps.getArrayRef(), false};
    closure.collect();
    for (auto op : closure.scalarOps) {
      matchedOps.insert(op);
    }
    CVPipeline::cloneScalarOpsForCrossBlockUses(bm, matchedOps, targetBlockId);
    if (!applyFixpipeOpt(matchedOps, targetBlockId, memDepGraph, bm)) {
      for (Operation *op : matchedOps) {
        LOG_DEBUG("Cannot set block id for op: " << *op);
      }
      LOG_DEBUG("Cannot set one Block Id, may be because cycle");
    }
  }

  LOG_DEBUG("== FixpipeOpt Pass Complete ==\n");
}

std::unique_ptr<OperationPass<ModuleOp>> createFixpipeOptPass() {
  return std::make_unique<FixpipeOptPass>();
}

} // namespace triton
} // namespace mlir
