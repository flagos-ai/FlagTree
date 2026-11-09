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

#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "exp-subf-pattern";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace arith;
using namespace math;

namespace {

static math::ExpOp matchExpFromSubf(arith::SubFOp subfOp) {
  auto subfResult = subfOp.getResult();
  if (!subfOp->hasOneUse()) {
    LOG_DEBUG("SubF has multiple users, skipping");
    return nullptr;
  }

  Operation *user = *subfResult.getUsers().begin();
  auto expOp = dyn_cast<math::ExpOp>(user);
  if (!expOp) {
    LOG_DEBUG("SubF user is not ExpOp, skipping");
    return nullptr;
  }
  if (expOp->getBlock() != subfOp->getBlock()) {
    LOG_DEBUG("SubF and ExpOp not in the same Block, skipping");
    return nullptr;
  }
  return expOp;
}

static bool isValidExt(arith::SubFOp subfOp, arith::ExtFOp extOp) {
  if (extOp->getBlock() != subfOp->getBlock()) {
    LOG_DEBUG("SubF and ExtfOp not in the same Block, skipping");
    return false;
  }
  auto inType = extOp.getIn().getType();
  if (!inType.isF16()) {
    LOG_DEBUG("ExtF input types are not both f16, skipping extended pattern");
    return false;
  }
  auto outType = extOp.getOut().getType();
  if (!outType.isF32()) {
    LOG_DEBUG("ExtF output types are not both f32, skipping extended pattern");
    return false;
  }

  auto extRes = extOp.getResult();
  if (!extRes.hasOneUse() || *extRes.getUsers().begin() != subfOp) {
    LOG_DEBUG("ExtF has multiple users or not used by subf, skipping");
    return false;
  }

  return true;
}

static std::optional<SmallVector<arith::ExtFOp, 2>>
matchExtfFromSubf(arith::SubFOp subfOp) {
  auto lhs = subfOp.getLhs();
  auto rhs = subfOp.getRhs();

  auto lhsDef = lhs.getDefiningOp<arith::ExtFOp>();
  auto rhsDef = rhs.getDefiningOp<arith::ExtFOp>();

  if (!lhsDef || !rhsDef) {
    LOG_DEBUG(
        "SubF operands are not both from ExtFOp, skipping extended pattern");
    return std::nullopt;
  }

  if (!isValidExt(subfOp, lhsDef) || !isValidExt(subfOp, rhsDef)) {
    return std::nullopt;
  }

  SmallVector<arith::ExtFOp, 2> extfOps;
  extfOps.push_back(lhsDef);
  extfOps.push_back(rhsDef);
  return extfOps;
}

static void processSubfOp(arith::SubFOp subfOp,
                          CVPipeline::ComputeBlockIdManager &bm) {
  int subfBlockId = bm.getBlockIdByOp(subfOp);

  auto expOp = matchExpFromSubf(subfOp);
  if (!expOp) {
    return;
  }

  int expBlockId = bm.getBlockIdByOp(expOp);
  if (expBlockId != subfBlockId) {
    LOG_DEBUG("Updating exp blockId from " << expBlockId << " to "
                                           << subfBlockId);
    bm.updateBlockId(expOp, subfBlockId);
  } else {
    LOG_DEBUG("Exp blockId already matches subf, skipping update");
  }

  auto extfOps = matchExtfFromSubf(subfOp);
  if (!extfOps) {
    return;
  }

  for (auto extfOp : *extfOps) {
    int extfBlockId = bm.getBlockIdByOp(extfOp);
    if (extfBlockId != subfBlockId) {
      LOG_DEBUG("Updating extf blockId from " << extfBlockId << " to "
                                              << subfBlockId);
      bm.updateBlockId(extfOp, subfBlockId);
    } else {
      LOG_DEBUG("ExtF blockId already matches subf, skipping update");
    }
  }
}

} // anonymous namespace

class ExpSubfPatternPass
    : public PassWrapper<ExpSubfPatternPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpSubfPatternPass)

  ExpSubfPatternPass() = default;

  StringRef getArgument() const override { return "exp-subf-pattern"; }

  StringRef getDescription() const override {
    return "Match exp-subf pattern and unify blockId for exp and optional extf "
           "operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LOG_DEBUG("Input mlir:\n " << module);
    CVPipeline::ComputeBlockIdManager bm(module);

    module.walk([&](arith::SubFOp subfOp) {
      LOG_DEBUG("Process subf: " << subfOp << "\n");
      processSubfOp(subfOp, bm);
    });
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createExpSubfPatternPass() {
  return std::make_unique<ExpSubfPatternPass>();
}

} // namespace triton
} // namespace mlir
