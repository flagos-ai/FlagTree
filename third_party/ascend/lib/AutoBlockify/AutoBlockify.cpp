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

#include "AutoBlockify/AutoBlockify.h"
#include "AutoBlockify/Utils.h"
#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "Utils/Utils.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "auto-blockify"

using namespace mlir;
using namespace triton;

PropagateUnrealizedCastDown::PropagateUnrealizedCastDown(MLIRContext *context,
                                                         Value logicalBlockId,
                                                         Value logicalBlockNum,
                                                         int autoBlockifySize)
    : OpRewritePattern<UnrealizedConversionCastOp>(context),
      logicalBlockId(logicalBlockId), logicalBlockNum(logicalBlockNum),
      autoBlockifySize(autoBlockifySize) {}

LogicalResult
PropagateUnrealizedCastDown::matchAndRewrite(UnrealizedConversionCastOp op,
                                             PatternRewriter &rewriter) const {
  if (op.getInputs().size() != 2)
    return failure();
  auto funcOp = op->getParentOfType<triton::FuncOp>();
  auto input = op.getInputs()[0];
  auto res = op->getResult(0);
  SmallPtrSet<Operation *, 8> users(op->user_begin(), op->user_end());
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Handling UnrealizedConversionCastOp:\n" << op << "\n";
    os << "Users:\n";
    for (auto *user : users)
      os << *user << "\n";
  });
  for (auto *user : users) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(user);
    if (auto uccOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
      if (uccOp->getResultTypes()[0] != input.getType()) {
        LLVM_DEBUG({
          auto &os = llvm::dbgs();
          os << *user << "\n";
        });
        return op.emitError("UnrealizedConversionCastOp cannot be resolved\n");
      }
      rewriter.replaceOp(user, input);
    } else if (auto blockifyLoop = getBlockifyLoop(user)) {
      handleBlockifyLoop(blockifyLoop.value(), user, rewriter);
    } else if (auto splatOp = dyn_cast<triton::SplatOp>(user)) {
      rewriteSplat(op, splatOp, rewriter);
    } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(user)) {
      rewriteExpandDims(op, expandDimsOp, rewriter);
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(user)) {
      rewriteReduce(op, reduceOp, rewriter);
    } else if (auto scanOp = dyn_cast<triton::ScanOp>(user)) {
      rewriteScan(op, scanOp, rewriter);
    } else if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
      rewriteLoad(op, loadOp, rewriter);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
      rewriteStore(op, storeOp, rewriter);
    } else if (auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(user)) {
      rewriteAtomicRMW(op, atomicRMWOp, rewriter);
    } else if (auto assertOp = dyn_cast<triton::AssertOp>(user)) {
      rewriteAssert(op, assertOp, rewriter);
    } else if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
      rewriteExtractSlice(op, extractSliceOp, rewriter);
    } else if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(user)) {
      rewriteInsertSlice(op, insertSliceOp, rewriter);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      rewriteWhile(op, whileOp, rewriter);
    } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
      rewriteLoop(op, loopOp, rewriter);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      rewriteYield(op, yieldOp, rewriter);
    } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      rewriteCondition(op, conditionOp, rewriter);
    } else if (user->hasTrait<OpTrait::Elementwise>() ||
               isa<triton::BroadcastOp, triton::JoinOp, triton::ReshapeOp,
                   triton::PrintOp, triton::ascend::AnnotationOp>(user)) {
      rewriteGeneraleOp(op, user, rewriter);
    } else if (isa<triton::AtomicCASOp>(user)) {
      auto *newOp =
          createBlockifyLoop(user, op, logicalBlockId, logicalBlockNum,
                             autoBlockifySize, rewriter);
      rewriter.setInsertionPoint(newOp);
      handleBlockifyLoop(*getBlockifyLoop(newOp), newOp, rewriter);
    } else {
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Unhandled Op\n" << *user << "\n";
      });
      llvm_unreachable("Unhandled operation");
    }
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "After successful conversion\n";
    os << funcOp << "\n";
  });
  rewriter.eraseOp(op);
  return success();
}

AutoBlockifyPass::AutoBlockifyPass(const AutoBlockifyOptions &options)
    : AutoBlockifyBase(options) {}

bool AutoBlockifyPass::checkBlockifiable(Value v) {
  if (!checkedValues.insert(v).second)
    return true;
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Checking blockifiable:\n" << v << "\n";
  });
  for (auto &use : v.getUses()) {
    auto *user = use.getOwner();
    auto opNum = use.getOperandNumber();
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "User:\n" << *user << "\n";
    });
    if (isa<cf::CondBranchOp, triton::IntToPtrOp, scf::WhileOp, triton::DotOp>(
            user) ||
        llvm::any_of(user->getOperandTypes(), isTensorPtrType))
      return false;
    if (auto ifOp = dyn_cast<scf::IfOp>(user)) {
      user->setAttr(autoBlockifyRegionOpAttr, UnitAttr::get(v.getContext()));
      return true;
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      if (!checkBlockifiable(whileOp.getBeforeArguments()[opNum]))
        return false;
    } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
      auto regionIterArg = loopOp.getTiedLoopRegionIterArg(&use);
      auto loopResult = loopOp.getTiedLoopResult(&use);
      if (!regionIterArg || !loopResult) {
        user->setAttr(autoBlockifyRegionOpAttr, UnitAttr::get(v.getContext()));
        return true;
      }
      if (!checkBlockifiable(regionIterArg) || !checkBlockifiable(loopResult))
        return false;
    } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      auto whileOp = cast<scf::WhileOp>(user->getParentOp());
      if (opNum == 0) {
        whileOp->setAttr(autoBlockifyRegionOpAttr,
                         UnitAttr::get(v.getContext()));
        return true;
      }
      if (!checkBlockifiable(whileOp.getAfterArguments()[opNum - 1]) ||
          !checkBlockifiable(whileOp->getResult(opNum - 1)))
        return false;
    } else if (auto conditionOp = dyn_cast<scf::YieldOp>(user)) {
      if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user->getParentOp());
          loopOp && !checkBlockifiable(loopOp.getInits()[opNum]))
        return false;
    } else {
      for (auto res : user->getResults()) {
        if (!checkBlockifiable(res))
          return false;
      }
    }
  }
  return true;
}

void AutoBlockifyPass::preProcess(triton::FuncOp func) {
  IRRewriter rewriter(func.getContext());
  rewriter.setInsertionPointToStart(&func.getBody().front());
  auto loc = rewriter.getUnknownLoc();
  // Get logical block num
  auto xNum =
      rewriter.create<triton::GetNumProgramsOp>(loc, triton::ProgramIDDim::X);
  auto yNum =
      rewriter.create<triton::GetNumProgramsOp>(loc, triton::ProgramIDDim::Y);
  auto zNum =
      rewriter.create<triton::GetNumProgramsOp>(loc, triton::ProgramIDDim::Z);
  auto yzNum = rewriter.create<arith::MulIOp>(loc, yNum, zNum);
  logicalBlockNum = rewriter.create<arith::MulIOp>(loc, yzNum, xNum);

  // Get logical block id
  auto xDim =
      rewriter.create<triton::GetProgramIdOp>(loc, triton::ProgramIDDim::X);
  auto yDim =
      rewriter.create<triton::GetProgramIdOp>(loc, triton::ProgramIDDim::Y);
  auto zDim =
      rewriter.create<triton::GetProgramIdOp>(loc, triton::ProgramIDDim::Z);
  xDim->setAttr(logicalBlockIdAttr, rewriter.getUnitAttr());
  yDim->setAttr(logicalBlockIdAttr, rewriter.getUnitAttr());
  zDim->setAttr(logicalBlockIdAttr, rewriter.getUnitAttr());
  auto xFlatten = rewriter.create<arith::MulIOp>(loc, xDim, yzNum);
  auto yFlatten = rewriter.create<arith::MulIOp>(loc, yDim, zNum);
  logicalBlockId = rewriter.create<arith::AddIOp>(loc, xFlatten, yFlatten);
  logicalBlockId = rewriter.create<arith::AddIOp>(loc, logicalBlockId, zDim);

  // get blockified block id
  auto blockifyTensorType =
      RankedTensorType::get({autoBlockifySize}, rewriter.getI32Type());
  auto blockfyRange = rewriter.create<triton::MakeRangeOp>(
      loc, blockifyTensorType, 0, autoBlockifySize);
  auto splatedLogicalBlockId = rewriter.create<triton::SplatOp>(
      loc, blockfyRange.getType(), logicalBlockId);
  Value blockifiedId =
      rewriter.create<arith::AddIOp>(loc, splatedLogicalBlockId, blockfyRange);

  // get mask
  auto splatedBlockNum = rewriter.create<triton::SplatOp>(
      loc, blockfyRange.getType(), logicalBlockNum);
  auto upperboundMask = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, blockifiedId, splatedBlockNum);
  auto splatedZero = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(blockifyTensorType,
                                  rewriter.getI32IntegerAttr(0)));
  auto lowerboundMask = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, blockifiedId, splatedZero);
  Value blockifiedIdMask =
      rewriter.create<arith::OrIOp>(loc, upperboundMask, lowerboundMask);

  blockifiedId = rewriter
                     .create<UnrealizedConversionCastOp>(
                         loc, logicalBlockId.getType(),
                         ValueRange({blockifiedId, blockifiedIdMask}))
                     ->getResult(0);

  // replace program id to be computed from blockified id
  SmallVector<triton::GetProgramIdOp> toReplace;
  func.walk([&](triton::GetProgramIdOp id) {
    if (id->hasAttr(logicalBlockIdAttr))
      return;
    toReplace.push_back(id);
  });
  for (auto id : toReplace) {
    rewriter.setInsertionPoint(id);
    Value newId;
    if (id.getAxis() == triton::ProgramIDDim::X) {
      newId = rewriter.create<arith::DivSIOp>(id.getLoc(), blockifiedId, yzNum);
      newId = rewriter.create<arith::RemSIOp>(id.getLoc(), newId, xNum);
    } else if (id.getAxis() == triton::ProgramIDDim::Y) {
      newId = rewriter.create<arith::DivSIOp>(id.getLoc(), blockifiedId, zNum);
      newId = rewriter.create<arith::RemSIOp>(id.getLoc(), newId, yNum);
    } else {
      newId = rewriter.create<arith::RemSIOp>(id.getLoc(), blockifiedId, zNum);
    }
    rewriter.replaceOp(id, newId);
  }

  // Create for loop for region ops
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->hasAttr(autoBlockifyRegionOpAttr)) {
      auto *newOp = createBlockifyLoop(
          op, blockifiedId.getDefiningOp<UnrealizedConversionCastOp>(),
          logicalBlockId, logicalBlockNum, autoBlockifySize, rewriter);
      newOp->removeAttr(autoBlockifyRegionOpAttr);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

void AutoBlockifyPass::runOnOperation() {
  if (autoBlockifySize == 1)
    return;
  ModuleOp moduleOp = getOperation();
  if (autoBlockifySize <= 0) {
    moduleOp->emitWarning("[AutoBlockify V2] AutoBlockifySize cannot be "
                          "negative integer, skipping.");
    return signalPassFailure();
  }

  MLIRContext *ctx = &getContext();

  moduleOp.walk([&](triton::FuncOp func) {
    LogicalResult result = success();
    func.walk([&](triton::GetProgramIdOp id) {
      if (!checkBlockifiable(id.getResult())) {
        result = failure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failed(result)) {
      func->emitWarning("Cannot apply auto blockify");
      return WalkResult::skip();
    }
    preProcess(func);

    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "After preprocess:\n" << func << "\n";
    });

    RewritePatternSet patterns(ctx);
    patterns.add<PropagateUnrealizedCastDown>(
        ctx, logicalBlockId, logicalBlockNum, autoBlockifySize);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      moduleOp->emitError("failed to apply Patterns");
      signalPassFailure();
      return WalkResult::interrupt();
    }

    IRRewriter rewriter(ctx);
    func->walk([&](UnrealizedConversionCastOp op) {
      rewriter.setInsertionPoint(op);
      auto input = op.getInputs()[0];
      auto resType = cast<RankedTensorType>(op->getResultTypes()[0]);
      if (auto constantOp = input.getDefiningOp<arith::ConstantOp>()) {
        Attribute val = constantOp.getValue();
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(val))
          val = denseAttr.getSplatValue<Attribute>();
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            op, DenseElementsAttr::get(resType, val));
      } else if (auto tensorType =
                     dyn_cast<RankedTensorType>(input.getType())) {
        input = rewriter.create<triton::ExpandDimsOp>(input.getLoc(), input, 0);
        rewriter.replaceOpWithNewOp<triton::BroadcastOp>(op, resType, input);
      } else {
        rewriter.replaceOpWithNewOp<triton::SplatOp>(op, resType, input);
      }
    });
    func->setAttr(autoBlockifySizeAttr,
                  rewriter.getI32IntegerAttr(autoBlockifySize));
    return WalkResult::skip();
  });

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, moduleOp))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createAutoBlockifyPass(const AutoBlockifyOptions &options) {
  return std::make_unique<AutoBlockifyPass>(options);
}
