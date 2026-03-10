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
#include <functional>
#include <memory>
#include <queue>
#include <set>
#include <utility>

#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Utility.h"

#include "Analysis/AxisInfoEx.h"
#include "Dialect/MathExt/IR/MathExt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_GCUTRITONFUSIONPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gcu;

namespace {

static const char *const kIsContinual = "IsContinual";

struct LoadOptimizationPatter : public OpRewritePattern<triton::LoadOp> {
  explicit LoadOptimizationPatter(MLIRContext *context,
                                  ModuleAxisInfoExAnalysis &analysis)
      : OpRewritePattern<triton::LoadOp>(context), analysis(analysis) {}
  ModuleAxisInfoExAnalysis &analysis;
  mlir::LogicalResult
  matchAndRewrite(triton::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto addPtrOp = op.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!addPtrOp) {
      return failure();
    }
    auto rankedTensorTy = dyn_cast<RankedTensorType>(op.getPtr().getType());
    if (!rankedTensorTy || rankedTensorTy.getRank() != 1) {
      return failure();
    }
    if (dyn_cast<triton::PointerType>(rankedTensorTy.getElementType())
            .getPointeeType()
            .isInteger(64)) {
      return failure();
    }
    auto axisInfoEx = analysis.getAxisInfoEx(op.getPtr());
    bool isContinual = axisInfoEx->getContinualInterval(0) == 1;

    auto mask = op.getMask();
    auto other = op.getOther();
    auto offset = addPtrOp.getOffset();
    auto ptr = addPtrOp.getPtr();

    while (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
      ptr = addPtrOp.getPtr();
      auto promoteType =
          mlir::triton::gcu::getBpe(getElementTypeOrSelf(offset.getType())) >
                  mlir::triton::gcu::getBpe(
                      getElementTypeOrSelf(addPtrOp.getOffset().getType()))
              ? offset.getType()
              : addPtrOp.getOffset().getType();
      if (promoteType != offset.getType()) {
        offset = rewriter.create<arith::ExtUIOp>(loc, promoteType, offset);
      }
      offset = rewriter.create<arith::AddIOp>(
          loc, offset,
          addPtrOp.getOffset().getType() == promoteType
              ? addPtrOp.getOffset()
              : rewriter.create<arith::ExtUIOp>(loc, promoteType,
                                                addPtrOp.getOffset()));
    }

    if (auto splatOp = ptr.getDefiningOp<triton::SplatOp>()) {
      ptr = splatOp.getSrc();
      while (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
        auto promoteType =
            mlir::triton::gcu::getBpe(getElementTypeOrSelf(offset.getType())) >
                    mlir::triton::gcu::getBpe(
                        getElementTypeOrSelf(addPtrOp.getOffset().getType()))
                ? offset.getType()
                : cast<RankedTensorType>(offset.getType())
                      .clone(addPtrOp.getOffset().getType());
        if (promoteType != offset.getType()) {
          offset = rewriter.create<arith::ExtUIOp>(loc, promoteType, offset);
        }
        Value accOffset = rewriter.create<triton::SplatOp>(
            loc,
            cast<RankedTensorType>(offset.getType())
                .clone(getElementTypeOrSelf(addPtrOp.getOffset().getType())),
            addPtrOp.getOffset());
        if (promoteType != accOffset.getType()) {
          accOffset =
              rewriter.create<arith::ExtUIOp>(loc, promoteType, accOffset);
        }

        offset = rewriter.create<arith::AddIOp>(loc, offset, accOffset);

        ptr = addPtrOp.getPtr();
      }
      auto maskedLoadOp = rewriter.create<triton::gcu::MaskedLoadOp>(
          loc, ptr, offset, mask, other);
      if (isContinual) {
        maskedLoadOp->setAttr(kIsContinual, rewriter.getBoolAttr(isContinual));
      }

      rewriter.replaceOp(op, maskedLoadOp);
      return success();
    }
    return failure();
  }
};

struct StoreOptimizationPatter : public OpRewritePattern<triton::StoreOp> {
  explicit StoreOptimizationPatter(MLIRContext *context,
                                   ModuleAxisInfoExAnalysis &analysis)
      : OpRewritePattern<triton::StoreOp>(context), analysis(analysis) {}
  ModuleAxisInfoExAnalysis &analysis;
  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto addPtrOp = op.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!addPtrOp) {
      return failure();
    }
    auto rankedTensorTy = dyn_cast<RankedTensorType>(op.getPtr().getType());
    if (!rankedTensorTy || rankedTensorTy.getRank() != 1) {
      return failure();
    }
    if (dyn_cast<triton::PointerType>(rankedTensorTy.getElementType())
            .getPointeeType()
            .isInteger(64)) {
      return failure();
    }

    auto axisInfoEx = analysis.getAxisInfoEx(op.getPtr());
    bool isContinual = axisInfoEx->getContinualInterval(0) == 1;

    auto offset = addPtrOp.getOffset();
    auto mask = op.getMask();
    auto value = op.getValue();
    auto ptr = addPtrOp.getPtr();

    while (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
      ptr = addPtrOp.getPtr();
      auto promoteType =
          mlir::triton::gcu::getBpe(getElementTypeOrSelf(offset.getType())) >
                  mlir::triton::gcu::getBpe(
                      getElementTypeOrSelf(addPtrOp.getOffset().getType()))
              ? offset.getType()
              : addPtrOp.getOffset().getType();
      if (promoteType != offset.getType()) {
        offset = rewriter.create<arith::ExtUIOp>(loc, promoteType, offset);
      }
      offset = rewriter.create<arith::AddIOp>(
          loc, offset,
          addPtrOp.getOffset().getType() == promoteType
              ? addPtrOp.getOffset()
              : rewriter.create<arith::ExtUIOp>(loc, promoteType,
                                                addPtrOp.getOffset()));
    }

    if (auto splatOp = ptr.getDefiningOp<triton::SplatOp>()) {
      ptr = splatOp.getSrc();
      while (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
        auto promoteType =
            mlir::triton::gcu::getBpe(getElementTypeOrSelf(offset.getType())) >
                    mlir::triton::gcu::getBpe(
                        getElementTypeOrSelf(addPtrOp.getOffset().getType()))
                ? offset.getType()
                : cast<RankedTensorType>(offset.getType())
                      .clone(addPtrOp.getOffset().getType());
        if (promoteType != offset.getType()) {
          offset = rewriter.create<arith::ExtUIOp>(loc, promoteType, offset);
        }
        Value accOffset = rewriter.create<triton::SplatOp>(
            loc,
            cast<RankedTensorType>(offset.getType())
                .clone(getElementTypeOrSelf(addPtrOp.getOffset().getType())),
            addPtrOp.getOffset());
        if (promoteType != accOffset.getType()) {
          accOffset =
              rewriter.create<arith::ExtUIOp>(loc, promoteType, accOffset);
        }

        offset = rewriter.create<arith::AddIOp>(loc, offset, accOffset);

        ptr = addPtrOp.getPtr();
      }

      auto maskedStoreOp = rewriter.create<triton::gcu::MaskedStoreOp>(
          loc, ptr, offset, value, mask);
      if (isContinual) {
        maskedStoreOp->setAttr(kIsContinual, rewriter.getBoolAttr(isContinual));
      }
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct ConvertClampFOp : public OpRewritePattern<triton::ClampFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ClampFOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (stringifyPropagateNan(op.getPropagateNan()) == "all") {
      auto newOp = rewriter.create<arith::MaximumFOp>(
          loc, op.getMin(),
          rewriter.create<arith::MinimumFOp>(loc, op.getX(), op.getMax()));
      rewriter.replaceOp(op, newOp);
    } else {
      auto newOp = rewriter.create<arith::MaxNumFOp>(
          loc, op.getMin(),
          rewriter.create<arith::MinNumFOp>(loc, op.getX(), op.getMax()));
      rewriter.replaceOp(op, newOp);
    }
    return success();
  }
};

struct ConvertPreciseDivFOp : public OpRewritePattern<triton::PreciseDivFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::PreciseDivFOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newOp = rewriter.create<arith::DivFOp>(loc, op.getX(), op.getY());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertFpToFpOp : public OpRewritePattern<triton::FpToFpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FpToFpOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    mlir::Value newOp = nullptr;
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto resType = cast<RankedTensorType>(op.getResult().getType());
    if (getElementBitWidth(resType) >
        getElementBitWidth(srcType)) { // Cast from floating-point
                                       // to wider floating-point, fp8->fp32
      newOp = rewriter.create<arith::ExtFOp>(loc, op.getType(), op.getSrc());
    } else { // Cast from floating-point to narrower floating-point, fp32->fp8
      newOp = rewriter.create<arith::TruncFOp>(loc, op.getType(), op.getSrc());
    }

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertPreciseSqrtOp : public OpRewritePattern<triton::PreciseSqrtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::PreciseSqrtOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newOp = rewriter.create<math::SqrtOp>(loc, op.getX());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertMulhiUIOpOp : public OpRewritePattern<triton::MulhiUIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::MulhiUIOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newOp =
        rewriter.create<mlir::math_ext::UmulhiOp>(loc, op.getX(), op.getY());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

} // namespace

namespace {

struct FusionInfo {
  SmallVector<Operation *> fusionOps;
  SetVector<Value> fusionOperands;
  SetVector<Value> fusionResults;
  ArrayRef<int64_t> shape;

  void collectInfo();
};

struct GCUTritonFusionPass
    : public mlir::impl::GCUTritonFusionPassBase<GCUTritonFusionPass> {
  using Base::Base;

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ModuleAxisInfoExAnalysis axisInfoExAnalysis(
        module->getParentOfType<ModuleOp>());
    DominanceInfo dominanceInfo(module);
    patterns.add<LoadOptimizationPatter, StoreOptimizationPatter>(
        ctx, axisInfoExAnalysis);
    patterns.add<ConvertClampFOp, ConvertPreciseDivFOp, ConvertPreciseSqrtOp,
                 ConvertFpToFpOp, ConvertMulhiUIOpOp>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();

    auto cmp = [&dominanceInfo](Operation *lhs, Operation *rhs) {
      return lhs != rhs && dominanceInfo.dominates(lhs, rhs);
    };
    std::set<Operation *, decltype(cmp)> opSet(cmp);
    std::queue<Operation *> workList;
    IRMapping mapper;
    DenseSet<Operation *> LoadStoreOpSet;
    module->walk([&](Operation *op) {
      OpBuilder builder(op);
      if (isa<triton::gcu::MaskedLoadOp>(op) && LoadStoreOpSet.count(op) == 0) {
        LoadStoreOpSet.insert(op);
        auto maskedLoadOp = cast<triton::gcu::MaskedLoadOp>(op);
        if (auto defOp = maskedLoadOp.getOffset().getDefiningOp()) {
          workList.push(defOp);
          opSet.insert(defOp);
        }
        auto result = maskedLoadOp.getResult();
        Operation *firstUser = nullptr;
        for (auto user : result.getUsers()) {
          if (user->getBlock() == op->getBlock() &&
              (!firstUser || user->isBeforeInBlock(firstUser))) {
            firstUser = user;
          }
        }
        if (firstUser && ++Block::iterator(op) != Block::iterator(firstUser)) {
          op->moveBefore(firstUser);
          builder.setInsertionPoint(op);
        }
        if (maskedLoadOp.getMask()) {
          if (auto defOp = maskedLoadOp.getMask().getDefiningOp()) {
            workList.push(defOp);
            opSet.insert(defOp);
          }
        }
        if (maskedLoadOp.getOther()) {
          if (auto defOp = maskedLoadOp.getOther().getDefiningOp()) {
            workList.push(defOp);
            opSet.insert(defOp);
          }
        }
        while (!workList.empty()) {
          auto o = workList.front();
          workList.pop();
          for (auto operand : o->getOperands()) {
            auto defOp = operand.getDefiningOp();
            if (defOp && canFuse(defOp) && opSet.count(defOp) == 0) {
              workList.push(defOp);
              opSet.insert(defOp);
            }
          }
        }
        for (auto o : opSet) {
          auto cloneOp = builder.clone(*o, mapper);
          for (auto [result, newResult] :
               llvm::zip(o->getResults(), cloneOp->getResults())) {
            mapper.map(result, newResult);
          }
        }
        if (auto offset = mapper.lookupOrNull(maskedLoadOp.getOffset())) {
          maskedLoadOp.setOperand(1, offset);
        }
        if (auto mask = mapper.lookupOrNull(maskedLoadOp.getMask())) {
          maskedLoadOp.setOperand(2, mask);
        }
        if (auto other = mapper.lookupOrNull(maskedLoadOp.getOther())) {
          maskedLoadOp.setOperand(3, other);
        }
        for (auto iter = opSet.rbegin(); iter != opSet.rend(); ++iter) {
          auto op = *iter;
          if (op->use_empty()) {
            op->erase();
          }
        }
        opSet.clear();
        mapper.clear();
      } else if (isa<triton::gcu::MaskedStoreOp>(op) &&
                 LoadStoreOpSet.count(op) == 0) {
        LoadStoreOpSet.insert(op);
        auto maskedStoreOp = dyn_cast<triton::gcu::MaskedStoreOp>(op);
        if (auto defOp = maskedStoreOp.getOffset().getDefiningOp()) {
          workList.push(defOp);
          opSet.insert(defOp);
        }
        if (maskedStoreOp.getMask()) {
          if (auto defOp = maskedStoreOp.getMask().getDefiningOp()) {
            workList.push(defOp);
            opSet.insert(defOp);
          }
        }
        while (!workList.empty()) {
          auto o = workList.front();
          workList.pop();
          for (auto operand : o->getOperands()) {
            auto defOp = operand.getDefiningOp();
            if (defOp && canFuse(defOp) && opSet.count(defOp) == 0) {
              workList.push(defOp);
              opSet.insert(defOp);
            }
          }
        }
        for (auto o : opSet) {
          auto cloneOp = builder.clone(*o, mapper);
          for (auto [result, newResult] :
               llvm::zip(o->getResults(), cloneOp->getResults())) {
            mapper.map(result, newResult);
          }
        }
        if (auto offset = mapper.lookupOrNull(maskedStoreOp.getOffset())) {
          maskedStoreOp.setOperand(1, offset);
        }
        if (auto mask = mapper.lookupOrNull(maskedStoreOp.getMask())) {
          maskedStoreOp.setOperand(3, mask);
        }
        for (auto iter = opSet.rbegin(); iter != opSet.rend(); ++iter) {
          auto op = *iter;
          if (op->use_empty()) {
            op->erase();
          }
        }
        opSet.clear();
        mapper.clear();
      }
    });

    module->walk([&](Operation *op) {
      if (canFuse(op)) {
        for (auto operand : op->getOperands()) {
          auto def = operand.getDefiningOp();
          if (def) {
            if (isa<arith::ConstantOp>(def) || isa<triton::SplatOp>(def) ||
                isa<triton::MakeRangeOp>(def) ||
                (isa<triton::BroadcastOp>(def) && canFuse(def))) {
              if (def->getBlock() != op->getBlock() ||
                  ++Block::iterator(def) != Block::iterator(op)) {
                OpBuilder builder(op);
                op->replaceUsesOfWith(operand,
                                      builder.clone(*def)->getResult(0));
              }
            }
          }
        }
      }
    });

    for (auto func : module.getOps<triton::FuncOp>()) {
      for (auto &region : func->getRegions()) {
        runFuse(region);
      }
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TritonGCUDialect, arith::ArithDialect, math::MathDialect,
                    triton::TritonDialect>();
  }

private:
  bool canFuse(Operation *op);
  void runFuse(Region &region);
  void runFuse(Block &block);
  void fuseOps(std::unique_ptr<FusionInfo> fusionRegion);
  bool tryDeepFuse(SmallVector<std::unique_ptr<FusionInfo>> &fusionRegions);
};
} // namespace

void FusionInfo::collectInfo() {
  DenseSet<Operation *> fusionOpSet(fusionOps.begin(), fusionOps.end());
  fusionOperands.clear();
  fusionResults.clear();
  for (auto op : fusionOps) {
    for (auto v : op->getOperands()) {
      if (!fusionOpSet.count(v.getDefiningOp())) {
        fusionOperands.insert(v);
      }
    }
    for (auto result : op->getResults()) {
      for (auto user : result.getUsers()) {
        if (!fusionOpSet.count(user)) {
          fusionResults.insert(result);
        }
      }
    }
  }
}

bool GCUTritonFusionPass::tryDeepFuse(
    SmallVector<std::unique_ptr<FusionInfo>> &fusionRegions) {
  auto numFusionRegion = fusionRegions.size();
  for (unsigned i = 0; i < numFusionRegion - 1; ++i) {
    Operation *firstUser = nullptr;
    auto &fusionResults = fusionRegions[i]->fusionResults;
    auto block = fusionRegions[i]->fusionOps[0]->getBlock();
    auto region = fusionRegions[i]->fusionOps[0]->getParentRegion();

    for (auto result : fusionResults) {
      for (auto user : result.getUsers()) {
        while (user && user->getBlock() != block &&
               user->getParentRegion() != region) {
          user = user->getParentOp();
        }
        if (user->getBlock() != block) {
          continue;
        }
        assert(user != nullptr);
        if (firstUser == nullptr) {
          firstUser = user;
        } else if (user->isBeforeInBlock(firstUser)) {
          firstUser = user;
        }
      }
    }
    if (firstUser) {
      for (unsigned j = i + 1; j < numFusionRegion; ++j) {
        auto &fusionOps = fusionRegions[j]->fusionOps;
        if (fusionRegions[i]->shape == fusionRegions[j]->shape &&
            (firstUser == fusionOps.front() || firstUser == fusionOps.back() ||
             (firstUser->isBeforeInBlock(fusionOps.back()) &&
              fusionOps.front()->isBeforeInBlock(firstUser))) &&
            !(std::any_of(
                  Block::iterator(fusionRegions[i]->fusionOps.back()),
                  Block::iterator(fusionRegions[j]->fusionOps.front()),
                  [](auto &op) { return isa<mlir::gpu::BarrierOp>(op); }) &&
              llvm::any_of(fusionRegions[i]->fusionOps,
                           [](auto op) { return !isMemoryEffectFree(op); }))) {
          for (auto op : fusionRegions[i]->fusionOps) {
            op->moveBefore(fusionOps.front());
          }
          fusionRegions[j]->fusionOps.insert(
              fusionOps.begin(), fusionRegions[i]->fusionOps.begin(),
              fusionRegions[i]->fusionOps.end());
          fusionRegions[j]->collectInfo();
          fusionRegions.erase(fusionRegions.begin() + i);
          return true;
        }
      }
    }

    for (unsigned j = i + 1; j < numFusionRegion; ++j) {
      auto &fusionOperands = fusionRegions[j]->fusionOperands;
      if (fusionRegions[i]->shape == fusionRegions[j]->shape &&
          llvm::any_of(fusionOperands,
                       [&](auto operand) {
                         return llvm::any_of(fusionResults,
                                             [&](auto result) {
                                               return result == operand;
                                             }) ||
                                llvm::any_of(
                                    fusionRegions[i]->fusionOperands,
                                    [&](auto v) { return v == operand; });
                       }) &&
          llvm::all_of(fusionOperands,
                       [&](auto operand) {
                         auto defOp = operand.getDefiningOp();
                         return !defOp || defOp->getBlock() != block ||
                                defOp->isBeforeInBlock(
                                    fusionRegions[i]->fusionOps.back()) ||
                                defOp == fusionRegions[i]->fusionOps.back();
                       }) &&
          !(std::any_of(
                Block::iterator(fusionRegions[i]->fusionOps.back()),
                Block::iterator(fusionRegions[j]->fusionOps.front()),
                [](auto &op) { return isa<mlir::gpu::BarrierOp>(op); }) &&
            llvm::any_of(fusionRegions[j]->fusionOps,
                         [](auto op) { return !isMemoryEffectFree(op); }))) {
        for (auto op : fusionRegions[j]->fusionOps) {
          op->moveAfter(fusionRegions[i]->fusionOps.back());
        }
        fusionRegions[i]->fusionOps.append(fusionRegions[j]->fusionOps.begin(),
                                           fusionRegions[j]->fusionOps.end());
        fusionRegions[i]->collectInfo();
        fusionRegions.erase(fusionRegions.begin() + j);
        return true;
      }
    }
  }
  return false;
}

void GCUTritonFusionPass::runFuse(Block &block) {
  SmallVector<std::unique_ptr<FusionInfo>> fusionRegions;
  fusionRegions.emplace_back(std::make_unique<FusionInfo>());
  for (auto &op : llvm::make_early_inc_range(block.getOperations())) {
    auto &fusionOps = fusionRegions.back()->fusionOps;
    if (canFuse(&op)) {
      if (fusionOps.empty()) {
        if (!isa<triton::gpu::ConvertLayoutOp>(op)) {
          fusionOps.push_back(&op);
          fusionRegions.back()->shape =
              cast<RankedTensorType>(op.getResultTypes().front()).getShape();
        }
      } else {
        if (isa<triton::gpu::ConvertLayoutOp>(op) &&
            llvm::all_of(fusionOps, [&](auto innerOp) {
              return llvm::none_of(innerOp->getUsers(),
                                   [&](auto user) { return user == &op; });
            })) {
          op.moveBefore(fusionOps.front());
        } else {
          auto curType =
              isa<triton::gcu::MaskedStoreOp>(op)
                  ? cast<triton::gcu::MaskedStoreOp>(op).getValue().getType()
                  : cast<RankedTensorType>(op.getResultTypes().front());
          auto preType =
              isa<triton::gcu::MaskedStoreOp>(fusionOps.back())
                  ? cast<triton::gcu::MaskedStoreOp>(fusionOps.back())
                        .getValue()
                        .getType()
                  : cast<RankedTensorType>(
                        fusionOps.back()->getResultTypes().front());
          if (curType.getShape() == preType.getShape() &&
              getElemsPerThread(curType) == getElemsPerThread(preType)) {
            fusionOps.push_back(&op);
          } else {
            fusionRegions.back()->collectInfo();
            auto fusionInfo = std::make_unique<FusionInfo>();
            fusionInfo->fusionOps.push_back(&op);
            fusionInfo->shape = curType.getShape();
            fusionRegions.emplace_back(std::move(fusionInfo));
          }
        }
      }
    } else if (!fusionOps.empty()) {
      if ((op.hasTrait<OpTrait::Elementwise>() ||
           isa<triton::gcu::LoadOp>(op)) &&
          llvm::all_of(fusionOps, [&](auto innerOp) {
            return llvm::none_of(innerOp->getUsers(),
                                 [&](auto user) { return user == &op; });
          })) {
        op.moveBefore(fusionOps.front());
      } else {
        fusionRegions.back()->collectInfo();
        fusionRegions.emplace_back(std::make_unique<FusionInfo>());
      }
    }
  }

  if (fusionRegions.back()->fusionOps.empty()) {
    fusionRegions.pop_back();
  } else {
    fusionRegions.back()->collectInfo();
  }

  if (fusionRegions.empty()) {
    return;
  }

  bool changed = true;
  do {
    changed = tryDeepFuse(fusionRegions);
  } while (changed);

  for (auto &fusionRegion : fusionRegions) {
    fuseOps(std::move(fusionRegion));
  }
}

void GCUTritonFusionPass::runFuse(Region &region) {
  for (auto &block : region) {
    for (auto &op : block) {
      for (auto &region : op.getRegions()) {
        runFuse(region);
      }
    }
    runFuse(block);
  }
}

bool GCUTritonFusionPass::canFuse(Operation *op) {
  if (!op) {
    return false;
  }
  if (!llvm::all_of(op->getResultTypes(), [](auto type) {
        auto rankedTensorTy = dyn_cast<RankedTensorType>(type);
        if (rankedTensorTy) {
          auto elementTy = rankedTensorTy.getElementType();
          return isa<Float8E5M2Type>(elementTy) ||
                 isa<Float8E4M3FNType>(elementTy) || elementTy.isBF16() ||
                 elementTy.isF16() || elementTy.isF32() ||
                 elementTy.isInteger(1) || elementTy.isInteger(8) ||
                 elementTy.isInteger(16) || elementTy.isInteger(32) ||
                 elementTy.isInteger(64);
        }
        return false;
      })) {
    return false;
  }

  if (!llvm::all_of(op->getResultTypes(), llvm::IsaPred<RankedTensorType>)) {
    return false;
  }
  if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    auto valueAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
    return valueAttr && valueAttr.isSplat();
  }
  if (isa<triton::MakeRangeOp, triton::BitcastOp, triton::SplatOp,
          triton::ExternElementwiseOp, triton::gcu::MaskedLoadOp,
          triton::gcu::MaskedStoreOp>(op)) {
    return true;
  }
  if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(op)) {
    auto srcType = broadcastOp.getSrc().getType();
    auto resultType = broadcastOp.getType();
    auto rank = srcType.getRank();
    unsigned broadcastAxis = -1;
    if (getElementTypeOrSelf(srcType).isInteger(1)) {
      return false;
    }
    for (unsigned i = 0; i < rank; ++i) {
      if (srcType.getDimSize(i) != resultType.getDimSize(i)) {
        if (broadcastAxis == -1) {
          broadcastAxis = i;
        } else {
          return false;
        }
      }
    }
    if (broadcastAxis != 0 && broadcastAxis != rank - 1) {
      return false;
    }
    auto elemsPerThread = broadcastAxis == 0
                              ? triton::gcu::getElemsPerThread(srcType)
                              : triton::gcu::getElemsPerThread(resultType);
    auto sizeInBytes =
        std::accumulate(elemsPerThread.begin() + broadcastAxis,
                        elemsPerThread.end(), 1, std::multiplies<int64_t>()) *
        triton::gcu::getBpe(getElementTypeOrSelf(srcType));
    static constexpr unsigned oaccSizeInBytes = 512;
    static constexpr unsigned loopUnrollTime = 16;
    auto numOacc = sizeInBytes / oaccSizeInBytes;
    if (numOacc >= 1 && numOacc <= 4 * loopUnrollTime) {
      return true;
    }
  } else if (auto cvtLayoutOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cvtLayoutOp.getSrc().getType();
    auto dstTy = cvtLayoutOp.getType();
    auto srcNumElems = triton::gcu::getElemsPerThread(srcTy);
    auto dstNumElems = triton::gcu::getElemsPerThread(dstTy);
    auto srcEncoding = srcTy.getEncoding();
    auto dstEncoding = dstTy.getEncoding();
    return srcNumElems == dstNumElems &&
           isa<triton::gpu::SliceEncodingAttr,
               triton::gpu::BlockedEncodingAttr>(srcEncoding) &&
           isa<triton::gpu::SliceEncodingAttr,
               triton::gpu::BlockedEncodingAttr>(dstEncoding);
  }
  return OpTrait::hasElementwiseMappableTraits(op);
}

void GCUTritonFusionPass::fuseOps(std::unique_ptr<FusionInfo> fusionRegion) {
  auto &ops = fusionRegion->fusionOps;
  auto &fusionResults = fusionRegion->fusionResults;
  SetVector<Value> fusionOperands;
  DenseSet<Operation *> fusionOpSet(ops.begin(), ops.end());
  for (auto op : ops) {
    for (auto v : op->getOperands()) {
      if (!fusionOpSet.count(v.getDefiningOp())) {
        fusionOperands.insert(v);
      }
    }
  }

  OpBuilder builder(ops.front());
  auto loc = ops.front()->getLoc();
  auto operands = fusionOperands.takeVector();
  auto results = fusionResults.takeVector();

  auto resultTypes = llvm::to_vector(
      llvm::map_range(results, [](auto result) { return result.getType(); }));

  auto fusedOp = builder.create<mlir::triton::gcu::ElementwiseFusionRegionOp>(
      loc, resultTypes, operands);
  auto &entryBlock = fusedOp.getRegion().emplaceBlock();
  {
    IRMapping map;
    for (auto operand : operands) {
      auto arg = entryBlock.addArgument(operand.getType(), loc);
      map.map(operand, arg);
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&entryBlock);
    for (auto op : ops) {
      auto newOp = builder.clone(*op, map);
      for (auto [result, newResult] :
           llvm::zip(op->getResults(), newOp->getResults())) {
        map.map(result, newResult);
      }
    }
    auto fusedResults = llvm::to_vector(llvm::map_range(
        results, [&map](auto result) { return map.lookup(result); }));
    builder.create<mlir::triton::gcu::YieldOp>(loc, fusedResults);
  }

  for (auto [result, fusedResult] : llvm::zip(results, fusedOp.getResults())) {
    result.replaceAllUsesWith(fusedResult);
  }

  for (auto op : ops) {
    op->dropAllUses();
    op->erase();
  }
}
