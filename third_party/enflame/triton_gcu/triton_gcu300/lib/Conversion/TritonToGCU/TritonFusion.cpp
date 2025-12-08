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
#include <utility>

#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Utils.h"

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

struct LoadOptimizationPatter : public OpRewritePattern<triton::LoadOp> {
  explicit LoadOptimizationPatter(MLIRContext *context)
      : OpRewritePattern<triton::LoadOp>(context) {}
  mlir::LogicalResult
  matchAndRewrite(triton::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto addPtrOp = op.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!addPtrOp) {
      return failure();
    }

    auto splatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
    if (!splatOp) {
      return failure();
    }
    auto ptr = splatOp.getSrc();
    auto offset = addPtrOp.getOffset();
    if (!getElementTypeOrSelf(offset.getType()).isInteger(32)) {
      return failure();
    }

    while (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
      offset = rewriter.create<arith::AddIOp>(
          loc, offset,
          rewriter.create<triton::SplatOp>(loc, offset.getType(),
                                           addPtrOp.getOffset()));
      ptr = addPtrOp.getPtr();
    }

    auto mask = op.getMask();
    auto other = op.getOther();

    rewriter.replaceOp(op, rewriter.create<triton::gcu::MaskedLoadOp>(
                               loc, ptr, offset, mask, other));
    return ::mlir::success();
  }
};

struct StoreOptimizationPatter : public OpRewritePattern<triton::StoreOp> {
  explicit StoreOptimizationPatter(MLIRContext *context)
      : OpRewritePattern<triton::StoreOp>(context) {}
  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto addPtrOp = op.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!addPtrOp) {
      return failure();
    }

    auto splatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
    if (!splatOp) {
      return failure();
    }

    auto offset = addPtrOp.getOffset();
    if (!getElementTypeOrSelf(offset.getType()).isInteger(32)) {
      return failure();
    }
    auto ptr = splatOp.getSrc();
    while (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
      offset = rewriter.create<arith::AddIOp>(
          loc, offset,
          rewriter.create<triton::SplatOp>(loc, offset.getType(),
                                           addPtrOp.getOffset()));
      ptr = addPtrOp.getPtr();
    }

    auto mask = op.getMask();
    auto value = op.getValue();

    rewriter.create<triton::gcu::MaskedStoreOp>(loc, ptr, offset, value, mask);
    rewriter.eraseOp(op);
    return ::mlir::success();
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

} // namespace

namespace {
struct GCUTritonFusionPass
    : public mlir::impl::GCUTritonFusionPassBase<GCUTritonFusionPass> {
  using Base::Base;

  void runOnOperation() override {
    dotZeroBiasFusion();
    // should do bias fusion before constant zero be fusioned
    auto module = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<ConvertClampFOp, ConvertPreciseDivFOp, ConvertPreciseSqrtOp,
                 ConvertFpToFpOp>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();

    auto hasNoSideEffects = [](Operation *op) {
      return !isa<arith::DivSIOp, arith::DivUIOp>(op);
    };

    DenseSet<Operation *> eraseOpSet;
    for (Operation *func : module.getOps<triton::FuncOp>()) {
      func->walk([&](Operation *op) {
        if (isElementwiseOp(op) && canVectorize(op)) {
          for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            auto def = op->getOperand(i).getDefiningOp();
            if (def && ((isa<arith::ConstantOp>(def) && hasNoSideEffects(op)) ||
                        isa<triton::SplatOp>(def))) {
              OpBuilder builder(op);
              eraseOpSet.insert(def);
              op->replaceUsesOfWith(op->getOperand(i),
                                    builder.clone(*def)->getResult(0));
            }
          }
        }
      });
    }

    for (auto op : eraseOpSet) {
      if (op->getUses().empty()) {
        op->erase();
      }
    }

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
  bool isElementwiseOp(Operation *op);
  bool canVectorize(Operation *op);
  void runFuse(Region &region);
  void fuseOps(ArrayRef<Operation *> ops);
  void dotZeroBiasFusion();
};
} // namespace

void GCUTritonFusionPass::runFuse(Region &region) {
  SmallVector<SmallVector<Operation *>> fusionOps;
  fusionOps.emplace_back();

  for (auto &block : region) {
    if (!fusionOps.back().empty()) {
      fusionOps.emplace_back();
    }
    for (auto &op : llvm::make_early_inc_range(block.getOperations())) {
      for (auto &region : op.getRegions()) {
        runFuse(region);
      }
      auto &fusionOp = fusionOps.back();
      if (isElementwiseOp(&op) && canVectorize(&op)) {
        if (fusionOp.empty()) {
          fusionOp.push_back(&op);
        } else {
          auto curType =
              isa<triton::gcu::MaskedStoreOp>(op)
                  ? cast<triton::gcu::MaskedStoreOp>(op).getValue().getType()
                  : cast<RankedTensorType>(op.getResultTypes().front());
          auto preType = isa<triton::gcu::MaskedStoreOp>(fusionOp.back())
                             ? cast<triton::gcu::MaskedStoreOp>(fusionOp.back())
                                   .getValue()
                                   .getType()
                             : cast<RankedTensorType>(
                                   fusionOp.back()->getResultTypes().front());
          if (curType.getShape() == preType.getShape() &&
              getElemsPerThread(curType) == getElemsPerThread(preType)) {
            fusionOp.push_back(&op);
          } else {
            fusionOps.emplace_back().push_back(&op);
          }
        }
      } else if (!fusionOp.empty()) {
        if (op.hasTrait<OpTrait::Elementwise>() &&
            llvm::all_of(fusionOp, [&](auto innerOp) {
              return llvm::none_of(innerOp->getUsers(),
                                   [&](auto user) { return user == &op; });
            })) {
          op.moveBefore(fusionOp.front());
        } else {
          fusionOps.emplace_back();
        }
      }
    }
  }

  for (auto fusionOp : fusionOps) {
    if (!fusionOp.empty())
      fuseOps(fusionOp);
  }
}

bool GCUTritonFusionPass::isElementwiseOp(Operation *op) {
  if (!op) {
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

  return OpTrait::hasElementwiseMappableTraits(op);
}

bool GCUTritonFusionPass::canVectorize(Operation *op) {
  if (!llvm::all_of(op->getResultTypes(), llvm::IsaPred<RankedTensorType>)) {
    return false;
  }
  return llvm::all_of(op->getResultTypes(), [](auto type) {
    auto elementTy = cast<RankedTensorType>(type).getElementType();
    return elementTy.isBF16() || elementTy.isF16() || elementTy.isF32() ||
           elementTy.isInteger(1) || elementTy.isInteger(8) ||
           elementTy.isInteger(16) || elementTy.isInteger(32);
  });
}

void GCUTritonFusionPass::fuseOps(ArrayRef<Operation *> ops) {
  DenseSet<Operation *> fusionOp(ops.begin(), ops.end());

  SetVector<Value> fusionOperands;
  SetVector<Value> fusionResults;
  for (auto op : ops) {
    for (auto v : op->getOperands()) {
      if (!fusionOp.count(v.getDefiningOp())) {
        fusionOperands.insert(v);
      }
    }
    for (auto result : op->getResults()) {
      for (auto user : result.getUsers()) {
        if (!fusionOp.count(user)) {
          fusionResults.insert(result);
        }
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

void GCUTritonFusionPass::dotZeroBiasFusion() {
  auto trionModule = getOperation();
  llvm::SmallVector<DotOp> dotList;
  trionModule.walk([&](DotOp dot) { dotList.push_back(dot); });
  for (auto &dot : dotList) {
    auto retType = dyn_cast<RankedTensorType>(dot.getType());
    int rank = retType.getShape().size();
    if (rank > 2) {
      // need test case for 3D dot
      continue;
    }
    // fusion dot and result cast
    OpBuilder rewriter(dot);
    auto dotUsers = dot.getResult().getUsers();
    auto userNumber = std::distance(dotUsers.begin(), dotUsers.end());
    bool isOnlyMatmul = false;
    auto CTensor = dot.getC().getDefiningOp();
    if (CTensor && isa<arith::ConstantOp>(CTensor)) {
      auto constantC = llvm::cast<arith::ConstantOp>(CTensor);
      if (auto splatAttr =
              llvm::dyn_cast<SplatElementsAttr>(constantC.getValue())) {
        mlir::Type elementType = splatAttr.getElementType();
        if (elementType.isInteger()) {
          if (splatAttr.getSplatValue<APInt>().isZero()) {
            isOnlyMatmul = true;
          }
        } else if (elementType.isBF16() || elementType.isF16() ||
                   elementType.isTF32() || elementType.isF32()) {
          if (splatAttr.getSplatValue<APFloat>().isZero()) {
            isOnlyMatmul = true;
          }
        }
      }
    }
    if (userNumber == 1 && isOnlyMatmul &&
        llvm::isa<arith::SIToFPOp>(*dotUsers.begin())) {
      auto castDotResult = llvm::cast<arith::SIToFPOp>(*dotUsers.begin());
      auto newDot = rewriter.create<mlir::triton::gcu::MatmulOp>(
          dot.getLoc(), castDotResult.getType(), dot.getA(), dot.getB());
      castDotResult.getResult().replaceAllUsesWith(newDot.getResult());
      castDotResult.erase();
      dot.erase();
      continue;
    }
    if (isOnlyMatmul) {
      auto newDot = rewriter.create<mlir::triton::gcu::MatmulOp>(
          dot.getLoc(), dot.getType(), dot.getA(), dot.getB());
      dot.getResult().replaceAllUsesWith(newDot.getResult());
      dot.erase();
      continue;
    }
    // maybe we can dot other fusion about dot in future as trion gpu' combine
  }
}
