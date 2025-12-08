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
#include <algorithm>
#include <optional>
#include <utility>

#include "Conversion/TritonToGCU/TritonToGCUPass.h"

#include "ConstantUtil.h"

#include "Analysis/MaskAnalysis.h"
#include "Analysis/OpFoldResultUtils.h"
#include "Analysis/PtrAnalysis.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// todo_AT gpu moduleop
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "triton-loadstore-to-dma"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTRITONLOADSTORETOGCUDMAPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

struct ConvertTritonLoadStoreToDmaPass
    : public mlir::impl::ConvertTritonLoadStoreToGCUDmaPassBase<
          ConvertTritonLoadStoreToDmaPass> {
  using Base::Base;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect,
                triton::TritonDialect, mlir::triton::gcu::TritonGCUDialect>();
  }
};

struct PreprocessForOp : public OpRewritePattern<scf::ForOp> {
  llvm::SmallDenseMap<Value, gcu::PtrState> &knownPtrs;
  llvm::SmallDenseMap<Value, gcu::MaskState> &knownMasks;
  llvm::SmallVector<Operation *, 8> &candidateOps;
  llvm::SmallDenseMap<Operation *, SmallVector<int32_t>> &candidateHints;

  explicit PreprocessForOp(
      MLIRContext *context,
      llvm::SmallDenseMap<Value, gcu::PtrState> &knownPtrs,
      llvm::SmallDenseMap<Value, gcu::MaskState> &knownMasks,
      llvm::SmallVector<Operation *, 8> &candidateOps,
      llvm::SmallDenseMap<Operation *, SmallVector<int32_t>> &candidateHints)
      : OpRewritePattern<scf::ForOp>(context), knownPtrs(knownPtrs),
        knownMasks(knownMasks), candidateOps(candidateOps),
        candidateHints(candidateHints) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (gcu::PtrAnalysis::byPassForOp(rewriter, op, candidateOps))
      return failure();
    return gcu::PtrAnalysis::rewriteForOp(rewriter, op, knownPtrs, knownMasks,
                                          candidateOps, candidateHints);
  }
};

struct PostprocessForOp : public OpRewritePattern<scf::ForOp> {
  llvm::SmallDenseMap<Value, gcu::PtrState> &knownPtrs;

  explicit PostprocessForOp(
      MLIRContext *context,
      llvm::SmallDenseMap<Value, gcu::PtrState> &knownPtrs)
      : OpRewritePattern<scf::ForOp>(context), knownPtrs(knownPtrs) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    gcu::PtrAnalysis::foldAwayForOp(rewriter, op, knownPtrs);
    return failure();
  }
};

bool IsStaticStride(SmallVector<int32_t> &candidateHints) {
  bool bStaticStride = true;
  int32_t rank = candidateHints.size();
  for (int32_t i = 0; i < rank; ++i) {
    if (candidateHints[i] == -1) {
      bStaticStride = false;
      break;
    }
  }

  return bStaticStride;
}

bool IsStaticReshape(SmallVector<int32_t> &candidateHints) {
  bool isReshape = true;
  int32_t rank = candidateHints.size();
  if (IsStaticStride(candidateHints)) {
    for (int32_t i = 0; i < rank; ++i) {
      if (candidateHints[i] == 1 || candidateHints[i] == 0) {
        isReshape = false;
        break;
      }
    }
  } else {
    isReshape = false;
  }

  return isReshape;
}

SmallVector<int32_t> GetOrderByHint(SmallVector<int32_t> &candidateHints) {
  SmallVector<int32_t> orderHint;
  int32_t rank = candidateHints.size();
  assert(IsStaticStride(candidateHints) &&
         "dynamic stride not support get static order");

  SmallVector<int32_t> broadcastDims;
  for (int32_t i = 0; i < rank; ++i)
    if (candidateHints[i] == 0)
      broadcastDims.push_back(i);

  for (int32_t i = 0; i < rank; ++i)
    if (candidateHints[i] != 0)
      orderHint.push_back(i);

  std::sort(orderHint.begin(), orderHint.end(), [&](int32_t a, int32_t b) {
    return (candidateHints[a] > candidateHints[b]);
  });

  if (orderHint.size() < static_cast<unsigned>(rank))
    for (auto dim : broadcastDims)
      orderHint.insert(orderHint.begin() + dim, dim);

  SmallVector<int32_t> transOrder(rank, 0);
  for (int32_t i = 0; i < rank; ++i)
    transOrder[orderHint[i]] = i;

  for (int32_t i = 0; i < rank; ++i)
    LLVM_DEBUG(llvm::dbgs() << "dim: " << i << "\n"
                            << "order: " << orderHint[i] << "\n");

  for (int32_t i = 0; i < rank; ++i)
    LLVM_DEBUG(llvm::dbgs() << "trans order: " << i << "\n"
                            << "order: " << transOrder[i] << "\n");

  return transOrder;
}

struct ConvertLoadOpToDma : public OpRewritePattern<triton::LoadOp> {
  llvm::SmallDenseMap<Value, gcu::PtrState> &knownPtrs;
  llvm::SmallDenseMap<Value, gcu::MaskState> &knownMasks;
  llvm::SmallVector<Operation *, 8> &candidateOps;
  llvm::SmallDenseMap<Operation *, SmallVector<int32_t>> &candidateHints;

  explicit ConvertLoadOpToDma(
      MLIRContext *context,
      llvm::SmallDenseMap<Value, gcu::PtrState> &knownPtrs,
      llvm::SmallDenseMap<Value, gcu::MaskState> &knownMasks,
      llvm::SmallVector<Operation *, 8> &candidateOps,
      llvm::SmallDenseMap<Operation *, SmallVector<int32_t>> &candidateHints)
      : OpRewritePattern<triton::LoadOp>(context), knownPtrs(knownPtrs),
        knownMasks(knownMasks), candidateOps(candidateOps),
        candidateHints(candidateHints) {}
  LogicalResult rewriteTensorLoad(triton::LoadOp op,
                                  PatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    // 2. Analyze the mask operand to determine at runtime the size of the data
    // we are moving.
    gcu::MaskState mstate;
    if (op.getMask()) {
      LLVM_DEBUG(llvm::dbgs() << "=== analyze load mask state ===\n");
      gcu::MaskAnalysis::parse(rewriter, loc, op.getMask(), mstate, knownMasks);
      assert(!mstate.isEmpty() &&
             "expect valid mask state after analysis succeed\n");
    }

    // 3. Get ptr info
    LLVM_DEBUG(llvm::dbgs() << "=== analyze load ptr state ===\n");
    gcu::PtrState pstate;
    gcu::PtrAnalysis::visitOperand(rewriter, loc, op.getPtr(), pstate,
                                   knownPtrs);

    // 4. Analyze the other operand to get a scalar value
    Value defaultValue;
    auto tType = dyn_cast<RankedTensorType>(op.getType());
    if (op.getOther()) {
      auto scalarValue = gcu::getScalarValue(rewriter, loc, op.getOther());
      assert(scalarValue.has_value() &&
             "other value used in masked load produced by "
             "unsupported instruction");
      defaultValue = scalarValue.value();
    } else {
      defaultValue =
          gcu::createConstantZero(rewriter, loc, tType.getElementType());
    }

    auto rank = pstate.getRank();
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto ptrInfo = pstate.getPtrInfo(rewriter, loc, mstate);
    bool bNeedBroadCast = false;
    auto resultShape = tType.getShape();
    auto elemType = tType.getElementType();
    for (auto dim : ptrInfo.broadcastDims) {
      if (resultShape[dim] > 1) {
        bNeedBroadCast = true;
        break;
      }
    }

    assert(candidateHints.find(op.getOperation()) != candidateHints.end() &&
           "get order failed");
    auto opHint = candidateHints[op.getOperation()];
    // dynamic stride process
    if (!IsStaticStride(opHint)) {
      LLVM_DEBUG(llvm::dbgs() << "=== dynamic stride process ===\n");
      SmallVector<Value> updateStrides;
      SmallVector<Value> updateShapes;
      SmallVector<int32_t> dynamicOpHint;
      for (int32_t i = 0; i < rank; ++i) {
        if (!ptrInfo.broadcastDims.count(i)) {
          updateStrides.push_back(ptrInfo.strides[i]);
          updateShapes.push_back(ptrInfo.shape[i]);
          dynamicOpHint.push_back(opHint[i]);
        } else {
          updateShapes.push_back(one);
          if (i == rank - 1) {
            dynamicOpHint.push_back(1);
            updateStrides.push_back(one);
          } else {
            dynamicOpHint.push_back(opHint[i]);
            updateStrides.push_back(ptrInfo.strides[i]);
          }
        }
      }

      if (bNeedBroadCast) {
        SmallVector<int64_t, 4> sliceShape(rank);
        for (unsigned int i = 0; i < rank; i++) {
          if (ptrInfo.broadcastDims.count(i))
            sliceShape[i] = 1;
          else
            sliceShape[i] = resultShape[i];
        }
        auto sliceType =
            RankedTensorType::get(sliceShape, elemType, tType.getEncoding());
        auto load = rewriter.create<mlir::triton::gcu::LoadOp>(
            loc, sliceType, ptrInfo.base, updateShapes, updateStrides,
            ptrInfo.offsets, defaultValue, dynamicOpHint);
        auto broadcastOp =
            rewriter.create<triton::BroadcastOp>(loc, op.getType(), load);
        rewriter.replaceOp(op, broadcastOp);
        return success();
      } else {
        auto load = rewriter.create<mlir::triton::gcu::LoadOp>(
            loc, tType, ptrInfo.base, updateShapes, updateStrides,
            ptrInfo.offsets, defaultValue, dynamicOpHint);
        rewriter.replaceOp(op, load);
        return success();
      }
    } else { // static stride process will be delete for pingpong support
             // dynamic
      LLVM_DEBUG(llvm::dbgs() << "=== static stride process ===\n");
      auto staticOrder = GetOrderByHint(opHint);
      // if (IsStaticReshape(opHint)) {
      //   for (int i = 0; i < rank; ++i)
      //     staticOrder[i]++;
      // }

      assert(static_cast<int64_t>(staticOrder.size()) == rank &&
             "the order size and rank mismatch \n");
      bool bNeedTranspose = false;
      for (uint32_t i = 0; i < rank; ++i) {
        if (staticOrder[i] != static_cast<int>(i)) {
          bNeedTranspose = true;
          break;
        }
      }

      SmallVector<int32_t> staticDefaultOrder;
      for (int i = 0; i < rank; ++i)
        staticDefaultOrder.push_back(i);

      if (IsStaticReshape(opHint)) {
        for (int i = 0; i < rank; ++i)
          staticDefaultOrder[i]++;
      }

      SmallVector<Value> orderStrides(rank);
      SmallVector<Value> orderShapes(rank);
      for (int i = 0; i < rank; ++i) {
        if (!bNeedTranspose) {
          orderStrides[i] = ptrInfo.strides[i];
          orderShapes[i] = ptrInfo.shape[i];
        } else {
          orderStrides[staticOrder[i]] = ptrInfo.strides[i];
          orderShapes[staticOrder[i]] = ptrInfo.shape[i];
        }
      }

      // update broadcast dim stride
      for (int i = rank - 1; i >= 0; --i) {
        if (ptrInfo.broadcastDims.count(i)) {
          if (i == rank - 1)
            orderStrides[i] = one;
          else
            orderStrides[i] = rewriter.create<arith::MulIOp>(
                loc, orderShapes[i + 1], orderStrides[i + 1]);
        }
      }

      SmallVector<int64_t, 4> updateResultShape(rank);
      for (int i = 0; i < rank; ++i) {
        if (ptrInfo.broadcastDims.count(i))
          updateResultShape[i] = 1;
        else
          updateResultShape[i] = resultShape[i];
      }

      if (bNeedTranspose) {
        SmallVector<int64_t, 4> sliceResultShape(rank);
        for (unsigned int i = 0; i < rank; i++)
          sliceResultShape[staticOrder[i]] = updateResultShape[i];

        auto warpsPerCTA = triton::gpu::getWarpsPerCTA(tType.getEncoding());
        auto order = triton::gpu::getOrder(tType);
        auto ctaLayout = triton::gpu::getCTALayout(tType.getEncoding());

        SmallVector<unsigned> sliceWarpsPerCTA(rank);
        SmallVector<unsigned> sliceOrder(rank);
        for (unsigned int i = 0; i < rank; ++i) {
          sliceWarpsPerCTA[staticOrder[i]] = warpsPerCTA[i];
          sliceOrder[order[i]] = staticOrder[i];
        }

        auto sliceOutType = RankedTensorType::get(
            sliceResultShape, elemType,
            triton::gpu::BlockedEncodingAttr::get(
                getContext(), SmallVector<unsigned>(rank, 1),
                SmallVector<unsigned>(rank, 1), sliceWarpsPerCTA, sliceOrder,
                ctaLayout));

        auto load = rewriter.create<mlir::triton::gcu::LoadOp>(
            loc, sliceOutType, ptrInfo.base, orderShapes, orderStrides,
            ptrInfo.offsets, defaultValue, staticDefaultOrder);
        if (bNeedBroadCast) {
          auto transType = RankedTensorType::get(updateResultShape, elemType,
                                                 tType.getEncoding());
          auto transpose = rewriter.create<triton::TransOp>(loc, transType,
                                                            load, staticOrder);
          auto broadcastOp = rewriter.create<triton::BroadcastOp>(
              loc, op.getType(), transpose);
          rewriter.replaceOp(op, broadcastOp);
          return success();
        } else {
          auto transpose = rewriter.create<triton::TransOp>(loc, op.getType(),
                                                            load, staticOrder);
          rewriter.replaceOp(op, transpose);
          return success();
        }
      } else if (bNeedBroadCast) {
        auto loadType = RankedTensorType::get(updateResultShape, elemType,
                                              tType.getEncoding());
        auto load = rewriter.create<mlir::triton::gcu::LoadOp>(
            loc, loadType, ptrInfo.base, orderShapes, orderStrides,
            ptrInfo.offsets, defaultValue, staticDefaultOrder);
        auto broadcastOp =
            rewriter.create<triton::BroadcastOp>(loc, op.getType(), load);
        rewriter.replaceOp(op, broadcastOp);
        return success();
      } else {
        auto load = rewriter.create<mlir::triton::gcu::LoadOp>(
            loc, tType, ptrInfo.base, orderShapes, orderStrides,
            ptrInfo.offsets, defaultValue, staticDefaultOrder);
        rewriter.replaceOp(op, load);
        return success();
      }
    }
  }

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Analyze the ptr operand to check whether it is continuous.
    LLVM_DEBUG(llvm::dbgs() << "=== check load ptr contiguous ===\n");
    if (std::find(candidateOps.begin(), candidateOps.end(),
                  op.getOperation()) == candidateOps.end()) {
      return failure();
    }
    return rewriteTensorLoad(op, rewriter);
  }
};

struct ConvertStoreOpToDma : public OpRewritePattern<triton::StoreOp> {
  llvm::SmallDenseMap<Value, gcu::PtrState> &knownPtrs;
  llvm::SmallDenseMap<Value, gcu::MaskState> &knownMasks;
  llvm::SmallVector<Operation *, 8> &candidateOps;
  llvm::SmallDenseMap<Operation *, SmallVector<int32_t>> &candidateHints;

  explicit ConvertStoreOpToDma(
      MLIRContext *context,
      llvm::SmallDenseMap<Value, gcu::PtrState> &knownPtrs,
      llvm::SmallDenseMap<Value, gcu::MaskState> &knownMasks,
      llvm::SmallVector<Operation *, 8> &candidateOps,
      llvm::SmallDenseMap<Operation *, SmallVector<int32_t>> &candidateHints)
      : OpRewritePattern<triton::StoreOp>(context), knownPtrs(knownPtrs),
        knownMasks(knownMasks), candidateOps(candidateOps),
        candidateHints(candidateHints) {}
  LogicalResult rewriteTensorStore(triton::StoreOp op,
                                   PatternRewriter &rewriter) const {
    // 2. Analyze the mask operand to determine at runtime the size of the data
    // we are moving.
    auto loc = op.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "=== analyze store mask state ===\n");
    gcu::MaskState mstate;
    if (op.getMask()) {
      gcu::MaskAnalysis::parse(rewriter, loc, op.getMask(), mstate, knownMasks);
      assert(!mstate.isEmpty() &&
             "expect valid mask state after analysis succeed\n");
    }

    // 3. Get ptr info
    LLVM_DEBUG(llvm::dbgs() << "=== analyze store ptr state ===\n");
    gcu::PtrState pstate;
    gcu::PtrAnalysis::visitOperand(rewriter, loc, op.getPtr(), pstate,
                                   knownPtrs);

    auto tType = dyn_cast<RankedTensorType>(op.getValue().getType());
    assert(tType && "the store value type is null\n");
    auto rank = pstate.getRank();
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto ptrInfo = pstate.getPtrInfo(rewriter, loc, mstate);
    auto storeShape = tType.getShape();
    auto elemType = tType.getElementType();

    SmallVector<Value> vSliceShape;
    for (unsigned int i = 0; i < rank; i++) {
      if (ptrInfo.broadcastDims.count(i))
        vSliceShape.push_back(one);
      else
        vSliceShape.push_back(ptrInfo.shape[i]);
    }
    assert(candidateHints.find(op.getOperation()) != candidateHints.end() &&
           "get order failed");
    auto opHint = candidateHints[op.getOperation()];
    // dynamic stride process
    if (!IsStaticStride(opHint)) {
      // update broadcast dim stride(move to tritontogcu pass)
      SmallVector<Value> updateStrides;
      SmallVector<int32_t> dynamicOpHint;
      for (int32_t i = 0; i < rank; ++i) {
        if (!ptrInfo.broadcastDims.count(i)) {
          updateStrides.push_back(ptrInfo.strides[i]);
          dynamicOpHint.push_back(opHint[i]);
        } else {
          if (i == rank - 1) {
            dynamicOpHint.push_back(1);
            updateStrides.push_back(one);
          } else {
            dynamicOpHint.push_back(opHint[i]);
            updateStrides.push_back(ptrInfo.strides[i]);
          }
        }
      }

      auto store = rewriter.create<mlir::triton::gcu::StoreOp>(
          loc, op.getValue(), ptrInfo.base, vSliceShape, updateStrides,
          ptrInfo.offsets, dynamicOpHint);
      rewriter.replaceOp(op, store);
      return success();
    } else { // static stride process will be delete for pingpong support
             // dynamic
      auto staticOrder = GetOrderByHint(opHint);
      // if (IsStaticReshape(opHint)) {
      //   for (int i = 0; i < rank; ++i)
      //     staticOrder[i]++;
      // }
      assert(static_cast<int64_t>(staticOrder.size()) == rank &&
             "the order size and rank mismatch \n");
      bool bNeedTranspose = false;
      for (uint32_t i = 0; i < rank; ++i) {
        if (staticOrder[i] != static_cast<int>(i)) {
          bNeedTranspose = true;
          break;
        }
      }

      SmallVector<int32_t> staticDefaultOrder;
      for (int i = 0; i < rank; ++i)
        staticDefaultOrder.push_back(i);

      if (IsStaticReshape(opHint)) {
        for (int i = 0; i < rank; ++i)
          staticDefaultOrder[i]++;
      }

      SmallVector<Value> orderStrides(rank);
      SmallVector<Value> orderShapes(rank);
      for (int i = 0; i < rank; ++i) {
        if (!bNeedTranspose) {
          orderStrides[i] = ptrInfo.strides[i];
          orderShapes[i] = vSliceShape[i];
        } else {
          orderStrides[staticOrder[i]] = ptrInfo.strides[i];
          orderShapes[staticOrder[i]] = vSliceShape[i];
        }
      }

      // update broadcast dim stride
      for (int i = rank - 1; i >= 0; --i) {
        if (ptrInfo.broadcastDims.count(i)) {
          if (i == rank - 1)
            orderStrides[i] = one;
          else
            orderStrides[i] = rewriter.create<arith::MulIOp>(
                loc, orderShapes[i + 1], orderStrides[i + 1]);
        }
      }

      if (bNeedTranspose) {
        SmallVector<int64_t, 4> transShapes(rank);
        for (unsigned int i = 0; i < rank; i++)
          transShapes[staticOrder[i]] = storeShape[i];

        auto warpsPerCTA = triton::gpu::getWarpsPerCTA(tType.getEncoding());
        auto order = triton::gpu::getOrder(tType);
        auto ctaLayout = triton::gpu::getCTALayout(tType.getEncoding());

        SmallVector<unsigned> transWarpsPerCTA(rank);
        SmallVector<unsigned> transOrder(rank);
        for (unsigned int i = 0; i < rank; ++i) {
          transWarpsPerCTA[staticOrder[i]] = warpsPerCTA[i];
          transOrder[order[i]] = staticOrder[i];
        }

        auto transType = RankedTensorType::get(
            transShapes, elemType,
            triton::gpu::BlockedEncodingAttr::get(
                getContext(), SmallVector<unsigned>(rank, 1),
                SmallVector<unsigned>(rank, 1), transWarpsPerCTA, transOrder,
                ctaLayout));

        auto transpose = rewriter.create<triton::TransOp>(
            loc, transType, op.getValue(), staticOrder);
        auto store = rewriter.create<mlir::triton::gcu::StoreOp>(
            loc, transpose, ptrInfo.base, orderShapes, orderStrides,
            ptrInfo.offsets, staticDefaultOrder);
        rewriter.replaceOp(op, store);
        return success();
      } else {
        auto store = rewriter.create<mlir::triton::gcu::StoreOp>(
            loc, op.getValue(), ptrInfo.base, orderShapes, orderStrides,
            ptrInfo.offsets, staticDefaultOrder);
        rewriter.replaceOp(op, store);
        return success();
      }
    }
  }

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Analyze the ptr operand to check whether it is continuous.
    LLVM_DEBUG(llvm::dbgs() << "=== check store ptr contiguous ===\n");
    if (std::find(candidateOps.begin(), candidateOps.end(),
                  op.getOperation()) == candidateOps.end()) {
      return failure();
    }
    return rewriteTensorStore(op, rewriter);
  }
};

} // namespace

void ConvertTritonLoadStoreToDmaPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "ConvertTritonLoadStoreToDmaPass\n");

  auto *ctx = &getContext();
  auto op = getOperation();

  // 1. collect load/store ops
  auto moduleOp = op->getParentOfType<ModuleOp>();
  llvm::SmallVector<Operation *, 8> candidateOps;
  llvm::SmallDenseMap<Operation *, SmallVector<int32_t>> candidateHints;
  gcu::PtrAnalysis::collectCandidateLoadStoreOps(moduleOp, candidateOps,
                                                 candidateHints);

  // 2. Pre-process some ops
  GreedyRewriteConfig rewriteConfig;
  rewriteConfig.strictMode = GreedyRewriteStrictness::ExistingOps;

  llvm::SmallDenseMap<Value, gcu::MaskState> knowMasks;
  llvm::SmallDenseMap<Value, gcu::PtrState> knownPtrs;
  RewritePatternSet prePatterns(ctx);
  prePatterns.add<PreprocessForOp>(ctx, knownPtrs, knowMasks, candidateOps,
                                   candidateHints);

  if (applyPatternsGreedily(op, std::move(prePatterns), rewriteConfig).failed())
    signalPassFailure();

  // 3. Start to process load/store op
  RewritePatternSet patterns(ctx);
  patterns.add<ConvertLoadOpToDma, ConvertStoreOpToDma>(
      ctx, knownPtrs, knowMasks, candidateOps, candidateHints);
  if (applyPatternsGreedily(op, std::move(patterns)).failed())
    signalPassFailure();

  // 4. Post-process some ops
  RewritePatternSet postPatterns(ctx);
  postPatterns.add<PostprocessForOp>(ctx, knownPtrs);
  if (applyPatternsGreedily(op, std::move(postPatterns), rewriteConfig)
          .failed())
    signalPassFailure();
}
