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
#include <utility>

#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Utility.h"

#include "Analysis/AxisInfoEx.h"
#include "Dialect/MathExt/IR/MathExt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
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

const char *const kIsContinual = "IsContinual";
constexpr unsigned kOaccSizeInBytes = 512;
constexpr unsigned kLoopUnrollTimes = 16;

static bool hasSideEffectBetween(Operation *from, Operation *to) {
  assert(from && to && from->getBlock() == to->getBlock() &&
         "from and to must be in the same block");
  return llvm::any_of(
      llvm::make_range(Block::iterator(from), Block::iterator(to)),
      [](auto &op) { return !isMemoryEffectFree(&op); });
}

static bool mightHaveWriteEffectBetween(Operation *from, Operation *to) {
  assert(from && to && from->getBlock() == to->getBlock() &&
         "from and to must be in the same block");
  return llvm::any_of(
      llvm::make_range(Block::iterator(from), Block::iterator(to)),
      [](auto &op) { return mightHaveEffect<MemoryEffects::Write>(&op); });
}

struct EliminateForOpInductionVars : public OpRewritePattern<scf::ForOp> {
  explicit EliminateForOpInductionVars(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}
  mlir::LogicalResult
  matchAndRewrite(scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto eliminableInductionVarInfos = collectEliminableIndVarInfos(op);
    if (eliminableInductionVarInfos.empty()) {
      return failure();
    }

    bool needExternalReplacement =
        llvm::any_of(eliminableInductionVarInfos.keys(), [&](auto iter) {
          return !op.getResult(iter).use_empty();
        });

    // We only require explicit step positivity check when materializing
    // external replacements. For no-use eliminated iter args, we rely on
    // scf.for legality invariants guaranteed by upstream pipeline.
    if (needExternalReplacement &&
        failed(verifyPositiveStepForExternalReplacement(op, rewriter))) {
      return failure();
    }

    auto canonicalizedResult =
        createCanonicalizedForOp(rewriter, op, eliminableInductionVarInfos);
    for (unsigned i = 0, j = 0; i < op.getNumResults(); i++) {
      if (eliminableInductionVarInfos.contains(i)) {
        if (op.getResult(i).use_empty()) {
          continue;
        }
        auto it = canonicalizedResult.eliminatedResultReplacements.find(i);
        if (it == canonicalizedResult.eliminatedResultReplacements.end()) {
          return rewriter.notifyMatchFailure(
              op, "missing replacement for eliminated result");
        }
        rewriter.replaceAllUsesWith(op.getResult(i), it->second);
      } else {
        rewriter.replaceAllUsesWith(op.getResult(i),
                                    canonicalizedResult.forOp.getResult(j++));
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  struct EliminableInductionVarInfo {
    enum class UpdateKind : uint8_t { Add, Sub } kind;
    Value stride;
  };

  struct CanonicalizedForOpResult {
    scf::ForOp forOp;
    DenseMap<unsigned, Value> eliminatedResultReplacements;
  };

  static LogicalResult
  verifyPositiveStepForExternalReplacement(scf::ForOp op,
                                           PatternRewriter &rewriter);

  static std::optional<EliminableInductionVarInfo>
  tryGetEliminableIndVarInfo(scf::ForOp forOp, unsigned iter);
  static llvm::DenseMap<unsigned, EliminableInductionVarInfo>
  collectEliminableIndVarInfos(scf::ForOp forOp);
  static Value materializeIndVar(OpBuilder &builder, Location loc,
                                 const EliminableInductionVarInfo &info,
                                 Value init, Value tripCount);
  static CanonicalizedForOpResult createCanonicalizedForOp(
      OpBuilder &builder, scf::ForOp op,
      const llvm::DenseMap<unsigned, EliminableInductionVarInfo>
          &eliminableInductionVarInfos);
};

std::optional<EliminateForOpInductionVars::EliminableInductionVarInfo>
EliminateForOpInductionVars::tryGetEliminableIndVarInfo(scf::ForOp forOp,
                                                        unsigned iter) {
  auto arg = forOp.getRegionIterArg(iter);
  if (!isa<TensorType>(arg.getType())) {
    return std::nullopt;
  }
  Value stride;
  EliminableInductionVarInfo::UpdateKind updateKind;
  Value updateValue = forOp.getBody()->getTerminator()->getOperand(iter);
  if (auto addOp = updateValue.getDefiningOp<arith::AddIOp>()) {
    if (addOp.getLhs() == arg) {
      stride = addOp.getRhs();
    } else if (addOp.getRhs() == arg) {
      stride = addOp.getLhs();
    } else {
      return std::nullopt;
    }
    updateKind = EliminableInductionVarInfo::UpdateKind::Add;
  } else if (auto subOp = updateValue.getDefiningOp<arith::SubIOp>()) {
    if (subOp.getLhs() != arg) {
      return std::nullopt;
    }
    stride = subOp.getRhs();
    updateKind = EliminableInductionVarInfo::UpdateKind::Sub;
  } else {
    return std::nullopt;
  }
  if (!forOp.isDefinedOutsideOfLoop(stride)) {
    return std::nullopt;
  }
  return EliminableInductionVarInfo{updateKind, stride};
}

llvm::DenseMap<unsigned,
               EliminateForOpInductionVars::EliminableInductionVarInfo>
EliminateForOpInductionVars::collectEliminableIndVarInfos(scf::ForOp forOp) {
  llvm::DenseMap<unsigned, EliminableInductionVarInfo> infos;
  for (unsigned iter = 0; iter < forOp.getNumRegionIterArgs(); ++iter) {
    if (auto info = tryGetEliminableIndVarInfo(forOp, iter)) {
      infos.insert({iter, *info});
    }
  }
  return infos;
}

LogicalResult
EliminateForOpInductionVars::verifyPositiveStepForExternalReplacement(
    scf::ForOp op, PatternRewriter &rewriter) {
  auto cstStep = op.getStep().getDefiningOp<arith::ConstantOp>();
  if (!cstStep) {
    return rewriter.notifyMatchFailure(
        op, "external replacement requires constant positive step");
  }
  auto intAttr = dyn_cast<IntegerAttr>(cstStep.getValue());
  if (!intAttr || !intAttr.getValue().isStrictlyPositive()) {
    return rewriter.notifyMatchFailure(
        op, "external replacement requires positive step");
  }
  return success();
}

Value EliminateForOpInductionVars::materializeIndVar(
    OpBuilder &builder, Location loc, const EliminableInductionVarInfo &info,
    Value init, Value tripCount) {
  auto splatTrip =
      builder.create<triton::SplatOp>(loc, init.getType(), tripCount);
  auto scaled = builder.create<arith::MulIOp>(loc, info.stride, splatTrip);
  return info.kind == EliminableInductionVarInfo::UpdateKind::Add
             ? builder.create<arith::AddIOp>(loc, init, scaled).getResult()
             : builder.create<arith::SubIOp>(loc, init, scaled).getResult();
}

EliminateForOpInductionVars::CanonicalizedForOpResult
EliminateForOpInductionVars::createCanonicalizedForOp(
    OpBuilder &builder, scf::ForOp op,
    const llvm::DenseMap<unsigned, EliminableInductionVarInfo>
        &eliminableInductionVarInfos) {
  SmallVector<Value> initArgs;
  auto yieldOp = op.getBody()->getTerminator();
  for (auto [iter, arg] : llvm::enumerate(op.getInitArgs())) {
    if (!eliminableInductionVarInfos.contains(iter)) {
      initArgs.push_back(arg);
    }
  }

  auto forOp = builder.create<scf::ForOp>(
      op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
      initArgs,
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
        IRMapping mapper;
        mapper.map(op.getInductionVar(), iv);
        for (unsigned i = 0, j = 0; i < op.getNumRegionIterArgs(); i++) {
          if (!eliminableInductionVarInfos.contains(i)) {
            mapper.map(op.getRegionIterArg(i), args[j++]);
          }
        }
        auto ivMinusLowerBound =
            builder.create<arith::SubIOp>(loc, iv, op.getLowerBound());
        auto tripCount = builder.create<arith::DivSIOp>(loc, ivMinusLowerBound,
                                                        op.getStep());
        DenseSet<Operation *> bypassedOps;
        for (auto [iter, info] : eliminableInductionVarInfos) {
          mapper.map(op.getRegionIterArg(iter), op.getInitArgs()[iter]);
          auto updateOp = yieldOp->getOperand(iter).getDefiningOp();
          if (updateOp->hasOneUse()) {
            bypassedOps.insert(updateOp);
          }
          // Replace iterative update `arg = arg +/- stride` with closed-form:
          //   init +/- stride * ((iv - lb) / step)
          mapper.map(op.getRegionIterArg(iter),
                     materializeIndVar(builder, loc, info,
                                       mapper.lookup(op.getRegionIterArg(iter)),
                                       tripCount));
        }
        for (auto &o : op.getBody()->without_terminator()) {
          if (bypassedOps.contains(&o)) {
            continue;
          }
          Operation *cloneOp = builder.clone(o, mapper);
          for (auto [result, newResult] :
               llvm::zip_equal(o.getResults(), cloneOp->getResults())) {
            mapper.map(result, newResult);
          }
        }
        SmallVector<Value> results;
        for (unsigned i = 0; i < yieldOp->getNumOperands(); i++) {
          if (!eliminableInductionVarInfos.contains(i)) {
            results.push_back(mapper.lookup(yieldOp->getOperand(i)));
          }
        }
        builder.create<scf::YieldOp>(loc, results);
      });

  if (llvm::all_of(eliminableInductionVarInfos.keys(),
                   [&](auto iter) { return op.getResult(iter).use_empty(); })) {
    return CanonicalizedForOpResult{forOp, DenseMap<unsigned, Value>{}};
  }
  auto loc = op.getLoc();
  DenseMap<unsigned, Value> eliminatedResultReplacements;
  auto span = builder.create<arith::SubIOp>(loc, op.getUpperBound(),
                                            op.getLowerBound());
  auto zero = builder.create<arith::ConstantOp>(
      loc, span.getType(), builder.getIntegerAttr(span.getType(), 0));
  auto tripCount = builder.create<arith::CeilDivSIOp>(
      loc, builder.create<arith::MaxSIOp>(loc, span, zero), op.getStep());
  for (auto [iter, info] : eliminableInductionVarInfos) {
    if (op.getResult(iter).use_empty()) {
      continue;
    }
    eliminatedResultReplacements[iter] = materializeIndVar(
        builder, loc, info, op.getInitArgs()[iter], tripCount);
  }
  return CanonicalizedForOpResult{forOp, eliminatedResultReplacements};
}

struct PtrDecomposeResult {
  Value basePtr;
  Value offset;
};

static FailureOr<PtrDecomposeResult>
tryDecomposeLoadStorePtr(OpBuilder &builder, Location loc, Value ptr) {
  auto bitcastOp = ptr.getDefiningOp<triton::BitcastOp>();
  if (bitcastOp) {
    ptr = bitcastOp.getSrc();
  }
  auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp) {
    return failure();
  }
  auto rankedTensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!rankedTensorTy || rankedTensorTy.getRank() != 1) {
    return failure();
  }
  if (cast<triton::PointerType>(rankedTensorTy.getElementType())
          .getPointeeType()
          .isInteger(64)) {
    return failure();
  }
  auto offset = addPtrOp.getOffset();
  auto basePtr = addPtrOp.getPtr();
  // Triton AddPtr offsets here are expected to be signless integer
  // tensors/scalars.
  auto accumulateOffset = [&](Value baseOffset, Value offsetToAdd,
                              bool needSplat = false) -> Value {
    if (needSplat) {
      offsetToAdd = builder.create<triton::SplatOp>(
          loc,
          cast<RankedTensorType>(baseOffset.getType())
              .clone(offsetToAdd.getType()),
          offsetToAdd);
    }
    auto promotedType =
        mlir::triton::gcu::getBpe(getElementTypeOrSelf(baseOffset.getType())) >
                mlir::triton::gcu::getBpe(
                    getElementTypeOrSelf(offsetToAdd.getType()))
            ? baseOffset.getType()
            : offsetToAdd.getType();
    if (promotedType != baseOffset.getType()) {
      baseOffset =
          builder.create<arith::ExtSIOp>(loc, promotedType, baseOffset);
    }
    if (promotedType != offsetToAdd.getType()) {
      offsetToAdd =
          builder.create<arith::ExtSIOp>(loc, promotedType, offsetToAdd);
    }
    return builder.create<arith::AddIOp>(loc, baseOffset, offsetToAdd);
  };
  while (auto addPtrOp = basePtr.getDefiningOp<triton::AddPtrOp>()) {
    basePtr = addPtrOp.getPtr();
    offset = accumulateOffset(offset, addPtrOp.getOffset());
  }
  if (auto splatOp = basePtr.getDefiningOp<triton::SplatOp>()) {
    basePtr = splatOp.getSrc();
    while (auto addPtrOp = basePtr.getDefiningOp<triton::AddPtrOp>()) {
      basePtr = addPtrOp.getPtr();
      offset = accumulateOffset(offset, addPtrOp.getOffset(), true);
    }
    if (bitcastOp) {
      auto pointeeType =
          cast<triton::PointerType>(
              cast<RankedTensorType>(bitcastOp.getType()).getElementType())
              .getPointeeType();
      auto origPtrType = cast<triton::PointerType>(basePtr.getType());
      auto newPtrType =
          triton::PointerType::get(pointeeType, origPtrType.getAddressSpace());
      basePtr = builder.create<triton::BitcastOp>(loc, newPtrType, basePtr);
    }
    return PtrDecomposeResult{basePtr, offset};
  }
  return failure();
}

struct ConvertTritonLoadOp : public OpRewritePattern<triton::LoadOp> {
  explicit ConvertTritonLoadOp(MLIRContext *context,
                               ModuleAxisInfoExAnalysis &analysis)
      : OpRewritePattern<triton::LoadOp>(context), analysis(analysis) {}
  ModuleAxisInfoExAnalysis &analysis;
  mlir::LogicalResult
  matchAndRewrite(triton::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto result = tryDecomposeLoadStorePtr(rewriter, loc, op.getPtr());
    if (succeeded(result)) {
      auto [basePtr, offset] = *result;
      auto mask = op.getMask();
      auto other = op.getOther();
      auto axisInfoEx = analysis.getAxisInfoEx(op.getPtr());
      auto isContinual = axisInfoEx && axisInfoEx->getContinualInterval(0) == 1;
      auto maskedLoadOp = rewriter.create<triton::gcu::MaskedLoadOp>(
          loc, basePtr, offset, mask, other);
      if (isContinual) {
        maskedLoadOp->setAttr(kIsContinual, rewriter.getBoolAttr(isContinual));
      }
      rewriter.replaceOp(op, maskedLoadOp);
      return success();
    }
    return failure();
  }
};

struct ConvertTritonStoreOp : public OpRewritePattern<triton::StoreOp> {
  explicit ConvertTritonStoreOp(MLIRContext *context,
                                ModuleAxisInfoExAnalysis &analysis)
      : OpRewritePattern<triton::StoreOp>(context), analysis(analysis) {}
  ModuleAxisInfoExAnalysis &analysis;
  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto result = tryDecomposeLoadStorePtr(rewriter, loc, op.getPtr());
    if (succeeded(result)) {
      auto [basePtr, offset] = *result;
      auto mask = op.getMask();
      auto value = op.getValue();
      auto axisInfoEx = analysis.getAxisInfoEx(op.getPtr());
      auto isContinual = axisInfoEx && axisInfoEx->getContinualInterval(0) == 1;
      auto maskedStoreOp = rewriter.create<triton::gcu::MaskedStoreOp>(
          loc, basePtr, offset, value, mask);
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
    Operation *newOp = nullptr;
    if (stringifyPropagateNan(op.getPropagateNan()) == "all") {
      newOp = rewriter.create<arith::MaximumFOp>(
          loc, op.getMin(),
          rewriter.create<arith::MinimumFOp>(loc, op.getX(), op.getMax()));
    } else {
      newOp = rewriter.create<arith::MaxNumFOp>(
          loc, op.getMin(),
          rewriter.create<arith::MinNumFOp>(loc, op.getX(), op.getMax()));
    }
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertFpToFpOp : public OpRewritePattern<triton::FpToFpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FpToFpOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Operation *newOp = nullptr;
    auto srcType = getElementTypeOrSelf(op.getSrc().getType());
    auto resType = getElementTypeOrSelf(op.getResult().getType());

    if (resType.getIntOrFloatBitWidth() > srcType.getIntOrFloatBitWidth()) {
      // Cast from floating-point to wider floating-point, fp8->fp32
      newOp = rewriter.create<arith::ExtFOp>(loc, op.getType(), op.getSrc());
    } else {
      // Cast from floating-point to narrower floating-point, fp32->fp8
      newOp = rewriter.create<arith::TruncFOp>(loc, op.getType(), op.getSrc());
    }
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

template <typename FT, typename TT>
struct ConvertTritonOp : public OpRewritePattern<FT> {
  using OpRewritePattern<FT>::OpRewritePattern;

  LogicalResult matchAndRewrite(FT op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto newOp = rewriter.create<TT>(loc, op->getOperands());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// restore arith.constant(#ttg.dot_op) to constant(#parent) + ttg.cvt(parent)
struct RestoreDotOperandSplatConstantPattern
    : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = dyn_cast<RankedTensorType>(op.getType());
    if (!tensorType)
      return failure();
    auto dotEncoding = llvm::dyn_cast<triton::gpu::DotOperandEncodingAttr>(
        tensorType.getEncoding());
    if (!dotEncoding)
      return failure();
    auto valueAttr = op.getValue();
    auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr);
    if (!denseAttr || !denseAttr.isSplat())
      return failure();
    Attribute parentEncoding = dotEncoding.getParent();
    if (!parentEncoding)
      return failure();
    TypedAttr splatTyped = denseAttr.getSplatValue<TypedAttr>();
    auto parentType = tensorType.cloneWithEncoding(parentEncoding);
    auto newAttr =
        DenseElementsAttr::get(cast<ShapedType>(parentType), splatTyped);
    auto loc = op.getLoc();
    auto newConstOp = rewriter.create<arith::ConstantOp>(loc, newAttr);
    auto convertOp = rewriter.create<triton::gpu::ConvertLayoutOp>(
        loc, tensorType, newConstOp.getResult());
    rewriter.replaceOp(op, convertOp.getResult());

    return success();
  }
};

static bool isSupportedElementType(Type elementType) {
  return isa<Float8E5M2Type>(elementType) ||
         isa<Float8E4M3FNType>(elementType) || elementType.isBF16() ||
         elementType.isF16() || elementType.isF32() ||
         elementType.isInteger(1) || elementType.isInteger(8) ||
         elementType.isInteger(16) || elementType.isInteger(32) ||
         elementType.isInteger(64);
}

static bool canFuseReshapeOp(triton::ReshapeOp op) {
  return !triton::gcu::isExpensiveView(op.getSrc().getType(), op.getType());
}

static bool canFuseBroadcast(triton::BroadcastOp op) {
  auto srcType = op.getSrc().getType();
  auto resultType = op.getType();
  auto rank = srcType.getRank();

  // TODO(peng.tian): support i1 broadcast
  if (getElementTypeOrSelf(srcType).isInteger(1)) {
    return false;
  }

  std::optional<unsigned> broadcastAxis;
  for (unsigned i = 0; i < rank; ++i) {
    if (srcType.getDimSize(i) != resultType.getDimSize(i)) {
      if (broadcastAxis) {
        return false;
      }
      broadcastAxis = i;
    }
  }
  if (!broadcastAxis || (*broadcastAxis != 0 && *broadcastAxis != rank - 1)) {
    return false;
  }

  auto elemsPerThread = broadcastAxis == 0
                            ? triton::gcu::getElemsPerThread(srcType)
                            : triton::gcu::getElemsPerThread(resultType);
  auto sizeInBytes =
      std::accumulate(elemsPerThread.begin() + *broadcastAxis,
                      elemsPerThread.end(), 1, std::multiplies<int64_t>()) *
      triton::gcu::getBpe(getElementTypeOrSelf(srcType));
  auto numOacc = sizeInBytes / kOaccSizeInBytes;
  // TODO(peng.tian): need to support general implementation
  return numOacc >= 1 && numOacc <= 4 * kLoopUnrollTimes;
}

static bool canFuseConvertLayout(triton::gpu::ConvertLayoutOp op) {
  auto srcTy = op.getSrc().getType();
  auto dstTy = op.getType();
  if (triton::gcu::getElemsPerThread(srcTy) !=
      triton::gcu::getElemsPerThread(dstTy)) {
    return false;
  }
  auto srcEnc = srcTy.getEncoding();
  auto dstEnc = dstTy.getEncoding();
  if (isa<triton::gpu::BlockedEncodingAttr>(srcEnc) &&
      isa<triton::gpu::BlockedEncodingAttr>(dstEnc)) {
    return true;
  }

  if ((isa<triton::gpu::BlockedEncodingAttr>(srcEnc) &&
       isa<triton::gpu::SliceEncodingAttr>(dstEnc)) ||
      (isa<triton::gpu::SliceEncodingAttr>(srcEnc) &&
       isa<triton::gpu::BlockedEncodingAttr>(dstEnc))) {
    return triton::gcu::getWarpsPerCTA(srcEnc) ==
           triton::gcu::getWarpsPerCTA(dstEnc);
  }

  auto dstSliceEnc = dyn_cast<triton::gpu::SliceEncodingAttr>(dstEnc);
  auto srcSliceEnc = dyn_cast<triton::gpu::SliceEncodingAttr>(srcEnc);
  if (dstSliceEnc && srcSliceEnc) {
    return dstSliceEnc.getDim() == srcSliceEnc.getDim() &&
           triton::gcu::getWarpsPerCTA(srcEnc) ==
               triton::gcu::getWarpsPerCTA(dstEnc);
  }
  return false;
}

static bool canFuse(Operation *op) {
  if (!op || !llvm::all_of(op->getResultTypes(), [](auto type) {
        auto tensorType = dyn_cast<RankedTensorType>(type);
        return tensorType &&
               isSupportedElementType(tensorType.getElementType());
      })) {
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
    return canFuseBroadcast(broadcastOp);
  }
  if (auto cvtLayoutOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
    return canFuseConvertLayout(cvtLayoutOp);
  }
  if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(op)) {
    return canFuseReshapeOp(reshapeOp);
  }
  return OpTrait::hasElementwiseMappableTraits(op);
}

struct MaskedLoadStoreFusionContext {
  MaskedLoadStoreFusionContext(Operation *userOp, DominanceInfo &dominanceInfo)
      : userOp(userOp), dominanceInfo(dominanceInfo) {}
  void collectDepsFromOperands(ArrayRef<Value> depsToCollect) {
    for (auto value : depsToCollect) {
      collectDepsFromOperand(value);
    }
  }

  void collectDepsFromOperand(Value value) {
    if (!value)
      return;
    auto defOp = value.getDefiningOp();
    if (!defOp || !visitedOps.insert(defOp).second) {
      return;
    }
    if (!canFuse(defOp) || !isMemoryEffectFree(defOp)) {
      return;
    }
    if (!llvm::all_of(defOp->getOperands(), [&](auto operand) {
          return dominanceInfo.dominates(operand, userOp);
        })) {
      return;
    }
    for (auto operand : defOp->getOperands()) {
      collectDepsFromOperand(operand);
    }
    orderedDeps.push_back(defOp);
  }

  void sinkDepsToUserOp() {
    OpBuilder builder(userOp);
    for (auto op : orderedDeps) {
      auto cloneOp = builder.clone(*op, mapping);
      for (unsigned i = 0, numResults = op->getNumResults(); i != numResults;
           ++i) {
        mapping.map(op->getResult(i), cloneOp->getResult(i));
      }
    }
  }

  void rewriteOperands() {
    for (auto &operand : userOp->getOpOperands()) {
      if (auto mappingValue = mapping.lookupOrNull(operand.get())) {
        operand.set(mappingValue);
      }
    }
  }

  void eraseDeadDeps() {
    for (auto *op : llvm::reverse(orderedDeps)) {
      if (op->use_empty())
        op->erase();
    }
  }

  Operation *userOp;
  DominanceInfo &dominanceInfo;
  llvm::DenseSet<Operation *> visitedOps;
  SmallVector<Operation *> orderedDeps;
  IRMapping mapping;
};

static Operation *findEarliestSafeMovePoint(Value value,
                                            DominanceInfo &dominanceInfo) {
  auto *defOp = value.getDefiningOp();
  Operation *movePoint = nullptr;
  for (auto user : value.getUsers()) {
    if (user->getBlock() == value.getParentBlock() &&
        (!movePoint || user->isBeforeInBlock(movePoint))) {
      movePoint = user;
    }
  }
  if (movePoint) {
    for (auto user : value.getUsers()) {
      if (user->getBlock() != value.getParentBlock() &&
          !dominanceInfo.dominates(movePoint, user)) {
        movePoint = nullptr;
        break;
      }
    }
  }

  if (movePoint &&
      mightHaveWriteEffectBetween(defOp->getNextNode(), movePoint)) {
    return nullptr;
  }
  return movePoint;
}

static void gatherOpDepsForFusion(Operation *op, ArrayRef<Value> depsToCollect,
                                  DominanceInfo &dominanceInfo) {
  MaskedLoadStoreFusionContext ctx(op, dominanceInfo);
  ctx.collectDepsFromOperands(depsToCollect);
  ctx.sinkDepsToUserOp();
  ctx.rewriteOperands();
  ctx.eraseDeadDeps();
}

// Preprocess masked memory ops for subsequent fusion:
// 1) Move MaskedLoadOp to the earliest safe user point when possible.
// 2) Materialize fusible dependency chains (offset/mask/other) near each
//    MaskedLoadOp/MaskedStoreOp, improving producer-user locality.
// This increases the chance that masked memory ops and their elementwise
// dependencies are fused together later. Any temporary redundancy is left to
// standard cleanup passes (e.g. CSE/DCE/canonicalization).
static void preProcessMaskedLoadStore(triton::FuncOp func,
                                      DominanceInfo &dominanceInfo) {
  SmallVector<Operation *> worklist;
  func.walk([&](Operation *op) {
    if (isa<triton::gcu::MaskedLoadOp, triton::gcu::MaskedStoreOp>(op)) {
      worklist.push_back(op);
    }
  });

  for (auto op : worklist) {
    if (auto maskedLoadOp = dyn_cast<triton::gcu::MaskedLoadOp>(op)) {
      auto movePoint =
          findEarliestSafeMovePoint(maskedLoadOp.getResult(), dominanceInfo);
      if (movePoint && op->getNextNode() != movePoint) {
        op->moveBefore(movePoint);
      }
      gatherOpDepsForFusion(op,
                            {maskedLoadOp.getOffset(), maskedLoadOp.getMask(),
                             maskedLoadOp.getOther()},
                            dominanceInfo);
    } else if (auto maskedStoreOp = dyn_cast<triton::gcu::MaskedStoreOp>(op)) {
      gatherOpDepsForFusion(
          op, {maskedStoreOp.getOffset(), maskedStoreOp.getMask()},
          dominanceInfo);
    }
  }
}

static void sinkFusibleProducersToUses(triton::FuncOp func,
                                       DominanceInfo &dominanceInfo) {
  func.walk([&](Operation *op) {
    if (!canFuse(op)) {
      return;
    }
    auto operands = llvm::to_vector(op->getOperands());
    for (auto operand : operands) {
      auto defOp = operand.getDefiningOp();
      if (!isa_and_nonnull<arith::ConstantOp, triton::SplatOp,
                           triton::MakeRangeOp, triton::BroadcastOp>(defOp)) {
        continue;
      }
      if (isa<triton::BroadcastOp>(defOp) &&
          !canFuseBroadcast(cast<triton::BroadcastOp>(defOp))) {
        continue;
      }
      if (defOp->isAncestor(op)) {
        continue;
      }
      if (llvm::all_of(defOp->getOperands(), [&](auto operand) {
            return dominanceInfo.dominates(operand, op);
          })) {
        if (operand.hasOneUse()) {
          defOp->moveBefore(op);
        } else {
          OpBuilder builder(op);
          op->replaceUsesOfWith(operand, builder.clone(*defOp)->getResult(0));
        }
      }
    }
  });
}

} // namespace

namespace {

struct FusionTensorDesc {
  SmallVector<int64_t> shape;
  SmallVector<unsigned> elemsPerThread;
  bool operator==(const FusionTensorDesc &other) const {
    return shape == other.shape && elemsPerThread == other.elemsPerThread;
  }
  bool operator!=(const FusionTensorDesc &other) const {
    return !(*this == other);
  }
};

static FusionTensorDesc getFusionTensorDesc(RankedTensorType type) {
  FusionTensorDesc desc;
  desc.shape = llvm::to_vector(type.getShape());
  desc.elemsPerThread = getElemsPerThread(type);
  return desc;
}

struct FusionGroup {
  bool empty() const { return ops.empty(); }
  bool isUsedBy(Operation *op) const {
    return llvm::any_of(ops, [op](auto innerOp) {
      return llvm::any_of(innerOp->getUsers(),
                          [op](auto user) { return user == op; });
    });
  }
  bool tryInsert(Operation *op) {
    auto tensorType =
        isa<triton::gcu::MaskedStoreOp>(op)
            ? cast<triton::gcu::MaskedStoreOp>(op).getValue().getType()
            : cast<RankedTensorType>(op->getResultTypes().front());
    auto tensorDesc = getFusionTensorDesc(tensorType);
    if (ops.empty()) {
      ops.push_back(op);
      tensorDescs.emplace_back(tensorDesc);
      return true;
    }
    if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(op)) {
      if (!llvm::is_contained(
              tensorDescs, getFusionTensorDesc(reshapeOp.getSrc().getType()))) {
        return false;
      }
      if (!llvm::is_contained(tensorDescs, tensorDesc)) {
        tensorDescs.emplace_back(tensorDesc);
      }
      ops.push_back(op);
      return true;
    }
    if (!llvm::is_contained(tensorDescs, tensorDesc)) {
      return false;
    }
    ops.push_back(op);
    return true;
  }
  Operation *front() const { return ops.front(); }
  Operation *back() const { return ops.back(); }
  bool hasSideEffect() const {
    return llvm::any_of(ops, [](auto op) { return !isMemoryEffectFree(op); });
  }
  void resolveOperandsAndResults() {
    DenseSet<Operation *> opSet(ops.begin(), ops.end());
    operands.clear();
    results.clear();
    for (auto op : ops) {
      for (auto operand : op->getOperands()) {
        auto defOp = operand.getDefiningOp();
        if (!opSet.contains(defOp)) {
          operands.insert(operand);
        }
      }
      for (auto result : op->getResults()) {
        if (llvm::any_of(result.getUsers(),
                         [&](auto user) { return !opSet.contains(user); })) {
          results.insert(result);
        }
      }
    }
  }
  Operation *findFirstUserInBlock() const {
    assert(!ops.empty() && "cannot find user of an empty group");
    auto block = ops.front()->getBlock();
    auto region = ops.front()->getParentRegion();
    Operation *firstUser = nullptr;
    for (auto result : results) {
      for (auto user : result.getUsers()) {
        // if the user is inside a nested region (e.g., body of
        // scf.for/scf.if), walk up the parent chain until we find an
        // ancestor op in the same block, or reach the same region but a
        // different block (CFG control flow)
        while (user && user->getBlock() != block &&
               user->getParentRegion() != region) {
          user = user->getParentOp();
        }
        if (!user || user->getBlock() != block) {
          continue;
        }
        if (back()->isBeforeInBlock(user) &&
            (!firstUser || user->isBeforeInBlock(firstUser))) {
          firstUser = user;
        }
      }
    }
    return firstUser;
  }
  bool contains(Operation *op) const {
    return op == ops.front() || op == ops.back() ||
           (op->isBeforeInBlock(ops.back()) &&
            ops.front()->isBeforeInBlock(op));
  }
  void mergeIntoFrontOf(FusionGroup &other) {
    for (auto op : ops) {
      op->moveBefore(other.front());
    }
    for (auto tensorDesc : tensorDescs) {
      if (!llvm::is_contained(other.tensorDescs, tensorDesc)) {
        other.tensorDescs.push_back(tensorDesc);
      }
    }
    other.ops.insert(other.ops.begin(), ops.begin(), ops.end());
    other.resolveOperandsAndResults();
  }

  bool isDependentOn(const FusionGroup &other) const {
    return llvm::any_of(operands, [&](auto operand) {
      return other.operands.contains(operand) ||
             other.results.contains(operand);
    });
  }

  bool isSafeToMergeIntoBackOf(const FusionGroup &other) const {
    assert(!ops.empty() && !other.ops.empty() &&
           "fusion groups must be non-empty");
    auto block = ops.front()->getBlock();
    if (block != other.front()->getBlock() ||
        !other.back()->isBeforeInBlock(ops.front())) {
      return false;
    }
    return llvm::all_of(operands, [&](auto operand) {
      auto defOp = operand.getDefiningOp();
      return defOp == nullptr || defOp->getBlock() != block ||
             defOp == other.back() || defOp->isBeforeInBlock(other.back());
    });
  }

  void mergeIntoBackOf(FusionGroup &other) {
    for (auto op : llvm::reverse(ops)) {
      op->moveAfter(other.back());
    }
    for (auto tensorDesc : tensorDescs) {
      if (!llvm::is_contained(other.tensorDescs, tensorDesc)) {
        other.tensorDescs.push_back(tensorDesc);
      }
    }
    other.ops.append(ops.begin(), ops.end());
    other.resolveOperandsAndResults();
  }

  SmallVector<Operation *> ops;
  SetVector<Value> operands;
  SetVector<Value> results;
  SmallVector<FusionTensorDesc> tensorDescs;
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
    scf::populateUpliftWhileToForPatterns(patterns);
    patterns.add<RestoreDotOperandSplatConstantPattern>(ctx);
    patterns.add<EliminateForOpInductionVars>(ctx);
    patterns.add<ConvertTritonLoadOp, ConvertTritonStoreOp>(ctx,
                                                            axisInfoExAnalysis);
    patterns.add<ConvertTritonOp<triton::MulhiUIOp, mlir::math_ext::UmulhiOp>,
                 ConvertTritonOp<triton::PreciseSqrtOp, math::SqrtOp>,
                 ConvertTritonOp<triton::PreciseDivFOp, arith::DivFOp>,
                 ConvertClampFOp, ConvertFpToFpOp>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    for (auto func : module.getOps<triton::FuncOp>()) {
      DominanceInfo dominanceInfo(func);
      preProcessMaskedLoadStore(func, dominanceInfo);
      sinkFusibleProducersToUses(func, dominanceInfo);
      func.walk([&](Block *block) { runFuse(block); });
      func.walk([&](Block *block) { fuseReduceOps(block); });
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TritonGCUDialect, arith::ArithDialect, math::MathDialect,
                    triton::TritonDialect, math_ext::MathExtDialect>();
  }

private:
  void runFuse(Block *block);
  void fuseOps(std::unique_ptr<FusionGroup> fusionGroup);
  bool tryDeepFuse(SmallVector<std::unique_ptr<FusionGroup>> &fusionGroups);
  void fuseReduceOps(Block *block);
};
} // namespace

void GCUTritonFusionPass::fuseReduceOps(Block *block) {
  for (auto &op : block->getOperations()) {
    auto fusionOp = dyn_cast<mlir::triton::gcu::ElementwiseFusionRegionOp>(op);
    if (!fusionOp || fusionOp.getNumResults() != 1 ||
        !fusionOp.getResult(0).hasOneUse()) {
      continue;
    }
    auto reduceOp = dyn_cast<mlir::triton::ReduceOp>(
        *fusionOp.getResult(0).getUsers().begin());
    if (!reduceOp || reduceOp.getNumOperands() != 1) {
      continue;
    }
    ReduceGenerator reduceGenerator(reduceOp,
                                    fusionOp.getBody()->getArguments(),
                                    fusionOp.getBody()->without_terminator());
    if (!reduceGenerator.hasVectorizeImpl()) {
      continue;
    }
    auto terminator = fusionOp.getBody()->getTerminator();
    fusionOp->getResult(0).setType(reduceOp->getResult(0).getType());
    reduceOp->getResult(0).replaceAllUsesWith(fusionOp->getResult(0));
    reduceOp->setOperand(0, terminator->getOperand(0));
    reduceOp->moveBefore(terminator);
    terminator->setOperand(0, reduceOp->getResult(0));
  }
}

bool GCUTritonFusionPass::tryDeepFuse(
    SmallVector<std::unique_ptr<FusionGroup>> &fusionGroups) {
  auto numFusionGroup = fusionGroups.size();
  for (unsigned i = 0; i < numFusionGroup - 1; ++i) {
    if (auto firstUser = fusionGroups[i]->findFirstUserInBlock()) {
      for (unsigned j = i + 1; j < numFusionGroup; ++j) {
        if (llvm::none_of(fusionGroups[i]->tensorDescs,
                          [&](const FusionTensorDesc &desc) {
                            return llvm::is_contained(
                                fusionGroups[j]->tensorDescs, desc);
                          })) {
          continue;
        }
        if (!fusionGroups[j]->contains(firstUser)) {
          continue;
        }
        if (hasSideEffectBetween(fusionGroups[i]->back()->getNextNode(),
                                 fusionGroups[j]->front()) &&
            fusionGroups[i]->hasSideEffect()) {
          continue;
        }
        fusionGroups[i]->mergeIntoFrontOf(*fusionGroups[j]);
        fusionGroups.erase(fusionGroups.begin() + i);
        return true;
      }
    }
    for (unsigned j = i + 1; j < numFusionGroup; ++j) {
      if (llvm::none_of(
              fusionGroups[i]->tensorDescs, [&](const FusionTensorDesc &desc) {
                return llvm::is_contained(fusionGroups[j]->tensorDescs, desc);
              })) {
        continue;
      }
      if (!fusionGroups[j]->isDependentOn(*fusionGroups[i]) ||
          !fusionGroups[j]->isSafeToMergeIntoBackOf(*fusionGroups[i])) {
        continue;
      }
      if (hasSideEffectBetween(fusionGroups[i]->back()->getNextNode(),
                               fusionGroups[j]->front()) &&
          fusionGroups[j]->hasSideEffect()) {
        continue;
      }
      fusionGroups[j]->mergeIntoBackOf(*fusionGroups[i]);
      fusionGroups.erase(fusionGroups.begin() + j);
      return true;
    }
  }
  return false;
}

void GCUTritonFusionPass::runFuse(Block *block) {
  SmallVector<std::unique_ptr<FusionGroup>> fusionGroups;
  auto startNewGroup = [&]() {
    fusionGroups.emplace_back(std::make_unique<FusionGroup>());
  };
  auto finalizeCurrentGroup = [&]() {
    fusionGroups.back()->resolveOperandsAndResults();
    startNewGroup();
  };
  auto currentGroup = [&]() { return fusionGroups.back().get(); };
  startNewGroup();
  SmallVector<Operation *> operations;
  for (auto &op : block->getOperations()) {
    operations.push_back(&op);
  }
  for (auto op : operations) {
    if (canFuse(op)) {
      if (isa<triton::gpu::ConvertLayoutOp>(op) &&
          !currentGroup()->isUsedBy(op)) {
        if (!currentGroup()->empty()) {
          op->moveBefore(currentGroup()->front());
        }
        continue;
      }
      if (!currentGroup()->tryInsert(op)) {
        finalizeCurrentGroup();
        auto success = currentGroup()->tryInsert(op);
        assert(success);
        (void)success;
      }
    } else if (!currentGroup()->empty()) {
      if ((op->hasTrait<OpTrait::Elementwise>() ||
           isa<triton::gcu::LoadOp>(op)) &&
          !currentGroup()->isUsedBy(op)) {
        op->moveBefore(currentGroup()->front());
        continue;
      }
      finalizeCurrentGroup();
    }
  }
  if (currentGroup()->empty()) {
    fusionGroups.pop_back();
  } else {
    currentGroup()->resolveOperandsAndResults();
  }
  if (fusionGroups.empty()) {
    return;
  }
  bool changed = true;
  do {
    changed = tryDeepFuse(fusionGroups);
  } while (changed);
  for (auto &fusionGroup : fusionGroups) {
    fuseOps(std::move(fusionGroup));
  }
}

void GCUTritonFusionPass::fuseOps(std::unique_ptr<FusionGroup> fusionGroup) {
  auto &ops = fusionGroup->ops;
  assert(!ops.empty() && "fusion group must be non-empty");

  OpBuilder builder(ops.front());
  auto loc = ops.front()->getLoc();
  fusionGroup->resolveOperandsAndResults();
  auto operands = fusionGroup->operands.takeVector();
  auto results = fusionGroup->results.takeVector();

  auto resultTypes = llvm::to_vector(
      llvm::map_range(results, [](auto result) { return result.getType(); }));
  auto fusedOp = builder.create<mlir::triton::gcu::ElementwiseFusionRegionOp>(
      loc, resultTypes, operands);
  auto &entryBlock = fusedOp.getRegion().emplaceBlock();
  {
    IRMapping mapper;
    for (auto operand : operands) {
      auto arg = entryBlock.addArgument(operand.getType(), loc);
      mapper.map(operand, arg);
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&entryBlock);
    for (auto op : ops) {
      Operation *cloneOp = builder.clone(*op, mapper);
      // clone() may update mapper internally in some MLIR versions;
      // explicit mapping here for compatibility and clarity.
      for (unsigned i = 0, numResults = op->getNumResults(); i != numResults;
           ++i) {
        mapper.map(op->getResult(i), cloneOp->getResult(i));
      }
    }
    auto fusedResults = llvm::to_vector(llvm::map_range(
        results, [&mapper](auto result) { return mapper.lookup(result); }));
    builder.create<mlir::triton::gcu::YieldOp>(loc, fusedResults);
  }
  for (auto [result, fusedResult] :
       llvm::zip_equal(results, fusedOp.getResults())) {
    result.replaceAllUsesWith(fusedResult);
  }
  for (auto op : llvm::reverse(ops)) {
    op->erase();
  }
}
