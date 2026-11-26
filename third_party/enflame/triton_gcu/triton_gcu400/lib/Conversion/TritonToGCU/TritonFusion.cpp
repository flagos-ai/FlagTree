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
#include <optional>
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
      [](auto &op) {
#if TRITON_VERSION == 35
        if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(&op)) {
          SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
              effects;
          memEffect.getEffects(effects);
          return llvm::any_of(effects, [](const auto &effect) {
            return isa<MemoryEffects::Write>(effect.getEffect());
          });
        }
        return !isMemoryEffectFree(&op);
#else
        return mightHaveEffect<MemoryEffects::Write>(&op);
#endif
      });
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
            results.push_back(mapper.lookupOrDefault(yieldOp->getOperand(i)));
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

static Value accumulateOffset(OpBuilder &builder, Location loc,
                              Value baseOffset, Value offsetToAdd) {
  auto rankedTensorType = cast<RankedTensorType>(baseOffset.getType());
  auto elementType = getElementTypeOrSelf(offsetToAdd.getType());
  if (!isa<RankedTensorType>(offsetToAdd.getType())) {
    offsetToAdd = builder.create<triton::SplatOp>(
        loc, rankedTensorType.clone(elementType), offsetToAdd);
  } else if (cast<RankedTensorType>(offsetToAdd.getType()).getShape() !=
             rankedTensorType.getShape()) {
    offsetToAdd = builder.create<triton::BroadcastOp>(
        loc, rankedTensorType.clone(elementType), offsetToAdd);
  }
  auto promotedType =
      mlir::triton::gcu::getBpe(rankedTensorType.getElementType()) >
              mlir::triton::gcu::getBpe(elementType)
          ? baseOffset.getType()
          : offsetToAdd.getType();
  if (promotedType != baseOffset.getType()) {
    baseOffset = builder.create<arith::ExtSIOp>(loc, promotedType, baseOffset);
  }
  if (promotedType != offsetToAdd.getType()) {
    offsetToAdd =
        builder.create<arith::ExtSIOp>(loc, promotedType, offsetToAdd);
  }
  return builder.create<arith::AddIOp>(loc, baseOffset, offsetToAdd);
}

static FailureOr<PtrDecomposeResult>
tryDecomposeLoadStorePtr(OpBuilder &builder, Location loc, Value ptr) {
  auto bitcastOp = ptr.getDefiningOp<triton::BitcastOp>();
  if (bitcastOp)
    ptr = bitcastOp.getSrc();
  auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp)
    return failure();
  auto rankedTensorType = dyn_cast<RankedTensorType>(ptr.getType());
  if (!rankedTensorType ||
      (rankedTensorType.getRank() != 1 && rankedTensorType.getRank() != 2))
    return failure();
  auto offset = addPtrOp.getOffset();
  auto basePtr = addPtrOp.getPtr();
  while (auto addPtrOp = basePtr.getDefiningOp<triton::AddPtrOp>()) {
    basePtr = addPtrOp.getPtr();
    offset = accumulateOffset(builder, loc, offset, addPtrOp.getOffset());
  }
  if (rankedTensorType.getRank() == 2) {
    if (auto broadcastOp = basePtr.getDefiningOp<triton::BroadcastOp>()) {
      auto srcPtr = broadcastOp.getSrc();
      auto expandDimsOp = srcPtr.getDefiningOp<triton::ExpandDimsOp>();
      if (expandDimsOp)
        srcPtr = expandDimsOp.getSrc();
      auto result = tryDecomposeLoadStorePtr(builder, loc, srcPtr);
      if (failed(result)) {
        return failure();
      }
      basePtr = result->basePtr;
      auto offsetToAdd = result->offset;
      if (expandDimsOp)
        offsetToAdd = builder.create<triton::ExpandDimsOp>(
            loc, offsetToAdd, expandDimsOp.getAxis());
      offset = accumulateOffset(builder, loc, offset, offsetToAdd);
      if (bitcastOp) {
        auto pointeeType =
            cast<triton::PointerType>(
                cast<RankedTensorType>(bitcastOp.getType()).getElementType())
                .getPointeeType();
        auto origPtrType = cast<triton::PointerType>(basePtr.getType());
        auto newPtrType = triton::PointerType::get(
            pointeeType, origPtrType.getAddressSpace());
        basePtr = builder.create<triton::BitcastOp>(loc, newPtrType, basePtr);
      }
      return PtrDecomposeResult{basePtr, offset};
    }
  }
  if (auto splatOp = basePtr.getDefiningOp<triton::SplatOp>()) {
    basePtr = splatOp.getSrc();
    while (auto addPtrOp = basePtr.getDefiningOp<triton::AddPtrOp>()) {
      basePtr = addPtrOp.getPtr();
      offset = accumulateOffset(builder, loc, offset, addPtrOp.getOffset());
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
    if (llvm::any_of(op->getUsers(), [](auto user) {
          return isa<triton::gpu::LocalAllocOp>(user);
        })) {
      return failure();
    }
    if (auto selectOp = op.getPtr().getDefiningOp<arith::SelectOp>()) {
      auto condition = selectOp.getCondition();
      auto truePtr = selectOp.getTrueValue();
      auto falsePtr = selectOp.getFalseValue();
      auto buildBranchLoad = [&](mlir::OpBuilder &builder, Value ptr) -> Value {
        Operation *cloned = builder.clone(*op.getOperation());
        cast<triton::LoadOp>(cloned).getPtrMutable().assign(ptr);
        return cloned->getResult(0);
      };
      auto resultType = op.getType();
      if (auto splatOp = condition.getDefiningOp<triton::SplatOp>()) {
        condition = splatOp.getSrc();
        auto ifOp = rewriter.create<scf::IfOp>(loc, mlir::TypeRange{resultType},
                                               condition, true);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          auto res = buildBranchLoad(rewriter, truePtr);
          rewriter.create<scf::YieldOp>(loc, res);
        }
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
          Value res = buildBranchLoad(rewriter, falsePtr);
          rewriter.create<scf::YieldOp>(loc, res);
        }
        rewriter.replaceOp(op, ifOp.getResult(0));
        return success();
      } else {
        rewriter.setInsertionPoint(op);
        auto trueValue = buildBranchLoad(rewriter, truePtr);
        auto falseValue = buildBranchLoad(rewriter, falsePtr);
        rewriter.replaceOpWithNewOp<arith::SelectOp>(op, resultType, condition,
                                                     trueValue, falseValue);
        return success();
      }
    }
    auto result = tryDecomposeLoadStorePtr(rewriter, loc, op.getPtr());
    if (succeeded(result)) {
      auto [basePtr, offset] = *result;
      auto mask = op.getMask();
      auto other = op.getOther();
      auto maskedLoadOp = rewriter.create<triton::gcu::MaskedLoadOp>(
          loc, basePtr, offset, mask, other);
      auto axisInfoEx = analysis.getAxisInfoEx(op.getPtr());
      if (!axisInfoEx || maskedLoadOp.getResult().getType().getRank() != 1) {
        rewriter.replaceOp(op, maskedLoadOp);
        return success();
      }
      auto continualInterval = axisInfoEx->getContinualInterval(0);
      if (continualInterval == 1) {
        maskedLoadOp->setAttr(kIsContinual, rewriter.getBoolAttr(true));
      } else if (continualInterval == 0) {
        auto constancy = axisInfoEx->getConstancy(0);
        if (constancy *
                getBpe(getElementTypeOrSelf(op.getResult().getType())) >=
            kOaccSizeInBytes) {
          maskedLoadOp->setAttr(kConstancy,
                                rewriter.getI32IntegerAttr(constancy));
        }
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
      auto maskedStoreOp = rewriter.create<triton::gcu::MaskedStoreOp>(
          loc, basePtr, offset, value, mask);
      auto isContinual = axisInfoEx &&
                         axisInfoEx->getContinualInterval(0) == 1 &&
                         maskedStoreOp.getValue().getType().getRank() == 1;
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

static std::optional<unsigned> getUniqueBroadcastAxis(triton::BroadcastOp op) {
  auto srcType = op.getSrc().getType();
  auto resultType = op.getType();
  auto rank = srcType.getRank();
  std::optional<unsigned> broadcastAxis = std::nullopt;
  for (unsigned i = 0; i < rank; ++i) {
    if (srcType.getDimSize(i) != resultType.getDimSize(i)) {
      if (broadcastAxis) {
        return std::nullopt;
      }
      broadcastAxis = i;
    }
  }
  return broadcastAxis;
}

static inline bool isValidOaccCount(unsigned numOacc) {
  return numOacc >= 1 && numOacc <= 4 * kLoopUnrollTimes;
}

static bool canFuseReshapeOp(triton::ReshapeOp op) {
  return !triton::gcu::isExpensiveView(op);
}

static bool canFuseConvertLayout(triton::gpu::ConvertLayoutOp op) {
  return triton::gcu::isZeroCopy(op);
}

static bool canFuseBroadcast(triton::BroadcastOp op) {
  auto srcType = op.getSrc().getType();
  auto resultType = op.getType();
  auto rank = srcType.getRank();
  auto broadcastAxis = getUniqueBroadcastAxis(op);

  if (!broadcastAxis || (*broadcastAxis != 0 && *broadcastAxis != rank - 1)) {
    return false;
  }
  auto elementType = getElementTypeOrSelf(srcType);
  if (elementType.isInteger(1)) {
    return false;
  }
  auto bpe = triton::gcu::getBpe(elementType);
  auto elemsPerThread = broadcastAxis == 0
                            ? triton::gcu::getElemsPerThread(srcType)
                            : triton::gcu::getElemsPerThread(resultType);
  auto sizeInBytes =
      std::accumulate(elemsPerThread.begin() + *broadcastAxis,
                      elemsPerThread.end(), 1, std::multiplies<int64_t>()) *
      bpe;
  auto numOacc = sizeInBytes / kOaccSizeInBytes;
  // TODO(peng.tian): need to support general implementation
  return isValidOaccCount(numOacc);
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
  func.walk([&](Block *block) {
    for (auto &op : llvm::make_early_inc_range(block->getOperations())) {
      if (!canFuse(&op)) {
        continue;
      }
      auto operands = llvm::to_vector(op.getOperands());
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
        if (defOp->isAncestor(&op)) {
          continue;
        }
        if (llvm::all_of(defOp->getOperands(), [&](auto operand) {
              return dominanceInfo.dominates(operand, &op);
            })) {
          if (operand.hasOneUse()) {
            defOp->moveBefore(&op);
          } else {
            OpBuilder builder(&op);
            op.replaceUsesOfWith(operand, builder.clone(*defOp)->getResult(0));
          }
        }
      }
    }
  });
}

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
  static RankedTensorType getRankedTensorType(Operation *fusionOp) {
    if (auto maskedStoreOp = dyn_cast<triton::gcu::MaskedStoreOp>(fusionOp)) {
      return maskedStoreOp.getValue().getType();
    } else {
      return cast<RankedTensorType>(fusionOp->getResultTypes().front());
    }
  }
  bool empty() const { return ops.empty(); }
  bool isUsedBy(Operation *op) const {
    return llvm::any_of(ops, [op](auto innerOp) {
      return llvm::any_of(innerOp->getUsers(),
                          [op](auto user) { return user == op; });
    });
  }
  bool tryInsert(Operation *op) {
    auto tensorType = getRankedTensorType(op);
    auto tensorDesc = getFusionTensorDesc(tensorType);
    if (ops.empty()) {
      ops.push_back(op);
      tensorDescs.emplace_back(tensorDesc);
      if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(op)) {
        tensorDescs.emplace_back(
            getFusionTensorDesc(reshapeOp.getSrc().getType()));
      }
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
    if (ops.empty()) {
      return false;
    }
    auto block = ops.front()->getBlock();
    if (block != op->getBlock()) {
      return false;
    }
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

  SmallVector<Operation *> scheduleFusionRegion();

  SmallVector<Operation *> ops;
  SetVector<Value> operands;
  SetVector<Value> results;
  SmallVector<FusionTensorDesc> tensorDescs;
};

SmallVector<Operation *> FusionGroup::scheduleFusionRegion() {
  if (ops.size() <= 2) {
    return llvm::to_vector(ops);
  }
  if (llvm::any_of(ops,
                   [](Operation *op) { return !isMemoryEffectFree(op); })) {
    return llvm::to_vector(ops);
  }
  auto hasInRegionDependency = [this](Operation *op) {
    return llvm::any_of(op->getOperands(), [this](auto operand) {
      auto defOp = operand.getDefiningOp();
      return defOp && contains(defOp);
    });
  };
  SmallVector<Operation *> ordered;
  DenseSet<Operation *> emitted;
  std::function<void(Operation *)> emit;

  auto isLeaf = [&](Operation *op) { return !hasInRegionDependency(op); };

  auto tryEmitConsumer = [&](Operation *op) {
    if (!op || !contains(op) || emitted.contains(op)) {
      return;
    }
    if (!llvm::all_of(op->getOperands(), [&](auto operand) {
          auto defOp = operand.getDefiningOp();
          if (!defOp || !contains(defOp) || isLeaf(defOp)) {
            return true;
          }
          return emitted.contains(defOp);
        })) {
      return;
    }
    for (auto operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || !contains(defOp)) {
        continue;
      }
      if (isLeaf(defOp)) {
        emit(defOp);
      }
    }
    emit(op);
  };

  emit = [&](Operation *op) {
    if (!op || !contains(op) || !emitted.insert(op).second) {
      return;
    }
    for (bool emitLeaf : {false, true}) {
      for (auto operand : op->getOperands()) {
        auto *defOp = operand.getDefiningOp();
        if (!defOp || !contains(defOp))
          continue;
        if (isLeaf(defOp) == emitLeaf)
          emit(defOp);
      }
    }
    ordered.push_back(op);
    for (bool emitResult : {false, true}) {
      for (auto user : op->getUsers()) {
        if (llvm::all_of(user->getResults(), [&](auto result) {
              return results.contains(result);
            }) == emitResult) {
          tryEmitConsumer(user);
        }
      }
    }
  };

  for (auto result : results) {
    if (auto *defOp = result.getDefiningOp())
      emit(defOp);
  }
  for (auto *op : ops) {
    emit(op);
  }
  return ordered;
}

static bool isConstantFusion(Operation *op) {
  auto fusion = dyn_cast<triton::gcu::ElementwiseFusionRegionOp>(op);
  if (!fusion)
    return false;
  auto body = fusion.getBody()->without_terminator();
  return llvm::hasSingleElement(body) && isa<arith::ConstantOp>(body.begin());
}

bool canUseAllocaForValue(Value val) {
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
      return false;
    }
  }
  return true;
}

// Match a splat i32 constant and return its value.
static bool matchSplatConstantI32(Value value, int64_t &out) {
  auto cst = value.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  auto splat = dyn_cast<SplatElementsAttr>(cst.getValue());
  if (!splat || !splat.getElementType().isInteger(32))
    return false;
  out = splat.getSplatValue<APInt>().getSExtValue();
  return true;
}

// An i32 scalar operand captured while matching a prefix mask.  A splat
// constant has no scalar SSA value yet, so the constant is kept here and only
// materialised when the count operands are actually built.  Keeping it as an
// attribute also lets two masks that use the same literal compare equal.
struct ScalarI32 {
  Value value;
  std::optional<int64_t> constant;

  explicit operator bool() const { return value || constant; }
};

struct PrefixMaskMatch {
  ScalarI32 bound;
  ScalarI32 offset;
  bool unsignedPredicate = false;
};
static std::optional<PrefixMaskMatch> matchPrefixMask(Value mask);

// Extract the scalar shift from the `tt.splat` (or splat constant) that feeds
// the digit `shrui`.
static Value matchShiftScalar(Value value, IRRewriter &rewriter) {
  if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
    if (splat.getSrc().getType().isInteger(32))
      return splat.getSrc();
    return Value();
  }
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto splat = dyn_cast<SplatElementsAttr>(cst.getValue()))
      if (splat.getElementType().isInteger(32))
        return arith::ConstantOp::create(
            rewriter, cst.getLoc(),
            rewriter.getI32IntegerAttr(
                splat.getSplatValue<APInt>().getSExtValue()));
  }
  return Value();
}

// The histogram count loop reads one 32-bit key per lane.  Reject any other
// buffer element type here so an unsupported memdesc never reaches the
// lowering (an f32 buffer is fine: the region bitcasts it to i32 first).
static bool isHistogramKeyElement(Type type) {
  return type.isIntOrFloat() && type.getIntOrFloatBitWidth() == 32;
}

// Trace the histogram key back to the L1 buffer that already holds it:
//   key = tt.load(tle.local_pointers(%memdesc, make_range(0, N)))
// The load must be a linear read of the whole buffer view whose mask (if any)
// is a prefix range, so the backend can count directly from `%memdesc` and
// skip both the load and the digit tile.  Look through a bitcast between the
// load and the shift.
static Value traceHistogramKeyBuffer(Value keyValue, RankedTensorType srcType,
                                     Value *loadMask = nullptr) {
  if (!srcType || !isHistogramKeyElement(srcType.getElementType()))
    return Value();
  // Peel bitcasts and layout-only ops inserted by earlier passes.
  Value base = keyValue;
  while (Operation *def = base.getDefiningOp()) {
    if (auto bitcast = dyn_cast<arith::BitcastOp>(def)) {
      base = bitcast.getIn();
      continue;
    }
    if (auto bitcast = dyn_cast<triton::BitcastOp>(def)) {
      base = bitcast.getSrc();
      continue;
    }
    if (isa<triton::BroadcastOp, triton::gpu::ConvertLayoutOp>(def)) {
      base = def->getOperand(0);
      continue;
    }
    break;
  }

  // After tle-to-triton-gcu the L1 key load becomes `ttg.local_load` of the
  // memdesc, so the buffer view is the direct operand.
  if (auto localLoad = base.getDefiningOp<triton::gpu::LocalLoadOp>()) {
    Value memdesc = localLoad.getSrc();
    auto memdescType = dyn_cast<triton::gpu::MemDescType>(memdesc.getType());
    if (!memdescType || memdescType.getRank() != 1 ||
        !memdescType.hasStaticShape() ||
        !isHistogramKeyElement(memdescType.getElementType()))
      return Value();
    if (memdescType.getShape()[0] != srcType.getShape()[0])
      return Value();
    return memdesc;
  }

  auto load = base.getDefiningOp<triton::LoadOp>();
  if (!load)
    return Value();
  // A masked local load can still be counted in place when its mask is a
  // prefix range: the same prefix is folded into `size_in`, so the count loop
  // skips the load's tail lanes without materialising the loaded tensor.
  if (loadMask)
    *loadMask = load.getMask();
  if (load.getMask() && !matchPrefixMask(load.getMask()))
    return Value();

  // Walk through layout-only ops to the local-pointers producer.
  Operation *localPointers = load.getPtr().getDefiningOp();
  while (localPointers &&
         isa<triton::BroadcastOp, triton::gpu::ConvertLayoutOp>(localPointers))
    localPointers = localPointers->getOperand(0).getDefiningOp();
  if (!localPointers ||
      localPointers->getName().getStringRef() != "tle.local_pointers")
    return Value();
  if (localPointers->getNumOperands() != 2)
    return Value();

  Value memdesc = localPointers->getOperand(0);
  auto memdescType = dyn_cast<triton::gpu::MemDescType>(memdesc.getType());
  if (!memdescType || memdescType.getRank() != 1 ||
      !memdescType.hasStaticShape() ||
      !isHistogramKeyElement(memdescType.getElementType()))
    return Value();
  if (memdescType.getShape()[0] != srcType.getShape()[0])
    return Value();

  auto range =
      localPointers->getOperand(1).getDefiningOp<triton::MakeRangeOp>();
  if (!range || range.getStart() != 0 ||
      range.getEnd() != memdescType.getShape()[0])
    return Value();
  return memdesc;
}

// Match an i32 scalar, a splat of it, or a splat constant.  The constant form
// is common for compile-time tile bounds (`dense<127> : tensor<128xi32>`) and
// folding it is what keeps those masks on the `size_in` path.
static ScalarI32 matchScalarI32(Value value) {
  if (!value)
    return {};
  if (value.getType().isInteger(32))
    return {value, std::nullopt};
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    if (splat.getSrc().getType().isInteger(32))
      return {splat.getSrc(), std::nullopt};
  if (auto cst = value.getDefiningOp<arith::ConstantOp>())
    if (auto splat = dyn_cast<SplatElementsAttr>(cst.getValue()))
      if (splat.getElementType().isInteger(32))
        return {Value(), splat.getSplatValue<APInt>().getSExtValue()};
  return {};
}

// The compile-time value of a matched scalar, when it is one.
static std::optional<int64_t> constantScalarI32(const ScalarI32 &scalar) {
  if (scalar.constant)
    return scalar.constant;
  if (auto cst = scalar.value.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getInt();
  return std::nullopt;
}

static bool sameScalarI32(const ScalarI32 &lhs, const ScalarI32 &rhs) {
  if (lhs.value && lhs.value == rhs.value)
    return true;
  std::optional<int64_t> lhsConstant = constantScalarI32(lhs);
  std::optional<int64_t> rhsConstant = constantScalarI32(rhs);
  if (lhsConstant && rhsConstant)
    return *lhsConstant == *rhsConstant;
  return !lhs && !rhs;
}

// Match the prefix mask `(make_range(0, BLOCK) + offset) < n` (the sort's
// `valid`) and return the runtime bound and tile offset.  Such a mask only
// excludes tail lanes, so the backend can count a prefix and skip the mask
// buffer entirely.
//
// `ult` is only folded without an offset.  `lane <u n` is still a prefix, but
// `(lane + offset) <u n` stops being one as soon as the add wraps (the true
// lanes are no longer a prefix of [0, BLOCK)); such masks keep the i8 buffer.
static std::optional<PrefixMaskMatch> matchPrefixMask(Value mask) {
  auto cmp = mask.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;
  auto pred = cmp.getPredicate();
  bool unsignedPredicate = pred == arith::CmpIPredicate::ult;
  if (pred != arith::CmpIPredicate::slt && !unsignedPredicate)
    return std::nullopt;

  Value lhs = cmp.getLhs();
  ScalarI32 offset;
  if (auto add = lhs.getDefiningOp<arith::AddIOp>()) {
    Value a = add.getLhs();
    Value b = add.getRhs();
    Value offsetValue;
    auto range = a.getDefiningOp<triton::MakeRangeOp>();
    if (range && range.getStart() == 0) {
      offsetValue = b;
    } else if ((range = b.getDefiningOp<triton::MakeRangeOp>()) &&
               range.getStart() == 0) {
      offsetValue = a;
    }
    if (!range || range.getStart() != 0)
      return std::nullopt;
    offset = matchScalarI32(offsetValue);
    if (!offset)
      return std::nullopt;
    // A literal zero offset is the same prefix as no offset.
    if (auto offsetConstant = constantScalarI32(offset))
      if (*offsetConstant == 0)
        offset = ScalarI32();
    if (unsignedPredicate && offset)
      return std::nullopt;
  } else if (auto range = lhs.getDefiningOp<triton::MakeRangeOp>()) {
    if (range.getStart() != 0)
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  ScalarI32 n = matchScalarI32(cmp.getRhs());
  if (!n)
    return std::nullopt;
  return PrefixMaskMatch{n, offset, unsignedPredicate};
}

// Materialise the number of leading lanes covered by `match` inside a tile of
// `bufferLen` elements, i.e. `|{i in [0, bufferLen) : i + offset < n}|`.
// `n` is a tile-relative length and `offset` a tile base, so both are assumed
// not to wrap the add.
static Value materializePrefixSize(IRRewriter &rewriter, Location loc,
                                   const PrefixMaskMatch &match,
                                   int64_t bufferLen) {
  auto materialize = [&](const ScalarI32 &scalar) -> Value {
    if (scalar.value)
      return scalar.value;
    if (!scalar.constant)
      return Value();
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(*scalar.constant));
  };
  Value n = materialize(match.bound);
  Value limit = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(bufferLen));

  if (match.unsignedPredicate) {
    // `lane <u n`: count `min(n, len)`.  `n` may exceed INT32_MAX (a uint mask
    // bound), so both the compare and the min must stay unsigned.
    return rewriter.create<arith::MinUIOp>(loc, n, limit);
  }

  Value offset = materialize(match.offset);
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  if (!offset) {
    // `lane <s n`: a non-positive bound counts nothing.
    n = rewriter.create<arith::MaxSIOp>(loc, n, zero);
    return rewriter.create<arith::MinSIOp>(loc, n, limit);
  }
  // `(lane + offset) <s n`: `n - offset` lanes clamped to the tile.  `n <
  // offset` means no lane matches; the select keeps that exact even when
  // `n - offset` would wrap.
  Value diff = rewriter.create<arith::SubIOp>(loc, n, offset);
  Value before =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, n, offset);
  Value clamped = rewriter.create<arith::MinSIOp>(loc, diff, limit);
  return rewriter.create<arith::SelectOp>(loc, before, zero, clamped);
}

// --- general fused digit regions -------------------------------------------
//
// `(key >> shift) & (bins - 1)` is the tuned form and keeps its dedicated
// lowering.  Any other per-element integer chain can still be counted in
// place: the whole chain is lifted into the histogram's scalar region and
// gcu_compiler vectorises it onto a full MRADIX vector inside the count loop,
// so no digit tile is materialised either.  That is what keeps the fusion
// general instead of a fixed list of digit shapes.

static void eraseDeadChain(IRRewriter &rewriter, Value value,
                           bool allowLoads = false);

// Ops the scalar digit region can absorb.  `tt.splat` / `tt.broadcast` /
// `tt.expand_dims` are pure shape plumbing and are dropped, so only real
// per-element arithmetic reaches the region.
//
// The list is exactly what the backend region vectoriser can lift onto a
// 128-lane MRADIX vector.  In particular there is no narrowing/widening:
// `trunci` / `extui` / `extsi` are deliberately absent, so a chain that first
// narrows the key (top-k's `.to(uint16)`) is rejected as a whole instead of
// reaching the vectoriser half-supported.
static bool isScalarizableDigitOp(Operation *op) {
  return isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::AddIOp,
             arith::SubIOp, arith::MulIOp, arith::ShRUIOp, arith::ShRSIOp,
             arith::ShLIOp, arith::CmpIOp, arith::SelectOp, arith::BitcastOp,
             triton::BitcastOp, arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp,
             arith::MinUIOp, triton::SplatOp, triton::BroadcastOp,
             triton::ExpandDimsOp>(op);
}

// Emit one tensor-level digit op as its scalar counterpart in the region.
static void scalarizeDigitOp(OpBuilder &builder, Location loc, Operation *op,
                             IRMapping &mapping) {
  if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp>(op)) {
    mapping.map(op->getResult(0), mapping.lookupOrDefault(op->getOperand(0)));
    return;
  }
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    auto splat = cast<SplatElementsAttr>(cst.getValue());
    mapping.map(op->getResult(0),
                arith::ConstantOp::create(builder, loc,
                                          splat.getSplatValue<TypedAttr>()));
    return;
  }
  // A tensor bitcast has no scalar counterpart in its own dialect at this
  // stage.  Emit `arith.bitcast` on the scalar element so the backend can
  // vectorise it (and so an f32 key can be counted through its i32 bits).
  if (isa<arith::BitcastOp, triton::BitcastOp>(op)) {
    Type elemTy = cast<ShapedType>(op->getResult(0).getType()).getElementType();
    Value src = mapping.lookupOrDefault(op->getOperand(0));
    mapping.map(op->getResult(0),
                arith::BitcastOp::create(builder, loc, elemTy, src));
    return;
  }
  Operation *clone = builder.clone(*op, mapping);
  for (OpResult result : clone->getResults())
    if (auto shaped = dyn_cast<ShapedType>(result.getType()))
      result.setType(shaped.getElementType());
}

// Collect the elementwise chain feeding `value` (post-order) and the buffer
// load it is rooted at.  Only chains that read one key buffer are accepted: a
// second root, a block argument or an unsupported op rejects the chain so the
// generic histogram lowering still handles it.
static bool collectDigitChain(Value value, SmallVectorImpl<Operation *> &chain,
                              Value &root, llvm::DenseSet<Operation *> &seen) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return false;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  // Leaf of the chain: the key load.  It is either an unmasked (or prefix-
  // masked) read of an L1 buffer (counted in place) or a plain tensor load
  // (staged by the histogram lowering); both keep the digit chain in the
  // region.  A bitcast is *not* a leaf: keeping it in the chain lets a 32-bit
  // buffer feed an f32 key without inventing a second element type.
  if (!isa<arith::BitcastOp, triton::BitcastOp>(def) &&
      (traceHistogramKeyBuffer(value, tensorType) ||
       isa<triton::LoadOp, triton::gpu::LocalLoadOp>(def))) {
    if (root && root != value)
      return false;
    root = value;
    return true;
  }
  if (!seen.insert(def).second)
    return true;
  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    auto splat = dyn_cast<SplatElementsAttr>(cst.getValue());
    if (!splat || !splat.getElementType().isIntOrFloat())
      return false;
    chain.push_back(def);
    return true;
  }
  if (!isScalarizableDigitOp(def))
    return false;
  for (Value operand : def->getOperands())
    if (isa<RankedTensorType>(operand.getType()) &&
        !collectDigitChain(operand, chain, root, seen))
      return false;
  chain.push_back(def);
  return true;
}

// Split a mask expression into prefix-range leaves (which become `size_in`)
// and the remaining per-lane predicate (which becomes the mask buffer).  This
// is what turns top-k's `mask_n & matches` into `size_in = n_cols` plus a
// `matches` byte mask, instead of counting the whole tile.
static void collectMaskLeaves(Value mask,
                              SmallVectorImpl<PrefixMaskMatch> &prefixes,
                              SmallVectorImpl<Value> &residual) {
  if (auto andOp = mask.getDefiningOp<arith::AndIOp>()) {
    collectMaskLeaves(andOp.getLhs(), prefixes, residual);
    collectMaskLeaves(andOp.getRhs(), prefixes, residual);
    return;
  }
  if (auto prefix = matchPrefixMask(mask)) {
    prefixes.push_back(*prefix);
    return;
  }
  residual.push_back(mask);
}

// Native `tl.histogram` mask semantics: prefix ranges (`lane < N`) collapse
// into `size_in`; any other mask is widened to i8 and forwarded as a buffer.
static bool buildHistogramCountOperands(triton::HistogramOp hist,
                                        IRRewriter &rewriter,
                                        RankedTensorType srcType, Value &sizeIn,
                                        Value &mask) {
  Value histMask = hist.getMask();
  if (!histMask) {
    sizeIn = arith::ConstantOp::create(
        rewriter, hist.getLoc(),
        rewriter.getI32IntegerAttr(srcType.getShape()[0]));
    return true;
  }
  auto maskType = dyn_cast<RankedTensorType>(histMask.getType());
  if (!maskType || maskType.getRank() != 1)
    return false;

  SmallVector<PrefixMaskMatch, 2> prefixes;
  SmallVector<Value, 2> residual;
  collectMaskLeaves(histMask, prefixes, residual);
  if (!prefixes.empty()) {
    Value prefixSize = materializePrefixSize(
        rewriter, hist.getLoc(), prefixes.front(), srcType.getShape()[0]);
    for (const PrefixMaskMatch &prefix : llvm::drop_begin(prefixes))
      prefixSize = rewriter.create<arith::MinSIOp>(
          hist.getLoc(), prefixSize,
          materializePrefixSize(rewriter, hist.getLoc(), prefix,
                                srcType.getShape()[0]));
    sizeIn = prefixSize;
  } else {
    sizeIn = arith::ConstantOp::create(
        rewriter, hist.getLoc(),
        rewriter.getI32IntegerAttr(srcType.getShape()[0]));
  }

  if (residual.empty())
    return true;
  Value residualMask = residual.front();
  for (Value leaf : llvm::drop_begin(residual))
    residualMask =
        rewriter.create<arith::AndIOp>(hist.getLoc(), residualMask, leaf);
  auto maskI8Type = RankedTensorType::get(
      maskType.getShape(), rewriter.getI8Type(), maskType.getEncoding());
  mask =
      rewriter.create<arith::ExtUIOp>(hist.getLoc(), maskI8Type, residualMask);
  return true;
}

// Whether the histogram mask contains the exact prefix that masks the key
// load.  A masked key load can only be counted in place when the same prefix
// removes its tail lanes from the count; a different (even wider) histogram
// prefix would let the load's `other` tail values reach the bins.
static bool histogramMaskCoversLoadPrefix(triton::HistogramOp hist,
                                          Value loadMask) {
  if (!loadMask)
    return true;
  std::optional<PrefixMaskMatch> loadPrefix = matchPrefixMask(loadMask);
  if (!loadPrefix)
    return false;
  Value mask = hist.getMask();
  if (!mask)
    return false;
  SmallVector<PrefixMaskMatch, 2> prefixes;
  SmallVector<Value, 2> residual;
  collectMaskLeaves(mask, prefixes, residual);
  return llvm::any_of(prefixes, [&](const PrefixMaskMatch &prefix) {
    // The predicate matters: the same `n` yields a different prefix for
    // `slt` and `ult` once it crosses INT32_MAX.  A wider histogram prefix than
    // the load prefix would let the load's `other` tail values reach the bins,
    // and a narrower one would silently drop lanes.
    return prefix.unsignedPredicate == loadPrefix->unsignedPredicate &&
           sameScalarI32(prefix.bound, loadPrefix->bound) &&
           sameScalarI32(prefix.offset, loadPrefix->offset);
  });
}

// Fuse a general elementwise digit chain into `triton_gcu.histogram`, keeping
// the chain as a scalar region.
static LogicalResult fuseGeneralHistogramDigit(triton::HistogramOp hist,
                                               IRRewriter &rewriter) {
  auto srcType = dyn_cast<RankedTensorType>(hist.getSrc().getType());
  if (!srcType || srcType.getRank() != 1 ||
      !srcType.getElementType().isInteger(32))
    return failure();
  // The general region path currently vectorises whole 128-lane chunks; a
  // partial tail stays on the generic digits-in histogram fallback.
  if (srcType.getShape()[0] % 128 != 0)
    return failure();
  auto resType = dyn_cast<RankedTensorType>(hist.getType());
  if (!resType || resType.getRank() != 1 || resType.getShape()[0] <= 0 ||
      resType.getShape()[0] > 32 || !resType.getElementType().isInteger(32))
    return failure();

  SmallVector<Operation *> chain;
  Value root;
  llvm::DenseSet<Operation *> seen;
  if (!collectDigitChain(hist.getSrc(), chain, root, seen) || !root)
    return failure();
  // Narrowing/widening is rejected by `collectDigitChain` itself (see
  // `isScalarizableDigitOp`); the chain reaching here is vectorisable as is.
  auto rootType = dyn_cast<RankedTensorType>(root.getType());
  if (!rootType || rootType.getElementType().getIntOrFloatBitWidth() != 32)
    return failure();
  Value loadMask;
  Value keySource = traceHistogramKeyBuffer(root, rootType, &loadMask);
  if (!histogramMaskCoversLoadPrefix(hist, loadMask))
    keySource = Value();
  if (keySource) {
    auto memdescType = dyn_cast<triton::gpu::MemDescType>(keySource.getType());
    if (!memdescType)
      return failure();
    // The key buffer may be f32 while the digit chain starts from its i32
    // bits; the histogram lowering inserts the scalar bitcast when the two
    // element types differ.
  } else {
    // No provable L1 view (or the load mask is not fully represented by the
    // histogram prefix): the histogram lowering stages the loaded tensor,
    // which still avoids the tensor-level digit tile.
    keySource = root;
  }

  rewriter.setInsertionPoint(hist);
  Value sizeIn;
  Value mask;
  if (!buildHistogramCountOperands(hist, rewriter, srcType, sizeIn, mask))
    return failure();
  Value numBins = arith::ConstantOp::create(
      rewriter, hist.getLoc(),
      rewriter.getI32IntegerAttr(resType.getShape()[0]));

  auto fused = rewriter.create<triton::gcu::HistogramOp>(
      hist.getLoc(), hist.getType(), keySource, numBins, sizeIn, mask);

  // Rebuild the chain as a scalar region; the elementwise ops are pure, so the
  // tensor-level originals die with the chain below.
  //
  // Contract: the yielded digit must already be in `[0, num_bins)`.  The
  // backend vectorises this region onto the MRADIX count and the hardware only
  // looks at the low 5 radix bits, so an out-of-range digit would be counted
  // into a wrapped bin instead of being dropped.  `mask` / `size_in` are the
  // only lane filters; no per-lane range check is added here because it would
  // cost a compare and a select on every lane of the hot count loop.
  Block *body = new Block();
  fused.getRegion().push_back(body);
  body->addArgument(rootType.getElementType(), hist.getLoc());
  OpBuilder regionBuilder = OpBuilder::atBlockEnd(body);
  IRMapping mapping;
  mapping.map(root, body->getArgument(0));
  for (Operation *op : chain)
    scalarizeDigitOp(regionBuilder, hist.getLoc(), op, mapping);
  regionBuilder.create<triton::gcu::YieldOp>(
      hist.getLoc(), mapping.lookupOrDefault(hist.getSrc()));

  // Keep the TLE memory-space hint: it decides whether the count vector lives
  // in a private stack buffer (and can be SROA-promoted) or in the pool.
  if (auto memSpace = hist->getAttrOfType<StringAttr>(kTleHint))
    fused->setAttr(kTleHint, memSpace);

  Value histMask = hist.getMask();
  hist.getResult().replaceAllUsesWith(fused.getResult());
  rewriter.eraseOp(hist);
  for (Operation *op : llvm::reverse(chain))
    if (op->use_empty() && isMemoryEffectFree(op))
      rewriter.eraseOp(op);
  if (histMask)
    eraseDeadChain(rewriter, histMask);
  // When the key is now counted in place, the staged load that used to feed
  // the digit chain is dead too.  Drop it (and its pointer chain) explicitly:
  // a `tt.load` carries a read effect, so the greedy rewriter keeps it even
  // though the fusion made the L1 buffer the only key source.
  if (keySource != root)
    eraseDeadChain(rewriter, root, /*allowLoads=*/true);
  return success();
}

// Erase a dead producer chain (a folded mask, or the load the digit chain was
// rooted at) so no real tile or read survives the fusion.  Side-effect-free
// ops are always removed; a dead local/global load is removed only when
// `allowLoads` is set: its result is unused, so dropping the read cannot change
// the result, but the greedy rewriter keeps it because loads are not pure.
static void eraseDeadChain(IRRewriter &rewriter, Value value, bool allowLoads) {
  SmallVector<Operation *> worklist;
  llvm::DenseSet<Operation *> erased;
  if (Operation *def = value.getDefiningOp())
    worklist.push_back(def);
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val();
    // Checked before any dereference: `erased` holds ops already deleted here,
    // and the same producer can be pushed by two of its users.
    if (erased.contains(def))
      continue;
    if (!def->use_empty())
      continue;
    bool deadLoad = isa<triton::LoadOp, triton::gpu::LocalLoadOp>(def);
    if (!isMemoryEffectFree(def) && !(allowLoads && deadLoad))
      continue;
    SmallVector<Operation *> operands;
    for (Value operand : def->getOperands())
      if (Operation *producer = operand.getDefiningOp())
        operands.push_back(producer);
    erased.insert(def);
    rewriter.eraseOp(def);
    worklist.append(operands.begin(), operands.end());
  }
}

// Fold `tl.histogram(((key >> shift) & (bins - 1)), bins, mask)` into a single
// digit-shift histogram op.  When the key already lives in an L1 memdesc the op
// carries that buffer directly, so the elementwise digit tile and its store
// loop never exist.  The digit chain is rebuilt as a scalar region and the
// mask is forwarded as an op operand.
static LogicalResult fuseHistogramInputs(::mlir::gpu::GPUModuleOp module,
                                         MLIRContext *ctx) {
  SmallVector<triton::HistogramOp> histograms;
  module.walk([&](triton::HistogramOp op) { histograms.push_back(op); });
  if (histograms.empty())
    return success();

  IRRewriter rewriter(ctx);
  for (auto hist : histograms) {
    auto srcType = dyn_cast<RankedTensorType>(hist.getSrc().getType());
    auto resType = dyn_cast<RankedTensorType>(hist.getType());
    if (!srcType || !resType || srcType.getRank() != 1 ||
        resType.getRank() != 1)
      continue;
    // Both the digit chain and the counts are i32.  Anything else keeps the
    // generic histogram lowering instead of relying on an implicit 32-bit
    // assumption elsewhere in the match.
    if (!srcType.getElementType().isInteger(32) ||
        !resType.getElementType().isInteger(32))
      continue;

    auto andOp = hist.getSrc().getDefiningOp<arith::AndIOp>();
    int64_t binMask = 0;
    Value shrValue;
    bool tuned = false;
    if (andOp) {
      if (matchSplatConstantI32(andOp.getRhs(), binMask)) {
        shrValue = andOp.getLhs();
        tuned = true;
      } else if (matchSplatConstantI32(andOp.getLhs(), binMask)) {
        shrValue = andOp.getRhs();
        tuned = true;
      }
      if (tuned && (resType.getShape()[0] != binMask + 1 || binMask + 1 > 32))
        tuned = false;
    }
    if (!tuned) {
      // Not the tuned `(key >> shift) & (bins - 1)` shape: keep the whole
      // elementwise chain as a region so the backend still counts it in place.
      (void)fuseGeneralHistogramDigit(hist, rewriter);
      continue;
    }

    // `(key >> 0)` folds to `key`, so digit 0 arrives here as a bare
    // `key & (bins - 1)` with no shift op.  Treat the missing shift as the
    // constant 0 instead of falling back to the generic histogram, which
    // would materialise a digit tile and lose the in-place MRADIX count.
    auto shrOp = shrValue.getDefiningOp<arith::ShRUIOp>();
    Value keyValue = shrOp ? shrOp.getLhs() : shrValue;

    rewriter.setInsertionPoint(hist);
    Value shiftScalar;
    if (shrOp) {
      shiftScalar = matchShiftScalar(shrOp.getRhs(), rewriter);
      if (!shiftScalar) {
        (void)fuseGeneralHistogramDigit(hist, rewriter);
        continue;
      }
    } else {
      shiftScalar = arith::ConstantOp::create(rewriter, hist.getLoc(),
                                              rewriter.getI32IntegerAttr(0));
    }

    // Prefer the L1 buffer view so the backend can count in place.  When the
    // buffer view cannot be proven to be a full linear read (for example the
    // load was rewritten by an earlier pass), keep the chain as a region and
    // let the backend vectorise it instead of staging a digit tile.
    Value loadMask;
    Value keySource = traceHistogramKeyBuffer(keyValue, srcType, &loadMask);
    if (!keySource || !histogramMaskCoversLoadPrefix(hist, loadMask)) {
      (void)fuseGeneralHistogramDigit(hist, rewriter);
      continue;
    }

    // Native `tl.histogram` mask semantics: prefix ranges (`lane < N`) are
    // folded into `size_in`; the remaining per-lane predicate (for top-k,
    // `matches`) is widened to i8 and forwarded as a mask buffer.  Splitting
    // the two is what keeps the tail lanes out of bin 0 without materialising
    // a `safe_digit`.
    Value histMask = hist.getMask();
    if (histMask) {
      auto maskType = dyn_cast<RankedTensorType>(histMask.getType());
      if (!maskType || maskType.getRank() != 1)
        continue;
    }
    Value sizeIn;
    Value mask;
    if (!buildHistogramCountOperands(hist, rewriter, srcType, sizeIn, mask))
      continue;
    Value numBins = arith::ConstantOp::create(
        rewriter, hist.getLoc(),
        rewriter.getI32IntegerAttr(resType.getShape()[0]));

    auto fused = rewriter.create<triton::gcu::HistogramOp>(
        hist.getLoc(), hist.getType(), keySource, numBins, sizeIn, mask);

    // Rebuild the elementwise chain as a scalar region: one block argument is
    // the source element, the body is the digit slice, and the yield is the
    // value to count.  The loop-carried `shift` is captured from the enclosing
    // scope (the op is not isolated from above).
    {
      Block *body = new Block();
      fused.getRegion().push_back(body);
      Type keyElemTy = srcType.getElementType();
      if (auto memdescTy =
              dyn_cast<triton::gpu::MemDescType>(keySource.getType()))
        keyElemTy = memdescTy.getElementType();
      Value elem = body->addArgument(keyElemTy, hist.getLoc());
      OpBuilder regionBuilder = OpBuilder::atBlockEnd(body);
      Value counted = elem;
      if (keyElemTy != srcType.getElementType())
        counted = regionBuilder.create<arith::BitcastOp>(
            hist.getLoc(), srcType.getElementType(), counted);
      if (shrOp)
        counted = regionBuilder.create<arith::ShRUIOp>(hist.getLoc(), counted,
                                                       shiftScalar);
      Value binMaskCst = regionBuilder.create<arith::ConstantOp>(
          hist.getLoc(), regionBuilder.getI32IntegerAttr(binMask));
      counted = regionBuilder.create<arith::AndIOp>(hist.getLoc(), counted,
                                                    binMaskCst);
      regionBuilder.create<triton::gcu::YieldOp>(hist.getLoc(), counted);
    }

    // The TLE memory-space hint must survive the fusion: the backend uses it to
    // place the count vector in a private stack buffer (and hence let SROA
    // promote it) instead of a slice of the private pool.
    if (auto memSpace = hist->getAttrOfType<StringAttr>(kTleHint))
      fused->setAttr(kTleHint, memSpace);
    hist.getResult().replaceAllUsesWith(fused.getResult());
    rewriter.eraseOp(hist);

    // The digit chain (and the buffer load) is now dead unless the key is
    // still used elsewhere, for example by the scatter.
    if (andOp->use_empty())
      rewriter.eraseOp(andOp);
    if (shrOp && shrOp->use_empty())
      rewriter.eraseOp(shrOp);
    if (auto shiftSplat = shiftScalar.getDefiningOp<triton::SplatOp>())
      if (shiftSplat->use_empty())
        rewriter.eraseOp(shiftSplat);

    // A folded prefix mask is now dead unless it masks a load or store too.
    if (histMask)
      eraseDeadChain(rewriter, histMask);

    // The key buffer is counted in place, so the load that fed the digit chain
    // is dead unless the key is still used elsewhere (the sort's scatter).
    // Drop it explicitly: a `tt.load` has a read effect, so the greedy
    // rewriter would otherwise keep the whole pointer chain alive.
    if (keySource != keyValue)
      eraseDeadChain(rewriter, keyValue, /*allowLoads=*/true);
  }
  return success();
}

struct GCUTritonFusionPass
    : public mlir::impl::GCUTritonFusionPassBase<GCUTritonFusionPass> {
  using Base::Base;

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();

    // Fold the digit arithmetic into the histogram so the key buffer is read
    // directly and no digit tile is materialized.
    if (failed(fuseHistogramInputs(module, ctx))) {
      signalPassFailure();
      return;
    }

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
      auto fusibleI1Broadcast = collectFusibleI1Broadcast(func);
      func.walk([&](Block *block) { runFuse(block, fusibleI1Broadcast); });
      func.walk([&](Block *block) { fuseReduceOps(block); });
      privatizeSharedConstantInits(func);
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TritonGCUDialect, arith::ArithDialect, math::MathDialect,
                    triton::TritonDialect, math_ext::MathExtDialect>();
  }

private:
  void runFuse(Block *block, llvm::DenseSet<Operation *> &fusibleOp);
  void fuseOps(std::unique_ptr<FusionGroup> fusionGroup);
  bool tryDeepFuse(SmallVector<std::unique_ptr<FusionGroup>> &fusionGroups);
  void splitFusion(SmallVector<std::unique_ptr<FusionGroup>> &fusionGroups);
  llvm::DenseSet<Operation *> collectFusibleI1Broadcast(FuncOp op);
  void fuseReduceOps(Block *block);
  void privatizeSharedConstantInits(triton::FuncOp func);
};

llvm::DenseSet<Operation *>
GCUTritonFusionPass::collectFusibleI1Broadcast(FuncOp funcOp) {
  llvm::DenseSet<Operation *> fusibleOps;
  llvm::DenseSet<Operation *> visitedI1BroadcastOp;
  llvm::SmallVector<Operation *> worklist;
  llvm::DenseSet<Operation *> visited;
  DenseSet<Operation *> broadcastOnDim0;
  DenseSet<Operation *> broadcastOnLastDim;
  auto hasFusibleBroadcastAxis = [&](triton::BroadcastOp op) {
    auto srcType = op.getSrc().getType();
    auto rank = srcType.getRank();
    auto broadcastAxis = getUniqueBroadcastAxis(op);
    if (!broadcastAxis)
      return false;
    if (*broadcastAxis == 0) {
      visitedI1BroadcastOp.insert(op);
      broadcastOnDim0.insert(op);
      return true;
    }
    if (*broadcastAxis == rank - 1) {
      visitedI1BroadcastOp.insert(op);
      broadcastOnLastDim.insert(op);
      return true;
    }
    return false;
  };
  auto isChainOp = [&](Operation *op) {
    return isa<arith::AndIOp, arith::XOrIOp, arith::OrIOp>(op) ||
           (isa<triton::BroadcastOp>(op) &&
            hasFusibleBroadcastAxis(cast<triton::BroadcastOp>(op))) ||
           triton::gcu::isZeroCopyOp(op);
  };
  funcOp->walk([&](triton::BroadcastOp op) {
    auto srcType = op.getSrc().getType();
    auto elementType = srcType.getElementType();
    if (!elementType.isInteger(1) || visitedI1BroadcastOp.contains(op)) {
      return WalkResult::advance();
    }
    if (!hasFusibleBroadcastAxis(op)) {
      return WalkResult::advance();
    }
    worklist.push_back(op);
    visited.insert(op);
    bool fusible = true;
    unsigned producerMinBpe = 8;
    unsigned consumerMinBpe = 8;
    while (!worklist.empty()) {
      auto cur = worklist.pop_back_val();
      if (isa<arith::AndIOp, arith::XOrIOp, arith::OrIOp>(cur) ||
          broadcastOnDim0.contains(cur) || triton::gcu::isZeroCopyOp(cur)) {
        for (auto operand : cur->getOperands()) {
          auto defOp = operand.getDefiningOp();
          if (isChainOp(defOp)) {
            if (!visited.contains(defOp)) {
              visited.insert(defOp);
              worklist.push_back(defOp);
            }
          } else if (isa<arith::CmpIOp, arith::CmpFOp>(defOp)) {
            producerMinBpe = std::min(producerMinBpe,
                                      triton::gcu::getBpe(getElementTypeOrSelf(
                                          defOp->getOperandTypes()[0])));
          } else {
            fusible = false;
          }
        }
      }
      for (auto user : cur->getUsers()) {
        if (isChainOp(user)) {
          if (!visited.contains(user)) {
            visited.insert(user);
            worklist.push_back(user);
          }
        } else if (auto maskedStoreOp =
                       dyn_cast<triton::gcu::MaskedStoreOp>(user)) {
          consumerMinBpe = std::min(
              consumerMinBpe,
              triton::gcu::getBpe(
                  maskedStoreOp.getValue().getType().getElementType()));
        } else if (canFuse(user) &&
                   !cast<RankedTensorType>(user->getResultTypes()[0])
                        .getElementType()
                        .isInteger(1)) {
          consumerMinBpe =
              std::min(consumerMinBpe,
                       triton::gcu::getBpe(
                           cast<RankedTensorType>(user->getResultTypes()[0])
                               .getElementType()));
          continue;
        } else {
          fusible = false;
        }
      }
    }
    if (fusible) {
      for (auto op : broadcastOnDim0) {
        auto srcType = cast<triton::BroadcastOp>(op).getSrc().getType();
        auto elemsPerThread = triton::gcu::getElemsPerThread(srcType);
        auto sizeInBytes =
            std::accumulate(elemsPerThread.begin(), elemsPerThread.end(), 1,
                            std::multiplies<int64_t>()) *
            producerMinBpe;
        auto numOacc = sizeInBytes / kOaccSizeInBytes;
        if (isValidOaccCount(numOacc)) {
          continue;
        }
        fusible = false;
        break;
      }
    }
    if (fusible) {
      for (auto op : broadcastOnLastDim) {
        auto resultType = cast<triton::BroadcastOp>(op).getType();
        auto elemsPerThread = triton::gcu::getElemsPerThread(resultType);
        auto sizeInBytes = elemsPerThread.back() * consumerMinBpe;
        auto numOacc = sizeInBytes / kOaccSizeInBytes;
        if (isValidOaccCount(numOacc)) {
          continue;
        }
        fusible = false;
        break;
      }
    }
    if (fusible) {
      fusibleOps.insert_range(broadcastOnDim0);
      fusibleOps.insert_range(broadcastOnLastDim);
    }
    visited.clear();
    broadcastOnDim0.clear();
    broadcastOnLastDim.clear();
    return WalkResult::advance();
  });
  return fusibleOps;
}

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

void GCUTritonFusionPass::privatizeSharedConstantInits(triton::FuncOp func) {
  auto markPrivateInit = [](Operation *fusionOp) {
    fusionOp->setAttr("ttgcu.private_init", DistinctAttr::create(UnitAttr::get(
                                                fusionOp->getContext())));
  };

  using InitPosition = std::pair<scf::ForOp, unsigned>;
  llvm::DenseMap<Value, SmallVector<InitPosition>> sharedInits;

  func.walk([&](scf::ForOp forOp) {
    for (auto [initIndex, initArg] : llvm::enumerate(forOp.getInitArgs())) {
      auto *definingOp = initArg.getDefiningOp();
      if (!definingOp || !isConstantFusion(definingOp))
        continue;
      if (!canUseAllocaForValue(initArg))
        continue;
      auto type = initArg.getType();
      auto totalNumElems = triton::gcu::getTotalElemsPerThread(type);
      auto totalElemsInBytes =
          totalNumElems * getBpe(cast<RankedTensorType>(type).getElementType());
      if (totalElemsInBytes > kOaccSizeInBytes * kLoopUnrollTimes ||
          totalElemsInBytes < kOaccSizeInBytes)
        continue;
      sharedInits[initArg].push_back({forOp, initIndex});
    }
  });

  for (auto &[initArg, initPositions] : sharedInits) {
    auto *originalOp = initArg.getDefiningOp();
    auto resultIndex = cast<OpResult>(initArg).getResultNumber();
    markPrivateInit(originalOp);
    for (auto &[forOp, initIndex] : initPositions) {
      Operation *cloneOp = originalOp->clone();
      markPrivateInit(cloneOp);
      Block *forBlock = forOp->getBlock();
      forBlock->getOperations().insert(Block::iterator(forOp.getOperation()),
                                       cloneOp);
      forOp.getInitArgsMutable()[initIndex].set(
          cloneOp->getResult(resultIndex));
    }
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

void GCUTritonFusionPass::splitFusion(
    SmallVector<std::unique_ptr<FusionGroup>> &fusionGroups) {
  SmallVector<std::unique_ptr<FusionGroup>> newGroups;
  for (auto &fusionGroup : fusionGroups) {
    if (fusionGroup->operands.size() <= 1) {
      newGroups.emplace_back(std::move(fusionGroup));
      continue;
    }
    DenseMap<Operation *, SmallVector<Operation *>> dependenceInfo;
    DenseMap<Value, SmallVector<Operation *>> operandUsers;
    for (auto op : fusionGroup->ops) {
      for (auto operand : op->getOperands()) {
        auto defOp = operand.getDefiningOp();
        if (defOp && fusionGroup->contains(defOp)) {
          dependenceInfo[op].push_back(defOp);
          dependenceInfo[defOp].push_back(op);
        } else {
          operandUsers[operand].push_back(op);
        }
      }
    }
    for (const auto &users : operandUsers.values()) {
      for (unsigned i = 0; i < users.size(); ++i) {
        for (unsigned j = i + 1; j < users.size(); ++j) {
          dependenceInfo[users[i]].push_back(users[j]);
          dependenceInfo[users[j]].push_back(users[i]);
        }
      }
    }
    SmallVector<SmallVector<Operation *>> splitFusionGroups;
    DenseSet<Operation *> visited;
    for (auto op : fusionGroup->ops) {
      if (visited.contains(op)) {
        continue;
      }
      visited.insert(op);
      SmallVector<Operation *> worklist{op};
      SmallVector<Operation *> group;
      while (!worklist.empty()) {
        auto cur = worklist.pop_back_val();
        group.push_back(cur);
        for (auto dependency : dependenceInfo[cur]) {
          if (visited.insert(dependency).second) {
            worklist.push_back(dependency);
          }
        }
      }
      splitFusionGroups.emplace_back(std::move(group));
    }
    if (splitFusionGroups.size() == 1) {
      newGroups.emplace_back(std::move(fusionGroup));
      continue;
    }
    auto back = fusionGroup->back();
    auto iter = ++back->getIterator();
    auto block = back->getBlock();
    for (auto splitFusionGroup : splitFusionGroups) {
      auto newGroup = std::make_unique<FusionGroup>();
      llvm::sort(splitFusionGroup, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      for (auto op : splitFusionGroup) {
        newGroup->ops.push_back(op);
        op->moveBefore(block, iter);
        auto tensorType = FusionGroup::getRankedTensorType(op);
        auto tensorDesc = getFusionTensorDesc(tensorType);
        if (!llvm::is_contained(newGroup->tensorDescs, tensorDesc)) {
          newGroup->tensorDescs.push_back(tensorDesc);
        }
      }
      newGroup->resolveOperandsAndResults();
      newGroups.emplace_back(std::move(newGroup));
    }
  }
  fusionGroups = std::move(newGroups);
}

void GCUTritonFusionPass::runFuse(Block *block,
                                  llvm::DenseSet<Operation *> &fusibleOp) {
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
    if (canFuse(op) || fusibleOp.erase(op)) {
      if (triton::gcu::isZeroCopyOp(op) && !currentGroup()->isUsedBy(op)) {
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
  splitFusion(fusionGroups);
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
  auto orderedOps = fusionGroup->scheduleFusionRegion();
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
    for (auto op : orderedOps) {
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
} // namespace
