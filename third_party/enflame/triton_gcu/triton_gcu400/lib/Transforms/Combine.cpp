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

#include <string>
#include <type_traits>
#include <utility>

#include "Conversion/TritonToGCU/Constants.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_GCUCOMBINEOPS
#include "Transforms/Passes.h.inc"
} // namespace mlir

namespace {

const char *const symbol0 = R"(_{0}mix_{1})";
const char *const symbol1 = R"(_{0})";

// `tle.gpu.memory_space(...)` attaches `tt.memory_space` to the op that defines
// the hinted value. The combine patterns below replace such an op with a single
// extern_elementwise, so the hint must travel with the replacement: it is
// consumed later by gcu-tle-assign-memory-space, which runs after
// gcu-triton-fusion and would otherwise never see it (the original op is gone).
void copyMemorySpaceHint(Operation *from, Operation *to) {
  if (Attribute hint = from->getAttr(kTleHint))
    to->setAttr(kTleHint, hint);
}

template <typename Op>
mlir::LogicalResult matchMixedPrecisionPattern0(Value v0, Value v1,
                                                SmallVector<Value> &args,
                                                std::string &symbol) {
  auto defOp = v0.getDefiningOp();
  if (!isa<TensorType>(v0.getType()) || !defOp) {
    return failure();
  }
  auto elementTy0 = cast<mlir::TensorType>(v0.getType()).getElementType();
  bool match = false;
  std::string outputTy, mixedPrecisionTy;
  if constexpr (std::is_same_v<Op, arith::AddFOp> ||
                std::is_same_v<Op, arith::MulFOp>) {
    if (elementTy0.isF32()) {
      outputTy = "fp32";
      if (isa<arith::ExtFOp>(defOp)) {
        auto elementTy1 = cast<mlir::TensorType>(defOp->getOperand(0).getType())
                              .getElementType();
        if (elementTy1.isF16() || elementTy1.isBF16()) {
          match = true;
          mixedPrecisionTy = elementTy1.isF16() ? "fp16" : "bf16";
        }
      } else if (isa<arith::UIToFPOp, arith::SIToFPOp>(defOp)) {
        auto elementTy1 = cast<mlir::TensorType>(defOp->getOperand(0).getType())
                              .getElementType();
        if (elementTy1.isInteger(8) || elementTy1.isInteger(32)) {
          match = true;
          mixedPrecisionTy =
              (isa<arith::UIToFPOp>(defOp) ? "ui" : "si") +
              std::to_string(cast<IntegerType>(elementTy1).getWidth());
        }
      }
    } else if (elementTy0.isBF16() || elementTy0.isF16()) {
      if (isa<arith::UIToFPOp, arith::SIToFPOp>(defOp) &&
          cast<mlir::TensorType>(defOp->getOperand(0).getType())
              .getElementType()
              .isInteger(8)) {
        match = true;
        outputTy = elementTy0.isBF16() ? "bf16" : "fp16";
        mixedPrecisionTy = isa<arith::UIToFPOp>(defOp) ? "ui8" : "si8";
      }
    }
  }

  if constexpr (std::is_same_v<Op, arith::AddIOp> ||
                std::is_same_v<Op, arith::MulIOp>) {
    if (elementTy0.isInteger(32) &&
        isa<arith::ExtUIOp, arith::ExtSIOp>(defOp) &&
        cast<mlir::TensorType>(defOp->getOperand(0).getType())
            .getElementType()
            .isInteger(8)) {
      match = true;
      outputTy = "si32";
      mixedPrecisionTy = isa<arith::ExtUIOp>(defOp) ? "ui8" : "si8";
    }
  }

  if (match) {
    args.push_back(defOp->getOperand(0));
    args.push_back(v1);
    symbol = llvm::formatv(symbol0, outputTy, mixedPrecisionTy);
    return success();
  }
  return failure();
}

template <typename Op,
          typename = std::enable_if_t<std::is_same_v<Op, arith::AddIOp> ||
                                      std::is_same_v<Op, arith::MulIOp> ||
                                      std::is_same_v<Op, arith::MulFOp>>>
mlir::LogicalResult matchMixedPrecisionPattern1(Value v0, Value v1,
                                                SmallVector<Value> &args,
                                                std::string &symbol) {
  auto defOp0 = v0.getDefiningOp();
  auto defOp1 = v1.getDefiningOp();
  if (!isa<TensorType>(v0.getType()) || !defOp0 || !defOp1) {
    return failure();
  }
  auto elementTy0 = cast<mlir::TensorType>(v0.getType()).getElementType();
  if constexpr (std::is_same_v<Op, arith::AddIOp> ||
                std::is_same_v<Op, arith::MulIOp>) {
    if (elementTy0.isInteger(32) &&
        ((isa<arith::ExtUIOp>(defOp0) && isa<arith::ExtUIOp>(defOp1)) ||
         (isa<arith::ExtSIOp>(defOp0) && isa<arith::ExtSIOp>(defOp1))) &&
        defOp0->getOperand(0).getType() == defOp1->getOperand(0).getType() &&
        cast<mlir::TensorType>(defOp0->getOperand(0).getType())
            .getElementType()
            .isInteger(8)) {
      args.push_back(defOp0->getOperand(0));
      args.push_back(defOp1->getOperand(0));
      symbol = llvm::formatv(symbol0, "si32",
                             isa<arith::ExtUIOp>(defOp0) ? "ui8" : "si8");
      return success();
    }
  }
  if constexpr (std::is_same_v<Op, arith::MulFOp>) {
    if (elementTy0.isF32() && isa<arith::ExtFOp>(defOp0) &&
        isa<arith::ExtFOp>(defOp1) &&
        defOp0->getOperand(0).getType() == defOp1->getOperand(0).getType() &&
        (cast<mlir::TensorType>(defOp0->getOperand(0).getType())
             .getElementType()
             .isF16() ||
         cast<mlir::TensorType>(defOp0->getOperand(0).getType())
             .getElementType()
             .isBF16())) {
      args.push_back(defOp0->getOperand(0));
      args.push_back(defOp1->getOperand(0));
      symbol =
          llvm::formatv(symbol0, "fp32",
                        cast<mlir::TensorType>(defOp0->getOperand(0).getType())
                                .getElementType()
                                .isF16()
                            ? "fp16"
                            : "bf16");
      return success();
    }
  }
  return failure();
}

template <typename Op>
mlir::LogicalResult matchCombinePattern(Value v0, Value v1,
                                        SmallVector<Value> &args,
                                        std::string &symbol) {
  auto defOp = v0.getDefiningOp();
  if (!isa<TensorType>(v0.getType()) || !defOp) {
    return failure();
  }
  auto elementTy0 = cast<mlir::TensorType>(v0.getType()).getElementType();

  if constexpr (std::is_same_v<Op, arith::AddFOp> ||
                std::is_same_v<Op, arith::SubFOp>) {
    if (isa<arith::MulFOp>(defOp)) {
      auto operand0 = defOp->getOperand(0);
      auto operand1 = defOp->getOperand(1);
      if (succeeded(matchMixedPrecisionPattern1<arith::MulFOp>(
              operand0, operand1, args, symbol))) {
        symbol = symbol.substr(symbol.rfind("_"));
        args.push_back(v1);
        return success();
      } else if (succeeded(matchMixedPrecisionPattern0<arith::MulFOp>(
                     operand0, operand1, args, symbol)) ||
                 succeeded(matchMixedPrecisionPattern0<arith::MulFOp>(
                     operand1, operand0, args, symbol))) {
        if (llvm::any_of(args, [](auto arg) {
              return getElementTypeOrSelf(arg).isInteger(32);
            })) {
          args[0] = operand0;
          args[1] = operand1;
          args.push_back(v1);
          symbol = llvm::formatv(symbol1, "fp32");
          return success();
        }
        args.push_back(v1);
        return success();
      } else if (elementTy0.isF32()) {
        args.push_back(operand0);
        args.push_back(operand1);
        args.push_back(v1);
        symbol = llvm::formatv(symbol1, "fp32");
        return success();
      }
    }
  }

  if constexpr (std::is_same_v<Op, arith::AddIOp> ||
                std::is_same_v<Op, arith::SubIOp>) {
    if (isa<arith::MulIOp>(defOp)) {
      auto operand0 = defOp->getOperand(0);
      auto operand1 = defOp->getOperand(1);
      if (succeeded(matchMixedPrecisionPattern1<arith::MulIOp>(
              operand0, operand1, args, symbol))) {
        symbol = symbol.substr(symbol.rfind("_"));
        args.push_back(v1);
        return success();
      } else if (succeeded(matchMixedPrecisionPattern0<arith::MulIOp>(
                     operand0, operand1, args, symbol)) ||
                 succeeded(matchMixedPrecisionPattern0<arith::MulIOp>(
                     operand1, operand0, args, symbol))) {
        args.push_back(v1);
        return success();
      }
    }
  }
  return failure();
}

template <typename Op,
          typename = std::enable_if_t<std::is_same_v<Op, arith::AddFOp> ||
                                      std::is_same_v<Op, arith::AddIOp>>>
class CombineMACPattern : public OpRewritePattern<Op> {
public:
  explicit CombineMACPattern(MLIRContext *context)
      : OpRewritePattern<Op>(context, 10) {}
  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto loc = op->getLoc();
    SmallVector<Value> args;
    std::string symbol = "";
    if (succeeded(matchCombinePattern<Op>(lhs, rhs, args, symbol)) ||
        succeeded(matchCombinePattern<Op>(rhs, lhs, args, symbol))) {
      symbol = "__gcu_mac" + symbol;
      auto externElementwiseOp =
          rewriter.create<mlir::triton::ExternElementwiseOp>(
              loc, op->getResult(0).getType(), args,
              /*libname*/ rewriter.getStringAttr(""),
              /*libpath*/ rewriter.getStringAttr(""),
              /*symbol*/ rewriter.getStringAttr(symbol),
              /*pure*/ rewriter.getBoolAttr(true));
      copyMemorySpaceHint(op.getOperation(),
                          externElementwiseOp.getOperation());
      rewriter.replaceOp(op, externElementwiseOp);
      return success();
    }
    return failure();
  }
};

template <typename Op,
          typename = std::enable_if_t<std::is_same_v<Op, arith::SubFOp> ||
                                      std::is_same_v<Op, arith::SubIOp>>>
class CombineIMASMASPattern : public OpRewritePattern<Op> {
public:
  explicit CombineIMASMASPattern(MLIRContext *context)
      : OpRewritePattern<Op>(context, 10) {}
  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto loc = op->getLoc();
    SmallVector<Value> args;
    std::string symbol = "";
    if (succeeded(matchCombinePattern<Op>(lhs, rhs, args, symbol))) {
      symbol = "__gcu_imas" + symbol;
    } else if (succeeded(matchCombinePattern<Op>(rhs, lhs, args, symbol))) {
      symbol = "__gcu_mas" + symbol;
    } else {
      return failure();
    }
    auto externElementwiseOp =
        rewriter.create<mlir::triton::ExternElementwiseOp>(
            loc, op->getResult(0).getType(), args,
            /*libname*/ rewriter.getStringAttr(""),
            /*libpath*/ rewriter.getStringAttr(""),
            /*symbol*/ rewriter.getStringAttr(symbol),
            /*pure*/ rewriter.getBoolAttr(true));
    copyMemorySpaceHint(op.getOperation(), externElementwiseOp.getOperation());
    rewriter.replaceOp(op, externElementwiseOp);
    return success();
  }
};

template <typename Op,
          typename = std::enable_if_t<std::is_same_v<Op, arith::AddFOp> ||
                                      std::is_same_v<Op, arith::MulFOp> ||
                                      std::is_same_v<Op, arith::AddIOp> ||
                                      std::is_same_v<Op, arith::MulIOp>>>
class CombineMixedPrecisionPattern : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto loc = op->getLoc();
    SmallVector<Value> args;
    std::string symbol = "";
    if constexpr (std::is_same_v<Op, arith::AddIOp>) {
      if (succeeded(matchMixedPrecisionPattern1<Op>(lhs, rhs, args, symbol))) {
        symbol = "__gcu_wadd" + symbol;
      } else if (succeeded(
                     matchMixedPrecisionPattern0<Op>(lhs, rhs, args, symbol)) ||
                 succeeded(
                     matchMixedPrecisionPattern0<Op>(rhs, lhs, args, symbol))) {
        symbol = "__gcu_add" + symbol;
      } else {
        return failure();
      }
    }

    if constexpr (std::is_same_v<Op, arith::MulIOp>) {
      if (succeeded(matchMixedPrecisionPattern1<Op>(lhs, rhs, args, symbol))) {
        symbol = "__gcu_wmul" + symbol;
      } else if (succeeded(
                     matchMixedPrecisionPattern0<Op>(lhs, rhs, args, symbol)) ||
                 succeeded(
                     matchMixedPrecisionPattern0<Op>(rhs, lhs, args, symbol))) {
        symbol = "__gcu_mul" + symbol;
      } else {
        return failure();
      }
    }

    if constexpr (std::is_same_v<Op, arith::AddFOp>) {
      if (succeeded(matchMixedPrecisionPattern0<Op>(lhs, rhs, args, symbol)) ||
          succeeded(matchMixedPrecisionPattern0<Op>(rhs, lhs, args, symbol))) {
        symbol = "__gcu_add" + symbol;
      } else {
        return failure();
      }
    }

    if constexpr (std::is_same_v<Op, arith::MulFOp>) {
      if (succeeded(matchMixedPrecisionPattern0<Op>(lhs, rhs, args, symbol)) ||
          succeeded(matchMixedPrecisionPattern0<Op>(rhs, lhs, args, symbol))) {
        symbol = "__gcu_mul" + symbol;
      } else {
        return failure();
      }
    }

    auto externElementwiseOp =
        rewriter.create<mlir::triton::ExternElementwiseOp>(
            loc, op->getResult(0).getType(), args,
            /*libname*/ rewriter.getStringAttr(""),
            /*libpath*/ rewriter.getStringAttr(""),
            /*symbol*/ rewriter.getStringAttr(symbol),
            /*pure*/ rewriter.getBoolAttr(true));
    copyMemorySpaceHint(op.getOperation(), externElementwiseOp.getOperation());
    rewriter.replaceOp(op, externElementwiseOp);
    return success();
  }
};

class CombineWMULUSPattern : public OpRewritePattern<arith::MulIOp> {
public:
  explicit CombineWMULUSPattern(MLIRContext *context)
      : OpRewritePattern<arith::MulIOp>(context, 20) {}
  mlir::LogicalResult
  matchAndRewrite(arith::MulIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy = dyn_cast<TensorType>(op.getResult().getType());
    auto lhsDefOp = op.getLhs().getDefiningOp();
    auto rhsDefOp = op.getRhs().getDefiningOp();
    auto loc = op.getLoc();
    SmallVector<Value> args;
    auto match = [&args](Operation *defOp0, Operation *defOp1) {
      if (isa<arith::ExtUIOp>(defOp0) &&
          cast<mlir::TensorType>(defOp0->getOperandTypes()[0])
              .getElementType()
              .isInteger(8) &&
          isa<arith::ExtSIOp>(defOp1) &&
          cast<mlir::TensorType>(defOp1->getOperandTypes()[0])
              .getElementType()
              .isInteger(8)) {
        args.push_back(defOp0->getOperand(0));
        args.push_back(defOp1->getOperand(0));
        return true;
      }
      return false;
    };
    if (resultTy && resultTy.getElementType().isInteger(32) && lhsDefOp &&
        rhsDefOp &&
        (match(lhsDefOp, rhsDefOp) || (match(rhsDefOp, lhsDefOp)))) {
      auto externElementwiseOp =
          rewriter.create<mlir::triton::ExternElementwiseOp>(
              loc, op->getResult(0).getType(), args,
              /*libname*/ rewriter.getStringAttr(""),
              /*libpath*/ rewriter.getStringAttr(""),
              /*symbol*/ rewriter.getStringAttr("__gcu_wmulus_si32mix_ui8"),
              /*pure*/ rewriter.getBoolAttr(true));
      copyMemorySpaceHint(op.getOperation(),
                          externElementwiseOp.getOperation());
      rewriter.replaceOp(op, externElementwiseOp);
      return success();
    }
    return failure();
  }
};

class CombineSigmoidPattern : public OpRewritePattern<arith::DivFOp> {
public:
  explicit CombineSigmoidPattern(MLIRContext *context)
      : OpRewritePattern<arith::DivFOp>(context, 30) {}
  mlir::LogicalResult
  matchAndRewrite(arith::DivFOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultTy = dyn_cast<TensorType>(op.getResult().getType());
    if (!resultTy || !resultTy.getElementType().isF32()) {
      return failure();
    }
    auto rhs = op.getRhs();
    auto addOp = dyn_cast_or_null<arith::AddFOp>(rhs.getDefiningOp());
    if (!addOp) {
      return failure();
    }

    rhs = addOp.getRhs();
    auto lhs = addOp.getLhs();

    Attribute attr;
    if (!rhs.getDefiningOp() || !matchPattern(rhs, m_Constant(&attr))) {
      return failure();
    }

    auto elementAttr = dyn_cast<DenseElementsAttr>(attr);
    if (!elementAttr || !elementAttr.isSplat() ||
        !elementAttr.getElementType().isF32() ||
        elementAttr.getSplatValue<float>() != 1.0) {
      return failure();
    }

    auto defOp = lhs.getDefiningOp();
    if (!defOp || !(isa<math::ExpOp>(defOp) || [](auto op) {
          auto o = dyn_cast<triton::ExternElementwiseOp>(op);
          return o && o.getSymbol() == "__ef_expf";
        }(defOp))) {
      return failure();
    }

    auto subOp =
        dyn_cast_or_null<arith::SubFOp>(defOp->getOperand(0).getDefiningOp());

    if (!subOp) {
      return failure();
    }

    rhs = subOp.getRhs();
    lhs = subOp.getLhs();

    if (!lhs.getDefiningOp() || !matchPattern(lhs, m_Constant(&attr))) {
      return failure();
    }

    elementAttr = dyn_cast<DenseElementsAttr>(attr);
    if (!elementAttr || !elementAttr.isSplat() ||
        !elementAttr.getElementType().isF32() ||
        !elementAttr.getSplatValue<APFloat>().isZero()) {
      return failure();
    }
    auto sigmoidOp = rewriter.create<mlir::triton::ExternElementwiseOp>(
        loc, resultTy, ValueRange{rhs},
        /*libname*/ rewriter.getStringAttr(""),
        /*libpath*/ rewriter.getStringAttr(""),
        /*symbol*/ rewriter.getStringAttr("__gcu_sigmoid"),
        /*pure*/ rewriter.getBoolAttr(true));
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getLhs(),
                                               sigmoidOp.getResult());
    return success();
  }
};

template <typename Op,
          typename =
              std::enable_if_t<std::is_same_v<Op, math::LogOp> ||
                               std::is_same_v<Op, triton::ExternElementwiseOp>>>
class CombineSoftplusPattern : public OpRewritePattern<Op> {
public:
  explicit CombineSoftplusPattern(MLIRContext *context)
      : OpRewritePattern<Op>(context, 30) {}
  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    if constexpr (std::is_same_v<Op, triton::ExternElementwiseOp>) {
      auto o = cast<triton::ExternElementwiseOp>(op);
      if (o.getSymbol() != "__ef_logf") {
        return failure();
      }
    }

    auto loc = op->getLoc();
    auto resultTy = dyn_cast<TensorType>(op->getResult(0).getType());
    if (!resultTy || !resultTy.getElementType().isF32()) {
      return failure();
    }

    auto addOp =
        dyn_cast_or_null<arith::AddFOp>(op->getOperand(0).getDefiningOp());
    if (!addOp) {
      return failure();
    }

    auto lhs = addOp.getLhs();
    auto defOp = lhs.getDefiningOp();
    if (!defOp || !(isa<math::ExpOp>(defOp) || [](auto op) {
          auto o = dyn_cast<triton::ExternElementwiseOp>(op);
          return o && o.getSymbol() == "__ef_expf";
        }(defOp))) {
      return failure();
    }
    auto operand = defOp->getOperand(0);
    auto rhs = addOp.getRhs();

    Attribute attr;
    if (!rhs.getDefiningOp() || !matchPattern(rhs, m_Constant(&attr))) {
      return failure();
    }

    auto elementAttr = dyn_cast<DenseElementsAttr>(attr);
    if (!elementAttr || !elementAttr.isSplat() ||
        !elementAttr.getElementType().isF32() ||
        elementAttr.getSplatValue<float>() != 1.0) {
      return failure();
    }
    auto softPlusOp = rewriter.create<mlir::triton::ExternElementwiseOp>(
        loc, resultTy, ValueRange{operand},
        /*libname*/ rewriter.getStringAttr(""),
        /*libpath*/ rewriter.getStringAttr(""),
        /*symbol*/ rewriter.getStringAttr("__gcu_softplus"),
        /*pure*/ rewriter.getBoolAttr(true));
    copyMemorySpaceHint(op.getOperation(), softPlusOp.getOperation());
    rewriter.replaceOp(op, softPlusOp);
    return success();
  }
};

// DotOp + CvtOp + StoreOp -> DotOp + StoreOp
template <typename CvtOp>
class ElideCvtBeforeAccReuseStore : public OpRewritePattern<CvtOp> {
public:
  using OpRewritePattern<CvtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CvtOp cvtOp,
                                PatternRewriter &rewriter) const override {
    if (!cvtOp.getResult().hasOneUse())
      return failure();

    auto storeOp =
        dyn_cast<triton::gcu::StoreOp>(*cvtOp.getResult().getUsers().begin());
    if (!storeOp)
      return failure();

    // Check if the dot op is an acc reuse candidate
    Value input = cvtOp->getOperand(0);
    auto dotOp = getDotOp(input);
    if (!dotOp)
      return failure();

    if (!dotOp.getResult().hasOneUse())
      return failure();

    // Update oacc store mode
    const char *const kAccStore = "acc_store";
    if (!dotOp->hasAttr(kAccStore))
      return failure();

    if (dotOp->hasAttr("acc_reuse_candidate")) {
      if (mlir::cast<StringAttr>(dotOp->getAttr("acc_reuse_candidate"))
              .getValue() != "acc_reuse_oacc")
        return failure();
    } else if (dotOp->getBlock() != cvtOp->getBlock()) {
      return failure();
    }

    dotOp->setAttr(kAccStore,
                   StringAttr::get(dotOp.getContext(), "cvt_global"));

    rewriter.replaceOp(cvtOp, input);
    return success();
  }

private:
  static triton::DotOp getDotOp(Value val) {
    auto *defOp = val.getDefiningOp();
    if (!defOp)
      return nullptr;
    if (auto dotOp = dyn_cast<triton::DotOp>(defOp))
      return dotOp;

    if (auto forResult = dyn_cast<OpResult>(val)) {
      if (auto forOp = dyn_cast<scf::ForOp>(forResult.getOwner())) {
        unsigned resultIdx = forResult.getResultNumber();
        auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        Value yieldedVal = yieldOp.getOperand(resultIdx);
        if (auto dotOp = dyn_cast<triton::DotOp>(yieldedVal.getDefiningOp()))
          return dotOp;
      }
    }

    return nullptr;
  }
};

class FoldCmpMakeRange : public OpRewritePattern<arith::CmpIOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    auto rhs = op.getRhs();
    auto lhs = op.getLhs();
    auto makeRangeOp = lhs.getDefiningOp<triton::MakeRangeOp>();
    if (!makeRangeOp)
      return failure();

    auto constantOp = rhs.getDefiningOp<arith::ConstantOp>();
    if (!constantOp)
      return failure();

    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
    if (!denseAttr || !denseAttr.isSplat())
      return failure();

    auto start = makeRangeOp.getStartAttr().getInt();
    auto end = makeRangeOp.getEndAttr().getInt();
    auto constValue = denseAttr.getSplatValue<IntegerAttr>().getInt();

    bool allTrue = false;
    switch (op.getPredicate()) {
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      allTrue = (end <= constValue);
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      allTrue = (end - 1 <= constValue);
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      allTrue = (start > constValue);
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      allTrue = (start >= constValue);
      break;
    default:
      return failure();
    }

    if (!allTrue)
      return failure();

    auto resultType = cast<ShapedType>(op.getResult().getType());
    auto trueAttr =
        DenseElementsAttr::get(resultType, rewriter.getBoolAttr(true));
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, trueAttr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Integer division / remainder canonicalization.
//
// Triton lowers `//` to arith.divsi / arith.divui (C semantics: truncation
// toward zero).  The GCU400 lowering turns both into libcalls
// (tops_floordivsi / tops_floordivui), i.e. a real function call per element
// block.  Inductor emits them for bounds checks such as `(x // 2048) >= 18000`
// once per masked load/store; folding the constant compare here -- before
// fusion and before the libcall lowering -- removes both the call and the mask
// materialization around it.
//
// The only rewrite kept is an exact integer identity:
//
//   sdiv(x, C) cmp K   ->  x cmp K*C or x cmp (K+1)*C             (C > 0)
//   udiv(x, C) cmp K   ->  same, unsigned                         (C > 0)
//
// with one widening cast allowed between the division and the compare:
//
//   extsi(sdiv(x, C)) cmp K  ->  x cmp K*C, in the narrow type    (C > 0)
//   extui(udiv(x, C)) cmp K  ->  same, unsigned                   (C > 0)
//
// Inductor emits exactly that shape for `(x // 2048) >= 18000` (i32 division,
// extsi to i64 because the threshold constant is i64), and without peeling the
// cast the pattern never matched, which left the floordiv libcall alive.  Only
// one cast layer is peeled and it must agree in signedness with the compare;
// the rescaled bound additionally has to be representable in the narrow type,
// otherwise the fold is refused.
//
// It needs no non-negativity proof: for a positive divisor C,
// trunc(x/C) >= K <=> x >= K*C holds for every integer x and K (this is LLVM
// InstCombine's foldICmpDiv).  Overflow of a folded constant is rejected, and
// mul is treated as non-wrapping here, which matches the index arithmetic
// Triton emits (a wrap would already leave the address out of range).
//
// Every other constant-div/rem shape is deliberately NOT folded.
// Each extra index rewrite risks disturbing the affine/contiguity structure
// the address analysis relies on, so (x // a) // b, (x % c) % c and the
// power-of-two shifts/ands are all left alone.
// On the power-of-two forms the shift also hides a normalized index
// (observed as a swiglu (N, 2H) read-as-(N, H) regression), and they would
// additionally need a non-negativity proof.
//===----------------------------------------------------------------------===//

/// Returns the constant integer value of |v| when it is a scalar integer
/// constant or a splat integer tensor constant.
static std::optional<APInt> getIntegerConstant(Value v) {
  auto constantOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return std::nullopt;
  if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
    auto attr = dyn_cast<IntegerAttr>(constantOp.getValue());
    if (!attr || attr.getType() != intTy)
      return std::nullopt;
    return attr.getValue();
  }
  auto shapedTy = dyn_cast<ShapedType>(v.getType());
  if (!shapedTy || !shapedTy.getElementType().isInteger())
    return std::nullopt;
  auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
  if (!denseAttr || !denseAttr.isSplat())
    return std::nullopt;
  return denseAttr.getSplatValue<APInt>();
}

/// Creates an integer constant of type |ty| holding the splat value |value|.
static Value createIntegerConstant(PatternRewriter &rewriter, Location loc,
                                   Type ty, const APInt &value) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return rewriter.create<arith::ConstantOp>(loc,
                                              IntegerAttr::get(intTy, value));
  auto shapedTy = cast<ShapedType>(ty);
  auto elemTy = cast<IntegerType>(shapedTy.getElementType());
  auto attr = DenseElementsAttr::get(shapedTy, IntegerAttr::get(elemTy, value));
  return rewriter.create<arith::ConstantOp>(loc, attr);
}

/// `cmp(div(x, C), K)` -> `cmp(x, K')`, with C a positive constant and K a
/// constant.  Removes the division (a libcall on GCU400) and often the i64
/// promotion around the comparison.
///
/// The compare operand may also be one widening cast above the division
/// (`divsi(i32) -> extsi -> cmpi sge(i64)`, which is what Inductor emits for
/// `(x // 2048) >= 18000`).  The cast is peeled when its signedness matches the
/// compare, and the bound is then rescaled in the narrow type.  That is exact
/// only when the rescaled bound is representable in the narrow type, since the
/// compare can only ever see narrow values widened; anything else bails out and
/// leaves the division in place.
template <typename DivOp, bool IsSigned>
class FoldCmpDivByConst : public OpRewritePattern<arith::CmpIOp> {
public:
  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  /// Element bit width of an integer scalar or tensor type.
  static unsigned elemBitWidth(Type ty) {
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return intTy.getWidth();
    return cast<ShapedType>(ty).getElementType().getIntOrFloatBitWidth();
  }

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    Value divResult = op.getLhs();
    bool peeledCast = false;
    if (auto ext = divResult.getDefiningOp<arith::ExtSIOp>()) {
      if (!IsSigned)
        return failure();
      divResult = ext.getIn();
      peeledCast = true;
    } else if (auto ext = divResult.getDefiningOp<arith::ExtUIOp>()) {
      if (IsSigned)
        return failure();
      divResult = ext.getIn();
      peeledCast = true;
    }
    auto divOp = divResult.getDefiningOp<DivOp>();
    if (!divOp)
      return failure();
    auto divisor = getIntegerConstant(divOp.getRhs());
    auto bound = getIntegerConstant(op.getRhs());
    if (!divisor || !bound || divisor->isZero())
      return failure();
    if (IsSigned && divisor->isNegative())
      return failure();

    arith::CmpIPredicate pred = op.getPredicate();
    bool predIsSigned = pred == arith::CmpIPredicate::slt ||
                        pred == arith::CmpIPredicate::sle ||
                        pred == arith::CmpIPredicate::sgt ||
                        pred == arith::CmpIPredicate::sge;
    if (predIsSigned != IsSigned)
      return failure();

    // trunc(x/C) <  K  <=>  x <  K*C
    // trunc(x/C) <= K  <=>  x <  (K+1)*C
    // trunc(x/C) >  K  <=>  x >= (K+1)*C
    // trunc(x/C) >= K  <=>  x >= K*C
    bool addOne = false;
    arith::CmpIPredicate newPred = pred;
    switch (pred) {
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      break;
    case arith::CmpIPredicate::sle:
      newPred = arith::CmpIPredicate::slt;
      addOne = true;
      break;
    case arith::CmpIPredicate::ule:
      newPred = arith::CmpIPredicate::ult;
      addOne = true;
      break;
    case arith::CmpIPredicate::sgt:
      newPred = arith::CmpIPredicate::sge;
      addOne = true;
      break;
    case arith::CmpIPredicate::ugt:
      newPred = arith::CmpIPredicate::uge;
      addOne = true;
      break;
    default:
      return failure();
    }

    bool overflow = false;
    APInt threshold = *bound;
    // A peeled cast leaves the divisor in the narrow (division) type while the
    // bound and the compare live in the wide one; rescale at the wide width.
    // The divisor is known positive when signed, so either extension agrees.
    APInt scaled = IsSigned ? divisor->sextOrTrunc(threshold.getBitWidth())
                            : divisor->zextOrTrunc(threshold.getBitWidth());
    if (addOne) {
      APInt one(threshold.getBitWidth(), 1);
      threshold = IsSigned ? threshold.sadd_ov(one, overflow)
                           : threshold.uadd_ov(one, overflow);
      if (overflow)
        return failure();
    }
    APInt newBound = IsSigned ? threshold.smul_ov(scaled, overflow)
                              : threshold.umul_ov(scaled, overflow);
    if (overflow)
      return failure();

    // Rescale in the narrow (division) type when a widening cast was peeled.
    // A compare on widened values cannot see anything outside the narrow type's
    // range, so the fold is exact iff that range holds the new bound.
    Type cmpTy = op.getRhs().getType();
    if (peeledCast) {
      unsigned narrowBits = elemBitWidth(divResult.getType());
      if (IsSigned ? !newBound.isSignedIntN(narrowBits)
                   : !newBound.isIntN(narrowBits))
        return failure();
      cmpTy = divResult.getType();
      newBound = newBound.trunc(narrowBits);
    }

    Value newCst = createIntegerConstant(rewriter, op.getLoc(), cmpTy, newBound);
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, newPred, divOp.getLhs(),
                                               newCst);
    return success();
  }
};

struct GCUCombineOps : public impl::GCUCombineOpsBase<GCUCombineOps> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    auto gpuModuleOp = getOperation();

    // These conversion ops can be elided before acc reuse store. If added,
    // useMatrixStore must be updated and MATRIX_STORE_IMPL must be updated.
    // arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp, arith::UIToFPOp,
    // arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp
    patterns.add<ElideCvtBeforeAccReuseStore<arith::TruncFOp>,
                 ElideCvtBeforeAccReuseStore<arith::TruncIOp>>(context);
    patterns.add<CombineSigmoidPattern, CombineSoftplusPattern<math::LogOp>,
                 CombineSoftplusPattern<triton::ExternElementwiseOp>>(context);
    patterns.add<CombineMACPattern<arith::AddFOp>,
                 CombineIMASMASPattern<arith::SubFOp>>(context);
    if (enableIntegerMixedPrecision) {
      patterns.add<CombineMACPattern<arith::AddIOp>,
                   CombineIMASMASPattern<arith::SubIOp>>(context);
    }

    patterns
        .add<CombineWMULUSPattern, CombineMixedPrecisionPattern<arith::AddFOp>,
             CombineMixedPrecisionPattern<arith::MulFOp>>(context);
    if (enableIntegerMixedPrecision) {
      patterns.add<CombineWMULUSPattern,
                   CombineMixedPrecisionPattern<arith::AddIOp>,
                   CombineMixedPrecisionPattern<arith::MulIOp>>(context);
    }
    patterns.add<FoldCmpMakeRange>(context);

    // Mostly used for Inductor 2D split cases, where inputs are split and
    // treated as 1D. These patterns fold index comparisons for such scenarios.
    patterns.add<FoldCmpDivByConst<arith::DivSIOp, true>,
                 FoldCmpDivByConst<arith::DivUIOp, false>>(context);

    if (applyPatternsGreedily(gpuModuleOp, std::move(patterns)).failed())
      signalPassFailure();
  }
};
} // namespace
