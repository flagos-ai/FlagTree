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

#include "Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
    auto lhs = op.getLhs();

    Attribute attr;
    if (!lhs.getDefiningOp() || !matchPattern(lhs, m_Constant(&attr))) {
      return failure();
    }

    auto elementAttr = dyn_cast<DenseElementsAttr>(attr);
    if (!elementAttr || !elementAttr.isSplat() ||
        !elementAttr.getElementType().isF32() ||
        elementAttr.getSplatValue<float>() != 1.0) {
      return failure();
    }

    auto addOp = dyn_cast_or_null<arith::AddFOp>(rhs.getDefiningOp());
    if (!addOp) {
      return failure();
    }

    rhs = addOp.getRhs();
    lhs = addOp.getLhs();

    if (!rhs.getDefiningOp() || !matchPattern(rhs, m_Constant(&attr))) {
      return failure();
    }

    elementAttr = dyn_cast<DenseElementsAttr>(attr);
    if (!elementAttr || !elementAttr.isSplat() ||
        !elementAttr.getElementType().isF32() ||
        elementAttr.getSplatValue<float>() != 1.0) {
      return failure();
    }

    auto defOp = lhs.getDefiningOp();
    if (!defOp || !(isa<math::ExpOp>(defOp) || [](auto op) {
          auto o = dyn_cast<triton::ExternElementwiseOp>(op);
          return o.getSymbol() == "__nv_expf";
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
    rewriter.replaceOp(op, sigmoidOp);
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
      if (o.getSymbol() != "__nv_logf") {
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
          return o.getSymbol() == "__nv_expf";
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
    rewriter.replaceOp(op, softPlusOp);
    return success();
  }
};

struct GCUCombineOps : public impl::GCUCombineOpsBase<GCUCombineOps> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    auto gpuModuleOp = getOperation();

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

    if (applyPatternsGreedily(gpuModuleOp, std::move(patterns)).failed())
      signalPassFailure();
  }
};
} // namespace
