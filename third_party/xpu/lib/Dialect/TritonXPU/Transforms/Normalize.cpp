//===----------------------------------------------------------------------===//
// Pre-vectorization normalization.
//
// Split out of TritonXPUVectorize's prologue (redesign-v2 §4.2 step 1.5). None
// of this is vectorization: it rewrites scalar IR into the shapes the
// vectorizer knows how to match (libdevice calls instead of math dialect ops,
// fused max/min instead of [cmpf, cmpf, ori, select], compares that carry
// their i8 result in an i32 vector lane). Step 1.3 measured that these
// rewrites *create* vectorizability -- `boolfused` has closure=0 before them
// and closure=6 after -- so any analysis that predicts what Vectorize will do
// has to run after them. That is why they have to leave Vectorize before the
// state analysis can move ahead of it (step 1.5c).
//
// Position: immediately before tritonxpu-vectorize, i.e. exactly where the
// code used to run, with one difference -- Vectorize's own `vectorizeTLE` now
// runs *after* these rewrites instead of before them. That can only matter for
// a TLE kernel whose local-buffer store chain contains the NaN-aware max/min
// select pattern: `doMaximumFusion` folds it to arith.maximumf, which *is* in
// ARITH_BINARY_FLOAT_OP, so such a chain would newly vectorize. `vectorizeTLE`
// stays behind because it needs Vectorize.cpp's VOp<T> table; no probe covers
// the TLE path (golden.py has none), so this is stated, not measured.
//===----------------------------------------------------------------------===//

// clang-format off
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Analysis/VectorizabilityAnalysis.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// clang-format on

#define DEBUG_TYPE "tritonxpu-normalize"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUNORMALIZE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

// ext(logic_op(cmp ne(a, 0))) -> ext(locic_op(a))
template <typename LOGIC_OP_TYPE>
struct ConvertI1LogicOpToI8 : public OpRewritePattern<arith::ExtUIOp> {
  using OpRewritePattern<arith::ExtUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
                                PatternRewriter &rewriter) const override {
    auto i1TensorType = dyn_cast<RankedTensorType>(extOp.getIn().getType());
    auto i8TensorType = dyn_cast<RankedTensorType>(extOp.getType());
    if (!i1TensorType || !i8TensorType ||
        i1TensorType.getElementTypeBitWidth() != 1)
      return failure();

    auto logicOp = extOp.getIn().getDefiningOp<LOGIC_OP_TYPE>();
    if (!logicOp)
      return failure();

    auto getI8Source = [&](Value v) -> Value {
      auto cmpi = v.getDefiningOp<arith::CmpIOp>();
      if (!cmpi || cmpi.getPredicate() != arith::CmpIPredicate::ne)
        return nullptr;

      if (!getElementTypeOrSelf(cmpi.getLhs()).isInteger(8))
        return nullptr;

      if (isZeroConst(cmpi.getRhs()))
        return cmpi.getLhs();

      if (isZeroConst(cmpi.getLhs()))
        return cmpi.getRhs();

      return nullptr;
    };

    Value lhsSource = getI8Source(logicOp.getLhs());
    Value rhsSource = getI8Source(logicOp.getRhs());

    if (lhsSource && rhsSource) {
      rewriter.replaceOpWithNewOp<LOGIC_OP_TYPE>(extOp, lhsSource, rhsSource);
      return success();
    }

    return failure();
  }
};

// fold extui
struct BypassCmpIExtUI : public OpRewritePattern<arith::ExtUIOp> {
  using OpRewritePattern<arith::ExtUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
                                PatternRewriter &rewriter) const override {
    if (extOp.getIn().getType() == extOp.getOut().getType()) {
      rewriter.replaceOp(extOp, extOp.getIn());
      return success();
    }
    return failure();
  }
};

struct TritonXPUNormalize
    : public impl::TritonXPUNormalizeBase<TritonXPUNormalize> {
  using impl::TritonXPUNormalizeBase<TritonXPUNormalize>::TritonXPUNormalizeBase;

  void doMaximumFusion(arith::SelectOp selectOp) {
    if (auto orIOp = selectOp.getCondition().getDefiningOp<arith::OrIOp>()) {
      if (orIOp.getResult().hasOneUse()) {
        auto lhs = orIOp.getLhs().getDefiningOp<arith::CmpFOp>();
        auto rhs = orIOp.getRhs().getDefiningOp<arith::CmpFOp>();

        // The null check comes before getPredicate() here; in Vectorize it came
        // after, so an or of anything but two cmpf dereferenced null. Cannot
        // change the emitted code: such a module crashed instead of compiling.
        if (!lhs || !rhs || !lhs.getResult().hasOneUse() ||
            !rhs.getResult().hasOneUse())
          return;

        bool isMax = (lhs.getPredicate() == arith::CmpFPredicate::OGT &&
                      rhs.getPredicate() == arith::CmpFPredicate::UNE) ||
                     (lhs.getPredicate() == arith::CmpFPredicate::UNE &&
                      rhs.getPredicate() == arith::CmpFPredicate::OGT);
        bool isMin = (lhs.getPredicate() == arith::CmpFPredicate::OLT &&
                      rhs.getPredicate() == arith::CmpFPredicate::UNE) ||
                     (lhs.getPredicate() == arith::CmpFPredicate::UNE &&
                      rhs.getPredicate() == arith::CmpFPredicate::OLT);

        OpBuilder builder(selectOp);
        if (isMax) {
          auto newMaxFOp = builder.create<arith::MaximumFOp>(
              selectOp.getLoc(), selectOp.getType(), selectOp.getTrueValue(),
              selectOp.getFalseValue());
          selectOp->replaceAllUsesWith(newMaxFOp);
          selectOp->erase();
          orIOp->erase();
          lhs->erase();
          rhs->erase();
          LLVM_DEBUG(llvm::dbgs()
                     << "[Normalize]: Apply Maximum Fusion Optimization "
                        "For VVMax.\n");
        } else if (isMin) {
          auto newMinFOp = builder.create<arith::MinimumFOp>(
              selectOp.getLoc(), selectOp.getType(), selectOp.getTrueValue(),
              selectOp.getFalseValue());
          selectOp->replaceAllUsesWith(newMinFOp);
          selectOp->erase();
          orIOp->erase();
          lhs->erase();
          rhs->erase();
          LLVM_DEBUG(llvm::dbgs()
                     << "[Normalize]: Apply Minimum Fusion Optimization "
                        "For VVMin.\n");
        }
      }
    }
  }

  void doCompareExtUI8Fusion(arith::ExtUIOp extUIOp) {
    auto inTy = extUIOp.getIn().getType();
    auto outTy = extUIOp.getOut().getType();
    auto inElemTy = getElementTypeOrSelf(inTy);
    auto outElemTy = getElementTypeOrSelf(outTy);
    // Only Vectorize Do Fusion
    auto rowsPerCore = 1;
    if (auto outTensorTy = mlir::dyn_cast<RankedTensorType>(outTy)) {
      auto rank = outTensorTy.getShape().size();
      if (rank > 1) {
        rowsPerCore = mlir::cast<triton::xpu::ClusterLayoutAttr>(
                          outTensorTy.getEncoding())
                          .getSizePerCore()[0];
      }
    }
    unsigned numElems = getTotalElemsPerThread(outTy) / rowsPerCore;
    Type elemTy = getElementTypeOrSelf(outTy);
    auto elemWidth = elemTy.getIntOrFloatBitWidth();
    auto vectorWidth = 512 / elemWidth;
    if (numElems < vectorWidth || numElems % vectorWidth > 0 ||
        !vectorizedTyValid(elemTy))
      return;
    // Fuse CmpFOp(i1) + ExtUIOp(i8) + StoreOp = CmpFOp(i32) + StoreOp
    if (auto cmpFOp = extUIOp.getIn().getDefiningOp<arith::CmpFOp>()) {
      if (inElemTy.isInteger(1) && outElemTy.isInteger(8)) {
        for (auto user : extUIOp.getOut().getUsers()) {
          if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(user)) {
            if (auto outTensorTy = dyn_cast<RankedTensorType>(outTy)) {
              auto lhsTy = cmpFOp.getLhs().getType();
              auto lhsElemTy =
                  getElementTypeOrSelf(getElementTypeOrSelf(lhsTy));
              auto context = storeOp.getContext();
              if (lhsElemTy.isF32()) {
                auto dtype = DtypeAttr::get(context, Dtype::FP32);
                storeOp->setAttr("dtype", dtype);
              } else if (lhsElemTy.isF16()) {
                auto dtype = DtypeAttr::get(context, Dtype::FP16);
                storeOp->setAttr("dtype", dtype);
              } else {
                llvm_unreachable(
                    "CompareExtUI8Fusion only supports FP32 or FP16");
              }
              OpBuilder builder(extUIOp);
              auto newTensorTy = RankedTensorType::get(
                  outTensorTy.getShape(), builder.getIntegerType(32, false),
                  outTensorTy.getEncoding());
              auto newCmpFOp = builder.create<triton::xpu::CmpFOp>(
                  extUIOp.getLoc(), newTensorTy, cmpFOp.getPredicate(),
                  cmpFOp.getLhs(), cmpFOp.getRhs());
              extUIOp.getOut().replaceAllUsesWith(newCmpFOp.getResult());
              extUIOp.erase();
              cmpFOp.erase();
              bf16ToFP32VecOptOff = true;
            }
          }
        }
      }
    }
  }

  void doCompareTruncI8Fusion(arith::TruncIOp truncIOp) {
    auto inTy = truncIOp.getIn().getType();
    auto outTy = truncIOp.getOut().getType();
    auto inElemTy = getElementTypeOrSelf(inTy);
    auto outElemTy = getElementTypeOrSelf(outTy);
    // Only Vectorize Do Fusion
    auto rowsPerCore = 1;
    if (auto outTensorTy = mlir::dyn_cast<RankedTensorType>(outTy)) {
      auto rank = outTensorTy.getShape().size();
      if (rank > 1) {
        rowsPerCore = mlir::cast<triton::xpu::ClusterLayoutAttr>(
                          outTensorTy.getEncoding())
                          .getSizePerCore()[0];
      }
    }
    unsigned numElems = getTotalElemsPerThread(outTy) / rowsPerCore;
    Type elemTy = getElementTypeOrSelf(outTy);
    auto elemWidth = elemTy.getIntOrFloatBitWidth();
    auto vectorWidth = 512 / elemWidth;
    if (numElems < vectorWidth || numElems % vectorWidth > 0 ||
        !vectorizedTyValid(elemTy))
      return;
    // Fuse ExtElemwiseOp(i8) + TruncIOp(i8) + StoreOp = newExtElemwiseOp(i32) +
    // StoreOp
    if (auto extElemwiseOp =
            truncIOp.getIn().getDefiningOp<triton::ExternElementwiseOp>()) {
      if (extElemwiseOp.getSymbol() == "_ZN3xpu5isnanEf" &&
          inElemTy.isInteger(32) && outElemTy.isInteger(8)) {
        for (auto user : truncIOp.getOut().getUsers()) {
          if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(user)) {
            if (auto outTensorTy = dyn_cast<RankedTensorType>(outTy)) {
              auto inTy = extElemwiseOp.getOperands().front().getType();
              auto inElemTy = getElementTypeOrSelf(getElementTypeOrSelf(inTy));
              auto context = storeOp.getContext();
              if (inElemTy.isF32()) {
                auto dtype = DtypeAttr::get(context, Dtype::FP32);
                storeOp->setAttr("dtype", dtype);
              } else if (inElemTy.isF16()) {
                auto dtype = DtypeAttr::get(context, Dtype::FP16);
                storeOp->setAttr("dtype", dtype);
              } else {
                llvm_unreachable(
                    "CompareExtUI8Fusion only supports FP32 or FP16");
              }
              OpBuilder builder(truncIOp);
              auto newTensorTy = RankedTensorType::get(
                  outTensorTy.getShape(), builder.getIntegerType(32),
                  outTensorTy.getEncoding());
              auto newExtElemwiseOp =
                  builder.create<triton::ExternElementwiseOp>(
                      truncIOp.getLoc(), newTensorTy,
                      extElemwiseOp.getOperands().front(),
                      extElemwiseOp.getLibname(), extElemwiseOp.getLibpath(),
                      extElemwiseOp.getSymbol(), extElemwiseOp.getPure());

              truncIOp.getOut().replaceAllUsesWith(
                  newExtElemwiseOp.getResult());
              truncIOp.erase();
              extElemwiseOp.erase();
              bf16ToFP32VecOptOff = true;
            }
          }
        }
      }
    }
  }

  void runOnOperation() override {
    context = &getContext();
    ModuleOp mod = getOperation();

    LLVM_DEBUG(llvm::dbgs() << __FILE__ << " START\n" << mod << "\n");

    // Lower math.erf to xpu libdevice erf (ExternElementwiseOp). The
    // subsequent ExternElementwiseOp vectorization step will rewrite
    // "_ZN3xpu3erfEf" to its vectorized counterpart "_ZN3xpu4verfEDv16_f".
    //
    // FIXME(XPUTC-7517): Generalize this ad-hoc walker into a proper
    // PatternRewriter-based rewrite (e.g. OpRewritePattern<MathOp>) and
    // table-drive the math-dialect-op -> xpu-libdevice-symbol mapping
    // (math.erf -> _ZN3xpu3erfEf, math.tan -> _ZN3xpu4tanfEf, ...). This
    // will let us cover other math dialect ops (atan, isinf, isnan, rsqrt,
    // tanh, ...) without copy-pasting walker blocks here, and will compose
    // cleanly with the existing ExternElementwiseOp vectorization Case.
    mod.walk([&](math::ErfOp erfOp) {
      OpBuilder builder(erfOp);
      auto newExtElemwiseOp = builder.create<triton::ExternElementwiseOp>(
          erfOp.getLoc(), erfOp.getResult().getType(),
          ValueRange{erfOp.getOperand()}, /*libname=*/"", /*libpath=*/"",
          /*symbol=*/"_ZN3xpu3erfEf", /*pure=*/true);
      erfOp.getResult().replaceAllUsesWith(newExtElemwiseOp.getResult());
      erfOp.erase();
    });

    // Maximum Fusion Online
    // [cmpf, cmpf, ori, select] -> [fmax]
    if (maximumFusion) {
      mod.walk([&](arith::SelectOp selectOp) { doMaximumFusion(selectOp); });
    }

    // Compare Fusion
    // [cmpf, extui] -> [vcmpf(castToI8=True)]
    if (this->compareFusion) {
      mod.walk([&](arith::ExtUIOp extUIOp) { doCompareExtUI8Fusion(extUIOp); });
    }
    mod.walk(
        [&](arith::TruncIOp truncIOp) { doCompareTruncI8Fusion(truncIOp); });

    {
      RewritePatternSet patterns(context);
      patterns.add<ConvertI1LogicOpToI8<arith::AndIOp>,
                   ConvertI1LogicOpToI8<arith::OrIOp>,
                   ConvertI1LogicOpToI8<arith::XOrIOp>, BypassCmpIExtUI>(
          context);
      if (failed(applyPatternsGreedily(mod, std::move(patterns))))
        signalPassFailure();
    }

    // Cross-pass signal, replacing the `BF16ToFP32VecOpt` member the two
    // compare fusions used to clear inside Vectorize. Vectorize consumes and
    // *erases* the marker, so it never reaches the emitted IR -- which is what
    // keeps the `boolfused` probe (the only one where a compare fusion fires)
    // byte-identical across the split.
    if (bf16ToFP32VecOptOff) {
      mod->setAttr(kBF16ToFP32VecOptOffAttrName, UnitAttr::get(context));
    }

    LLVM_DEBUG(llvm::dbgs() << __FILE__ << " END\n" << mod << "\n");
  }

private:
  MLIRContext *context;
  bool maximumFusion = true;
  bool bf16ToFP32VecOptOff = false;
};

} // namespace xpu
} // namespace triton
} // namespace mlir
