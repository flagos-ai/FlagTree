//===----------------------------------------------------------------------===//
// Legalize unsupported tt.extern_elementwise ops for the non-SDNN XPU path.
//
// libdevice.fast_expf and libdevice.fast_dividef are not present in
// libdevice-xpu3.bc.  In the non-SDNN path these ops survive as unresolved
// external LLVM function calls and cause linker errors.  This pass rewrites
// them to native MLIR ops that are fully supported by TritonXPUToLLVM:
//
//   tt.extern_elementwise(%x)    {symbol="libdevice.fast_expf"}
//     -> math.exp(%x)
//       -> LLVM::Exp2Op(%x) -> XPU ASM: exp.f.rn = e^x  ✓
//
//   NOTE: Although the XPU backend maps math::ExpOp to LLVM::Exp2Op, the
//   hardware instruction exp.f.rn computes the natural exponential e^x
//   (not 2^x).  No log2(e) pre-scaling is needed.
//
//   tt.extern_elementwise(%a,%b) {symbol="libdevice.fast_dividef"}
//   tt.extern_elementwise(%a,%b) {symbol="__triton_houyi_fast_fdividef"}
//     -> arith.divf %a, %b
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-legalize-extern-ew"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPULEGALIZEEXTERNEW
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

// fast_expf(x) = e^x
//
// The XPU backend maps math::ExpOp -> LLVM::Exp2Op -> XPU ASM exp.f.rn.
// The hardware instruction exp.f.rn computes the natural exponential e^x
// directly (not 2^x), so no log2(e) pre-scaling is required.
//
// tt.extern_elementwise(%x) {symbol="libdevice.fast_expf"}
//   -> math.exp(%x)
struct FastExpfToMathExpPattern
    : public OpRewritePattern<triton::ExternElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSymbol() != "libdevice.fast_expf")
      return failure();
    auto srcs = op.getSrcs();
    if (srcs.size() != 1)
      return failure();

    rewriter.replaceOpWithNewOp<math::ExpOp>(op, srcs[0]);
    return success();
  }
};

// tt.extern_elementwise(%a, %b) {symbol="libdevice.fast_dividef"}
// tt.extern_elementwise(%a, %b) {symbol="__triton_houyi_fast_fdividef"}
//   -> arith.divf %a, %b
struct FastDividefToArithDivfPattern
    : public OpRewritePattern<triton::ExternElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    StringRef symbol = op.getSymbol();
    if (symbol != "libdevice.fast_dividef" &&
        symbol != "__triton_houyi_fast_fdividef")
      return failure();
    auto srcs = op.getSrcs();
    if (srcs.size() != 2)
      return failure();
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, srcs[0], srcs[1]);
    return success();
  }
};

} // namespace

struct TritonXPULegalizeExternEW
    : public impl::TritonXPULegalizeExternEWBase<TritonXPULegalizeExternEW> {

  using impl::TritonXPULegalizeExternEWBase<
      TritonXPULegalizeExternEW>::TritonXPULegalizeExternEWBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FastExpfToMathExpPattern, FastDividefToArithDivfPattern>(
        context);
    if (failed(applyPatternsGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
