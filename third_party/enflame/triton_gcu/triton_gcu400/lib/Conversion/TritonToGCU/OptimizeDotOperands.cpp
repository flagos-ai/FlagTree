/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
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

#include "Conversion/Passes.h"
#include "Conversion/TritonToGCU/Constants.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONGCUOPTIMIZEDOTOPERANDS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Pattern: tt.trans(order={1,0}) -> ttg.convert_layout -> tt.dot(rhs)
///
/// Eliminates the explicit transpose on the dot's right-hand operand by
/// creating a new convert_layout from the pre-transpose source with the
/// original (un-transposed) shape and columnMajor=true on the encoding.
/// The hardware gemm will read B in column-major order (physical [N,K]),
/// making the explicit transpose unnecessary.
class FuseTrans : public OpRewritePattern<triton::DotOp> {
public:
  using OpRewritePattern<triton::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    // Already annotated — nothing to do.
    if (dotOp->hasAttr(kRhsColumnMajor))
      return failure();

    // Look for: trans -> convert_layout -> dot(operand[1])
    auto bVal = dotOp.getB();
    auto cvtOp = bVal.getDefiningOp<triton::gpu::ConvertLayoutOp>();
    if (!cvtOp)
      return failure();

    auto transOp = cvtOp.getSrc().getDefiningOp<triton::TransOp>();
    if (!transOp)
      return failure();

    // Only standard 2D transposes.
    if (transOp.getOrder() != ArrayRef<int32_t>({1, 0}))
      return failure();

    // Must be the right-hand operand (opIdx == 1).
    auto cvtResultTy = cast<RankedTensorType>(cvtOp.getResult().getType());
    auto dotOpEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
        cvtResultTy.getEncoding());
    if (!dotOpEnc || dotOpEnc.getOpIdx() != 1)
      return failure();

    // The pre-transpose source keeps its original shape.
    // For rhs_column_major, B is physically stored as [N, K] so the
    // gemm hardware reads it column-major.  The parent encoding stays
    // the same (dot_op's parent = accumulator encoding).
    auto srcType = cast<RankedTensorType>(transOp.getSrc().getType());
    auto colMajorEnc = triton::gpu::DotOperandEncodingAttr::get(
        dotOpEnc.getContext(), dotOpEnc.getOpIdx(), dotOpEnc.getParent(),
        dotOpEnc.getKWidth(), /*columnMajor=*/true);
    auto newCvtType = RankedTensorType::get(
        srcType.getShape(), srcType.getElementType(), colMajorEnc);

    // Create the new convert_layout from pre-trans src.
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        cvtOp.getLoc(), newCvtType, transOp.getSrc());

    // Redirect the dot's B operand to the new convert_layout.
    dotOp.getBMutable().assign(newCvt.getResult());

    // The B encoding carries columnMajor=true; set attr on the dot so
    // verifyDims/verifyOutputDims (in Triton dialect) can read it.
    dotOp->setAttr(kRhsColumnMajor, rewriter.getUnitAttr());

    return success();
  }
};

struct TritonGCUOptimizeDotOperandsPass
    : public impl::TritonGCUOptimizeDotOperandsBase<
          TritonGCUOptimizeDotOperandsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseTrans>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
