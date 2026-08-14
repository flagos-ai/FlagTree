// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// flagtree tle

#include "IR/Dialect.h"
#include "Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::iluvatar_tle {

#define GEN_PASS_DEF_TRITONILUVATARTLEOPTIMIZEEXCLUSIVECUMSUMLAYOUTS
#include "Transforms/Passes.h.inc"

namespace {

class OptimizeExclusiveCumsumLayoutsPass
    : public impl::TritonIluvatarTleOptimizeExclusiveCumsumLayoutsBase<
          OptimizeExclusiveCumsumLayoutsPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ExclusiveCumsumOp> ops;
    module.walk([&](ExclusiveCumsumOp op) { ops.push_back(op); });

    for (ExclusiveCumsumOp op : ops) {
      // Only fold when the cumsum input is produced by a convert_layout, so we
      // can push the cumsum back onto the pre-convert (canonical) layout.
      auto srcCvt = op.getSrc().getDefiningOp<ttg::ConvertLayoutOp>();
      if (!srcCvt)
        continue;

      auto srcBaseTy = dyn_cast<RankedTensorType>(srcCvt.getSrc().getType());
      auto cumsumTy = dyn_cast<RankedTensorType>(op.getExclusive().getType());
      if (!srcBaseTy || !cumsumTy)
        continue;
      if (srcBaseTy.getShape() != cumsumTy.getShape() ||
          srcBaseTy.getElementType() != cumsumTy.getElementType())
        continue;
      if (srcBaseTy.getEncoding() == cumsumTy.getEncoding())
        continue;

      // Every use of the exclusive result must be a convert_layout back to the
      // canonical (pre-convert) layout, otherwise folding would change the
      // observable layout of a live value.
      SmallVector<ttg::ConvertLayoutOp> outCvts;
      bool allUsersMatch = true;
      for (OpOperand &use : op.getExclusive().getUses()) {
        auto outCvt = dyn_cast<ttg::ConvertLayoutOp>(use.getOwner());
        if (!outCvt || outCvt.getSrc() != op.getExclusive()) {
          allUsersMatch = false;
          break;
        }
        auto outTy = dyn_cast<RankedTensorType>(outCvt.getType());
        if (!outTy || outTy.getShape() != srcBaseTy.getShape() ||
            outTy.getElementType() != srcBaseTy.getElementType() ||
            outTy.getEncoding() != srcBaseTy.getEncoding()) {
          allUsersMatch = false;
          break;
        }
        outCvts.push_back(outCvt);
      }
      if (!allUsersMatch)
        continue;

      OpBuilder builder(op);
      auto newOp = builder.create<ExclusiveCumsumOp>(
          op.getLoc(), TypeRange{srcBaseTy, op.getTotal().getType()},
          srcCvt.getSrc(), op.getAxisAttr(), op.getReverseAttr());

      for (auto cvt : outCvts)
        cvt.replaceAllUsesWith(newOp.getExclusive());
      op.getTotal().replaceAllUsesWith(newOp.getTotal());

      for (auto cvt : outCvts)
        cvt.erase();
      op.erase();
      if (srcCvt->use_empty())
        srcCvt.erase();
    }
  }
};

} // namespace
} // namespace mlir::triton::iluvatar_tle
