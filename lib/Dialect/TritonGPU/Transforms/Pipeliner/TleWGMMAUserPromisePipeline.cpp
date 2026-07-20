/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
* Copyright 2025-     FlagOS Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifdef __TLE__
#include "TleWGMMAAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::gpu::detail {

static constexpr llvm::StringLiteral
    kTleExplicitWgmmaCommitAttr("tle.explicit_wgmma_commit");

void scheduleTleWgmmaUserPromisePipeline(scf::ForOp forOp) {
  IRRewriter builder(forOp.getContext());
  SmallVector<ttng::WarpGroupDotOp, 8> dots;
  forOp.getBody()->walk([&](ttng::WarpGroupDotOp dot) {
    if (dot->getParentOfType<scf::ForOp>() == forOp)
      dots.push_back(dot);
  });

  for (ttng::WarpGroupDotOp dot : llvm::make_early_inc_range(dots)) {
    dot.setIsAsync(true);
    dot->setAttr(kTleExplicitWgmmaCommitAttr, builder.getUnitAttr());

    Operation *next = dot->getNextNode();
    if (next && isa<ttng::WarpGroupDotCommitOp>(next))
      continue;

    builder.setInsertionPointAfter(dot);
    ttng::WarpGroupDotCommitOp::create(builder, dot.getLoc());
  }
}

} // namespace mlir::triton::gpu::detail
#endif // __TLE__
