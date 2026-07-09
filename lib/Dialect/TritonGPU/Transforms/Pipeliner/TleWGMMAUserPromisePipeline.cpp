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
