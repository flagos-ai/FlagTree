// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "triton/Analysis/Utility.h"

using namespace mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDFOLDTRUECMPI
#include "TritonAMDGPUTransforms/Passes.h.inc"

struct TritonAMDFoldTrueCmpIOpPass
    : impl::TritonAMDFoldTrueCmpIBase<TritonAMDFoldTrueCmpIOpPass> {

  void runOnOperation() override {
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    ModuleOp mod = getOperation();
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AMD::TritonIntegerRangeAnalysis *rangeAnalysis =
        solver->load<AMD::TritonIntegerRangeAnalysis>(
            assumptions, &getAnalysis<DominanceInfo>());
    AMD::initializeFuncOps(mod, rangeAnalysis);
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    RewritePatternSet patterns(&getContext());
    AMD::populateFoldTrueCmpIOpPatterns(patterns, solver.get());
    (void)applyPatternsGreedily(mod, std::move(patterns));
  }
};

} // namespace mlir
