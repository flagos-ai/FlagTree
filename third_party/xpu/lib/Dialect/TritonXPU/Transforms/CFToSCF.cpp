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

//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUCFTOSCF
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUCFToSCF : public impl::TritonXPUCFToSCFBase<TritonXPUCFToSCF> {

public:
  using impl::TritonXPUCFToSCFBase<TritonXPUCFToSCF>::TritonXPUCFToSCFBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    ControlFlowToSCFTransformation transformation;
    bool changed = false;
    m.walk([&](triton::FuncOp funcOp) {
      if (funcOp.getBody().empty())
        return WalkResult::advance();

      auto &domInfo = funcOp != m ? getChildAnalysis<DominanceInfo>(funcOp)
                                  : getAnalysis<DominanceInfo>();

      auto visitor = [&](Operation *innerOp) -> WalkResult {
        for (Region &reg : innerOp->getRegions()) {
          FailureOr<bool> changedFunc =
              transformCFGToSCF(reg, transformation, domInfo);
          if (failed(changedFunc))
            return WalkResult::interrupt();

          changed |= *changedFunc;
        }
        return WalkResult::advance();
      };

      if (funcOp->walk<WalkOrder::PostOrder>(visitor).wasInterrupted())
        return WalkResult::interrupt();

      return WalkResult::advance();
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
