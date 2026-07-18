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
