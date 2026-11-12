#include "Passes.h"

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::Intellifusion {

namespace {

class GPNPUCheckMatmulTwoCTAPass
    : public PassWrapper<GPNPUCheckMatmulTwoCTAPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPNPUCheckMatmulTwoCTAPass)

  StringRef getArgument() const final { return "gpnpu-check-matmul-two-cta"; }
  StringRef getDescription() const final {
    return "verify consistent two_ctas usage across GPNPU matmuls";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ttng::TritonNvidiaGPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Operation *firstMatmul = nullptr;
    bool firstTwoCTA = false;

    // Walk all MMAv5 ops using the interface
    WalkResult result = mod.walk([&](ttng::MMAv5OpInterface op) -> WalkResult {
      bool currentTwoCTA = op.getTwoCtas();
      if (!firstMatmul) {
        firstMatmul = op;
        firstTwoCTA = currentTwoCTA;
        return WalkResult::advance();
      }
      if (currentTwoCTA != firstTwoCTA) {
        auto diag = op.emitError()
                    << "inconsistent two_ctas setting across matmuls; "
                       "expected all matmuls to "
                    << (firstTwoCTA ? "enable" : "disable") << " two_ctas.";
        diag.attachNote(firstMatmul->getLoc())
            << "first matmul here has two_ctas="
            << (firstTwoCTA ? "true" : "false") << ".";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // FPSAN rewrites all `tcgen05` MMA ops but sets the flag so it can be
    // propagated.
    if (!firstMatmul && mod->hasAttr(ttng::AttrTwoCTAsName))
      return;
    bool twoCTAValue = firstMatmul ? firstTwoCTA : false;
    mod->setAttr(ttng::AttrTwoCTAsName,
                 BoolAttr::get(mod.getContext(), twoCTAValue));
  }
};

} // namespace

std::unique_ptr<Pass> createGPNPUCheckMatmulTwoCTAPass() {
  return std::make_unique<GPNPUCheckMatmulTwoCTAPass>();
}

} // namespace mlir::triton::Intellifusion
