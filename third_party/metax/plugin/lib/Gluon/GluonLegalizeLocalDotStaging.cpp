#include "Gluon/Passes.h"
#include "Gluon/GluonRegisterToShared.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONLEGALIZELOCALDOTSTAGING
#include "Gluon/Passes.h.inc"

namespace {

/// Materializes target-required shared-memory boundaries. Keeping this
/// transformation separate from final verification makes pipeline ownership
/// explicit and prevents a verifier from silently repairing invalid IR.
class TritonMETAXGPUGluonLegalizeLocalDotStagingPass
    : public impl::TritonMETAXGPUGluonLegalizeLocalDotStagingBase<
          TritonMETAXGPUGluonLegalizeLocalDotStagingPass> {
public:
  void runOnOperation() override {
    if (failed(triton::gpu::metax::gluon::legalizeLocalDotStaging(
            getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
createTritonMETAXGPUGluonLegalizeLocalDotStagingPass() {
  return std::make_unique<TritonMETAXGPUGluonLegalizeLocalDotStagingPass>();
}

} // namespace mlir
