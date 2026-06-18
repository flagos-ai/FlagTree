#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "nvidia/tle_raw/include/DeferredRawSourceRegistry.h"
#include "nvidia/tle_raw/include/Passes.h"
#include "tle/dialect/include/IR/Dialect.h"

#include "tle/utils/include/TleRawMaterialize.h"

namespace tle = mlir::triton::tle;
namespace nv_tle_raw = mlir::triton::nvidia::tle_raw;

namespace mlir {

#define GEN_PASS_DEF_NVIDIAMATERIALIZEDEFERREDRAW
#include "nvidia/tle_raw/include/Passes.h.inc"

class NvidiaMaterializeDeferredRawPass
    : public impl::NvidiaMaterializeDeferredRawBase<
          NvidiaMaterializeDeferredRawPass> {
public:
  using impl::NvidiaMaterializeDeferredRawBase<
      NvidiaMaterializeDeferredRawPass>::NvidiaMaterializeDeferredRawBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto &registry = nv_tle_raw::getDeferredRawSourceRegistry();
    if (registry.empty())
      return;

    static constexpr llvm::StringLiteral kSourceIdAttr = "tle_raw.source_id";
    WalkResult result = module.walk([&](tle::DSLRegionOp op) -> WalkResult {
      auto sourceIdAttr = op->getAttrOfType<StringAttr>(kSourceIdAttr);
      if (!sourceIdAttr)
        return WalkResult::advance();

      auto it = registry.find(sourceIdAttr.getValue());
      if (it == registry.end()) {
        op.emitError("missing pending raw source for id ")
            << sourceIdAttr.getValue();
        return WalkResult::interrupt();
      }

      const nv_tle_raw::DeferredRawSourceEntry &entry = it->second;
      if (!entry.externFuncName) {
        op.emitError("deferred raw source is missing extern_func_name");
        return WalkResult::interrupt();
      }
      if (entry.llvmIr.empty()) {
        op.emitError("deferred raw source is missing compiled LLVM IR");
        return WalkResult::interrupt();
      }

      if (failed(tle::raw::materializeDeferredDSLRegion(
              module, op, entry.llvmIr, *entry.externFuncName))) {
        op.emitError("failed to materialize deferred raw source ")
            << sourceIdAttr.getValue();
        return WalkResult::interrupt();
      }

      op->removeAttr(kSourceIdAttr);
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
    nv_tle_raw::clearDeferredRawSourceRegistry();
  }
};

} // namespace mlir
