#include "iluvatar/tle/dialect/include/IR/Dialect.h"
#include "iluvatar/tle_raw/include/DeferredRawSourceRegistry.h"
#include "iluvatar/tle_raw/include/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "iluvatar/tle/utils/include/TleRawMaterialize.h"

namespace iluvatar_tle = mlir::triton::iluvatar_tle;
namespace ilu_tle_raw = mlir::triton::iluvatar::tle_raw;

namespace mlir {

#define GEN_PASS_DEF_ILUVATARMATERIALIZEDEFERREDRAW
#include "iluvatar/tle_raw/include/Passes.h.inc"

class IluvatarMaterializeDeferredRawPass
    : public impl::IluvatarMaterializeDeferredRawBase<
          IluvatarMaterializeDeferredRawPass> {
public:
  using impl::IluvatarMaterializeDeferredRawBase<
      IluvatarMaterializeDeferredRawPass>::IluvatarMaterializeDeferredRawBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto &registry = ilu_tle_raw::getDeferredRawSourceRegistry();
    if (registry.empty())
      return;

    static constexpr llvm::StringLiteral kSourceIdAttr = "tle_raw.source_id";
    WalkResult result =
        module.walk([&](iluvatar_tle::DSLRegionOp op) -> WalkResult {
          auto sourceIdAttr = op->getAttrOfType<StringAttr>(kSourceIdAttr);
          if (!sourceIdAttr)
            return WalkResult::advance();

          auto it = registry.find(sourceIdAttr.getValue());
          if (it == registry.end()) {
            op.emitError("missing pending raw source for id ")
                << sourceIdAttr.getValue();
            return WalkResult::interrupt();
          }

          const ilu_tle_raw::DeferredRawSourceEntry &entry = it->second;
          if (!entry.externFuncName) {
            op.emitError("deferred raw source is missing extern_func_name");
            return WalkResult::interrupt();
          }
          if (entry.llvmIr.empty()) {
            op.emitError("deferred raw source is missing compiled LLVM IR");
            return WalkResult::interrupt();
          }

          if (failed(iluvatar_tle::raw::materializeDeferredDSLRegion(
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
    ilu_tle_raw::clearDeferredRawSourceRegistry();
  }
};

} // namespace mlir
