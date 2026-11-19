#include "hcu/tle_raw/include/CompiledRaw.h"
#include "hcu/tle_raw/include/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/utils/include/TleRawMaterialize.h"

#include <memory>

namespace tle = mlir::triton::tle;
namespace hcu_tle_raw = mlir::triton::HCU::tle_raw;

namespace mlir {

#define GEN_PASS_DEF_HCUMATERIALIZEDEFERREDRAW
#include "hcu/tle_raw/include/Passes.h.inc"

class HCUMaterializeDeferredRawPass
    : public impl::HCUMaterializeDeferredRawBase<
          HCUMaterializeDeferredRawPass> {
public:
  using impl::HCUMaterializeDeferredRawBase<
      HCUMaterializeDeferredRawPass>::HCUMaterializeDeferredRawBase;

  HCUMaterializeDeferredRawPass() = default;
  explicit HCUMaterializeDeferredRawPass(
      llvm::StringMap<hcu_tle_raw::CompiledRaw> sources)
      : sources(
            std::make_shared<const llvm::StringMap<hcu_tle_raw::CompiledRaw>>(
                std::move(sources))) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const auto &compiled = *sources;

    static constexpr llvm::StringLiteral kSourceIdAttr = "tle_raw.source_id";
    WalkResult result = module.walk([&](tle::DSLRegionOp op) -> WalkResult {
      auto sourceIdAttr = op->getAttrOfType<StringAttr>(kSourceIdAttr);
      if (!sourceIdAttr || !op.getBody().empty())
        return WalkResult::advance();

      auto it = compiled.find(sourceIdAttr.getValue());
      if (it == compiled.end()) {
        op.emitError("missing pending raw source for id ")
            << sourceIdAttr.getValue();
        return WalkResult::interrupt();
      }

      const hcu_tle_raw::CompiledRaw &entry = it->second;
      if (entry.externFuncName.empty()) {
        op.emitError("deferred raw source is missing extern_func_name");
        return WalkResult::interrupt();
      }
      if (entry.llvmIr.empty()) {
        op.emitError("deferred raw source is missing compiled LLVM IR");
        return WalkResult::interrupt();
      }

      if (failed(tle::raw::materializeDeferredDSLRegion(
              module, op, entry.llvmIr, entry.externFuncName))) {
        op.emitError("failed to materialize deferred raw source ")
            << sourceIdAttr.getValue();
        return WalkResult::interrupt();
      }

      op->removeAttr(kSourceIdAttr);
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }

private:
  std::shared_ptr<const llvm::StringMap<hcu_tle_raw::CompiledRaw>> sources =
      std::make_shared<const llvm::StringMap<hcu_tle_raw::CompiledRaw>>();
};

std::unique_ptr<Pass> createHCUMaterializeDeferredRaw(
    llvm::StringMap<hcu_tle_raw::CompiledRaw> sources) {
  return std::make_unique<HCUMaterializeDeferredRawPass>(std::move(sources));
}

} // namespace mlir
