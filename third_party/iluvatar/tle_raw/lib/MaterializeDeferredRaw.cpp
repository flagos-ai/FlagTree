#include "iluvatar/tle/dialect/include/IR/Dialect.h"
#include "iluvatar/tle_raw/include/CompiledRaw.h"
#include "iluvatar/tle_raw/include/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "iluvatar/tle/utils/include/TleRawMaterialize.h"

#include <memory>

namespace iluvatar_tle = mlir::triton::iluvatar_tle;
using mlir::triton::iluvatar::tle_raw::CompiledRaw;

namespace mlir {

#define GEN_PASS_DEF_ILUVATARMATERIALIZEDEFERREDRAW
#include "iluvatar/tle_raw/include/Passes.h.inc"

class IluvatarMaterializeDeferredRawPass
    : public impl::IluvatarMaterializeDeferredRawBase<
          IluvatarMaterializeDeferredRawPass> {
public:
  using impl::IluvatarMaterializeDeferredRawBase<
      IluvatarMaterializeDeferredRawPass>::IluvatarMaterializeDeferredRawBase;

  IluvatarMaterializeDeferredRawPass() = default;

  explicit IluvatarMaterializeDeferredRawPass(
      llvm::StringMap<CompiledRaw> sources)
      : sources(std::make_shared<const llvm::StringMap<CompiledRaw>>(
            std::move(sources))) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const auto &compiled = *sources;

    static constexpr llvm::StringLiteral kSourceIdAttr = "tle_raw.source_id";
    WalkResult result =
        module.walk([&](iluvatar_tle::DSLRegionOp op) -> WalkResult {
          auto sourceIdAttr = op->getAttrOfType<StringAttr>(kSourceIdAttr);
          if (!sourceIdAttr || !op.getBody().empty())
            return WalkResult::advance();

          auto it = compiled.find(sourceIdAttr.getValue());
          if (it == compiled.end()) {
            op.emitError("missing pending raw source for id ")
                << sourceIdAttr.getValue();
            return WalkResult::interrupt();
          }

          const CompiledRaw &entry = it->second;
          if (entry.externFuncName.empty()) {
            op.emitError("deferred raw source is missing extern_func_name");
            return WalkResult::interrupt();
          }
          if (entry.llvmIr.empty()) {
            op.emitError("deferred raw source is missing compiled LLVM IR");
            return WalkResult::interrupt();
          }

          if (failed(iluvatar_tle::raw::materializeDeferredDSLRegion(
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
  std::shared_ptr<const llvm::StringMap<CompiledRaw>> sources =
      std::make_shared<const llvm::StringMap<CompiledRaw>>();
};

std::unique_ptr<Pass>
createIluvatarMaterializeDeferredRaw(llvm::StringMap<CompiledRaw> sources) {
  return std::make_unique<IluvatarMaterializeDeferredRawPass>(
      std::move(sources));
}

} // namespace mlir
