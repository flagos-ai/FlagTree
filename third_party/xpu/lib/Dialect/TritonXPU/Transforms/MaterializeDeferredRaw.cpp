//===- MaterializeDeferredRaw.cpp - fill in deferred tle.raw payloads -----===//
//
// A `triton_xpu.raw` traced in deferred mode carries only a source id: at trace
// time the target arch is not known, so the payload cannot be compiled yet. The
// backend compiles the pending sources in `make_llir` and hands the
// {source_id: llvm_ir} map to this pass, which fills `llvm_ir` in before
// TritonXPU->LLVM conversion turns the op into a call.
//
// Mirrors ../../TritonSDNN/Transforms/MaterializeDeferredRaw.cpp, which does
// the same job on the SDNN path.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

using namespace mlir;

namespace mlir::triton::xpu {

#define GEN_PASS_DEF_TRITONXPUMATERIALIZEDEFERREDRAW
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

struct TritonXPUMaterializeDeferredRawPass
    : public impl::TritonXPUMaterializeDeferredRawBase<
          TritonXPUMaterializeDeferredRawPass> {

  TritonXPUMaterializeDeferredRawPass() = default;

  void setSources(const llvm::StringMap<std::string> &newSources) {
    for (auto &kv : newSources)
      sources.insert_or_assign(kv.getKey(), kv.getValue());
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    WalkResult result = mod.walk([&](RawOp op) -> WalkResult {
      auto sourceId = op->getAttrOfType<StringAttr>(kRawSourceIdAttrName);
      if (!sourceId)
        return WalkResult::advance();

      auto it = sources.find(sourceId.getValue());
      if (it == sources.end()) {
        op.emitError("triton_xpu.raw: no compiled payload for deferred source "
                     "id '")
            << sourceId.getValue()
            << "'; the backend did not register it before running "
               "tritonxpu-materialize-deferred-raw";
        return WalkResult::interrupt();
      }
      if (it->second.empty()) {
        op.emitError("triton_xpu.raw: deferred source id '")
            << sourceId.getValue() << "' compiled to an empty payload";
        return WalkResult::interrupt();
      }

      op.setLlvmIr(it->second);
      op->removeAttr(kRawSourceIdAttrName);
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }

private:
  llvm::StringMap<std::string> sources;
};

} // namespace

std::unique_ptr<mlir::Pass> createTritonXPUMaterializeDeferredRawWithSources(
    const llvm::StringMap<std::string> &sources) {
  auto pass = std::make_unique<TritonXPUMaterializeDeferredRawPass>();
  pass->setSources(sources);
  return pass;
}

} // namespace mlir::triton::xpu
