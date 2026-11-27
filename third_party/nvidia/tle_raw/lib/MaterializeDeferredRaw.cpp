#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "nvidia/tle_raw/include/CompiledRaw.h"
#include "nvidia/tle_raw/include/Passes.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/utils/include/TleRawMaterialize.h"

#include <memory>

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

  NvidiaMaterializeDeferredRawPass() = default;
  explicit NvidiaMaterializeDeferredRawPass(
      llvm::StringMap<nv_tle_raw::CompiledRaw> sources)
      : sources(
            std::make_shared<const llvm::StringMap<nv_tle_raw::CompiledRaw>>(
                std::move(sources))) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringMap<tle::DSLRegionOp> firstCallBySource;
    module.walk([&](tle::DSLRegionOp op) {
      auto id = op->getAttrOfType<StringAttr>("tle_raw.source_id");
      if (id && op.getBody().empty())
        firstCallBySource.try_emplace(id.getValue(), op);
    });

    for (auto &[id, call] : firstCallBySource) {
      auto entry = sources->find(id);
      if (entry == sources->end()) {
        call.emitOpError("missing compiled raw source ") << id;
        return signalPassFailure();
      }
      auto parsed =
          tle::raw::parseLLVMModule(module.getContext(), entry->second.llvmIr);
      if (!parsed) {
        call.emitOpError("cannot parse compiled raw source ") << id;
        return signalPassFailure();
      }

      // Keep wrapper names and shared globals stable across call sites.
      std::string prefix = "__tle_raw_fn_" + id.str() + "_";
      auto name_mapping =
          tle::raw::renameLLVMSymbols(*parsed, [&](Operation *op) {
            auto name = SymbolTable::getSymbolName(op);
            auto fn = dyn_cast<LLVM::LLVMFuncOp>(op);
            if (!fn || fn.isExternal() ||
                name.getValue().starts_with("__tle_raw_wrapper_"))
              return name;
            return StringAttr::get(module.getContext(),
                                   prefix + name.getValue().str());
          });
      if (failed(name_mapping)) {
        call.emitOpError("cannot rename symbols in raw source ") << id;
        return signalPassFailure();
      }

      Location fallbackLoc = call.getLoc();
      parsed->walk([fallbackLoc](Operation *op) {
        if (isa<UnknownLoc>(op->getLoc()))
          op->setLoc(fallbackLoc);
        auto fn = dyn_cast<LLVM::LLVMFuncOp>(op);
        if (!fn || fn.isExternal())
          return;
        fn.setLinkage(LLVM::Linkage::Internal);
        fn.setVisibility_(LLVM::Visibility::Default);
        SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
      });

      if (failed(tle::raw::cloneLLVMSymbols(module, *parsed)))
        return signalPassFailure();
    }
  }

private:
  std::shared_ptr<const llvm::StringMap<nv_tle_raw::CompiledRaw>> sources =
      std::make_shared<const llvm::StringMap<nv_tle_raw::CompiledRaw>>();
};

std::unique_ptr<Pass> createNvidiaMaterializeDeferredRaw(
    llvm::StringMap<nv_tle_raw::CompiledRaw> sources) {
  return std::make_unique<NvidiaMaterializeDeferredRawPass>(std::move(sources));
}

} // namespace mlir
