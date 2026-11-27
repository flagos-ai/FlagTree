#include "hcu/tle_raw/include/CompiledRaw.h"
#include "hcu/tle_raw/include/Passes.h"
#include "hcu/tle_raw/include/RawWrapperAdapter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/utils/include/TleRawMaterialize.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#include <memory>

namespace tle = mlir::triton::tle;
namespace hcu_tle_raw = mlir::triton::HCU::tle_raw;

namespace mlir {

#define GEN_PASS_DEF_HCUMATERIALIZEDEFERREDRAW
#include "hcu/tle_raw/include/Passes.h.inc"

static FailureOr<LLVM::LLVMFuncOp>
adaptWrapper(tle::DSLRegionOp call, ModuleOp source, LLVM::LLVMFuncOp wrapper) {
  auto kinds = call->getAttrOfType<ArrayAttr>("tle_raw.arg_info_list");
  unsigned inputCount = 0;
  SmallVector<hcu_tle_raw::RawFragmentOutput> outputs;
  for (auto [i, operand] : llvm::enumerate(call.getOperands())) {
    if (cast<StringAttr>(kinds[i]).getValue() == "tensor_fragment") {
      auto tensor = cast<RankedTensorType>(operand.getType());
      unsigned count = triton::gpu::getTotalElemsPerThread(tensor);
      // Duplicate result indices still share one trailing HIP output pointer.
      if (llvm::is_contained(call.getOutputOperandIndices(), i))
        outputs.push_back({inputCount, count});
      inputCount += count;
    } else if (auto memdesc =
                   dyn_cast<triton::gpu::MemDescType>(operand.getType())) {
      inputCount += 3 + 2 * memdesc.getRank();
    } else {
      ++inputCount;
    }
  }
  return hcu_tle_raw::createRawWrapperAdapter(source, wrapper, inputCount,
                                              outputs);
}

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

    llvm::StringMap<SmallVector<tle::DSLRegionOp>> callsBySource;
    module.walk([&](tle::DSLRegionOp op) {
      auto id = op->getAttrOfType<StringAttr>("tle_raw.source_id");
      if (id && op.getBody().empty())
        callsBySource[id.getValue()].push_back(op);
    });

    for (auto &[id, calls] : callsBySource) {
      auto entry = sources->find(id);
      if (entry == sources->end()) {
        calls.front().emitOpError("missing compiled raw source ") << id;
        return signalPassFailure();
      }
      auto parsed =
          tle::raw::parseLLVMModule(module.getContext(), entry->second.llvmIr);
      if (!parsed) {
        calls.front().emitOpError("cannot parse compiled raw source ") << id;
        return signalPassFailure();
      }

      Location fallbackLoc = calls.front().getLoc();
      parsed->walk([fallbackLoc](Operation *op) {
        if (isa<UnknownLoc>(op->getLoc()))
          op->setLoc(fallbackLoc);
      });

      // Keep wrapper names and shared globals stable across call sites.
      std::string prefix = "__tle_raw_fn_" + id.str() + "_";
      auto originals = tle::raw::renameLLVMSymbols(*parsed, [&](Operation *op) {
        auto name = SymbolTable::getSymbolName(op);
        auto fn = dyn_cast<LLVM::LLVMFuncOp>(op);
        if (!fn || fn.isExternal() ||
            name.getValue().starts_with("__tle_raw_wrapper_"))
          return name;
        return StringAttr::get(module.getContext(),
                               prefix + name.getValue().str());
      });
      if (failed(originals)) {
        calls.front().emitOpError("cannot rename symbols in raw source ") << id;
        return signalPassFailure();
      }

      llvm::StringMap<LLVM::LLVMFuncOp> adapters;
      for (auto call : calls) {
        auto symbol =
            call->getAttrOfType<FlatSymbolRefAttr>("tle_raw.wrapper_symbol");
        if (!symbol) {
          call.emitOpError("missing raw wrapper symbol");
          return signalPassFailure();
        }
        auto &adapter = adapters[symbol.getValue()];
        if (!adapter) {
          auto wrapper = dyn_cast_or_null<LLVM::LLVMFuncOp>(
              originals->lookup(symbol.getValue()));
          if (!wrapper) {
            call.emitOpError("cannot find raw wrapper ") << symbol;
            return signalPassFailure();
          }
          auto result = adaptWrapper(call, *parsed, wrapper);
          if (failed(result))
            return signalPassFailure();
          adapter = *result;
        }
        call->setAttr("tle_raw.wrapper_symbol",
                      FlatSymbolRefAttr::get(adapter));
      }

      for (auto fn : parsed->getOps<LLVM::LLVMFuncOp>()) {
        if (fn.isExternal())
          continue;
        fn.setLinkage(LLVM::Linkage::Internal);
        fn.setVisibility_(LLVM::Visibility::Default);
        SymbolTable::setSymbolVisibility(fn, SymbolTable::Visibility::Private);
      }

      if (failed(tle::raw::cloneLLVMSymbols(module, *parsed)))
        return signalPassFailure();
    }
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
