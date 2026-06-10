#include "epu/memory.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <cstdint>

#define GEN_PASS_DEF_BUFFERIZE
#include "evas/Transform/Linalg/Passes.h.inc"
namespace mlir::triton::ev {

namespace {
/// A pass to insert deallocations for allocated buffers after theirlast use.
using namespace mlir;

struct BufferizePass : public ::impl::BufferizeBase<BufferizePass> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    bufferization::OneShotBufferizationOptions bufferizeOption;
    bufferizeOption.bufferizeFunctionBoundaries = false;
    bufferizeOption.allowReturnAllocsFromLoops = true;
    bufferization::BufferizationState state;
    // default memory scope on ddr
    bufferizeOption.defaultMemorySpaceFn =
        [](TensorType t) -> std::optional<Attribute> {
      return IntegerAttr::get(IntegerType::get(t.getContext(), 64),
                              mlir::ev::MemScope::DDR);
    };
    bufferizeOption.opFilter.allowOperation([](Operation *op) {
      // If it's a function, only allow "kernel"
      if (auto funcOp = dyn_cast<func::FuncOp>(op))
        return funcOp.getSymName() == "kernel";
      // For other ops, check if they're inside "kernel"
      auto parentFunc = op->getParentOfType<func::FuncOp>();
      return parentFunc && parentFunc.getSymName() == "kernel";
    });
    // bufferizeOption.opFilter.denyDialect<arith::ArithDialect>();
    // bufferizeOption.opFilter.allowDialect<tensor::TensorDialect>();
    // bufferizeOption.opFilter.denyOperation<bufferization::DeallocTensorOp>();
    if (failed(bufferization::runOneShotBufferize(moduleOp, bufferizeOption,
                                                  state))) {
      signalPassFailure();
    }
  }
};

} // namespace
std::unique_ptr<Pass> createBufferizePass() {
  return std::make_unique<BufferizePass>();
}
} // namespace mlir::triton::ev
