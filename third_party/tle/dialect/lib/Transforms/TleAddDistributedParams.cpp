#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::tle {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONTLEADDDISTRIBUTEDPARAMS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

struct TritonTleAddDistributedParams
    : public impl::TritonTleAddDistributedParamsBase<
          TritonTleAddDistributedParams> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m.getContext());

    m->walk([&, this](triton::FuncOp func) {
      auto type = func.getFunctionType();
      llvm::errs() << type << "\n";
    });
  }
};

} // namespace
} // namespace mlir::triton::tle
