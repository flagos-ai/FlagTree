#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::xpu {

#define GEN_PASS_DEF_TRITONXPUADDDISTRIBUTEDPARAMS
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUAddDistributedParamsPass
    : public impl::TritonXPUAddDistributedParamsBase<
          TritonXPUAddDistributedParamsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    auto i64Ty = IntegerType::get(ctx, 64);
    module.walk([&](triton::FuncOp func) {
      auto oldType = func.getFunctionType();
      SmallVector<Type> inputs{i64Ty, i64Ty};
      llvm::append_range(inputs, oldType.getInputs());
      func.setFunctionType(
          FunctionType::get(ctx, inputs, oldType.getResults()));

      auto makeNameAttr = [&](StringRef name) {
        return DictionaryAttr::get(
            ctx, {NamedAttribute(StringAttr::get(ctx, "tt.name"),
                                 StringAttr::get(ctx, name))});
      };
      SmallVector<Attribute> argAttrs{makeNameAttr("device_comm_ptr"),
                                      makeNameAttr("device_mem_ptr")};
      if (auto oldAttrs = func.getArgAttrsAttr())
        llvm::append_range(argAttrs, oldAttrs);
      else
        argAttrs.append(oldType.getNumInputs(), DictionaryAttr::get(ctx));
      func.setArgAttrsAttr(ArrayAttr::get(ctx, argAttrs));

      Block &entry = func.front();
      entry.insertArgument(0u, i64Ty, func.getLoc());
      entry.insertArgument(1u, i64Ty, func.getLoc());
    });
    module->setAttr("xpu.implicit_distributed_params", UnitAttr::get(ctx));
  }
};

} // namespace mlir::triton::xpu
