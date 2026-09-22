#include "Transforms/Passes.h"

#include "IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::iluvatar_tle {

#define GEN_PASS_DEF_TRITONILUVATARTLEADDDISTRIBUTEDPARAMS
#include "Transforms/Passes.h.inc"

namespace {

void addDistributedParams(MLIRContext *ctx, mlir::triton::FuncOp func) {
  auto i64Ty = IntegerType::get(ctx, 64);

  //
  // 1. Update function type.
  //
  auto oldType = func.getFunctionType();

  SmallVector<Type> inputTypes;
  inputTypes.push_back(i64Ty); // device_comm_ptr
  inputTypes.push_back(i64Ty); // device_mem_ptr
  llvm::append_range(inputTypes, oldType.getInputs());

  func.setFunctionType(
      FunctionType::get(ctx, inputTypes, oldType.getResults()));

  //
  // 2. Update argument attributes.
  //
  auto makeNameAttr = [&](StringRef name) {
    return DictionaryAttr::get(ctx,
                               {NamedAttribute(StringAttr::get(ctx, "tt.name"),
                                               StringAttr::get(ctx, name))});
  };

  SmallVector<Attribute> argAttrs;
  argAttrs.push_back(makeNameAttr("device_comm_ptr"));
  argAttrs.push_back(makeNameAttr("device_mem_ptr"));

  if (auto oldAttrs = func.getArgAttrsAttr())
    llvm::append_range(argAttrs, oldAttrs);
  else
    // A function whose arguments carry no attributes has no arg_attrs array at
    // all, and tt.func requires one entry per argument once it exists.
    argAttrs.append(oldType.getNumInputs(), DictionaryAttr::get(ctx));

  func.setArgAttrsAttr(ArrayAttr::get(ctx, argAttrs));

  //
  // 3. Insert block arguments.
  //
  Block &entry = func.front();
  entry.insertArgument(static_cast<unsigned>(0), i64Ty, func.getLoc());
  entry.insertArgument(static_cast<unsigned>(1), i64Ty, func.getLoc());
}

struct TritonIluvatarTleAddDistributedParams
    : public impl::TritonIluvatarTleAddDistributedParamsBase<
          TritonIluvatarTleAddDistributedParams> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    m.walk([&](mlir::triton::FuncOp func) { addDistributedParams(ctx, func); });
  }
};

} // namespace
} // namespace mlir::triton::iluvatar_tle
