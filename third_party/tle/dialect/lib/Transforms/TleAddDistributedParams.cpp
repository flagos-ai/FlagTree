/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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

  func.setArgAttrsAttr(ArrayAttr::get(ctx, argAttrs));

  //
  // 3. Insert block arguments.
  //
  Block &entry = func.front();
  entry.insertArgument(static_cast<unsigned>(0), i64Ty, func.getLoc());
  entry.insertArgument(static_cast<unsigned>(1), i64Ty, func.getLoc());
}

struct TritonTleAddDistributedParams
    : public impl::TritonTleAddDistributedParamsBase<
          TritonTleAddDistributedParams> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    m.walk([&](mlir::triton::FuncOp func) { addDistributedParams(ctx, func); });
  }
};

} // namespace
} // namespace mlir::triton::tle
