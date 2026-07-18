// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUFUNCCONVERT
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUFuncConvert
    : public impl::TritonXPUFuncConvertBase<TritonXPUFuncConvert> {

public:
  using impl::TritonXPUFuncConvertBase<
      TritonXPUFuncConvert>::TritonXPUFuncConvertBase;

  TritonXPUFuncConvert() = default;
  TritonXPUFuncConvert(bool toMlir) { this->toMlir = toMlir; }

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    if (this->toMlir) {
      m.walk([&](triton::FuncOp func) {
        OpBuilder builder(func);

        auto name = func.getName();
        auto type = func.getFunctionType();

        SmallVector<DictionaryAttr> argAttrs, resAttrs;
        func.getAllArgAttrs(argAttrs);
        func.getAllResultAttrs(resAttrs);

        auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, type);
        funcFunc.setAllArgAttrs(argAttrs);
        funcFunc.setAllResultAttrs(resAttrs);

        auto &funcFuncBody = funcFunc.getBody();
        auto &funcBody = func.getBody();

        IRMapping map;
        funcBody.cloneInto(&funcFuncBody, map);

        for (Block &block : funcFuncBody.getBlocks()) {
          auto term = block.getTerminator();
          // Only convert to func.return if the terminator is a tt.return.
          // Otherwise, we will accidentally convert cf.br ops which are also
          // considered terminators.
          if (isa<triton::ReturnOp>(term)) {
            builder.setInsertionPoint(term);
            builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
            term->erase();
          }
        }
        func.erase();
      });
    } else {
      m.walk([&](func::FuncOp func) {
        OpBuilder builder(func);

        auto name = func.getName();
        auto type = func.getFunctionType();

        SmallVector<DictionaryAttr> argAttrs, resAttrs;
        func.getAllArgAttrs(argAttrs);
        func.getAllResultAttrs(resAttrs);

        auto tritonFunc =
            builder.create<triton::FuncOp>(func.getLoc(), name, type);
        tritonFunc.setAllArgAttrs(argAttrs);
        tritonFunc.setAllResultAttrs(resAttrs);

        auto &tritonFuncBody = tritonFunc.getBody();
        auto &funcBody = func.getBody();

        IRMapping map;
        funcBody.cloneInto(&tritonFuncBody, map);

        for (Block &block : tritonFuncBody.getBlocks()) {
          auto term = block.getTerminator();
          // Only convert to func.return if the terminator is a tt.return.
          // Otherwise, we will accidentally convert cf.br ops which are also
          // considered terminators.
          if (isa<func::ReturnOp>(term)) {
            builder.setInsertionPoint(term);
            builder.create<triton::ReturnOp>(func.getLoc(),
                                             term->getOperands());
            term->erase();
          }
        }
        func.erase();
      });
    }
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
