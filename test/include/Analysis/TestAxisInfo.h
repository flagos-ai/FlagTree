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

#pragma once

#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::test {

struct TestAxisInfoPass
    : public PassWrapper<TestAxisInfoPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAxisInfoPass);

  StringRef getArgument() const override { return "test-print-alignment"; }
  StringRef getDescription() const final {
    return "print the result of the alignment analysis pass";
  }

  void runOnOperation() override {
    Operation *operation = this->getOperation();
    ModuleOp moduleOp = cast<ModuleOp>(operation);
    auto moduleAxisInfoAnalysis = getAnalysis(moduleOp);
    moduleOp.walk([&](FuncOp funcOp) {
      funcOp.walk([&](Operation *op) {
        if (op->getNumResults() < 1)
          return;
        for (Value result : op->getResults()) {
          InFlightDiagnostic diag = mlir::emitRemark(op->getLoc());
          diag << result;
          diag << " => ";
          auto *axisInfo = moduleAxisInfoAnalysis.getAxisInfo(result);
          if (axisInfo) {
            std::string str;
            llvm::raw_string_ostream os(str);
            axisInfo->print(os);
            diag << str;
          }
        }
      });
    });
  }

protected:
  virtual ModuleAxisInfoAnalysis getAnalysis(ModuleOp moduleOp) const {
    return ModuleAxisInfoAnalysis(moduleOp);
  }
};

} // namespace mlir::test
