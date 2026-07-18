// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "RPU/IR/Dialect.h"
#include "RPUTransforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace mlir {
namespace rpu {
namespace {

class LegalizeRPUExecutableSoftmaxPass
    : public PassWrapper<LegalizeRPUExecutableSoftmaxPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeRPUExecutableSoftmaxPass)

  StringRef getArgument() const final {
    return "rpu-legalize-executable-softmax";
  }

  StringRef getDescription() const final {
    return "legalize executable RPU softmax ops to primitive executable ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<exec::RPUDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    SmallVector<exec::SoftmaxOp> softmaxOps;
    module.walk(
        [&](exec::SoftmaxOp softmax) { softmaxOps.push_back(softmax); });

    for (exec::SoftmaxOp softmax : softmaxOps) {
      StringRef consumer =
          "executable softmax legalization requires valid softmax op";
      if (failed(exec::expandExecutableSoftmaxOp(builder, softmax, consumer))) {
        signalPassFailure();
        return;
      }
    }

    if (failed(verify(module))) {
      module.emitError(
          "failed to verify executable RPU module after softmax legalization");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLegalizeRPUExecutableSoftmaxPass() {
  return std::make_unique<LegalizeRPUExecutableSoftmaxPass>();
}

} // namespace rpu
} // namespace mlir
