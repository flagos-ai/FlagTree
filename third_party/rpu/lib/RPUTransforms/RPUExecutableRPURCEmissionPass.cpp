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

#include "RPUTransforms/Passes.h"

#include "RPU/IR/Dialect.h"
#include "RPUExecutableEmitter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rpu {
namespace {

class EmitRPUExecutableToRPURCPass
    : public PassWrapper<EmitRPUExecutableToRPURCPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitRPUExecutableToRPURCPass)

  StringRef getArgument() const final { return "rpu-emit-executable-to-rpurc"; }

  StringRef getDescription() const final {
    return "emit RPUC .rc source metadata from executable RPU dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<exec::RPUDialect>();
  }

  void runOnOperation() final {
    FailureOr<RPUExecutableEmissionResult> result =
        emitRPUDSLFromExecutableModule(getOperation());
    if (failed(result)) {
      signalPassFailure();
      return;
    }

    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    module->setAttr("rpu.rpurc.kernel_name",
                    StringAttr::get(context, result->kernelName));
    module->setAttr("rpu.rpurc.source_kind",
                    StringAttr::get(context, result->sourceKind));
    module->setAttr("rpu.rpurc.source",
                    StringAttr::get(context, result->source));
  }
};

} // namespace

std::unique_ptr<Pass> createEmitRPUExecutableToRPURCPass() {
  return std::make_unique<EmitRPUExecutableToRPURCPass>();
}

} // namespace rpu
} // namespace mlir
