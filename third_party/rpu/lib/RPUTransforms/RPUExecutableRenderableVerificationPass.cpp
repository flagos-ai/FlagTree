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

#include "RPUTransforms/Passes.h"

#include "RPU/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rpu {
namespace {

class VerifyRPUExecutableRenderablePass
    : public PassWrapper<VerifyRPUExecutableRenderablePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VerifyRPUExecutableRenderablePass)

  StringRef getArgument() const final {
    return "rpu-verify-executable-renderable";
  }

  StringRef getDescription() const final {
    return "verify executable RPU dialect is renderable before RPURC emission";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<exec::RPUDialect>();
  }

  void runOnOperation() final {
    if (failed(exec::verifyExecutableModuleRenderable(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createVerifyRPUExecutableRenderablePass() {
  return std::make_unique<VerifyRPUExecutableRenderablePass>();
}

} // namespace rpu
} // namespace mlir
