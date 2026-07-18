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

#include "triton/Dialect/Gluon/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace triton;
namespace gluon = mlir::triton::gluon;

namespace mlir::triton::gluon {
#define GEN_PASS_DEF_GLUONINLINE
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"
} // namespace mlir::triton::gluon

namespace {
struct Inline : public gluon::impl::GluonInlineBase<Inline> {
  void runOnOperation() override;
};
} // namespace

void Inline::runOnOperation() {
  mlir::PassManager pm(&getContext());
  pm.addPass(createInlinerPass(/*opPipelines=*/{}, [](OpPassManager &pm) {
    pm.addPass(gluon::createGluonSimplifyControlFlow());
  }));
  if (failed(pm.run(getOperation())))
    return signalPassFailure();
}
