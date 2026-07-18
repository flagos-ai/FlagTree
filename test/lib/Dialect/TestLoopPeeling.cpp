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

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"

using namespace mlir;

namespace {

bool getPeelEpilogue(scf::ForOp forOp) {
  return forOp->hasAttr("__test_peel_epilogue");
}

struct TestLoopPeelingPass
    : public PassWrapper<TestLoopPeelingPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLoopPeelingPass);

  StringRef getArgument() const final { return "triton-test-loop-peeling"; }
  StringRef getDescription() const final {
    return "test the loop peeling pass";
  }

  void runOnOperation() override {
    IRRewriter rewriter(getOperation());
    getOperation().walk([&](scf::ForOp forOp) {
      if (getPeelEpilogue(forOp)) {
        mlir::triton::peelLoopEpilogue(forOp);
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestLoopPeelingPass() { PassRegistration<TestLoopPeelingPass>(); }
} // namespace test
} // namespace mlir
