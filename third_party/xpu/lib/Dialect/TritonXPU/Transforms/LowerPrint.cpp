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

//===----------------------------------------------------------------------===//
// lower triton::PrintOp to triton::xpu::XPUPrintOp
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineMap.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUPRINT
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUPrint : public impl::TritonXPUPrintBase<TritonXPUPrint> {

  using impl::TritonXPUPrintBase<TritonXPUPrint>::TritonXPUPrintBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    m.walk([&](PrintOp printOp) {
      OpBuilder b(printOp);
      GetProgramIdOp pidX = b.create<GetProgramIdOp>(
          printOp->getLoc(),
          ProgramIDDimAttr::get(b.getContext(), ProgramIDDim(0)));
      GetProgramIdOp pidY = b.create<GetProgramIdOp>(
          printOp->getLoc(),
          ProgramIDDimAttr::get(b.getContext(), ProgramIDDim(1)));
      GetProgramIdOp pidZ = b.create<GetProgramIdOp>(
          printOp->getLoc(),
          ProgramIDDimAttr::get(b.getContext(), ProgramIDDim(2)));
      Value outer_idx =
          b.create<arith::ConstantIntOp>(printOp->getLoc(), 0, 64);
      Value inner_idx =
          b.create<arith::ConstantIntOp>(printOp->getLoc(), 0, 64);
      Value uc_idx = b.create<arith::ConstantIntOp>(printOp->getLoc(), 0, 64);
      Value inner_bound =
          b.create<arith::ConstantIntOp>(printOp->getLoc(), 1, 64);
      Value uc_bound = b.create<arith::ConstantIntOp>(printOp->getLoc(), 1, 64);

      XPUPrintOp xpuPrintOp = b.create<XPUPrintOp>(
          printOp->getLoc(), pidX.getResult(), pidY.getResult(),
          pidZ.getResult(), outer_idx, inner_idx, uc_idx, inner_bound, uc_bound,
          b.getStringAttr(printOp.getPrefix()), b.getBoolAttr(printOp.getHex()),
          printOp.getOperands());

      printOp.erase();
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
