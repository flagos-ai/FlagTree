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
