#include "TritonHCUGPUToLLVM/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONHCUGPUSETFUNCINLINEATTRS
#include "TritonHCUGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct TritonHCUGPUSetFuncInlineAttrs
    : public mlir::triton::impl::TritonHCUGPUSetFuncInlineAttrsBase<
          TritonHCUGPUSetFuncInlineAttrs> {
public:
  TritonHCUGPUSetFuncInlineAttrs(StringRef arch)
      : TritonHCUGPUSetFuncInlineAttrsBase<TritonHCUGPUSetFuncInlineAttrs>() {
    this->arch = arch.str();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod->getContext();

    mod.walk([&](LLVM::LLVMFuncOp funcOp) {
      // Only touch non-kernel (internal linkage) helper functions.
      // Kernel entry points are handled separately in FuncOpToLLVM and
      // should not be affected here.
      if (funcOp.getLinkage() != LLVM::Linkage::Internal)
        return;

      if (auto noinlineAttr = funcOp->getAttrOfType<BoolAttr>("noinline")) {
        if (!noinlineAttr.getValue()) {
          funcOp.setPassthroughAttr(
              ArrayAttr::get(ctx, {StringAttr::get(ctx, "alwaysinline")}));
        }
      }
    });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Factory Function
//===----------------------------------------------------------------------===//
namespace mlir::triton::HCU {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonHCUGPUSetFuncInlineAttrsPass(StringRef targetArch) {
  return std::make_unique<TritonHCUGPUSetFuncInlineAttrs>(targetArch);
}

} // namespace mlir::triton::HCU
