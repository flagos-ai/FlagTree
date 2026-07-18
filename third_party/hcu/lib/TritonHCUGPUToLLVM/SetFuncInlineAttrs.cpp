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
