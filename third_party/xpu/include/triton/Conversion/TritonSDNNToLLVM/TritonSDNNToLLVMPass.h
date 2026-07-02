#ifndef TRITON_CONVERSION_TRITONSDNN_TO_LLVM_PASS_H
#define TRITON_CONVERSION_TRITONSDNN_TO_LLVM_PASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonSDNNToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonSDNNToLLVMPass(uint32_t xpu_arch);
} // namespace triton

} // namespace mlir

#endif
