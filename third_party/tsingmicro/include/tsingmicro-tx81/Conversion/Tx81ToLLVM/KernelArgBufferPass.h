//===- KernelArgBufferPass.h ----------------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//
//
// This pass transforms kernel function signatures by converting multiple
// arguments into a single void* buffer containing all the arguments.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_KERNEL_ARG_BUFFER_PASS_H
#define MLIR_KERNEL_ARG_BUFFER_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
class Pass;

namespace triton {
/// Creates a pass that transforms kernel functions by replacing multiple
/// arguments with a single void* buffer argument.
std::unique_ptr<Pass> createKernelArgBufferPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/KernelArgBufferPass.h.inc"

} // namespace triton
} // namespace mlir

#endif // MLIR_KERNEL_ARG_BUFFER_PASS_H
