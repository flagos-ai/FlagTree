//===- KernelArgBufferPass.td ---------------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//

#ifndef KERNEL_ARG_BUFFER_PASS
#define KERNEL_ARG_BUFFER_PASS

include "mlir/Pass/PassBase.td"

def KernelArgBufferPass : Pass<"kernel-arg-buffer", "ModuleOp"> {
  let summary = "Convert kernel arguments to a single buffer argument";
  let description = [{
    This pass transforms kernel function signatures by converting multiple
    arguments into a single void* buffer containing all the arguments.

    For example, a function like:
      add_kernel(uint64_t* arg1, uint64_t* arg2, int64_t size, int gridX, int x)

    Will be converted to:
      add_kernel(void* args)

    Where the args buffer contains pointers to arg1 and arg2, followed by the scalar
    values size, gridX, and x. Each scalar value occupies 8 bytes in the buffer.
  }];
  let constructor = "mlir::triton::createKernelArgBufferPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::func::FuncDialect"];
}

#endif // KERNEL_ARG_BUFFER_PASS
