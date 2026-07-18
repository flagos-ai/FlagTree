#ifndef TTX2LLVM_CONVERSION_PASSES
#define TTX2LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonXPUToLLVM: Pass<"convert-triton-xpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonXPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonXPUToLLVMPass()";

    let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

   let options = [
        Option<"xpu_arch", "xpu_arch",
               "uint32_t", /*default*/"3",
               "XPU Architecture">,
        Option<"buffer_size", "buffer_size",
               "uint32_t", /*default*/"128",
               "Buffer Size of LM">,
        Option<"isUseMaskZero", "is-use-mask-zero",
               "bool", /*default*/"0",
               "Whether Use Mask Zero Method">,
   ];
}

def AllocateXPUSharedMemory: Pass<"allocate-xpu-shared-memory", "mlir::ModuleOp"> {
    let summary = "Allocate shared memory for TritonXPU ops";
    let description = [{
      Attach allocation.offset and ttg.shared using XPU-specific scratch sizes.
    }];
    let constructor = "mlir::triton::createAllocateXPUSharedMemoryPass()";
    let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}


#endif // TTX2LLVM_CONVERSION_PASSES
