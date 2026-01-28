#ifndef TRITONCOMMONGPU_CONVERSION_PASSES
#define TRITONCOMMONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateSharedMemory : Pass<"allocate-shared-memory", "mlir::ModuleOp"> {
    let summary = "Add metadata for shared memory allocation";
    let constructor = "mlir::triton::gpu::createAllocateSharedMemoryPass()";
}

def ConvertPipelineIntrinsicsToLLVM : Pass<"convert-pipeline-intrinsics-to-llvm", "mlir::ModuleOp"> {
    let summary = "Lower pipeline synchronization intrinsics to LLVM/NVVM";
    let description = [{
      Converts pipeline intrinsics (init, acquire, commit, wait, release, flush)
      and async copy operations to LLVM IR primitives. On NVIDIA GPUs, this maps
      to cp.async and barrier operations. Provides fallback for non-NVIDIA backends.
    }];
    let constructor = "mlir::triton::createPipelineIntrinsicsToLLVMPass()";
    let dependentDialects = [
      "mlir::LLVM::LLVMDialect",
      "mlir::NVVM::NVVMDialect"
    ];
}

#endif
