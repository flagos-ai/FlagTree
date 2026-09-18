#ifndef TRITONMUSAGPU_CONVERSION_PASSES
#define TRITONMUSAGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonMUSAGPUToLLVM : Pass<"convert-triton-musagpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM for MUSA";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::musa::MUSADialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"31",
               "device compute capability">,
        Option<"enableFp8Burst2", "enable-fp8-burst2",
               "bool", /*default*/"false",
               "use burst2 (vector) fp8 conversion intrinsics">,
    ];
}

def AllocateMUSASharedMemory : Pass<"allocate-musa-shared-memory", "mlir::ModuleOp"> {
    let summary = "Allocate shared memory for MUSA";

    let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"31",
               "device compute capability">,
    ];
}

#endif
