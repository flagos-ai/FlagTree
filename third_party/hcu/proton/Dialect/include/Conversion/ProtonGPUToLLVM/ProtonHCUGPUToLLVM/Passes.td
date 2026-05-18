#ifndef PROTONHCUGPU_TO_LLVM_PASSES
#define PROTONHCUGPU_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertProtonHCUGPUToLLVM : Pass<"convert-proton-hcu-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ProtonGPU to LLVM";
    let description = [{
        Convert ProtonGPU to LLVM using HCU-specific lowering patterns.
    }];
    let constructor = "mlir::triton::proton::gpu::createConvertProtonHCUGPUToLLVMPass(\"\")";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::hcugpu::TritonHCUGPUDialect",
                             "mlir::triton::proton::ProtonDialect",
                             "mlir::triton::proton::gpu::ProtonGPUDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">
    ];
}

#endif // PROTONHCUGPU_TO_LLVM_PASSES
