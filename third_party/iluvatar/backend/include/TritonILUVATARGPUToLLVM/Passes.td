#ifndef TRITONILUVATARGPU_CONVERSION_PASSES
#define TRITONILUVATARGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonILUVATARGPUToLLVM : Pass<"convert-triton-iluvatargpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let constructor = "mlir::triton::createConvertTritonILUVATARGPUToLLVMPass(\"\", /*ftz=*/true)";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "target device architecture, e.g., ivcore11">,
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

#endif
