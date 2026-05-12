#ifndef MTGPU_CONVERSION_PASSES
#define MTGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMTGPUToLLVM : Pass<"convert-mtgpu-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert MTGPU dialect to LLVM for MUSA";

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::musa::MUSADialect",
                           "mlir::triton::mtgpu::MTGPUDialect"];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"31",
           "device compute capability">,
  ];
}

#endif
