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
        Option<"disableLoadVectorize", "disable-load-vectorize", "bool",
               /*default*/"false",
               "emit scalar global loads instead of vectorized ones; wide "
               "loads are load-issue-rate bound on this target, so read-bound "
               "kernels at high occupancy can be faster without them. Affects "
               "tt.load lowering only, not contiguity analysis, stores, "
               "atomics or SME async copies">,
    ];
}

def ILUVATARWarpSpecializeToLLVM : Pass<"convert-iluvatar-warp-specialize-to-llvm", "mlir::ModuleOp"> {
    let summary = "Lower ttg.warp_specialize to LLVM for the Iluvatar backend";
    let description = [{
        Ported from the NVIDIA `ConvertWarpSpecializeToLLVM` pass.
        Ivcore11 has no hardware named/partial barriers and no dynamic
        register reallocation (`setmaxnreg`). As a workaround:
          * structural warp-group entry/exit synchronization reuses the native whole-CTA barrier (`nvvm.barrier0`),
          * single-warp partition barriers are elided (lanes are lockstep),
          * multi-warp warp-group barriers are emulated with a shared-memory arrive/wait software barrier,
          * register reallocation is skipped.
    }];
    let constructor = "mlir::triton::createILUVATARWarpSpecializeToLLVMPass(\"\")";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "target device architecture, e.g., ivcore11">,
    ];
}

#endif
