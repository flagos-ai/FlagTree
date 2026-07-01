#ifndef TRITON_TO_TILEIR_CONVERSION_PASSES
#define TRITON_TO_TILEIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToCudaTile : Pass<"convert-triton-to-cuda-tile", "mlir::ModuleOp"> {
    let summary = "Convert Triton to cuda_tile/triton dialect";
    let description = [{
        A convert pass for convert the triton dialect into cuda_tile/triton dialect.
    }];
    let constructor = "mlir::triton::createConvertTritonToCudaTilePass()";
    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::cuda_tile::CudaTileDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::ub::UBDialect"
                             ];
    let options = [
    Option<"approxModifier", "approx-modifier", "bool",
           /*default=*/"false",
           "Set approx modifier on all the operations that support it in the module.">,
    Option<"flushToZeroModifier", "flush-to-zero-modifier", "bool",
          /*default=*/"false",
          "Set the flush to zero modifier on all the operations that support it in the module.">,
    Option<"computeCapability", "compute-capability", "int",
          /*default=*/"100",
          "Set the Compute Capability version that is supported in the module">,
    Option<"numCTAInCGA", "num-cta-in-cga", "int",
          /*default=*/"1",
          "number of CTA in CGA">,
    Option<"simtNumWarpsInCTA", "num-warps-in-cta", "int",
          /*default=*/"4",
          "number of SIMT warps in CTA">,
    Option<"occupancy", "occupancy", "int",
          /*default=*/"1",
          "number of ctas in one SM">,
    Option<"numStages", "num-stages", "int",
          /*default=*/"",
          "number of stages for the kernel">
    ];
}

#endif
