// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
