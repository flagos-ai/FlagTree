// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TT2TTX_CONVERSION_PASSES
#define TT2TTX_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonXPU: Pass<"convert-triton-to-triton-xpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonXPU";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonToTritonXPUPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             // TODO: Does this pass depend on SCF?
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             // "mlir::triton::gpu::TritonGPUDialect", // Does this pass depend on TritonGPU in Triton 3.0?
                             "mlir::triton::xpu::TritonXPUDialect"];

   let options = [
       Option<"xpu_arch", "xpu_arch",
               "uint32_t", /*default*/"3",
               "XPU Architecture">,
       Option<"buffer_size", "buffer_size",
              "uint32_t", /*default*/"512",
              "bytes for local memory buffer">,
       Option<"core_num", "core_num",
              "uint32_t", /*default*/"64",
              "xpu spec core_num">
   ];
}


#endif // TT2TTX_CONVERSION_PASSES
