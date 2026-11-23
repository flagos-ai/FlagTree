/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TILE_IR_TO_HIVM_CONVERSION_PASSES
#define TILE_IR_TO_HIVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def CommonIRToHIVM : Pass<"commonir-to-hivm", "mlir::ModuleOp"> {
  let summary = "Convert CommonIR dialect to HIVM dialect";
  let constructor = "triton::createCommonIRToHIVMPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "triton::tile::CommonIRDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::triton::TritonDialect",
  ];
}

def ReduceSumStrength : Pass<"reduce-sum-strength", "mlir::ModuleOp"> {
  let summary = "Reduce sum strength for profitable tl.sum-style add reductions";
  let description = [{
      This pass rewrites profitable add reductions by pre-adding two equal halves
      of the reduced dimension and reducing the half-sized tensor. It is owned by
      the Ascend ttir-to-ttadapter pipeline for backends where a full-width reduction
      is more expensive than an elementwise add plus a smaller reduction.
  }];
  let constructor = "triton::createReduceSumStrengthPass()";

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
      Option<"enable", "enable", "bool", /*default*/"true",
             "Enable the rewrite">,
      Option<"splitFactor", "split-factor", "int32_t", /*default*/"2",
             "Reduction-axis split factor; currently only 2 is supported">
  ];
}

#endif // TILE_IR_TO_HIVM_CONVERSION_PASSES
