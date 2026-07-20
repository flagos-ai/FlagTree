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

#ifndef TRITON_TARGET_LLVMIR_PASSES
#define TRITON_TARGET_LLVMIR_PASSES

include "mlir/Pass/PassBase.td"

def LLVMDIScope: Pass<"enable-line-info", "mlir::ModuleOp"> {
  let summary = "Materialize LLVM line info";
  let description = [{
    This pass materializes line mapping information for LLVM IR dialect operations.
  }];
}

def LLVMDILocalVariable: Pass<"extract-variable-info", "mlir::ModuleOp"> {
  let summary = "Pull out source variable info from Location to DILocalVariable";
  let description = [{
    This pass pulled out source vararible's debuginfo from LLVM IR dialect's Location
      into LLVM's DILocalVariable and fused it into previous Location so it can be passed to LLVM IR later in debugging mode.
  }];
}

#endif
