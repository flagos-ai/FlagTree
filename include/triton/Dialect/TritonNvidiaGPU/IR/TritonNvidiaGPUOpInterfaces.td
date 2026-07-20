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

#ifndef TRITON_NVIDIAGPU_OP_INTERFACES
#define TRITON_NVIDIAGPU_OP_INTERFACES

include "mlir/IR/OpBase.td"

def MMAv5OpInterface : OpInterface<"MMAv5OpInterface"> {
  let description = [{
     This interface is implemented by MMAv5 dot and dot scaled ops.
  }];

  let cppNamespace = "::mlir::triton::nvidia_gpu";

  // We can add more methods as needed.
  let methods = [
    InterfaceMethod<"Return the A operand.",
                    "::mlir::TypedValue<::mlir::triton::gpu::MemDescType>",
                    "getA">,
    InterfaceMethod<"Return the accumulator init flag.",
                    "::mlir::Value",
                    "useAccumulator">,
    InterfaceMethod<"Set the accumulator init flag.",
                    "void",
                    "setUseAccumulator",
                    (ins "::mlir::Value":$flag)>,
    InterfaceMethod<"Associate a new completion barrier to this MMAv5 op.",
                    "void",
                    "addCompletionBarrier",
                    (ins "::mlir::Value":$barrier, "::mlir::Value":$pred)>,
    InterfaceMethod<"Return the accumulator.",
                    "::mlir::TypedValue<::mlir::triton::gpu::MemDescType>",
                    "getAccumulator">,
    InterfaceMethod<"Set the accumulator.",
                    "void",
                    "setAccumulator",
                    (ins "::mlir::Value":$accum)>,
    InterfaceMethod<"Return the predicate of this op.",
                    "::mlir::Value",
                    "getPredicate">,
    InterfaceMethod<"Set the predicate of this op.",
                    "void",
                    "setPredicate",
                    (ins "::mlir::Value":$pred)>,
    InterfaceMethod<"Get the memory dependencies of the accumulator.",
                    "::mlir::Value",
                    "getAccDep">,
    InterfaceMethod<"Get the mutable memory dependencies of the accumulator.",
                    "::mlir::MutableOperandRange",
                    "getAccDepMutable">,
    InterfaceMethod<"Get the produced write dependency of the accumulator.",
                    "::mlir::Value",
                    "getToken">,
    InterfaceMethod<"Indicate that this MMA op executes asynchronously.",
                    "void",
                    "setIsAsync",
                    (ins "bool":$isAsync)>,
    InterfaceMethod<"Return true if this MMA op executes asynchronously.",
                    "bool",
                    "isAsync">
  ];

  let verify = [{
    return ::mlir::triton::nvidia_gpu::impl::verifyMMAv5Op($_op);
  }];
}
#endif // TRITON_NVIDIAGPU_OP_INTERFACES
