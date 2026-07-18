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

#ifndef NVWS_OP_INTERFACES
#define NVWS_OP_INTERFACES

include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"

def NVWS_DescriptorLoadOpInterface : OpInterface<"DescriptorLoadOpInterface", [TT_DescriptorOpInterface]> {
  let cppNamespace = "::mlir::triton::nvws";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the transaction counts",
      /*retType=*/"int",
      /*methodName=*/"getTxCount",
      /*args=*/(ins)>,
  ];
}

def NVWS_ArefStageInterface : OpInterface<"ArefStageInterface"> {
  let cppNamespace = "::mlir::triton::nvws";

  let description = [{
     This interface implements setStage/getStage for aref ops
  }];

  // We can add more methods as needed.
  let methods = [
    InterfaceMethod<"Return aref stage",
                    "::mlir::Value",
                    "getStage">,
    InterfaceMethod<"Set aref stage",
                    "void",
                    "setStage",
                    (ins "::mlir::Value":$stage)>,
  ];
}

#endif // NVWS_OP_INTERFACES
