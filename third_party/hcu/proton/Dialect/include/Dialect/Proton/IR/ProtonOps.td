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

#ifndef PROTON_OPS
#define PROTON_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "proton/Dialect/include/Dialect/Proton/IR/ProtonDialect.td"
include "proton/Dialect/include/Dialect/Proton/IR/ProtonAttrDefs.td"

class PT_Op<string mnemonic, list<Trait> traits = []> :
  Op<Proton_Dialect, mnemonic, !listconcat(traits, [])> {
}

def PT_RecordOp : PT_Op<"record", [
  MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
]> {
  let summary = "Record an event";

  let description = [{
    This operation annotates a region of IR where events are recorded.
    Events can be classified as hardware or software events.
    Hardware events are provided by the hardware performance counters obtained in later passes that convert Triton to target-specific IR.
    Software events are provided by the user or the compiler.

    Example:

    ```mlir
    proton.record start "name0"
    ...
    proton.record end "name0"
    ```

    Scope names cannot be reused within the same function.
  }];
  let arguments = (
    ins UnitAttr: $isStart,
    StrAttr: $name
  );

  let assemblyFormat = "(`start` $isStart^):(`end`)? $name attr-dict";
}

#endif // PROTON_OPS
