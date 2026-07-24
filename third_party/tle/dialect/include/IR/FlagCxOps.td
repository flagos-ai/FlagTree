/*
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

include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "tle/dialect/include/IR/TleDialect.td"
include "tle/dialect/include/IR/TleAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class Tle_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tle_Dialect, mnemonic,
         !listconcat(traits, [VerifyTensorLayoutsTrait])> {}

def Tle_LocalPointerResultType : AnyTypeOf<[TT_Tensor, TT_Ptr, TT_Int]>;

def Tle_GetLocalRankOp : Tle_Op<"get local rank"> {
  let summary = "Get local rank";
  let arguments = (ins
    Tle_LocalPointerResultType:$src
  );
  let results = (outs I32:$result);
  let hasVerifier = 1;
}


def Tle_DeviceIntraBarrierOp : Tle_Op<"device intra barrier"> {
  let summary = "Device-local synchronization barrier";
  let description = [{
    Performs a device-local synchronization barrier by lowering to one of the
    following runtime calls:

      * flagcxIntraBarrierArriveS(comm, coopKind, index, multimem, order)
      * flagcxIntraBarrierWaitS(comm, coopKind, index, multimem, order)
      * flagcxIntraBarrierSyncS(comm, coopKind, index, multimem, order)

    The specific runtime routine is selected according to the barrier operation
    (arrive, wait, or sync).

    The barrier is identified by `index`, scoped by `coopKind`, and obeys the
    specified memory ordering semantics (`order`). `multimem` indicates whether
    the barrier operates on multimem storage.

    Returns the runtime status code.
  }];

  let arguments = (ins
    Tle_LocalPointerResultType:$comm,
    OptionalAttr<StrAttr>:$barrier_type,
    OptionalAttr<I32Attr>:$coop_kind,
    OptionalAttr<I32Attr>:$index,
    OptionalAttr<I1Attr>:$multimem,
    OptionalAttr<I32Attr>:$order
  );

  let hasVerifier = 1;
}
