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


def Tle_FlagCxBarrierOp : Tle_Op<"flagcx_barrier"> {
  let summary = "FlagCX communicator barrier";
  let description = [{
    Performs a communicator barrier by lowering to one of the transport-
    transparent FlagCX unified device IR calls:

      * flagcxDevBarrierArrive(comm, teamKind, index, contextId, coopKind,
                              order, scope)
      * flagcxDevBarrierWait(comm, teamKind, index, contextId, coopKind,
                            order, scope)
      * flagcxDevBarrierSync(comm, teamKind, index, contextId, coopKind,
                            order, scope)

    The specific runtime routine is selected according to the barrier operation
    (arrive, wait, or sync). `team_kind` selects the intra-node, inter-node rail,
    or world communicator team. `index` identifies the barrier channel and
    `context_id` selects the pre-created network context used by inter/world
    barriers.
  }];

  let arguments = (ins
    Tle_LocalPointerResultType:$comm,
    StrAttr:$barrier_type,
    Tle_FlagCXTeamKindAttr:$team_kind,
    Tle_FlagCXCoopKindAttr:$coop_kind,
    I32Attr:$index,
    I32Attr:$context_id,
    Tle_MemoryOrderAttr:$order,
    Tle_SyncScopeAttr:$scope
  );

  let hasVerifier = 1;
}


def Tle_DeviceIntraBarrierOp : Tle_Op<"device intra barrier",
                                     [MemoryEffects<[MemRead, MemWrite]>]> {
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

    Alternatively, `dynamic_index` supplies a scalar i32 slot index. In that
    case `index` must be absent and `barrier_count` declares the minimum runtime
    slot capacity. The caller must guarantee 0 <= dynamic_index < barrier_count
    and one cooperating group per chip and slot at each rendezvous.

    The runtime routines return void; this operation has no results.
  }];

  let arguments = (ins
    Tle_LocalPointerResultType:$comm,
    Optional<I32>:$dynamic_index,
    OptionalAttr<StrAttr>:$barrier_type,
    OptionalAttr<I32Attr>:$coop_kind,
    OptionalAttr<I32Attr>:$index,
    OptionalAttr<I1Attr>:$multimem,
    OptionalAttr<I32Attr>:$order,
    OptionalAttr<I32Attr>:$barrier_count
  );

  let hasVerifier = 1;
}


def Tle_FlagCxSignalOp : Tle_Op<"flagcx signal", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Atomically update a remote FlagCX signal slot";
  let description = [{
    Obtains a pre-allocated FlagCX network context from `comm`, then either
    increments the selected remote signal slot by one or adds `value` to it.
    This operation only sends a signal; it does not transfer user data or wait
    for the remote signal to reach a value.
  }];

  let arguments = (ins
    Tle_LocalPointerResultType:$comm,
    I32:$peer,
    I32:$slot_id,
    Optional<I64>:$value,
    Tle_SignalOpKindAttr:$signal_op,
    Tle_FlagCXTeamKindAttr:$team_kind,
    Tle_FlagCXCoopKindAttr:$coop_kind,
    I32Attr:$context_id,
    Tle_SyncScopeAttr:$scope
  );

  let results = (outs);

  let assemblyFormat = [{
    `(` $signal_op `,` $team_kind `,` $coop_kind `,` $scope `)` $comm `,` $peer `,` $slot_id (`,` $value^)?  attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = true;
}

def Tle_FlagCxSignalWaitOp : Tle_Op<"flagcx signal_wait"> {
  let summary = "Wait for a remote FlagCX signal slot to become the target value";
  let description = [{
    Obtains a pre-allocated FlagCX network context from `comm`, then spin-waits the selected
    signal slot to reach the target value (actual value >= target),
    or waits the signal to reach its shadow value.
  }];

  let arguments = (ins
    Tle_LocalPointerResultType:$comm,
    I32:$slot_id,
    Tle_SignalWaitKindAttr:$wait_kind,
    Optional<I64>:$target,
    Tle_FlagCXCoopKindAttr:$coop_kind,
    I32Attr:$context_id,
    Tle_MemoryOrderAttr:$order
  );

  let results = (outs);

  let assemblyFormat = [{
    `(` $wait_kind `,` $coop_kind `,` $order `)` $comm `,` $slot_id (`,` $target^)?  attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = true;
}
