#ifndef ILUVATAR_TLE_FLAGCX_OPS
#define ILUVATAR_TLE_FLAGCX_OPS

#ifdef __ILUVATAR_TLE__
include "mlir/Interfaces/SideEffectInterfaces.td"
include "IluvatarTleDialect.td"
include "IluvatarTleAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

class IluvatarTle_Op<string mnemonic, list<Trait> traits = []>
    : Op<IluvatarTle_Dialect, mnemonic,
         !listconcat(traits, [VerifyTensorLayoutsTrait])> {}

def IluvatarTle_FlagCxCommType : AnyTypeOf<[TT_Tensor, TT_Ptr, TT_Int]>;

def IluvatarTle_GetLocalRankOp : IluvatarTle_Op<"get_local_rank"> {
  let summary = "Get local rank";
  let arguments = (ins
    IluvatarTle_FlagCxCommType:$src
  );
  let results = (outs I32:$result);
  let hasVerifier = 1;
}

def IluvatarTle_DeviceIntraBarrierOp : IluvatarTle_Op<"device_intra_barrier"> {
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
    IluvatarTle_FlagCxCommType:$comm,
    OptionalAttr<StrAttr>:$barrier_type,
    OptionalAttr<IluvatarTle_FlagCXCoopKindAttr>:$coop_kind,
    OptionalAttr<I32Attr>:$index,
    OptionalAttr<I1Attr>:$multimem,
    OptionalAttr<IluvatarTle_MemoryOrderAttr>:$order
  );

  let hasVerifier = 1;
}
#endif // __ILUVATAR_TLE__

#endif // ILUVATAR_TLE_FLAGCX_OPS
